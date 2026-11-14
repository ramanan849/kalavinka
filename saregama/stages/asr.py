"""Stage 4a - automatic lyrics: transcribe the sung vocal when there is no lyric sheet.

The film-song pipeline needs the words before it can align them.  When a song
has no lyric sheet, this stage writes one from ASR:

1. **Phrases.** The separated vocal is cut at unvoiced gaps (from the F0 track)
   into sung phrases; short ones merge into a neighbour and long ones are split
   at their longest internal pause.  Each phrase becomes one lyric line, so the
   aligner's line structure (and its star token between lines) follows the song.
2. **Transcription.** Each phrase, with a little context, goes through an ASR
   backend in batches:

   ``qwen3``    Qwen3-ASR (Alibaba, Apache-2.0).  The default: it is trained on
                singing and songs with accompaniment, and on Hindi.  Its forced
                aligner does not cover Hindi, so timing comes from the MMS_FA
                stage (``stages.align``), not from Qwen.
   ``whisper``  any Hugging Face Whisper checkpoint (``openai/whisper-large-v3``,
                or a Hinglish fine-tune such as
                ``Oriserve/Whisper-Hindi2Hinglish-Prime``, which writes Roman
                Hinglish).  Can also return Whisper's own word timestamps.

3. **Clean-up.** Hallucinations typical of ASR on music are removed: text far
   too fast for the phrase, n-gram loops, subtitle boiler-plate.  Gurmukhi
   (Punjabi lines in Hindi films) is transliterated to Devanagari, Perso-Arabic
   script is dropped (the G2P cannot read it), and the result is NFC.

The transcript is written next to the song (``asr.json`` with phrase times and
``lyrics.asr.txt`` in lyric-sheet format) and used as the lyric sheet.  ASR on
ornamented singing is not perfect, so the pipeline flags such songs
``lyrics_asr_unverified``; correct ``lyrics.asr.txt``, point songs.csv at it and
re-run to turn the song into training data.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..audio import SR, resample
from ..config import AsrCfg

ASR_SR = 16000
DEFAULT_MODELS = {"qwen3": "Qwen/Qwen3-ASR-1.7B", "whisper": "openai/whisper-large-v3"}
# Qwen3-ASR takes language names, Whisper takes codes / names it knows
QWEN_LANG = {"hi": "Hindi", "en": "English", "ur": "Urdu", "mr": "Marathi", "ta": "Tamil",
             "pa": "Punjabi", "bn": "Bengali"}


@dataclass
class AsrPhrase:
    """One transcribed phrase: time span (s), cleaned text, raw model text and
    optional model word timestamps ``[{"text", "start", "end"}]`` in song time."""
    start: float
    end: float
    text: str
    raw: str = ""
    words: List[Dict] = field(default_factory=list)
    dropped: str = ""            # why the text was discarded, if it was


# ---------------------------------------------------------------------------
# phrase planning
# ---------------------------------------------------------------------------
def plan_phrases(voiced: np.ndarray, fps: float, cfg: AsrCfg) -> List[Tuple[float, float]]:
    """Sung phrases as (start, end) seconds from a voicing mask on a ``fps`` grid.

    Voiced runs closer than ``min_gap_s`` are one phrase; phrases shorter than
    ``min_phrase_s`` join the nearer neighbour when the gap is under 1.5 s;
    phrases longer than ``max_phrase_s`` are split at their longest pause.
    """
    from .align import _phrases
    runs = [list(r) for r in _phrases(voiced, min_gap=max(1, int(cfg.min_gap_s * fps)),
                                      min_len=max(1, int(0.15 * fps)))]
    min_len, max_gap = cfg.min_phrase_s * fps, 1.5 * fps
    k = 0
    while k < len(runs):                       # merge short phrases into a neighbour
        a, b = runs[k]
        if b - a < min_len and len(runs) > 1:
            gl = a - runs[k - 1][1] if k > 0 else np.inf
            gr = runs[k + 1][0] - b if k + 1 < len(runs) else np.inf
            if min(gl, gr) <= max_gap:
                if gl <= gr:
                    runs[k - 1][1] = b
                else:
                    runs[k + 1][0] = a
                runs.pop(k)
                continue
        k += 1
    out: List[Tuple[int, int]] = []
    max_len = int(cfg.max_phrase_s * fps)
    stack = [tuple(r) for r in runs][::-1]
    while stack:                               # split over-long phrases at the longest pause
        a, b = stack.pop()
        if b - a <= max_len:
            out.append((a, b))
            continue
        seg = voiced[a:b]
        best, best_len, k = None, 0, int(0.2 * (b - a))
        while k < int(0.8 * (b - a)):          # search the middle 60 % only
            if not seg[k]:
                j = k
                while j < len(seg) and not seg[j]:
                    j += 1
                if j - k > best_len:
                    best, best_len = (k + j) // 2, j - k
                k = j
            else:
                k += 1
        cut = a + (best if best is not None else (b - a) // 2)
        stack.extend([(cut, b), (a, cut)])
    return [(a / fps, b / fps) for a, b in out]


# ---------------------------------------------------------------------------
# text clean-up
# ---------------------------------------------------------------------------
_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_GURMUKHI = re.compile(r"[਀-੿]")
_BRACKETS = re.compile(r"[\[\(\{<][^\]\)\}>]*[\]\)\}>]")
_KEEP = re.compile(r"[^A-Za-z\s'ऀ-ॣ०-ॿ]")     # Latin + Devanagari, no dandas
_BOILERPLATE = ("subscribe", "सब्सक्राइब", "धन्यवाद", "thanks for watching", "amara.org",
                "subtitles", "उपशीर्षक", "copyright", "♪")


def gurmukhi_to_devanagari(text: str) -> str:
    """Transliterate Gurmukhi letters to Devanagari (Punjabi lines in Hindi films)."""
    if not _GURMUKHI.search(text):
        return text
    from indic_transliteration import sanscript  # type: ignore
    return sanscript.transliterate(text, sanscript.GURMUKHI, sanscript.DEVANAGARI)


def collapse_loops(words: List[str], max_repeat: int = 4) -> List[str]:
    """Cut n-gram loops (n = 4..1) repeated more than ``max_repeat`` times in a
    row down to ``max_repeat`` copies: the classic decoder hallucination.
    Songs repeat words too ("na na na"), hence the generous default."""
    out = list(words)
    for n in (4, 3, 2, 1):
        res, k = [], 0
        while k < len(out):
            gram = out[k:k + n]
            reps = 1
            if len(gram) == n:
                while out[k + reps * n:k + (reps + 1) * n] == gram:
                    reps += 1
            if reps > max_repeat:
                res.extend(gram * max_repeat)
                k += reps * n
            else:
                res.append(out[k])
                k += 1
        out = res
    return out


def clean_text(raw: str, dur_s: float, max_chars_per_s: float) -> Tuple[str, str]:
    """Cleaned lyric line and, if it must be discarded, the reason ("" if kept)."""
    t = unicodedata.normalize("NFC", raw or "").strip()
    low = t.lower()
    if any(b in low for b in _BOILERPLATE):
        return "", "boilerplate"
    t = gurmukhi_to_devanagari(_BRACKETS.sub(" ", t))
    had_arabic = bool(_ARABIC.search(t))
    t = _ARABIC.sub(" ", t)
    t = _KEEP.sub(" ", t.replace("’", "'"))
    t = re.sub(r"[\d_]+", " ", t)
    words = collapse_loops(t.split())
    t = " ".join(w.lower() if w.isascii() else w for w in words)
    if not t:
        return "", "perso_arabic_script" if had_arabic else "empty"
    letters = len(re.sub(r"\s|[ा-्ंःँ]", "", t))   # vowel signs don't count
    if letters > max_chars_per_s * max(dur_s, 0.5):
        return "", f"too_fast_{letters / max(dur_s, 0.5):.0f}cps"
    return t, ""


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
class AsrBackend:
    """Transcribes a batch of 16 kHz mono clips; ``words`` asks for model word
    timestamps (seconds relative to each clip) where the backend has them."""
    name = "base"
    has_word_times = False

    def __call__(self, clips: Sequence[np.ndarray], lang: str, words: bool = False
                 ) -> List[Tuple[str, List[Dict]]]:
        """[(text, word timestamps)] per clip; implemented by subclasses."""
        raise NotImplementedError


class Qwen3Asr(AsrBackend):
    """Qwen3-ASR through the ``qwen-asr`` package (transformers backend)."""
    name = "qwen3"

    def __init__(self, cfg: AsrCfg):
        """Load ``cfg.model`` (default Qwen/Qwen3-ASR-1.7B) in bfloat16 on ``cfg.device``."""
        import torch
        from qwen_asr import Qwen3ASRModel  # type: ignore
        self.cfg = cfg
        self.model_id = cfg.model or DEFAULT_MODELS["qwen3"]
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=cfg.device,
            max_inference_batch_size=cfg.batch_size, max_new_tokens=256)

    def __call__(self, clips, lang, words=False):
        """Transcribe ``clips``; Qwen3-ASR's aligner has no Hindi, so no word times."""
        language = QWEN_LANG.get(lang, lang) if lang else None
        res = self.model.transcribe(audio=[(c, ASR_SR) for c in clips],
                                    language=[language] * len(clips) if language else None)
        return [(r.text or "", []) for r in res]


class WhisperAsr(AsrBackend):
    """A Hugging Face Whisper checkpoint through the transformers ASR pipeline."""
    name = "whisper"
    has_word_times = True

    def __init__(self, cfg: AsrCfg):
        """Load ``cfg.model`` (default openai/whisper-large-v3) in float16."""
        import torch
        from transformers import pipeline  # type: ignore
        self.cfg = cfg
        self.model_id = cfg.model or DEFAULT_MODELS["whisper"]
        self.pipe = pipeline("automatic-speech-recognition", model=self.model_id,
                             dtype=torch.float16, device=cfg.device)
        tok = self.pipe.tokenizer
        if tok is not None and tok.pad_token_id is None:     # some fine-tunes ship without one;
            tok.pad_token = tok.eos_token                    # batching needs it
        # word timestamps need the cross-attention alignment heads, which many
        # fine-tunes (e.g. the Hinglish ones) do not ship
        self.has_word_times = bool(getattr(self.pipe.model.generation_config, "alignment_heads", None))

    def __call__(self, clips, lang, words=False):
        """Transcribe ``clips`` (each under 30 s); ``words`` adds cross-attention
        word timestamps when the checkpoint supports them.  Beam search curbs loops;
        batches are smaller with word timestamps (attention maps use memory)."""
        words = words and self.has_word_times
        # word timestamps keep cross-attention maps for every beam and token
        # (tens of GB for large-v3), so they decode greedily
        gen = {"task": "transcribe", "num_beams": 1 if words else 5}
        if lang:
            gen["language"] = lang
        inputs = [{"raw": c.astype(np.float32), "sampling_rate": ASR_SR} for c in clips]
        bs = max(1, self.cfg.batch_size // 4) if words else self.cfg.batch_size
        out = self.pipe(inputs, batch_size=bs, generate_kwargs=gen,
                        return_timestamps="word" if words else False)
        res = []
        for o in out:
            wt = [{"text": c["text"].strip(), "start": c["timestamp"][0], "end": c["timestamp"][1]}
                  for c in o.get("chunks", [])] if words else []
            res.append((o["text"], wt))
        return res


def load_backend(cfg: AsrCfg) -> AsrBackend:
    """The ASR backend named by ``cfg.backend``."""
    if cfg.backend == "qwen3":
        return Qwen3Asr(cfg)
    if cfg.backend == "whisper":
        return WhisperAsr(cfg)
    raise ValueError(f"unknown asr backend {cfg.backend!r} (qwen3 | whisper)")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def transcribe_song(audio: np.ndarray, sr: int, voiced: np.ndarray, fps: float,
                    backend: AsrBackend, cfg: AsrCfg, words: bool = False) -> List[AsrPhrase]:
    """Transcribe ``audio`` phrase by phrase (phrases from the ``voiced`` mask).

    Returns every phrase, including discarded ones (``dropped`` says why), so the
    transcript can be audited.  Word timestamps, when asked for, are in song time.
    """
    spans = plan_phrases(voiced, fps, cfg)
    x16 = resample(audio, sr, ASR_SR) if sr != ASR_SR else audio
    clips, offs = [], []
    for a, b in spans:
        lo = max(0, int((a - cfg.pad_s) * ASR_SR))
        hi = min(len(x16), int((b + cfg.pad_s) * ASR_SR))
        clips.append(x16[lo:hi].astype(np.float32))
        offs.append(lo / ASR_SR)
    raw = backend(clips, cfg.language, words=words) if clips else []
    out = []
    for (a, b), off, (txt, wt) in zip(spans, offs, raw):
        text, why = clean_text(txt, b - a, cfg.max_chars_per_s)
        wt = [{"text": w["text"], "start": round(off + (w["start"] or 0.0), 3),
               "end": round(off + (w["end"] if w["end"] is not None else (w["start"] or 0.0)), 3)}
              for w in wt]
        out.append(AsrPhrase(round(a, 3), round(b, 3), text, (txt or "").strip(), wt, why))
    return out


def write_transcript(phrases: List[AsrPhrase], song_dir: str | Path, model: str) -> List[str]:
    """Write ``asr.json`` (all phrases with times) and ``lyrics.asr.txt`` (the kept
    lines, lyric-sheet format) into ``song_dir``; return the kept lines."""
    d = Path(song_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "asr.json").write_text(json.dumps({"model": model, "phrases": [asdict(p) for p in phrases]},
                                           ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [p.text for p in phrases if p.text]
    header = (f"# Lyrics transcribed automatically by {model} (one line per sung phrase).\n"
              "# Correct every line, then set this file as the song's `lyrics` in songs.csv\n"
              "# and re-run with --force to align the corrected words.\n")
    (d / "lyrics.asr.txt").write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return lines


def load_transcript(song_dir: str | Path) -> Optional[Tuple[str, List[AsrPhrase]]]:
    """(model, phrases) from ``asr.json`` in ``song_dir``, or None."""
    p = Path(song_dir) / "asr.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["model"], [AsrPhrase(**x) for x in d["phrases"]]
