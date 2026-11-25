"""KalavinkaSinger: the one object the notebook, the command line and the web UI all use.

What it does: loads the final Kalavinka model (SoulX-Singer + our LoRA and Hindi phone rows, run L4 = "V4-L"),
and sings

* **new lyrics** (Devanagari) to the tune of a bundled held-out song line, in a bundled voice, or
* a bundled line with its **original lyrics** (``resing``), next to the real singer for comparison.

Two pitch modes:

* ``"score"`` (default; best in our listening test): SoulX gets the notes and makes its own pitch;
* ``"expressive"``: the two-stage system. Our stage-1 contour model (C-flow, temperature 0.85) draws a
  vibrato-rich pitch curve from the notes and SoulX sings that curve (melody mode).

It runs on a **bundle directory** made by ``kalavinka.deploy.build_bundle`` (code under ``vendor/``, our weights
under ``weights/``, tunes and voices under ``data/``). SoulX's own 2.7 GB ``model.pt`` is not in the bundle: on first
use it is downloaded once from Hugging Face (``Soul-AILab/SoulX-Singer``) into
``vendor/SoulX-Singer/pretrained_models/SoulX-Singer/``.

Device: ``"auto"`` picks CUDA, then Apple's MPS, then CPU. If a step fails on MPS (an operator PyTorch does not support
there yet) the singer moves to CPU and retries, so it never just crashes on a Mac.

Usage::

    from kalavinka_app.singer import KalavinkaSinger
    s = KalavinkaSinger("/path/to/bundle")
    wav, sr, info = s.sing("ठंडी हवा में गीत नया")              # auto-picked tune, the tune's own voice
    wav, sr, info = s.sing("...", tune=3, voice=7, mode="expressive")
    out = s.resing(3)                                          # {"model": (wav, sr), "real": (wav, sr), ...}
    wav, sr, info = s.sing("...", model="base")                # the released SoulX-Singer, for comparison

``model="base"`` is our zero-shot baseline (V0): the released SoulX-Singer with every Hindi phone mapped to its
nearest English phone. It is loaded on first use (another ~2 GB of memory) and works with every call.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HF_REPO = "Soul-AILab/SoulX-Singer"
SR = 24000
TAU = 0.85            # C-flow temperature chosen on the validation split
FLOW_STEPS = 10
GAP_S = 0.3           # silence between lines when several lines are sung


class SingError(ValueError):
    """A problem with the request (not a crash): bad script, line too long, unknown tune or voice."""


def use_soundfile_loader() -> None:
    """Make ``torchaudio.load`` read wav files with ``soundfile``. New torchaudio (>= 2.9) needs the extra
    TorchCodec + FFmpeg packages for ``load``; our clips are plain wav, so soundfile is enough everywhere."""
    import soundfile as sf
    import torch
    import torchaudio

    def _load(path, *args, **kwargs):
        """(channels x samples float tensor, sample rate), like torchaudio.load."""
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr
    torchaudio.load = _load


def pick_device(want: str = "auto") -> str:
    """``auto`` -> cuda, else mps, else cpu; anything else is returned unchanged."""
    if want != "auto":
        return want
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_devanagari(s: str) -> bool:
    """True if the text has at least one Devanagari letter and no Latin letters."""
    has_dev = any("ऀ" <= ch <= "ॿ" for ch in s)
    has_latin = any(("a" <= ch.lower() <= "z") for ch in s)
    return has_dev and not has_latin


def span_hash(sylls: Sequence[str]) -> bytes:
    """8-byte hash of a run of syllable texts (for the novelty check; the lyrics themselves are not shipped)."""
    return hashlib.blake2b("".join(sylls).encode("utf-8"), digest_size=8).digest()


SPAN = 4              # novelty: share of the line's 4-syllable phrases that also occur in the training lyrics


def split_lines(lyrics: str) -> List[str]:
    """One phrase per line (newline or ``|`` separates); empty lines dropped."""
    return [x.strip() for x in lyrics.replace("|", "\n").splitlines() if x.strip()]


def base_weights_path(bundle: str | Path) -> Path:
    """Where SoulX's model.pt lives inside a bundle."""
    return Path(bundle).expanduser().resolve() / "vendor" / "SoulX-Singer" / "pretrained_models" / "SoulX-Singer" / "model.pt"


def fetch_base_weights(bundle: str | Path, log=print) -> Path:
    """Download SoulX's model.pt (2.7 GB) into the bundle once; returns its path."""
    path = base_weights_path(bundle)
    if path.exists() and path.stat().st_size > 1e9:
        return path
    log("[kalavinka] downloading SoulX-Singer base weights (2.7 GB, only the first time) ...")
    from huggingface_hub import hf_hub_download
    path.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(HF_REPO, "model.pt", local_dir=str(path.parent))
    if Path(got).resolve() != path.resolve():
        os.replace(got, path)
    log("[kalavinka] base weights ready")
    return path


def looks_like_mps_error(e: BaseException) -> bool:
    """True for errors that mean "this operation does not run on the Mac GPU" (then the CPU is used), not for bugs
    in our own code (those are raised as they are)."""
    msg = str(e).lower()
    return isinstance(e, NotImplementedError) or "mps" in msg or "not implemented" in msg or "not supported" in msg \
        or "metal" in msg or "placeholder storage" in msg


def default_steps(device: str) -> int:
    """Flow steps: 32 on a GPU; 16 on CPU (twice as fast, slightly rougher)."""
    return 16 if device == "cpu" else 32


class KalavinkaSinger:
    """Load once, then call ``sing`` / ``resing`` as often as you like (see the module doc)."""

    def __init__(self, bundle: str | Path, device: str = "auto", verbose: bool = True):
        """Set up paths, fetch the base weights if needed, and load the model."""
        self.bundle = Path(bundle).expanduser().resolve()
        self.verbose = verbose
        soulx = self.bundle / "vendor" / "SoulX-Singer"
        if (self.bundle / "vendor").is_dir():
            os.environ["SAREGAMA_SOULX"] = str(soulx)
            sys.path.insert(0, str(self.bundle / "vendor"))
            sys.path.insert(0, str(soulx))
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        data = self.bundle / "data"
        self.table = data / "phone_set_indic.json"
        self._tunes = [json.loads(l) for l in (data / "tunes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        self._voices = [json.loads(l) for l in (data / "voices.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        nov = data / "novelty.bin"
        raw = nov.read_bytes() if nov.exists() else b""
        self._seen = {raw[i:i + 8] for i in range(0, len(raw), 8)}
        fetch_base_weights(self.bundle, log=lambda m: self.log(m.replace("[kalavinka] ", "")))
        use_soundfile_loader()
        self.device = pick_device(device)
        self._lock = threading.RLock()        # one render at a time (web page users, tabs, threads share the model)
        self.steps = default_steps(self.device)
        try:
            self._load(self.device)
        except Exception as e:                  # the model could not be built on the Mac GPU: use the CPU instead
            if self.device != "mps":
                raise
            self.log(f"could not load on the Mac GPU ({str(e).splitlines()[0][:120]}); using the CPU")
            self._load("cpu")
            self.steps = default_steps("cpu")
        self._cflow = None
        self._base = None

    # ---------------------------------------------------------------- setup
    def log(self, msg: str) -> None:
        """Print a progress line (unless ``verbose=False``)."""
        if self.verbose:
            print(f"[kalavinka] {msg}", flush=True)

    def _load(self, device: str) -> None:
        """Build the fine-tuned model (V4-L) and the data processor on ``device``."""
        import torch
        from kalavinka.eval.render import HOP, load_system
        from kalavinka.sx.adapter import IndicDataProcessor
        t0 = time.time()
        self.log(f"loading the Kalavinka model on {device} ...")
        self.model, _ = load_system(str(self.bundle / "weights" / "kalavinka_v4l"), self.table, device)
        self.proc = IndicDataProcessor(hop_size=HOP, sample_rate=SR, device=device, phoneset_path=str(self.table))
        self.device = device
        torch.set_grad_enabled(False)
        self.log(f"ready in {time.time() - t0:.0f} s")

    def _to_cpu(self) -> None:
        """Fall back to CPU (after an MPS failure)."""
        self.log("this step is not supported on the Mac GPU (MPS) yet; switching to CPU")
        self.model.to("cpu")
        if hasattr(self.model, "vocoder") and self.model.vocoder is not None:
            self.model.vocoder.to("cpu")
        from kalavinka.eval.render import HOP
        from kalavinka.sx.adapter import IndicDataProcessor
        self.proc = IndicDataProcessor(hop_size=HOP, sample_rate=SR, device="cpu", phoneset_path=str(self.table))
        if self._base is not None:
            from kalavinka.sx.adapter import UPSTREAM_PHONES
            self._base[0].to("cpu")
            self._base = (self._base[0], IndicDataProcessor(hop_size=HOP, sample_rate=SR, device="cpu",
                                                            phoneset_path=str(UPSTREAM_PHONES)))
        if self.steps == default_steps("mps"):
            self.steps = default_steps("cpu")
            self.log(f"using {self.steps} steps on the CPU (pass --steps to choose)")
        self.device = "cpu"

    def system(self, model: str = "kalavinka"):
        """(model, data processor, remap_english) for ``kalavinka`` (V4-L) or ``base`` (zero-shot SoulX, loaded lazily)."""
        if model == "kalavinka":
            return self.model, self.proc, False
        if model != "base":
            raise SingError("model must be 'kalavinka' or 'base'")
        if self._base is None:
            from kalavinka.eval.render import HOP, load_system
            from kalavinka.sx.adapter import UPSTREAM_PHONES, IndicDataProcessor
            t0 = time.time()
            self.log(f"loading the base SoulX-Singer (zero-shot, Hindi phones -> nearest English) on {self.device} ...")
            m, _ = load_system("zero-shot", self.table, self.device)
            self._base = (m, IndicDataProcessor(hop_size=HOP, sample_rate=SR, device=self.device,
                                                phoneset_path=str(UPSTREAM_PHONES)))
            self.log(f"base model ready in {time.time() - t0:.0f} s")
        return self._base[0], self._base[1], True

    def _contour_model(self):
        """The stage-1 C-flow model (loaded on first use; small, so it always runs on CPU)."""
        if self._cflow is None:
            import torch
            from kalavinka.ir.model import ContourNet
            ck = torch.load(self.bundle / "weights" / "contour_cflow" / "best.pt", map_location="cpu")
            a = ck["args"]
            m = ContourNet("flow", int(a["d"]), int(a["layers"]), int(a["heads"]))
            m.load_state_dict(ck["model"])
            self._cflow = m.eval()
        return self._cflow

    # ---------------------------------------------------------------- library
    def tunes(self) -> List[Dict]:
        """The bundled tunes: number, song, year, syllables, seconds, original lyric."""
        return [{k: t[k] for k in ("no", "song", "year", "syllables", "seconds", "lyric", "voice")} for t in self._tunes]

    def voices(self) -> List[Dict]:
        """The bundled voices: number, description, and the song the voice clip comes from."""
        return [{k: v[k] for k in ("no", "name", "song", "year", "register")} for v in self._voices]

    def _tune(self, no: int) -> Dict:
        """Tune record by number (1-based)."""
        for t in self._tunes:
            if t["no"] == int(no):
                return t
        raise SingError(f"there is no tune {no}; pick 1 to {len(self._tunes)} (see the tune list)")

    def _voice(self, no: int) -> Dict:
        """Voice record by number (1-based)."""
        for v in self._voices:
            if v["no"] == int(no):
                return v
        raise SingError(f"there is no voice {no}; pick 1 to {len(self._voices)} (see the voice list)")

    # ---------------------------------------------------------------- text
    def syllables(self, line: str) -> List[Dict]:
        """Devanagari line -> syllables [{text, phones}] with the same G2P that built the dataset."""
        if not is_devanagari(line):
            raise SingError("please type the lyrics in Hindi (Devanagari script), e.g. ठंडी हवा में गीत नया. "
                            "Tip: Google Input Tools or your phone's Hindi keyboard can convert Hinglish.")
        from saregama.g2p import line_to_syllables
        known = {x[3:] for x in json.loads(self.table.read_text(encoding="utf-8")) if x.startswith("in_")}
        try:
            sylls = [{"text": s.text, "phones": list(s.phones)} for s in line_to_syllables(line, "hi")]
        except ValueError as e:
            raise SingError(f"cannot read the lyrics: {e}") from e
        bad = sorted({p for s in sylls for p in s["phones"] if p not in known})
        if bad:
            raise SingError(f"the model has no sound for {bad} in: {line}")
        if not sylls:
            raise SingError("no singable syllables found in the lyrics")
        return sylls

    def novelty(self, sylls: Sequence[Dict]) -> float:
        """Share of the line's 4-syllable phrases that also appear in the training lyrics (0 = completely new)."""
        texts = [s["text"] for s in sylls]
        spans = [texts[i:i + SPAN] for i in range(max(1, len(texts) - SPAN + 1))]
        return sum(span_hash(sp) in self._seen for sp in spans) / len(spans) if self._seen else 0.0

    def auto_tune(self, n_syll: int, seed: int = 0) -> Dict:
        """A tune that fits ``n_syll`` syllables: closest fit first (n..n+4 syllables), seeded choice among those."""
        fits = [t for t in self._tunes if n_syll <= t["syllables"] <= n_syll + 4]
        if not fits:
            longer = sorted((t for t in self._tunes if t["syllables"] >= n_syll), key=lambda t: t["syllables"])
            if not longer:
                most = max(t["syllables"] for t in self._tunes)
                raise SingError(f"this line has {n_syll} syllables but the longest tune has {most}. "
                                "Split it into shorter lines (one per line, or separated by |).")
            fits = longer[:1]
        return random.Random(seed).choice(fits)

    # ---------------------------------------------------------------- singing
    def _render(self, target: Dict, voice: Dict, f0: Optional[List[float]], seed: int, shift: bool,
                model: str = "kalavinka") -> np.ndarray:
        """One phrase. ``f0=None``: score mode (SoulX makes its own pitch); else melody mode following ``f0``.
        ``model``: kalavinka or base. Runs SoulX with an MPS -> CPU fallback."""
        import torch
        from kalavinka.eval.render import side
        with self._lock:
            return self._render_locked(target, voice, f0, seed, shift, model, torch, side)

    def _render_locked(self, target, voice, f0, seed, shift, model, torch, side) -> np.ndarray:
        """``_render`` under the lock; moves to the CPU once if the Mac GPU cannot run a step."""
        for attempt in (0, 1):
            try:
                net, proc, remap = self.system(model)
                meta = {"prompt": side(proc, voice["item"], self.bundle / "data" / voice["item"]["wav"], remap, None, self.device),
                        "target": side(proc, target, None, remap, f0, self.device)}
                torch.manual_seed(seed)
                wav = net.infer(meta, auto_shift=shift, pitch_shift=0, n_steps=self.steps, cfg=3.0,
                                       control="score" if f0 is None else "melody")
                return wav[0].float().cpu().numpy().reshape(-1)
            except (RuntimeError, NotImplementedError, TypeError) as e:
                if self.device != "mps" or attempt or not looks_like_mps_error(e):
                    raise
                self.log(f"MPS error: {str(e).splitlines()[0][:120]}")
                self._to_cpu()
        raise RuntimeError("unreachable")

    def contour(self, item: Dict, seed: int = 0) -> List[float]:
        """Stage 1: an expressive F0 curve (Hz, 50 fps) for an item's notes, sampled by C-flow at tau = 0.85."""
        import torch
        from kalavinka.ir.features import batchify, scaffold, to_f0
        sc = scaffold(item)
        arrs, mask = batchify([sc])
        b = {k: torch.from_numpy(v) for k, v in arrs.items()}
        gen = torch.Generator().manual_seed(seed)
        r, v = self._contour_model().generate(b, torch.from_numpy(mask), steps=FLOW_STEPS, gen=gen, temperature=TAU)
        n = len(sc["f0"])
        return [round(float(x), 1) for x in to_f0(sc["flat_hz"], r[0, :n].numpy(), v[0, :n].numpy())]

    def sing(self, lyrics: str, tune: Optional[int] = None, voice: Optional[int] = None, mode: str = "score",
             steps: Optional[int] = None, seed: int = 0, model: str = "kalavinka") -> Tuple[np.ndarray, int, Dict]:
        """Sing new lyrics (``model``: kalavinka or base). One voice sings every line: ``voice``, or else the first
        line's tune singer (later lines on other tunes are transposed into that voice's range). Several lines (newline or ``|``) are sung one after another, each on its own tune
        (consecutive tunes if ``tune`` is given). Returns (waveform float32, 24000, info)."""
        from kalavinka.eval.sing_new import set_lyrics
        if mode not in ("score", "expressive"):
            raise SingError("mode must be 'score' (model's own pitch) or 'expressive' (two-stage)")
        lines = split_lines(lyrics)
        if not lines:
            raise SingError("type some lyrics first")
        if tune is not None and not 1 <= int(tune) <= len(self._tunes):
            raise SingError(f"there is no tune {tune}; pick 1 to {len(self._tunes)} (see the tune list)")
        self.steps = int(steps or default_steps(self.device))
        t0 = time.time()
        pieces, info = [], {"lines": [], "mode": mode, "model": model, "device": self.device}
        for k, line in enumerate(lines):
            sylls = self.syllables(line)
            if tune is not None:
                tu = self._tune((int(tune) - 1 + k) % len(self._tunes) + 1)
                if tu["syllables"] < len(sylls):
                    raise SingError(f"tune {tu['no']} has {tu['syllables']} syllables but \"{line}\" needs {len(sylls)}. "
                                    "Pick a longer tune, leave the tune on auto, or shorten the line.")
            else:
                tu = self.auto_tune(len(sylls), seed + k)
            if k == 0:
                # one singer for the whole song: the chosen voice, else the first line's tune singer
                song_voice = self._voice(voice) if voice is not None else self._voice(tu["voice"])
            vo = song_voice
            target = set_lyrics(tu["item"], sylls)
            seed_k = zlib.crc32(f"{line}|{tu['no']}|{seed}".encode()) % (2 ** 31)
            f0 = self.contour(target, seed_k) if mode == "expressive" else None
            pieces.append(self._render(target, vo, f0, seed_k, shift=vo["no"] != tu["voice"], model=model))
            info["lines"].append({"line": line, "syllables": " ".join(s["text"] for s in sylls), "tune": tu["no"],
                                  "tune_song": f"{tu['song']} ({tu['year']})", "voice": vo["no"],
                                  "seen_in_training": round(self.novelty(sylls), 2),
                                  "seconds": round(len(pieces[-1]) / SR, 2)})
        gap = np.zeros(int(GAP_S * SR), np.float32)
        wav = np.concatenate([x for p in pieces for x in (p, gap)][:-1]).astype(np.float32)
        info["seconds"] = round(len(wav) / SR, 2)
        info["took_s"] = round(time.time() - t0, 1)
        return wav, SR, info

    def resing(self, tune: int, mode: str = "score", steps: Optional[int] = None, seed: int = 0,
               model: str = "kalavinka") -> Dict:
        """Sing a bundled line with its original lyrics, plus the real singer's clip for comparison.
        ``mode``: score | expressive | real-pitch (SoulX given the real singer's own pitch curve).
        Returns {"model": (wav, sr), "real": (wav, sr), "model_name", "lyric", "song", "mode", "took_s"}."""
        import soundfile as sf
        tu = self._tune(tune)
        self.steps = int(steps or default_steps(self.device))
        vo = self._voice(tu["voice"])
        t0 = time.time()
        if mode not in ("score", "expressive", "real-pitch"):
            raise SingError("mode must be score, expressive or real-pitch")
        f0 = {"score": None, "expressive": lambda: self.contour(tu["item"], seed),
              "real-pitch": lambda: [float(x) for x in tu["item"]["f0"].split()]}[mode]
        wav = self._render(tu["item"], vo, f0() if f0 else None, seed, shift=False, model=model)
        real, sr = sf.read(self.bundle / "data" / tu["item"]["wav"], dtype="float32")
        return {"model": (wav, SR), "real": (real, sr), "lyric": tu["lyric"], "song": f"{tu['song']} ({tu['year']})",
                "mode": mode, "model_name": model, "took_s": round(time.time() - t0, 1)}


def save_wav(path: str | Path, wav: np.ndarray, sr: int = SR) -> Path:
    """Write a float waveform to ``path`` (creates folders)."""
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, sr)
    return path
