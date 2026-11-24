"""Phone-level pronunciation scoring with a language-model-free IPA recognizer.

Why: the lyric CER in ``kalavinka.eval.score`` cannot see the Hindi contrasts
the articulatory embedding (B) is about.  Its loose spelling folds retroflex
into dental and aspirated into plain stops, and Whisper's language model
spells a known word correctly however it was sung.  This module uses
``facebook/wav2vec2-xlsr-53-espeak-cv-ft``, a CTC recognizer that outputs IPA
phones (including ʈ ɖ ɳ ɽ ʂ and aspirated ʰ forms) with no language model,
decoded greedily.

Per clip:

* the target IPA sequence comes from the item's ``phoneme`` tokens
  (``in_d-a-r-d`` -> phones -> ``saregama.phoneset`` IPA; the nasalisation
  mark ``nas`` and non-Indic tokens are skipped);
* recognised and target sequences are aligned by a weighted edit distance
  (substitution 0 for equal phones, 0.5 for the same base phone that differs
  only by retroflexion / aspiration / length / dental mark, else 1);
* **feature recall** for a class is, over target phones of that class, the
  share aligned to a recognised phone that has the feature (retroflex:
  ʈ ɖ ɳ ɽ ɻ ʂ ɭ; aspirated: any ʰ/ʱ form); the **false-alarm** rate is the
  share of target phones *without* the feature aligned to one that has it.
  ``held`` restricts the retroflex+aspirate recall to the held-out phones of
  the holdout experiment (Th, Dh, Rh by default).  PER is the plain phone
  error rate (unit costs).

Run it on the real clips first (``--real``): that is the gate.  The metric is
only usable if real singing clearly realises the features more often than
the zero-shot renders do.

Writes ``<render dir>/phones.jsonl`` (per-clip counts) and ``phones.json``
(pooled summary); with ``--real`` they go to ``--out``.

Usage::

    python -m kalavinka.eval.phone_eval --render <PROJECT>/runs/eval_v1/R1_melody --export <PROJECT>/export/v1
    python -m kalavinka.eval.phone_eval --real --export <PROJECT>/export/v1 --split test --out <dir>
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
RETRO = set("ʈɖɳɽɻʂɭ")
ASP_MARKS = ("ʰ", "ʱ")
ASP_TOKENS = {"th", "kh", "ph", "tsh", "tɕh", "ts.h"}         # the vocab's other aspiration spelling
HELD_DEFAULT = ("Th", "Dh", "Rh")
RETRO_NAMES = {"T", "Th", "D", "Dh", "N", "R", "Rh", "L", "Sh", "zh"}
ASP_NAMES = {"kh", "gh", "ch", "jh", "Th", "Dh", "th", "dh", "ph", "bh", "Rh"}


def norm_ipa(p: str) -> str:
    """Canonical form of one IPA phone (NFC, ɡ->g, ʱ->ʰ)."""
    return unicodedata.normalize("NFC", p).replace("ɡ", "g").replace("ʱ", "ʰ")


def is_retro(p: str) -> bool:
    """Does a recognised phone carry retroflexion?"""
    return any(ch in RETRO for ch in p)


def is_asp(p: str) -> bool:
    """Does a recognised phone carry aspiration / breathy release?"""
    return any(m in p for m in ASP_MARKS) or p in ASP_TOKENS


def base(p: str) -> str:
    """Phone with retroflexion, aspiration, length and dental marks removed (for the 0.5 cost)."""
    q = norm_ipa(p)
    for a, b in (("ʈ", "t"), ("ɖ", "d"), ("ɳ", "n"), ("ɽ", "ɾ"), ("ɻ", "ɾ"), ("ʂ", "ʃ"), ("ɭ", "l")):
        q = q.replace(a, b)
    if q in ASP_TOKENS:                                       # "th" style aspiration spelling
        q = q[:-1]
    for m in ("ʰ", "ː", "̪"):
        q = q.replace(m, "")
    return q


def target_phones(phoneme: str) -> List[Tuple[str, str]]:
    """(phone name, IPA) of the Indic phones in an item's ``phoneme`` field."""
    from saregama.phoneset import BY_NAME
    out = []
    for tok in phoneme.split():
        if not tok.startswith("in_"):
            continue
        for name in tok[3:].split("-"):
            if name == "nas" or name not in BY_NAME:
                continue
            out.append((name, norm_ipa(BY_NAME[name].ipa)))
    return out


def align(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Weighted Levenshtein alignment -> list of (ref index | None, hyp index | None)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    sub = lambda a, b: 0.0 if a == b else (0.5 if base(a) == base(b) else 1.0)   # noqa: E731
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + sub(ref[i - 1], hyp[j - 1]))
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and np.isclose(d[i, j], d[i - 1, j - 1] + sub(ref[i - 1], hyp[j - 1])):
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and np.isclose(d[i, j], d[i - 1, j] + 1):
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, j - 1)); j -= 1
    return out[::-1]


def clip_counts(target: Sequence[Tuple[str, str]], hyp: Sequence[str], held: Sequence[str] = HELD_DEFAULT) -> Dict:
    """Per-clip counts: PER parts and feature hits / false alarms."""
    ref = [ipa for _, ipa in target]
    pairs = align(ref, hyp)
    c = {k: 0 for k in ("n_ref", "errors", "retro_n", "retro_hit", "retro_neg", "retro_fa",
                        "asp_n", "asp_hit", "asp_neg", "asp_fa", "held_n", "held_retro_hit", "held_asp_hit")}
    c["n_ref"] = len(ref)
    for ri, hi in pairs:
        if ri is None:
            c["errors"] += 1
            continue
        h = hyp[hi] if hi is not None else ""
        if h != ref[ri]:
            c["errors"] += 1
        name = target[ri][0]
        if name in RETRO_NAMES:
            c["retro_n"] += 1; c["retro_hit"] += is_retro(h)
        else:
            c["retro_neg"] += 1; c["retro_fa"] += is_retro(h)
        if name in ASP_NAMES:
            c["asp_n"] += 1; c["asp_hit"] += is_asp(h)
        else:
            c["asp_neg"] += 1; c["asp_fa"] += is_asp(h)
        if name in held:
            c["held_n"] += 1; c["held_retro_hit"] += is_retro(h)
            c["held_asp_hit"] += is_asp(h)
    return c


def pooled(rows: Sequence[Dict]) -> Dict:
    """Ratio-of-sums summary over clips."""
    s = {k: sum(r[k] for r in rows) for k in rows[0]} if rows else {}
    div = lambda a, b: round(s[a] / s[b], 4) if s.get(b) else None   # noqa: E731
    return {"clips": len(rows), "phones": s.get("n_ref", 0), "per": div("errors", "n_ref"),
            "retro_recall": div("retro_hit", "retro_n"), "retro_false_alarm": div("retro_fa", "retro_neg"),
            "asp_recall": div("asp_hit", "asp_n"), "asp_false_alarm": div("asp_fa", "asp_neg"),
            "held_n": s.get("held_n", 0), "held_retro_recall": div("held_retro_hit", "held_n"),
            "held_asp_recall": div("held_asp_hit", "held_n")}


class Recognizer:
    """Greedy CTC IPA phone recognizer."""

    def __init__(self, device: str = "cuda"):
        """Load the model, feature extractor and vocab."""
        import torch
        from huggingface_hub import snapshot_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        path = snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt"])
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(path)
        self.model = Wav2Vec2ForCTC.from_pretrained(path).to(device).eval()
        vocab = json.loads((Path(path) / "vocab.json").read_text())
        self.inv = {v: k for k, v in vocab.items()}
        self.skip = {vocab.get(t) for t in ("<pad>", "<s>", "</s>", "<unk>")}
        self.device, self.torch = device, torch

    def __call__(self, wav16: np.ndarray) -> List[str]:
        """IPA phones of a 16 kHz mono signal."""
        x = self.fe(wav16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with self.torch.no_grad():
            ids = self.model(x).logits.argmax(-1)[0].tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i not in self.skip:
                out.append(norm_ipa(self.inv[i]))
            prev = i
        return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Score a render dir (or the real clips with ``--real``)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--render", type=Path, default=None)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--split", default="test", help="split(s) for --real / item lookup, comma list")
    p.add_argument("--real", action="store_true", help="score the real clips listed in --render (or the split)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--held", default=",".join(HELD_DEFAULT))
    a = p.parse_args(argv)
    from saregama.audio import load_audio
    items = {}
    for s in a.split.split(","):
        for line in (a.export / f"{s}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                items[it["index"]] = it
    if a.render:
        todo = [json.loads(l) for l in (a.render / "render.jsonl").read_text().splitlines() if l.strip()]
    else:
        todo = [{"index": k, "ref_wav": v["wav"]} for k, v in items.items()]
    out = a.out or a.render
    out.mkdir(parents=True, exist_ok=True)
    rec, held = Recognizer(), tuple(a.held.split(","))
    rows = []
    with open(out / "phones.jsonl", "w", encoding="utf-8") as f:
        for r in todo:
            wav = (a.export / r["ref_wav"]) if a.real else (a.render / f"{r['index']}.wav")
            hyp = rec(load_audio(wav, 16000))
            c = clip_counts(target_phones(items[r["index"]]["phoneme"]), hyp, held)
            rows.append(c)
            f.write(json.dumps({"index": r["index"], "hyp": " ".join(hyp), **c}, ensure_ascii=False) + "\n")
    summ = pooled(rows)
    (out / "phones.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
