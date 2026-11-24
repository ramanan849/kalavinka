"""Minimal-pair pronunciation scoring: does the singing realise retroflexion and aspiration?

Why: the lyric CER cannot see these Hindi contrasts (its loose spelling folds
them, and Whisper's language model fixes spellings).  A multilingual IPA
recogniser (``kalavinka.eval.phone_eval``) was tried first and failed its gate:
on real sung, separated vocals it never outputs a retroflex or aspirated
phone.  This module instead asks a Hindi **character-level** CTC model
(``Harveenchadha/vakyansh-wav2vec2-hindi-him-4200``, Devanagari output, used
without its KenLM) a forced-choice question at each place where the lyric has
a contrastive consonant.

Per clip:

1. the lyric (item ``text``, NFD, Devanagari only, no spaces) is force-aligned
   to the CTC log-probabilities (``torchaudio.functional.forced_align``);
2. at each consonant that has a minimal-pair partner, the score is
   ``p(target) / (p(target) + p(partner))`` from the posteriors over the
   frames aligned to it (mean of per-frame probabilities, then the ratio):

   * place: ट-त, ठ-थ, ड-द, ढ-ध, ण-न (retroflex vs dental), in both directions;
   * aspiration: ख-क, घ-ग, छ-च, झ-ज, ठ-ट, ढ-ड, थ-त, ध-द, फ-प, भ-ब, both directions.

Buckets: ``place_retro`` (target retroflex), ``place_dental``, ``asp_asp``
(target aspirated), ``asp_plain``, and ``held_place`` / ``held_asp`` for ठ and ढ
(the phones Th, Dh, Rh held out in the phone-holdout runs).  0.5 is chance;
real singing is the reference (the gate: real clips must sit clearly above
the zero-shot renders, else the metric is not usable).

Writes ``contrast.jsonl`` (per-clip sums) and ``contrast.json`` (pooled
means) into the render dir (or ``--out`` with ``--real``).

Usage::

    python -m kalavinka.eval.contrast_eval --render <dir> --export <PROJECT>/export/v1 [--split test]
    python -m kalavinka.eval.contrast_eval --real --render <dir> --export ... --out <dir>
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

MODEL_ID = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
PLACE = [("ट", "त"), ("ठ", "थ"), ("ड", "द"), ("ढ", "ध"), ("ण", "न")]
ASP = [("ख", "क"), ("घ", "ग"), ("छ", "च"), ("झ", "ज"), ("ठ", "ट"), ("ढ", "ड"), ("थ", "त"), ("ध", "द"),
       ("फ", "प"), ("भ", "ब")]
HELD = {"ठ", "ढ"}
BUCKETS = ("place_retro", "place_dental", "asp_asp", "asp_plain", "held_place", "held_asp")


def contrast_tests(ch: str) -> List[tuple]:
    """(bucket, partner) tests that apply to target character ``ch``."""
    out = []
    for r, d in PLACE:
        if ch == r:
            out.append(("place_retro", d))
        elif ch == d:
            out.append(("place_dental", r))
    for a, p in ASP:
        if ch == a:
            out.append(("asp_asp", p))
        elif ch == p:
            out.append(("asp_plain", a))
    if ch in HELD:
        out += [("held_" + b.split("_")[0], q) for b, q in out if b in ("place_retro", "asp_asp")]
    return out


def lyric_chars(text: str) -> str:
    """The clip lyric as NFD Devanagari letters only (no spaces, no <SP>)."""
    t = unicodedata.normalize("NFD", text.replace("<SP>", " "))
    return "".join(ch for ch in t if "ऀ" <= ch <= "ॿ")


def score_clip(lp: np.ndarray, chars: str, vocab: Dict[str, int], spans: Sequence[Sequence[int]]) -> Dict:
    """Per-bucket (n, sum of scores, n above 0.5) for one clip.

    ``lp``: (T, V) log-probs; ``spans[i]``: frames aligned to the i-th kept character."""
    out = {f"{b}_{k}": 0.0 for b in BUCKETS for k in ("n", "sum", "win")}
    kept = [c for c in chars if c in vocab]
    for c, fr in zip(kept, spans):
        if not len(fr):
            continue
        p = np.exp(lp[list(fr)]).mean(0)
        for bucket, partner in contrast_tests(c):
            if partner not in vocab:
                continue
            s = float(p[vocab[c]] / max(p[vocab[c]] + p[vocab[partner]], 1e-12))
            out[f"{bucket}_n"] += 1; out[f"{bucket}_sum"] += s; out[f"{bucket}_win"] += s > 0.5
    return out


def pooled(rows: Sequence[Dict]) -> Dict:
    """Mean score and win rate per bucket over all positions."""
    res = {"clips": len(rows)}
    for b in BUCKETS:
        n = sum(r[f"{b}_n"] for r in rows)
        res[b] = {"n": int(n), "mean": round(sum(r[f"{b}_sum"] for r in rows) / n, 4) if n else None,
                  "win": round(sum(r[f"{b}_win"] for r in rows) / n, 4) if n else None}
    return res


class Aligner:
    """Hindi character CTC model + forced alignment."""

    def __init__(self, device: str = "cuda"):
        """Load model, feature extractor, vocab."""
        import torch
        from huggingface_hub import snapshot_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        path = snapshot_download(MODEL_ID, allow_patterns=["*.json", "pytorch_model.bin"])
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(path)
        self.model = Wav2Vec2ForCTC.from_pretrained(path).to(device).eval()
        self.vocab = json.loads((Path(path) / "vocab.json").read_text())
        self.blank = self.model.config.pad_token_id
        self.device, self.torch = device, torch

    def __call__(self, wav16: np.ndarray, chars: str):
        """(log-probs (T, V), frame spans per kept character) or None when it cannot align."""
        import torchaudio.functional as F
        torch = self.torch
        x = self.fe(wav16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            lp = torch.log_softmax(self.model(x).logits.float(), -1)[0]
        ids = [self.vocab[c] for c in chars if c in self.vocab]
        if not ids or len(ids) > lp.shape[0]:
            return None
        ali, _ = F.forced_align(lp[None], torch.tensor([ids], device=self.device), blank=self.blank)
        ali = ali[0].tolist()
        spans: List[List[int]] = [[] for _ in ids]
        k, prev = -1, self.blank
        for t, a in enumerate(ali):
            if a != self.blank and (a != prev or k < 0):
                k += 1                      # a new token starts (CTC: repeats need a blank between)
            if a != self.blank and 0 <= k < len(ids):
                spans[k].append(t)
            prev = a
        return lp.cpu().numpy(), spans


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Score a render dir (or the real clips with ``--real``)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--render", type=Path, required=True, help="render dir (its render.jsonl lists the clips)")
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--split", default="test", help="split(s) to look items up in, comma list")
    p.add_argument("--real", action="store_true", help="score the real reference clips instead of the renders")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    from saregama.audio import load_audio
    items = {}
    for s in a.split.split(","):
        for line in (a.export / f"{s}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                items[it["index"]] = it
    todo = [json.loads(l) for l in (a.render / "render.jsonl").read_text().splitlines() if l.strip()]
    out = a.out or a.render
    out.mkdir(parents=True, exist_ok=True)
    al, rows = Aligner(), []
    with open(out / "contrast.jsonl", "w", encoding="utf-8") as f:
        for r in todo:
            wav = (a.export / r["ref_wav"]) if a.real else (a.render / f"{r['index']}.wav")
            chars = lyric_chars(items[r["index"]]["text"])
            res = al(load_audio(wav, 16000), chars)
            if res is None:
                continue
            c = score_clip(res[0], chars, al.vocab, res[1])
            rows.append(c)
            f.write(json.dumps({"index": r["index"], **c}, ensure_ascii=False) + "\n")
    summ = pooled(rows)
    (out / "contrast.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
