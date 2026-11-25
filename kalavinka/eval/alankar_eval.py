"""Alankar benchmark: base SoulX (zero-shot) vs Kalavinka (V4-L) on sargam exercises.

Reads the takes logged by the bundle's history (``<bundle>/outputs/history.jsonl``, kind ``sargam``; made with
``python -m kalavinka_app alankar all --model both --voice V``) and reports per model:

* **pitch** (already in each take's info, from ``kalavinka_app.sargam.pitch_report``): share of notes within 50 cents
  of the written swara ("right swara"), within 25 cents ("in tune"), mean |error| in cents, voiced share;
* **syllables**: Whisper-large-v3 (Hindi) transcribes each take; the character error rate against the sung swara
  names (सा रे ग म प ध नि), after the repo's loose phonetic spelling (as in ``kalavinka.eval.score``). Sargam
  syllables are not words, so Whisper's absolute CER is high for both models; only the difference is meaningful.

Paired comparison: every (alankar, voice, seed) is sung by both models, so differences are paired; 95% CIs by
paired bootstrap over takes.

Usage (annotation venv, which has Whisper)::

    python -m kalavinka.eval.alankar_eval --history <bundle>/outputs/history.jsonl --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from kalavinka.eval.score import cer, loose_string


def boot_ci(d: np.ndarray, n: int = 5000, seed: int = 0):
    """Paired bootstrap 95% CI of the mean of ``d``."""
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(d, len(d)).mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def expected_text(info: Dict) -> str:
    """Devanagari swara names of a take, from its sargam text."""
    dev = {"sa": "सा", "re": "रे", "ga": "ग", "ma": "म", "pa": "प", "dha": "ध", "ni": "नि"}
    out = []
    for tok in info["sargam"].split():
        name = tok.strip(".'").split("(")[0].split(":")[0].lower()
        if name in dev:
            out.append(dev[name])
    return " ".join(out)


def main(argv=None) -> None:
    """Score the logged alankar takes and write alankar_bench.json / .md."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-asr", action="store_true")
    a = p.parse_args(argv)
    recs = [json.loads(l) for l in a.history.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if r["kind"] == "sargam" and r["request"].get("syllables") == "sargam" and r["request"].get("voice")
            and r["request"]["title"].startswith("alankar")]
    latest: Dict[tuple, Dict] = {}
    for r in recs:
        q = r["request"]
        latest[(q["title"], q["voice"], q["seed"], q["model"])] = r
    asr = None
    if not a.no_asr:
        import torch
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16,
                       device="cuda:0")
    rows: List[Dict] = []
    for (title, voice, seed, model), r in sorted(latest.items(), key=lambda kv: kv[0]):
        wav = a.history.parent / r["wav"]
        row = {"alankar": title.split(":")[0], "voice": voice, "seed": seed, "model": model, **{
            k: r["info"]["pitch"][k] for k in ("within_50c", "within_25c", "mean_abs_cents", "voiced_share")}}
        if asr is not None:
            import librosa
            import soundfile as sf
            x, sr = sf.read(wav, dtype="float32")
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            hyp = asr({"raw": x, "sampling_rate": 16000}, generate_kwargs={"language": "hi", "task": "transcribe"})["text"]
            row["asr"] = hyp
            row["cer"] = min(1.0, cer(loose_string(expected_text(r["info"])), loose_string(hyp)))
        rows.append(row)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "alankar_bench.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                                               encoding="utf-8")
    keys = ["within_50c", "within_25c", "mean_abs_cents", "voiced_share"] + (["cer"] if asr is not None else [])
    by = {m: {(x["alankar"], x["voice"], x["seed"]): x for x in rows if x["model"] == m} for m in ("kalavinka", "base")}
    common = sorted(set(by["kalavinka"]) & set(by["base"]))
    summary = {"pairs": len(common), "metrics": {}}
    lines = [f"# Alankar benchmark: Kalavinka (V4-L) vs base SoulX (zero-shot)\n",
             f"{len(common)} paired takes (alankar x voice x seed), score mode, swara names sung.\n",
             "| metric | base | kalavinka | kalavinka - base [95% CI] |", "|---|---|---|---|"]
    for k in keys:
        b = np.array([by["base"][c][k] for c in common], float)
        kv = np.array([by["kalavinka"][c][k] for c in common], float)
        lo, hi = boot_ci(kv - b)
        sig = "*" if lo > 0 or hi < 0 else ""
        summary["metrics"][k] = {"base": b.mean(), "kalavinka": kv.mean(), "diff": (kv - b).mean(), "ci": [lo, hi]}
        lines.append(f"| {k} | {b.mean():.3f} | {kv.mean():.3f} | {(kv - b).mean():+.3f}{sig} [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "Per alankar (mean over voices): within_50c / mean_abs_cents" + (" / cer" if asr else ""), "",
              "| alankar | base | kalavinka |", "|---|---|---|"]
    for al in sorted({c[0] for c in common}, key=lambda s: int(s.split()[1])):
        cell = {}
        for m in ("base", "kalavinka"):
            xs = [by[m][c] for c in common if c[0] == al]
            cell[m] = f"{np.mean([x['within_50c'] for x in xs]) * 100:.0f}% / {np.mean([x['mean_abs_cents'] for x in xs]):.1f} c" + (
                f" / {np.mean([x['cer'] for x in xs]):.2f}" if asr else "")
        lines.append(f"| {al} | {cell['base']} | {cell['kalavinka']} |")
    (a.out / "alankar_bench.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (a.out / "alankar_bench.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
