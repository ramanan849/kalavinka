"""Score rendered clips (annotation venv, which has Whisper, RMVPE and WavLM).

For every line of ``<render dir>/render.jsonl``:

* **CER** -- Whisper-large-v3 (language ``hi``) transcribes the render and the
  real vocal clip; both and the reference lyrics go through the repo's loose
  phonetic spelling (``saregama.lyrics_eval.loose_words``, script-independent:
  Devanagari, Urdu-style spellings and romanised Hindi all compare), and CER
  is the character edit distance on those strings.  The real clip's CER is
  the ceiling a render can reach with this ASR;
* **F0** -- RMVPE (via ``saregama.stages.f0.F0Estimator``) at 50 fps on the
  render, against the F0 it was asked to sing (melody modes) and against the
  real singer's F0 (all modes): cents RMSE, ornament-frame RMSE (reference
  slope > 300 c/s), voicing F1 (``kalavinka.ir.metrics``);
* **SIM** -- WavLM-base-plus-sv speaker-embedding cosine between the render
  and the prompt clip (and the real clip and the prompt, as the ceiling).

Writes ``scores.jsonl`` and ``summary.json`` next to ``render.jsonl``.

Usage::

    python -m kalavinka.eval.score --render <dir> --export <PROJECT>/export/v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from kalavinka.ir.metrics import pair_metrics
from saregama.audio import load_audio
from saregama.lyrics_eval import _edit_ops, loose_words

SR, FPS = 24000, 50


def loose_string(text: str) -> str:
    """Loose phonetic character string of a lyric/ASR text (rests and spaces dropped)."""
    return "".join(loose_words(text.replace("<SP>", " ").replace("~", " ")))


def cer(ref: str, hyp: str) -> float:
    """Character error rate of ``hyp`` against ``ref``."""
    if not ref:
        return float("nan")
    ops = _edit_ops(list(ref), list(hyp))
    return sum(1 for o in ops if o[0] != "ok") / len(ref)


class Scorer:
    """Lazily loaded Whisper, RMVPE and WavLM-SV on one device."""

    def __init__(self, device: str = "cuda"):
        """Load the three models."""
        from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline
        from saregama.config import F0Cfg
        from saregama.stages.f0 import F0Estimator
        self.device = device
        self.asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                            torch_dtype=torch.float16, device=device)
        self.f0 = F0Estimator(F0Cfg(backends=["rmvpe"], device=device), fps=FPS)
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()

    def transcribe(self, wavs16: List[np.ndarray]) -> List[str]:
        """Whisper transcripts (Hindi) of 16 kHz clips."""
        out = self.asr([{"raw": w, "sampling_rate": 16000} for w in wavs16], batch_size=8,
                       generate_kwargs={"language": "hi", "task": "transcribe"})
        return [o["text"] for o in out]

    @torch.no_grad()
    def embed(self, wav16: np.ndarray) -> np.ndarray:
        """L2-normalised WavLM x-vector of a 16 kHz clip."""
        x = self.fe(wav16, sampling_rate=16000, return_tensors="pt").to(self.device)
        e = self.sv(**x).embeddings[0]
        return (e / e.norm()).cpu().numpy()

    def pitch(self, wav24: np.ndarray) -> np.ndarray:
        """50 fps F0 (Hz, 0 unvoiced) of a 24 kHz clip."""
        return np.asarray(self.f0(wav24, SR)["f0"], np.float32)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Score one render directory and write scores.jsonl + summary.json."""
    from saregama.audio import resample
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--render", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--split", default="test")
    a = p.parse_args(argv)
    rows = [json.loads(l) for l in (a.render / "render.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    items = {json.loads(l)["index"]: json.loads(l) for l in
             (a.export / f"{a.split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    sc = Scorer()
    gen = [load_audio(a.render / f"{r['index']}.wav", SR) for r in rows]
    ref = [load_audio(a.export / r["ref_wav"], SR) for r in rows]
    prm = [load_audio(a.export / r["prompt_wav"], SR) for r in rows]
    g16, r16, p16 = ([resample(x, SR, 16000) for x in xs] for xs in (gen, ref, prm))
    hyp_g, hyp_r = sc.transcribe(g16), sc.transcribe(r16)
    out = []
    for k, r in enumerate(rows):
        it = items[r["index"]]
        lref = loose_string(it["text"])
        f0g = sc.pitch(gen[k])
        real = np.array([float(x) for x in it["f0"].split()], np.float32)
        rec = {"index": r["index"], "cer": cer(lref, loose_string(hyp_g[k])), "cer_real": cer(lref, loose_string(hyp_r[k])),
               "asr": hyp_g[k], "sim": float(sc.embed(g16[k]) @ sc.embed(p16[k])),
               "sim_real": float(sc.embed(r16[k]) @ sc.embed(p16[k]))}
        rec.update({f"vs_real_{m}": v for m, v in pair_metrics(f0g, real).items()})
        if r.get("f0_given"):
            rec.update({f"vs_given_{m}": v for m, v in pair_metrics(f0g, np.array(r["f0_given"], np.float32)).items()})
        out.append(rec)
    with open(a.render / "scores.jsonl", "w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    keys = sorted({k for r in out for k, v in r.items() if isinstance(v, float)})
    summary = {"n": len(out), **{k: round(float(np.nanmean([r.get(k, np.nan) for r in out])), 4) for k in keys}}
    (a.render / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
