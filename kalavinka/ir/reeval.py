"""Re-evaluate a trained contour run on the test split with the current ``train.evaluate``.

Why: on 25 Nov 2026 the evaluation gained fair flat-MIDI baselines (``flat_midi`` now uses its own note voicing,
``flat_midi_same_frames`` is scored on exactly the model's frames; the old reference-voicing baseline is kept as
``flat_midi_ref_voicing``) and the flow sampler's time label was corrected (``--t-label``). This writes
``<run>/test_metrics_v2.json`` without retraining, so the report can quote the corrected comparison.

Usage::

    python -m kalavinka.ir.reeval --run <PROJECT>/runs/I1_v1 --export <PROJECT>/export/v1 [--tau 1.0] [--t-label left]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from .features import scaffold
from .model import ContourNet
from .train import evaluate, load_split


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--t-label", choices=["left", "mid"], default="left")
    p.add_argument("--split", default="test")
    a = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.run / "best.pt", map_location=device)
    args = ck["args"]
    model = ContourNet(args["mode"], int(args["d"]), int(args["layers"]), int(args["heads"])).to(device)
    model.load_state_dict(ck["model"])
    items = load_split(a.export / f"{a.split}.jsonl")
    res, _ = evaluate(model, [scaffold(it) for it in items], items, device, int(args.get("max_frames", 12000)),
                      int(args.get("flow_steps", 10)), temperature=a.tau, t_label=a.t_label)
    res["settings"] = {"tau": a.tau, "t_label": a.t_label, "split": a.split, "mode": args["mode"]}
    out = a.run / f"{a.split}_metrics_v2.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {m: round(v, 3) for m, v in r.items() if m in ("cents_rmse", "orn_rmse", "voicing_f1")}
                      for k, r in res.items() if k != "settings"}, indent=1))


if __name__ == "__main__":
    main()
