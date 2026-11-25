"""Sampling-temperature sweep for the flow contour model (I2).

Why: the flow model samples ornaments instead of averaging them, so it has
realistic vibrato statistics but a larger pointwise error than the regression
model.  Scaling its starting noise by a temperature ``tau`` < 1 moves it
towards the mean contour.  This sweep traces that trade-off (pointwise cents
RMSE against the vibrato-statistics distance) and picks ``tau*`` **on the val
split**, so the test numbers stay untouched by the choice.

Sampler time label (``--t-label``): ``left`` (default since 25 Nov 2026, the correct Euler label) writes to
``<run>/sweep_left/``; ``mid`` (the earlier, half-step-ahead label) reproduces ``<run>/sweep/``.

For every ``tau`` it writes ``<run>/sweep*/val_tau{tau}.json`` and
``test_tau{tau}.json`` (metrics) and ``test_f0_tau{tau}.json`` (index -> F0,
for ``kalavinka.eval.render --control melody-ir``), plus ``sweep.json`` with
the table and the chosen ``tau*``.

Choice rule (``pick_tau``): the smallest pointwise error among the
temperatures whose vibrato distance (extent W1 + 10 x rate W1) is at most
``--vib-slack`` times the best one.  It keeps the sampler's realism and buys
back as much accuracy as it can.

Usage::

    python -m kalavinka.ir.sweep --run <PROJECT>/runs/I2_v1 --export <PROJECT>/export/v1 \\
        [--taus 0.3,0.5,0.7,0.85,1.0] [--vib-slack 1.5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .features import scaffold
from .model import ContourNet
from .train import evaluate, load_split


def vib_distance(m: Dict) -> float:
    """One number for how far the vibrato statistics are from the real ones."""
    return float(m["vib_extent_w1"] + 10.0 * m["vib_rate_w1"])


def pick_tau(rows: List[Dict], slack: float) -> float:
    """``tau`` with the lowest val cents RMSE among those within ``slack`` x the best vibrato distance."""
    best_vib = min(r["val_vib"] for r in rows)
    ok = [r for r in rows if r["val_vib"] <= slack * best_vib + 1e-9]
    return min(ok, key=lambda r: r["val_cents_rmse"])["tau"]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the sweep for one flow run dir."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--taus", default="0.3,0.5,0.7,0.85,1.0")
    p.add_argument("--vib-slack", type=float, default=1.5)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=12000)
    p.add_argument("--t-label", choices=["left", "mid"], default="left",
                   help="sampler time label (left = correct Euler; mid = the pre-25-Nov label, writes to sweep/)")
    a = p.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ContourNet("flow", a.d, a.layers, a.heads).to(device)
    model.load_state_dict(torch.load(a.run / "best.pt", map_location=device)["model"])
    data = {s: load_split(a.export / f"{s}.jsonl") for s in ("val", "test")}
    scs = {s: [scaffold(it) for it in v] for s, v in data.items()}
    out = a.run / ("sweep" if a.t_label == "mid" else "sweep_left")
    out.mkdir(exist_ok=True)
    rows = []
    for tau in [float(t) for t in a.taus.split(",")]:
        row = {"tau": tau}
        for split in ("val", "test"):
            res, f0s = evaluate(model, scs[split], data[split], device, a.max_frames, a.flow_steps, temperature=tau,
                                t_label=a.t_label)
            (out / f"{split}_tau{tau}.json").write_text(json.dumps(res, indent=1))
            if split == "test":
                (out / f"test_f0_tau{tau}.json").write_text(json.dumps(f0s))
            m = res["model"]
            row.update({f"{split}_cents_rmse": m["cents_rmse"], f"{split}_orn_rmse": m["orn_rmse"],
                        f"{split}_vib": vib_distance(m), f"{split}_vib_extent": m["vib_extent_pred"],
                        f"{split}_vib_rate": m["vib_rate_pred"]})
        rows.append(row)
        print(json.dumps(row), flush=True)
    chosen = pick_tau(rows, a.vib_slack)
    (out / "sweep.json").write_text(json.dumps({"rows": rows, "tau_star": chosen, "rule": f"vib_slack={a.vib_slack}",
                                                "t_label": a.t_label}, indent=1))
    print(f"tau* = {chosen}")


if __name__ == "__main__":
    main()
