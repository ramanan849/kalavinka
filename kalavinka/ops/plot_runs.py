"""Live training plots for Kalavinka runs: one PNG per run plus an overlay of all val curves.

Reads ``<runs>/<run>/metrics.jsonl`` (written by ``kalavinka.sx.train`` and
``kalavinka.ir.train``) and writes ``<runs>/<run>/live.png`` (train loss,
val loss, grad norm, LR, and the articulatory shared-update share when
present) and ``<runs>/monitoring/val_overlay.png``.  ``--every N`` repeats
every N seconds (run it in a tmux window).

Usage: python plot_runs.py --runs <PROJECT>/runs R1 R2 R3 R4 [--every 300]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path: Path):
    """(train records, val records) from a metrics.jsonl."""
    tr, va = [], []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            (va if "val" in r else tr).append(r)
    return tr, va


def plot_run(run: Path) -> None:
    """live.png for one run."""
    tr, va = read(run / "metrics.jsonl")
    if not tr and not va:
        return
    keys = [k for k in ("loss", "gnorm", "lr", "shared_share", "resid", "voicing") if tr and k in tr[-1]]
    fig, ax = plt.subplots(1, len(keys) + 1, figsize=(4 * (len(keys) + 1), 3))
    ax[0].plot([r["step"] for r in va], [r["val"] for r in va], "o-", color="C1")
    ax[0].set_title(f"{run.name} val (best {min((r['val'] for r in va), default=float('nan')):.4f})")
    for a, k in zip(ax[1:], keys):
        a.plot([r["step"] for r in tr], [r[k] for r in tr], lw=0.8)
        a.set_title(k)
        if k in ("gnorm",):
            a.set_yscale("log")
    fig.tight_layout()
    fig.savefig(run / "live.png", dpi=90)
    plt.close(fig)


def overlay(runs, out: Path) -> None:
    """All runs' val curves in one figure."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in runs:
        _, va = read(run / "metrics.jsonl")
        if va:
            ax.plot([r["step"] for r in va], [r["val"] for r in va], "o-", ms=3, label=run.name)
    ax.set_xlabel("step"); ax.set_ylabel("val loss"); ax.legend(); fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main() -> None:
    """CLI: plot once or every --every seconds."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("names", nargs="+")
    p.add_argument("--every", type=int, default=0)
    a = p.parse_args()
    while True:
        runs = [a.runs / n for n in a.names]
        for r in runs:
            plot_run(r)
        overlay(runs, a.runs / "monitoring" / "val_overlay.png")
        if not a.every:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
