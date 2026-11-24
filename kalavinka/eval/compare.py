"""Paired bootstrap comparisons between systems, from per-clip result files.

Every evaluation module writes one JSON line per clip (``scores.jsonl`` from
``score``, ``contrast.jsonl`` from ``contrast_eval``).  This module pairs two
systems on the clips both have and bootstraps the difference B - A over
clips (5,000 resamples, seeded), giving the mean difference and a 95 % CI.

Metric specs:

* ``key`` -- mean of a per-clip value (e.g. ``vs_real_cents_rmse``);
* ``key@cap`` -- the same with each clip's value capped (``cer@1``: Whisper
  sometimes loops and gives CER >> 1);
* ``num/den`` -- ratio of sums over clips (e.g. ``place_retro_sum/place_retro_n``
  for the contrast scores, which have different numbers of positions per clip).

Usage::

    python -m kalavinka.eval.compare --file contrast.jsonl --metric place_retro_sum/place_retro_n \\
        --pair <dirA> <dirB> [--pair ...] [--out table.md]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def load(path: Path) -> Dict[str, Dict]:
    """index -> per-clip record of a jsonl file."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["index"]] = r
    return out


def stat(rows: Sequence[Dict], metric: str) -> float:
    """The metric over a list of clip records (see the module doc for the spec)."""
    if "/" in metric:
        num, den = metric.split("/")
        d = sum(r[den] for r in rows)
        return sum(r[num] for r in rows) / d if d else float("nan")
    key, _, cap = metric.partition("@")
    vals = [min(r[key], float(cap)) if cap else r[key] for r in rows]
    return sum(vals) / len(vals) if vals else float("nan")


def paired_bootstrap(a: Dict[str, Dict], b: Dict[str, Dict], metric: str, n: int = 5000,
                     seed: int = 0) -> Tuple[float, float, float, float, float, int]:
    """(A value, B value, B-A, CI low, CI high, n clips) over the shared clips."""
    ids = sorted(set(a) & set(b))
    ra, rb = [a[i] for i in ids], [b[i] for i in ids]
    va, vb = stat(ra, metric), stat(rb, metric)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        k = [rng.randrange(len(ids)) for _ in ids]
        diffs.append(stat([rb[i] for i in k], metric) - stat([ra[i] for i in k], metric))
    diffs.sort()
    return va, vb, vb - va, diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1], len(ids)


def table(pairs: Sequence[Tuple[Path, Path]], file: str, metric: str, n: int = 5000) -> List[str]:
    """Markdown rows for each (A dir, B dir) pair; ``*`` marks a CI that excludes 0."""
    rows = [f"| A | B | {metric} A | B | B-A | 95% CI | n |", "|---|---|---|---|---|---|---|"]
    for pa, pb in pairs:
        va, vb, d, lo, hi, k = paired_bootstrap(load(pa / file), load(pb / file), metric, n)
        sig = " *" if lo > 0 or hi < 0 else ""
        rows.append(f"| {pa.name} | {pb.name} | {va:.3f} | {vb:.3f} | {d:+.3f}{sig} | [{lo:+.3f}, {hi:+.3f}] | {k} |")
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: print (and optionally append to ``--out``) a comparison table."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, help="per-clip jsonl name inside each dir")
    p.add_argument("--metric", required=True)
    p.add_argument("--pair", nargs=2, type=Path, action="append", required=True)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    lines = table(a.pair, a.file, a.metric, a.n)
    print("\n".join(lines))
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    main()
