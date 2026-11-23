"""Pick GPUs with enough free memory for a P1 / Kalavinka job.

This machine's 8 H100s are shared with other long runs, so jobs never take a
card that lacks room: a GPU qualifies when its free memory is at least
``--min-free-gb``.  Prints the qualifying indices (comma separated), most free
first, at most ``--max`` of them.  Read-only: it only queries nvidia-smi.

Usage: python gpu_pick.py --min-free-gb 20 [--max 8]
"""
from __future__ import annotations

import argparse
import subprocess
from typing import List, Tuple


def free_memory() -> List[Tuple[int, float]]:
    """(index, free GiB) for every visible GPU, from nvidia-smi."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        i, free = [x.strip() for x in line.split(",")]
        rows.append((int(i), float(free) / 1024))
    return rows


def pick(rows: List[Tuple[int, float]], min_free_gb: float, max_n: int) -> List[int]:
    """Indices with at least ``min_free_gb`` free, most free first, capped at ``max_n``."""
    ok = sorted([r for r in rows if r[1] >= min_free_gb], key=lambda r: -r[1])
    return [i for i, _ in ok[:max_n]]


def main() -> None:
    """CLI entry: print the chosen GPU indices."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-free-gb", type=float, default=20.0)
    p.add_argument("--max", type=int, default=8)
    a = p.parse_args()
    print(",".join(map(str, pick(free_memory(), a.min_free_gb, a.max))))


if __name__ == "__main__":
    main()
