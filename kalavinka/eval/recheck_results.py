"""Recompute every comparison table of a RESULTS.md with the song-level bootstrap and Holm correction.

Why: the first comparisons (``compare.py`` before 25 Nov) resampled single clips, but the 105 test clips come from
18 songs whose clips share singer, prompt, key and recording, so the intervals were too narrow, and every table was
starred at an uncorrected 95 %. This tool finds each table (``| A | B | <metric> A | B | B-A | 95% CI | n |``),
re-runs it on the same per-clip files with ``compare.table`` (song bootstrap, two-sided p, Holm over the table's rows)
and writes a new markdown file listing, for every row, the old and the new verdict, plus a summary of the stars that
changed. The per-clip scores are not touched; only the statistics change.

Usage::

    python -m kalavinka.eval.recheck_results --results <PROJECT>/runs/eval_v1/RESULTS.md \\
        --out <PROJECT>/runs/eval_v1/RESULTS_song_bootstrap.md [--n 5000]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .compare import holm, load, paired_bootstrap

HEAD = re.compile(r"^\| A \| B \| (\S+) A \|")


def tables(md: str) -> List[Dict]:
    """[{metric, title, rows: [(A, B, old_star, old_line)]}] for every comparison table in the markdown."""
    out, cur, title = [], None, ""
    for line in md.split("\n"):
        if line.startswith("#"):
            title = line.lstrip("# ").strip()
        m = HEAD.match(line)
        if m:
            cur = {"metric": m.group(1), "title": title, "rows": []}
            out.append(cur)
            continue
        if cur is not None:
            if line.startswith("|---"):
                continue
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                cur["rows"].append((cells[0], cells[1], "*" in cells[4], line))
            else:
                cur = None
    return out


def file_for(metric: str) -> str:
    """Per-clip file of a metric: contrast scores (``*_sum/*_n``) or the main scores."""
    return "contrast.jsonl" if "_sum/" in metric else "scores.jsonl"


def recheck(results: Path, n: int = 5000) -> Tuple[str, List[str]]:
    """(new markdown, list of changed verdicts)."""
    root = results.parent
    md = results.read_text(encoding="utf-8")
    lines_out = [f"# Song-level bootstrap re-check of {results.name}", "",
                 "Same per-clip scores; whole songs resampled (n = %d), two-sided bootstrap p, Holm over each table's "
                 "rows; `*` = Holm-corrected p < 0.05. `old` = the clip-bootstrap star in the original table." % n, ""]
    changes = []
    for t in tables(md):
        f = file_for(t["metric"])
        res = []
        for a, b, old, _ in t["rows"]:
            pa, pb = root / a, root / b
            if not (pa / f).exists() or not (pb / f).exists():
                res.append((a, b, old, None))
                continue
            res.append((a, b, old, paired_bootstrap(load(pa / f), load(pb / f), t["metric"], n, unit="song")))
        ps = [r[3][6] for r in res if r[3] is not None]
        adj = iter(holm(ps))
        lines_out += [f"## {t['title']}: {t['metric']}", "",
                      "| A | B | A | B | B-A | 95% CI (songs) | p | p Holm | new | old | n (songs) |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, b, old, r in res:
            if r is None:
                lines_out.append(f"| {a} | {b} | | | | | | | missing | {'*' if old else ''} | |")
                continue
            va, vb, d, lo, hi, k, p, g = r
            ph = next(adj)
            new = ph < 0.05
            lines_out.append(f"| {a} | {b} | {va:.3f} | {vb:.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} | "
                             f"{ph:.3f} | {'*' if new else ''} | {'*' if old else ''} | {k} ({g}) |")
            if new != old:
                changes.append(f"{t['title']} / {t['metric']}: {a} vs {b}: {'*' if old else 'ns'} -> "
                               f"{'*' if new else 'ns'} (B-A {d:+.3f}, CI [{lo:+.3f}, {hi:+.3f}], p Holm {ph:.3f})")
        lines_out.append("")
    lines_out += ["## Verdicts that changed", ""] + ([f"- {c}" for c in changes] or ["- none"])
    return "\n".join(lines_out) + "\n", changes


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=5000)
    a = p.parse_args(argv)
    md, changes = recheck(a.results, a.n)
    a.out.write_text(md, encoding="utf-8")
    print(f"{len(changes)} verdicts changed -> {a.out}")
    for c in changes:
        print("  " + c)


if __name__ == "__main__":
    main()
