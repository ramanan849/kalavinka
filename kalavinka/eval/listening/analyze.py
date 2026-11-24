"""Analyse the listening test: naturalness per condition with bootstrap CIs.

Inputs: the private ``key.json`` written by ``build.py`` and the responses,
either one JSON file per listener (what the page offers when it cannot save
online) or one JSON file holding a list of response documents (an export of
the page's ``responses`` store), or a directory of such files.

Outputs (stdout and ``--out``):

* per condition: mean rating with a 95 % CI from a two-way bootstrap
  (resampling listeners and clips together, 5,000 rounds, seeded);
* paired differences between conditions (same listener, same clip) with
  their CIs, for the pairs given by ``--pairs`` (default: every pair against
  the first condition);
* listener screening: each listener's mean rating of the hidden real clip;
  listeners whose real-clip mean is below ``--min-real`` are reported and,
  with ``--screen``, left out (the anchor tells whether someone was listening).

Only complete responses count (every stimulus rated), unless ``--partial``.

Usage::

    python -m kalavinka.eval.listening.analyze --key <dir>/key.json --responses <file or dir> [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def load_responses(path: Path) -> List[Dict]:
    """Response documents from a file (one doc or a list) or every *.json in a directory."""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    out = []
    for f in files:
        d = json.loads(f.read_text())
        docs = d if isinstance(d, list) else [d]
        for doc in docs:
            body = doc.get("data", doc) if isinstance(doc, dict) else doc
            if isinstance(body, dict) and "ratings" in body:
                out.append(body)
    return out


def table(key: Dict, responses: Sequence[Dict], partial: bool = False) -> List[Tuple[int, str, str, int]]:
    """(listener, clip, condition, rating) rows from complete (or all) responses."""
    stim = key["stimuli"]
    rows = []
    for li, r in enumerate(responses):
        rated = {s: v for s, v in r["ratings"].items() if s in stim}
        if not partial and len(rated) < len(stim):
            continue
        rows += [(li, stim[s]["clip"], stim[s]["cond"], int(v)) for s, v in rated.items()]
    return rows


def means(rows: Sequence[Tuple[int, str, str, int]]) -> Dict[str, float]:
    """Mean rating per condition."""
    acc: Dict[str, List[int]] = {}
    for _, _, c, v in rows:
        acc.setdefault(c, []).append(v)
    return {c: sum(v) / len(v) for c, v in acc.items()}


def bootstrap(rows: Sequence[Tuple[int, str, str, int]], pairs: Sequence[Tuple[str, str]], n: int = 5000,
              seed: int = 0) -> Dict:
    """Two-way (listener x clip) bootstrap of condition means and paired differences."""
    listeners = sorted({r[0] for r in rows})
    clips = sorted({r[1] for r in rows})
    cell: Dict[Tuple[int, str], Dict[str, int]] = {}
    for li, cl, c, v in rows:
        cell.setdefault((li, cl), {})[c] = v
    rng = random.Random(seed)
    conds = sorted({r[2] for r in rows})
    draws = {c: [] for c in conds}
    ddraws = {f"{b}-{a}": [] for a, b in pairs}
    for _ in range(n):
        L = [rng.choice(listeners) for _ in listeners]
        C = [rng.choice(clips) for _ in clips]
        sums = {c: [0.0, 0] for c in conds}
        dsum = {k: [0.0, 0] for k in ddraws}
        for li in L:
            for cl in C:
                cc = cell.get((li, cl))
                if not cc:
                    continue
                for c, v in cc.items():
                    sums[c][0] += v; sums[c][1] += 1
                for a, b in pairs:
                    if a in cc and b in cc:
                        k = f"{b}-{a}"; dsum[k][0] += cc[b] - cc[a]; dsum[k][1] += 1
        for c in conds:
            if sums[c][1]:
                draws[c].append(sums[c][0] / sums[c][1])
        for k in ddraws:
            if dsum[k][1]:
                ddraws[k].append(dsum[k][0] / dsum[k][1])

    def ci(xs: List[float]) -> List[float]:
        """2.5 / 97.5 percentiles."""
        xs = sorted(xs)
        return [round(xs[int(0.025 * len(xs))], 3), round(xs[int(0.975 * len(xs)) - 1], 3)] if xs else [None, None]
    m = means(rows)
    out = {"listeners": len(listeners), "clips": len(clips),
           "conditions": {c: {"mean": round(m[c], 3), "ci95": ci(draws[c])} for c in conds}, "pairs": {}}
    for a, b in pairs:
        k = f"{b}-{a}"
        paired = [(cell[x][b] - cell[x][a]) for x in cell if a in cell[x] and b in cell[x]]
        mean = sum(paired) / len(paired) if paired else None
        lo, hi = ci(ddraws[k])
        out["pairs"][k] = {"mean": round(mean, 3) if mean is not None else None, "ci95": [lo, hi],
                           "significant": lo is not None and (lo > 0 or hi < 0)}
    return out


def screen(rows: Sequence[Tuple[int, str, str, int]], min_real: float) -> Tuple[List[int], Dict[int, float]]:
    """Listeners whose mean rating of the real clip is below ``min_real``, and every listener's real mean."""
    acc: Dict[int, List[int]] = {}
    for li, _, c, v in rows:
        if c == "real":
            acc.setdefault(li, []).append(v)
    real = {li: sum(v) / len(v) for li, v in acc.items()}
    return [li for li, m in real.items() if m < min_real], real


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--pairs", default="", help="comma list of a:b (difference b-a); default: all vs the first condition")
    p.add_argument("--min-real", type=float, default=3.0)
    p.add_argument("--screen", action="store_true")
    p.add_argument("--partial", action="store_true")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    key = json.loads(a.key.read_text())
    rows = table(key, load_responses(a.responses), a.partial)
    flagged, real = screen(rows, a.min_real)
    if a.screen:
        rows = [r for r in rows if r[0] not in flagged]
    conds = key["conditions"]
    pairs = [tuple(x.split(":")) for x in a.pairs.split(",") if x] or [(conds[0], c) for c in conds[1:]]
    res = bootstrap(rows, pairs, a.n)
    res["screening"] = {"min_real": a.min_real, "flagged": flagged, "applied": a.screen,
                        "real_means": {str(k): round(v, 2) for k, v in real.items()}}
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
