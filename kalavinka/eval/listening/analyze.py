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
  the first condition). With fewer than ``MIN_LISTENERS`` (8) listeners the
  listener resampling has almost no variance to draw on, so pairs are reported
  as **descriptive only** (no "significant" flag); from 8 listeners on, each
  pair gets a two-sided bootstrap p and a Holm-corrected p over the pairs, and
  "significant" means Holm p < 0.05;
* duplicate response files (the same listener's answers saved twice, e.g. the
  online store and a downloaded file) are counted once;
* listener screening: each listener's mean rating of the hidden real clip;
  listeners whose real-clip mean is below ``--min-real`` (default 3.0 on the
  1-5 scale) are reported and, only with ``--screen``, left out (the anchor
  tells whether someone was listening). Screening is off by default because it
  can also drop listeners who genuinely heard the separated real vocals as
  imperfect; the report states the threshold and whether it was applied;
* listeners' free-text comments: per page (resolved to the clip, never to a
  condition, since a comment can refer to any of the five versions) and overall.

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


MIN_LISTENERS = 8


def load_responses(path: Path, dedupe: bool = True) -> List[Dict]:
    """Response documents from a file (one doc or a list) or every *.json in a directory; identical answer sets
    (same initials and same ratings) are kept once."""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    out, seen = [], set()
    for f in files:
        d = json.loads(f.read_text())
        docs = d if isinstance(d, list) else [d]
        for doc in docs:
            body = doc.get("data", doc) if isinstance(doc, dict) else doc
            if isinstance(body, dict) and "ratings" in body:
                fp = json.dumps([body.get("initials", ""), body["ratings"]], sort_keys=True)
                if dedupe and fp in seen:
                    continue
                seen.add(fp)
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
    enough = len(listeners) >= MIN_LISTENERS
    out = {"listeners": len(listeners), "clips": len(clips),
           "inference": "bootstrap p + Holm" if enough else f"descriptive only (fewer than {MIN_LISTENERS} listeners)",
           "conditions": {c: {"mean": round(m[c], 3), "ci95": ci(draws[c])} for c in conds}, "pairs": {}}
    ps = {}
    for a, b in pairs:
        k = f"{b}-{a}"
        xs = ddraws[k]
        ps[k] = min(1.0, 2 * min(sum(x <= 0 for x in xs), sum(x >= 0 for x in xs)) / len(xs)) if xs else None
    keys = [k for k in ps if ps[k] is not None]
    order = sorted(keys, key=lambda k: ps[k])
    adj, run = {}, 0.0
    for rank, k in enumerate(order):                 # Holm
        run = max(run, min(1.0, (len(order) - rank) * ps[k]))
        adj[k] = run
    for a, b in pairs:
        k = f"{b}-{a}"
        paired = [(cell[x][b] - cell[x][a]) for x in cell if a in cell[x] and b in cell[x]]
        mean = sum(paired) / len(paired) if paired else None
        lo, hi = ci(ddraws[k])
        out["pairs"][k] = {"mean": round(mean, 3) if mean is not None else None, "ci95": [lo, hi],
                           "p": None if ps[k] is None else round(ps[k], 4),
                           "p_holm": None if k not in adj else round(adj[k], 4),
                           "significant": (k in adj and adj[k] < 0.05) if enough else None}
    return out


def screen(rows: Sequence[Tuple[int, str, str, int]], min_real: float) -> Tuple[List[int], Dict[int, float]]:
    """Listeners whose mean rating of the real clip is below ``min_real``, and every listener's real mean."""
    acc: Dict[int, List[int]] = {}
    for li, _, c, v in rows:
        if c == "real":
            acc.setdefault(li, []).append(v)
    real = {li: sum(v) / len(v) for li, v in acc.items()}
    return [li for li, m in real.items() if m < min_real], real


def comments(key: Dict, responses: Sequence[Dict]) -> List[Dict]:
    """Free-text comments as {listener, initials, clip, text}; clip is None for the overall comment."""
    stim = key["stimuli"]
    out = []
    for li, r in enumerate(responses):
        for tid, text in (r.get("comments") or {}).items():
            if not isinstance(text, str) or not text.strip():
                continue
            ids = (r.get("stimOrder") or {}).get(tid) or []
            clip = next((stim[s]["clip"] for s in ids if s in stim), None)
            out.append({"listener": li, "initials": r.get("initials", ""), "clip": None if tid == "_overall" else clip,
                        "text": text.strip()})
    return out


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
    responses = load_responses(a.responses)
    rows = table(key, responses, a.partial)
    flagged, real = screen(rows, a.min_real)
    if a.screen:
        rows = [r for r in rows if r[0] not in flagged]
    conds = key["conditions"]
    pairs = [tuple(x.split(":")) for x in a.pairs.split(",") if x] or [(conds[0], c) for c in conds[1:]]
    res = bootstrap(rows, pairs, a.n)
    res["screening"] = {"min_real": a.min_real, "flagged": flagged, "applied": a.screen,
                        "real_means": {str(k): round(v, 2) for k, v in real.items()}}
    res["comments"] = comments(key, responses)
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
