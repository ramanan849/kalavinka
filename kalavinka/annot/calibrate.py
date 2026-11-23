"""Gold-set calibration of the P1 clip-acceptance thresholds (PLAN.md §8, step C3).

What it does
------------
The P1 pipeline plans clips and gives each one confidence features
(``clips.json``: ``good_src_share``, ``asr1_share``, ``mean_conf``,
``f0_agree``, ``unlabelled_voiced``); ``saregama.p1.export.accept_band`` turns
them into accept / review / reject with *provisional* thresholds.  This module
fits those thresholds on the hand-corrected gold set:

1. For every gold song (``gold/corrected/<id>.TextGrid`` -- also
   ``gold/pack/corrected/``, where the gold README's relative path leads),
   take its 75 s window (``gold/pack/<id>/meta.json``).
2. For every planned clip mostly inside that window, score the **current**
   predictions (``syllables.json``) against the corrected tiers inside the
   clip span: word errors (loose phonetic words, Levenshtein), syllable-onset
   errors (each gold onset vs the nearest predicted onset), and the share of
   the clip the gold marks as several voices (``AB`` in the singers tier).
3. Grid-search the two thresholds that carry most of the signal
   (``min_mean_conf``, ``min_good_src``; the other three stay at their defaults
   to avoid overfitting ~100-150 clips) for the most accepted clip time such
   that the pooled accepted clips reach the targets (default: word accuracy
   >= 0.95 and >= 90 % of onsets within 50 ms).
4. Bootstrap over songs for confidence intervals, and (if the whole run's
   ``clips.json`` files exist) the dataset-wide accepted hours at the choice.

Gold songs whose annotator record (``gold/.annotator/state.json``) shows less
than ``--min-listened`` % listened are treated as unreviewed and left out
unless ``--include-unreviewed`` (sensitivity check).

Outputs (``--out``, default ``<export>/calibration``): ``calibration.json``
(thresholds + metrics), ``clips.tsv`` (per-clip scores), ``report.md``.

Usage::

    python -m kalavinka.annot.calibrate --work <PROJECT>/work --gold <PROJECT>/gold \
        [--out DIR] [--word-acc 0.95] [--onset50 0.90] [--include-unreviewed]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from saregama.audio import FPS
from saregama.lyrics_eval import _edit_ops, loose_word
from saregama.p1.export import DEFAULT_QC, accept_band
from saregama.p1.state import read_json
from saregama.review import read_textgrid

GRID_CONF = [round(x, 2) for x in np.arange(-4.0, -0.95, 0.1)]
GRID_SRC = [round(x, 2) for x in np.arange(0.5, 1.001, 0.05)]


@dataclass
class ClipScore:
    """One planned clip inside a gold window, with its features and gold scores."""
    song_id: str
    clip: int
    start: int
    end: int
    features: Dict
    gold_words: int = 0
    word_errors: int = 0
    gold_onsets: int = 0
    onsets_50ms: int = 0
    onset_errors_ms: List[float] = field(default_factory=list)
    overlap_share: float = 0.0
    reviewed: bool = True

    @property
    def dur_s(self) -> float:
        """Clip length in seconds."""
        return (self.end - self.start) / FPS


# ---------------------------------------------------------------------------
# gold files
# ---------------------------------------------------------------------------
def corrected_files(gold: Path) -> Dict[str, Path]:
    """song_id -> corrected TextGrid, from gold/corrected/ and gold/pack/corrected/
    (the first wins when both exist)."""
    out: Dict[str, Path] = {}
    for d in (gold / "corrected", gold / "pack" / "corrected"):
        for p in sorted(d.glob("*.TextGrid")):
            out.setdefault(p.stem, p)
    return out


def reviewed_songs(gold: Path, min_listened: float) -> Dict[str, bool]:
    """song_id -> reviewed? from the annotator's state.json (listened_pct)."""
    st = read_json(gold / ".annotator" / "state.json", {}) or {}
    return {k: float(v.get("listened_pct", 0) or 0) >= min_listened for k, v in st.items()}


def _in_span(t0: int, ivs, a: int, b: int) -> List[Tuple[float, float, str]]:
    """Labelled intervals (seconds rel. to the window) whose midpoint falls in frames [a, b),
    returned in absolute frames."""
    out = []
    for x0, x1, txt in ivs:
        if not txt.strip():
            continue
        f0, f1 = t0 + x0 * FPS, t0 + x1 * FPS
        if a <= (f0 + f1) / 2 < b:
            out.append((f0, f1, txt.strip()))
    return out


def predicted_words(sylls: Sequence[Dict], a: int, b: int) -> List[str]:
    """Predicted lyric words (one per (phrase, word)) whose first syllable starts in [a, b)."""
    words, last = [], None
    for s in sylls:
        if s.get("kind") != "lyric":
            continue
        key = (s.get("phrase"), s.get("word"))
        if key == last:
            continue
        last = key
        if a <= s["start"] < b:
            words.append(s.get("word_text") or s.get("text", ""))
    return words


def _loose(ws: Sequence[str]) -> List[str]:
    """Loose phonetic forms (Latin words lower-cased; unreadable words kept verbatim)."""
    return [loose_word(w) or w.lower() for w in ws]


def score_clip(gold_tiers: Dict, t0: int, sylls: Sequence[Dict], a: int, b: int) -> Dict:
    """Word errors, onset errors and multi-voice share of predicted frames [a, b) vs gold."""
    gw = [t for _, _, t in _in_span(t0, gold_tiers.get("words", []), a, b) if t != "?"]
    pw = predicted_words(sylls, a, b)
    ops = _edit_ops(_loose(gw), _loose(pw))
    errs = sum(1 for o in ops if o[0] != "ok")
    gon = [f0 for f0, _, _ in _in_span(t0, gold_tiers.get("syllables", []), a, b)]
    pon = np.array([s["start"] for s in sylls if s.get("kind") in ("lyric", "vocable")
                    and a - FPS <= s["start"] < b + FPS], dtype=float)
    errs_ms = [float(np.min(np.abs(pon - g)) / FPS * 1000) if len(pon) else 1e9 for g in gon]
    sing = _in_span(t0, gold_tiers.get("singers", []), a, b)
    multi = sum(min(f1, b) - max(f0, a) for f0, f1, t in sing if len(t) > 1)
    return {"gold_words": len(gw), "word_errors": int(min(errs, max(len(gw), errs))),
            "gold_onsets": len(gon), "onsets_50ms": int(sum(e <= 50 for e in errs_ms)),
            "onset_errors_ms": [round(e, 1) for e in errs_ms],
            "overlap_share": round(float(multi) / max(b - a, 1), 3)}


def collect(work: Path, gold: Path, min_overlap: float = 0.8, min_listened: float = 90.0
            ) -> List[ClipScore]:
    """ClipScores for every planned clip that lies (>= ``min_overlap`` of it) inside a gold window."""
    rev = reviewed_songs(gold, min_listened)
    out: List[ClipScore] = []
    for sid, tg in corrected_files(gold).items():
        meta = read_json(gold / "pack" / sid / "meta.json", None)
        sdir = work / "songs" / sid
        cj, sj = read_json(sdir / "clips.json", None), read_json(sdir / "syllables.json", None)
        if not meta or not cj or not sj:
            continue
        t0, t1 = int(meta["start_frame"]), int(meta["end_frame"])
        tiers = read_textgrid(tg)
        sylls = sorted(sj["syllables"], key=lambda s: s["start"])
        for k, c in enumerate(cj["clips"]):
            a, b = int(c["start"]), int(c["end"])
            inside = max(0, min(b, t1) - max(a, t0))
            if inside < min_overlap * (b - a):
                continue
            sc = score_clip(tiers, t0, sylls, max(a, t0), min(b, t1))
            out.append(ClipScore(sid, k, a, b, c["features"], reviewed=rev.get(sid, False), **sc))
    return out


# ---------------------------------------------------------------------------
# pooled metrics and the threshold search
# ---------------------------------------------------------------------------
def pooled(clips: Sequence[ClipScore]) -> Dict:
    """Pooled word accuracy, onset-within-50 ms share and multi-voice share of ``clips``."""
    gw = sum(c.gold_words for c in clips)
    go = sum(c.gold_onsets for c in clips)
    dur = sum(c.dur_s for c in clips)
    return {"clips": len(clips), "seconds": round(dur, 1),
            "word_acc": round(1 - sum(c.word_errors for c in clips) / gw, 4) if gw else None,
            "onset50": round(sum(c.onsets_50ms for c in clips) / go, 4) if go else None,
            "multi_voice": round(sum(c.overlap_share * c.dur_s for c in clips) / dur, 4) if dur else None}


def qc_with(min_mean_conf: float, min_good_src: float, base: Optional[Dict] = None) -> Dict:
    """DEFAULT_QC (or ``base``) with the two searched thresholds replaced."""
    return {**DEFAULT_QC, **(base or {}), "min_mean_conf": min_mean_conf, "min_good_src": min_good_src}


def meets(m: Dict, word_acc: float, onset50: float) -> bool:
    """Do pooled metrics ``m`` reach both targets?"""
    return (m["word_acc"] is not None and m["word_acc"] >= word_acc
            and m["onset50"] is not None and m["onset50"] >= onset50)


def search(clips: Sequence[ClipScore], word_acc: float, onset50: float) -> Tuple[Dict, Dict, List[Dict]]:
    """Grid search: (best qc, its pooled metrics, the whole grid).  Best = most
    accepted seconds among settings meeting the targets; ties -> the stricter one.
    If no setting meets them, the setting with the highest word accuracy wins."""
    grid = []
    for mc in GRID_CONF:
        for gs in GRID_SRC:
            q = qc_with(mc, gs)
            acc = [c for c in clips if accept_band(c.features, q) == "accept"]
            m = pooled(acc)
            grid.append({"min_mean_conf": mc, "min_good_src": gs, **m, "ok": meets(m, word_acc, onset50)})
    ok = [g for g in grid if g["ok"]]
    if ok:
        best = max(ok, key=lambda g: (g["seconds"], g["min_mean_conf"], g["min_good_src"]))
    else:
        best = max(grid, key=lambda g: (g["word_acc"] or 0, g["seconds"]))
    q = qc_with(best["min_mean_conf"], best["min_good_src"])
    return q, best, grid


def bootstrap(clips: Sequence[ClipScore], q: Dict, n: int = 1000, seed: int = 0) -> Dict:
    """95 % CIs of the accepted clips' pooled metrics, resampling whole songs."""
    rng = np.random.default_rng(seed)
    songs = sorted({c.song_id for c in clips})
    by = {s: [c for c in clips if c.song_id == s and accept_band(c.features, q) == "accept"] for s in songs}
    wa, on = [], []
    for _ in range(n):
        pick = rng.choice(songs, size=len(songs), replace=True)
        m = pooled([c for s in pick for c in by[s]])
        if m["word_acc"] is not None:
            wa.append(m["word_acc"])
        if m["onset50"] is not None:
            on.append(m["onset50"])
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if v else None  # noqa: E731
    return {"word_acc_ci95": ci(wa), "onset50_ci95": ci(on), "n": n}


def dataset_yield(work: Path, q: Dict) -> Dict:
    """Clip time per band over every song with a clips.json, at thresholds ``q``."""
    sec = {"accept": 0.0, "review": 0.0, "reject": 0.0}
    songs = 0
    for cj_path in (work / "songs").glob("*/clips.json"):
        cj = read_json(cj_path, {}) or {}
        songs += 1
        for c in cj.get("clips", []):
            sec[accept_band(c["features"], q)] += c["features"]["dur_s"]
    return {"songs": songs, **{f"{k}_h": round(v / 3600, 2) for k, v in sec.items()}}


def calibrate(work: Path, gold: Path, out: Path, word_acc: float = 0.95, onset50: float = 0.90,
              include_unreviewed: bool = False, min_listened: float = 90.0) -> Dict:
    """Run the whole calibration and write calibration.json, clips.tsv, report.md into ``out``."""
    allc = collect(work, gold, min_listened=min_listened)
    clips = [c for c in allc if c.reviewed or include_unreviewed]
    q, best, grid = search(clips, word_acc, onset50)
    bands = {b: pooled([c for c in clips if accept_band(c.features, q) == b]) for b in ("accept", "review", "reject")}
    prov = {b: pooled([c for c in clips if accept_band(c.features, DEFAULT_QC) == b]) for b in ("accept", "review")}
    res = {"targets": {"word_acc": word_acc, "onset50": onset50},
           "gold_songs": sorted({c.song_id for c in clips}),
           "excluded_unreviewed": sorted({c.song_id for c in allc if not c.reviewed}) if not include_unreviewed else [],
           "qc": q, "met_targets": bool(best["ok"]), "at_choice": bands, "provisional_qc": DEFAULT_QC,
           "at_provisional": prov, "all_gold_clips": pooled(clips),
           "bootstrap": bootstrap(clips, q), "dataset_yield": dataset_yield(work, q),
           "dataset_yield_provisional": dataset_yield(work, DEFAULT_QC)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps({**res, "grid": grid}, indent=1, ensure_ascii=False))
    cols = ["song_id", "clip", "start", "end", "reviewed", "gold_words", "word_errors", "gold_onsets",
            "onsets_50ms", "overlap_share", "band_choice", "band_provisional"] + sorted(clips[0].features) if clips else []
    rows = ["\t".join(cols)]
    for c in clips:
        d = asdict(c)
        d.update({"band_choice": accept_band(c.features, q), "band_provisional": accept_band(c.features, DEFAULT_QC),
                  **c.features})
        rows.append("\t".join(str(d.get(k, "")) for k in cols))
    (out / "clips.tsv").write_text("\n".join(rows) + "\n")
    (out / "report.md").write_text(report_md(res))
    return res


def report_md(res: Dict) -> str:
    """A short human-readable summary of a calibration result."""
    L = ["# P1 threshold calibration on the gold set", "",
         f"Gold songs used: {len(res['gold_songs'])}; excluded as unreviewed: {res['excluded_unreviewed'] or 'none'}",
         f"Targets: word accuracy >= {res['targets']['word_acc']}, onsets within 50 ms >= {res['targets']['onset50']}"
         f" -> met: **{res['met_targets']}**", "",
         f"Chosen: min_mean_conf = {res['qc']['min_mean_conf']}, min_good_src = {res['qc']['min_good_src']}"
         f" (others at defaults)", "", "| band | clips | seconds | word acc | onset50 | multi-voice |",
         "|---|---|---|---|---|---|"]
    for name, m in [*(("chosen " + k, v) for k, v in res["at_choice"].items()),
                    *(("provisional " + k, v) for k, v in res["at_provisional"].items()),
                    ("all gold clips", res["all_gold_clips"])]:
        L.append(f"| {name} | {m['clips']} | {m['seconds']} | {m['word_acc']} | {m['onset50']} | {m['multi_voice']} |")
    b = res["bootstrap"]
    L += ["", f"Bootstrap over songs (accepted): word acc 95% CI {b['word_acc_ci95']}, onset50 CI {b['onset50_ci95']}",
          "", f"Dataset yield at the choice: {res['dataset_yield']}",
          f"Dataset yield at the provisional thresholds: {res['dataset_yield_provisional']}"]
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--gold", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--word-acc", type=float, default=0.95)
    p.add_argument("--onset50", type=float, default=0.90)
    p.add_argument("--min-listened", type=float, default=90.0)
    p.add_argument("--include-unreviewed", action="store_true")
    a = p.parse_args(argv)
    out = a.out or (a.work.parent / "export" / "calibration")
    res = calibrate(a.work, a.gold, out, a.word_acc, a.onset50, a.include_unreviewed, a.min_listened)
    print(report_md(res))


if __name__ == "__main__":
    main()
