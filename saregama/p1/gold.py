"""Gold set (PLAN.md step C2/C3): a pack of pre-filled Praat TextGrids for a
person to correct, the import of the corrections, and scoring against them.

``p1-goldpack``  picks one 75 s window per chosen song (the stretch with the
most aligned syllables), cuts the vocal and the mix, and writes a TextGrid
with the pipeline's predictions as tiers:

    words      one interval per sung word (correct text and boundaries; add
               missed words, delete invented ones)
    syllables  one interval per syllable, ``text`` (move boundaries; text only
               needs fixing where the word is wrong)
    vocables   aakar / humming spans: ``आ`` ``ओ`` ``म्म`` ... or ``mel`` for a
               melisma continuing the previous syllable (add missed ones)
    singers    one interval per singing turn: ``A``, ``B`` ... per singer, ``AB``
               where two or more voices overlap

``p1-goldeval`` scores the current predictions against the corrected files
(``gold/corrected/<song_id>.TextGrid``): word error rate, syllable-onset error,
vocable recall / precision, single-voice precision and singer-turn purity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..audio import FPS, SR, load_audio, save_audio
from ..review import read_textgrid, write_textgrid
from .manifest import Song
from .state import atomic_json, read_json

WINDOW_S = 75.0
README = """P1 gold set: correcting the pipeline's predictions
=================================================

Install Praat (https://www.praat.org).  For each folder:
  1. Praat > Open > Read from file: vocals.wav AND <song_id>.TextGrid
     (mix.wav is the original song, to listen when the vocal is unclear).
  2. Select both > View & Edit.  Play a stretch with Tab / click the bars.
  3. Correct the tiers (boundaries: drag; text: click the interval, type):
       words      every sung word, in the order sung, spelled as sung
                  (Devanagari for Hindi, Latin for English words).  Repeats and
                  half lines are written as sung.
       syllables  boundaries at syllable onsets; you only need to fix the TEXT
                  where you changed the word.
       vocables   every aakar / humming stretch that is not a word:
                  आ / ओ / ई / ए / ऊ / म्म (the vowel heard), or  mel  when it is a
                  long vowel continuing the previous syllable.
       singers    who sings: A, B, C ... (same letter = same person within this
                  song), AB when two or more voices sing at once, empty = nobody.
  4. Save: File > Save TextGrid as text file, into  ../corrected/<song_id>.TextGrid
     (same file name).  Copy the corrected/ folder back to Prajna:
        P1/gold/corrected/

Rough effort: 15-25 min per 75 s window.  Precision matters more than speed:
when you can't tell, leave the prediction as it is and add a word "?" in words.
"""


def _intervals(items: Sequence[Dict], t0: int, t1: int, label) -> List[Tuple[float, float, str]]:
    """Frames -> seconds intervals inside [t0, t1), relative to t0."""
    out = []
    for s in items:
        a, b = max(s["start"], t0), min(s["end"], t1)
        if b > a:
            out.append(((a - t0) / FPS, (b - t0) / FPS, label(s)))
    return out


def best_window(sylls: Sequence[Dict], n: int, win: int) -> int:
    """Start frame of the ``win``-frame window with the most syllables."""
    if n <= win:
        return 0
    starts = np.array([s["start"] for s in sylls])
    best, best_c = 0, -1
    for a in range(0, n - win, FPS * 5):
        c = int(((starts >= a) & (starts < a + win)).sum())
        if c > best_c:
            best, best_c = a, c
    return best


def make_pack(songs: Sequence[Song], work: Path, out: Path) -> List[str]:
    """Write the gold pack for ``songs`` into ``out/pack`` (+ README); returns ids."""
    from .text_stages import GOOD_SRC  # noqa: F401  (documents the src labels used)
    pack = out / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    (out / "corrected").mkdir(exist_ok=True)
    (out / "README.txt").write_text(README, encoding="utf-8")
    done = []
    for song in songs:
        sdir = work / "songs" / song.song_id
        sj = read_json(sdir / "syllables.json", None)
        if not sj:
            continue
        sylls = [s for s in sj["syllables"] if s["kind"] in ("lyric", "vocable")]
        n = sj["frames"]
        win = int(WINDOW_S * FPS)
        t0 = best_window(sylls, n, win)
        t1 = min(n, t0 + win)
        d = pack / song.song_id
        d.mkdir(exist_ok=True)
        voc = load_audio(sdir / "vocals.flac", SR)
        save_audio(d / "vocals.wav", voc[t0 * SR // FPS: t1 * SR // FPS], SR)
        mix = load_audio(song.audio, SR)
        save_audio(d / "mix.wav", mix[t0 * SR // FPS: t1 * SR // FPS], SR)
        lyr = [s for s in sylls if s["kind"] == "lyric"]
        words: List[Dict] = []
        for s in lyr:
            key = (s["phrase"], s["word"])
            if words and words[-1]["key"] == key:
                words[-1]["end"] = s["end"]
            else:
                words.append({"key": key, "start": s["start"], "end": s["end"], "text": s["word_text"]})
        voc_items = [s for s in sylls if s["kind"] == "vocable"] + \
                    [{"start": s["start"], "end": s["end"], "text": "mel"} for s in lyr if s.get("melisma")]
        vj = read_json(sdir / "voices.json", {}) or {}
        letters = "ABCDEFGHIJ"
        singers = [{"start": int(s["start"] * FPS), "end": int(s["end"] * FPS),
                    "text": letters[s["cluster"] % len(letters)]} for s in vj.get("segments", [])]
        xmax = (t1 - t0) / FPS
        tiers = [("words", _intervals(words, t0, t1, lambda s: s["text"])),
                 ("syllables", _intervals(lyr, t0, t1, lambda s: s["text"])),
                 ("vocables", _intervals(voc_items, t0, t1, lambda s: s["text"])),
                 ("singers", _intervals(singers, t0, t1, lambda s: s["text"]))]
        write_textgrid(d / f"{song.song_id}.TextGrid", xmax, tiers)
        atomic_json(d / "meta.json", {"song_id": song.song_id, "start_frame": t0, "end_frame": t1,
                                      "fps": FPS, "title": song.title})
        done.append(song.song_id)
    return done


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _labelled(tier: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """Non-empty intervals of a tier."""
    return [(a, b, t.strip()) for a, b, t in tier if t.strip()]


def score_song(gold_tg: Path, pred_tg: Path) -> Dict:
    """Metrics of one predicted TextGrid against its corrected version."""
    from ..lyrics_eval import _edit_ops, loose_word
    g, p = read_textgrid(gold_tg), read_textgrid(pred_tg)
    gw, pw = _labelled(g.get("words", [])), _labelled(p.get("words", []))
    gl = [loose_word(t) or t for _, _, t in gw]
    pl = [loose_word(t) or t for _, _, t in pw]
    ops = _edit_ops(gl, pl)
    errs = sum(1 for o in ops if o[0] != "ok")
    res: Dict = {"words": len(gw), "wer": round(errs / max(len(gw), 1), 3)}
    # syllable onsets: each gold syllable vs the nearest predicted onset
    gs = [a for a, _, _ in _labelled(g.get("syllables", []))]
    ps = np.array([a for a, _, _ in _labelled(p.get("syllables", []))])
    if gs and len(ps):
        e = np.array([np.min(np.abs(ps - a)) for a in gs])
        res.update({"syllables": len(gs), "onset_median_ms": round(float(np.median(e)) * 1000, 1),
                    "onset_within_50ms": round(float((e <= 0.05).mean()), 3),
                    "onset_within_100ms": round(float((e <= 0.1).mean()), 3)})
    # vocables: overlap-based recall / precision
    gv, pv = _labelled(g.get("vocables", [])), _labelled(p.get("vocables", []))

    def hit(x, ys):
        """Does interval x overlap any of ys by at least half of x?"""
        return any(min(x[1], y[1]) - max(x[0], y[0]) >= 0.5 * (x[1] - x[0]) for y in ys)
    res["vocable_recall"] = round(sum(hit(x, pv) for x in gv) / len(gv), 3) if gv else None
    res["vocable_precision"] = round(sum(hit(x, gv) for x in pv) / len(pv), 3) if pv else None
    # singers: single-voice precision (predicted single that gold calls one voice)
    gs_ = _labelled(g.get("singers", []))
    ps_ = _labelled(p.get("singers", []))
    if gs_ and ps_:
        t = np.arange(0, max(b for _, b, _ in gs_ + ps_), 0.02)

        def lab(ivs):
            """Label per 20 ms frame."""
            out = np.array([""] * len(t), dtype=object)
            for a, b, x in ivs:
                out[(t >= a) & (t < b)] = x
            return out
        G, P = lab(gs_), lab(ps_)
        pred_single = P != ""
        gold_single = np.array([len(x) == 1 for x in G])
        res["single_voice_precision"] = round(float(gold_single[pred_single].mean()), 3) if pred_single.any() else None
        # purity: within each predicted label, share of its majority gold singer
        pur = []
        for x in set(P[pred_single].tolist()):
            gg = G[(P == x) & gold_single]
            if len(gg):
                pur.append(max(np.sum(gg == y) for y in set(gg.tolist())) / len(gg))
        res["singer_purity"] = round(float(np.mean(pur)), 3) if pur else None
    return res


def evaluate(gold_dir: Path) -> Dict:
    """Score every corrected TextGrid against the pack's prediction; writes
    ``gold/eval.json`` with per-song metrics and medians."""
    per = {}
    for g in sorted((gold_dir / "corrected").glob("*.TextGrid")):
        p = gold_dir / "pack" / g.stem / g.name
        if p.exists():
            per[g.stem] = score_song(g, p)
    keys = sorted({k for v in per.values() for k, x in v.items() if isinstance(x, (int, float))})
    med = {k: round(float(np.median([v[k] for v in per.values() if v.get(k) is not None])), 3)
           for k in keys if any(v.get(k) is not None for v in per.values())}
    res = {"songs": len(per), "median": med, "per_song": per}
    atomic_json(gold_dir / "eval.json", res)
    return res
