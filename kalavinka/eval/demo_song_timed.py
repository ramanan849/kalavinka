"""Stitch a song's rendered lines with the song's *real* timing instead of a fixed gap.

Why: ``demo_song`` joins every line with the same ``--gap`` (0.6 s). Our clips are cut from the song at arbitrary
points, so some lines follow each other with no pause at all (e.g. ``तु झे झे अब मे`` | ``रा ...`` is the word
मेरा split across two clips). A fixed 0.6 s silence there sounds like the singer stops in the middle of a word,
in every system, base and ours alike. Here each gap is the real gap between the two clips in the song:

* consecutive clips (the next one starts within ``--max-gap`` s): the real gap; a gap of (almost) zero gets a short
  crossfade (``--xfade``) instead of a click;
* clips with skipped material between them (a later verse, a missing clip): ``--skip-gap`` s of silence.

Each line is still a separate render (one SoulX call per 3-16 s clip), so there can still be a change in breath or
level at a seam, but no invented pauses. Writes ``<out>/<song>_<name>_timed.wav`` per system and for the real
singer, plus ``<song>_timed_lines.json``.

Usage::

    python -m kalavinka.eval.demo_song_timed --export <PROJECT>/export/v1 --song 2017_tu_hi_hai__KyMNLIYRsR4 \\
        --out <PROJECT>/demo --sys zeroshot=<dir> --sys twostage_i2=<dir> [--max-gap 1.5 --skip-gap 1.0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from kalavinka.eval.demo_song import common_clips, load_mono
from kalavinka.eval.listening.build import normalise


def gaps_from_times(times: Sequence[Sequence[int]], max_gap: float, skip_gap: float) -> List[float]:
    """Seconds of silence after each clip but the last: the real gap (``time`` is in ms in the song) if the next
    clip follows within ``max_gap`` s, else ``skip_gap``. Overlaps count as 0."""
    out = []
    for (a0, a1), (b0, b1) in zip(times[:-1], times[1:]):
        g = max(0.0, (b0 - a1) / 1000.0)
        out.append(g if g <= max_gap else skip_gap)
    return out


def join(clips: Sequence[np.ndarray], gaps: Sequence[float], sr: int, xfade: float) -> Tuple[np.ndarray, List[float]]:
    """Join clips with the given gaps; gaps shorter than ``xfade`` become a crossfade of ``xfade`` s."""
    out = clips[0].astype(np.float32).copy()
    starts = [0.0]
    n = int(xfade * sr)
    for y, g in zip(clips[1:], gaps):
        y = y.astype(np.float32)
        if g < xfade and n > 0 and len(out) > n and len(y) > n:
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            starts.append((len(out) - n) / sr)
            out = np.concatenate([out[:-n], out[-n:] * (1 - ramp) + y[:n] * ramp, y[n:]])
        else:
            starts.append((len(out) + int(g * sr)) / sr)
            out = np.concatenate([out, np.zeros(int(g * sr), np.float32), y])
    return out, [round(s, 3) for s in starts]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--song", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sys", action="append", required=True, help="name=render dir (repeat)")
    p.add_argument("--max-gap", type=float, default=1.5, help="longer real gaps count as skipped material")
    p.add_argument("--skip-gap", type=float, default=1.0, help="silence where material was skipped")
    p.add_argument("--xfade", type=float, default=0.03)
    p.add_argument("--target-db", type=float, default=-20.0)
    a = p.parse_args(argv)
    systems = dict(s.split("=", 1) for s in a.sys)
    items = {}
    for line in (a.export / f"{a.split}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            items[it["index"]] = it
    clips = common_clips(a.song, items, [Path(d) for d in systems.values()])
    if not clips:
        raise SystemExit(f"no clips of {a.song} in all of {list(systems)}")
    gaps = gaps_from_times([items[c]["time"] for c in clips], a.max_gap, a.skip_gap)
    a.out.mkdir(parents=True, exist_ok=True)
    sources = {"real": [a.export / items[c]["wav"] for c in clips]}
    sources.update({n: [Path(d) / f"{c}.wav" for c in clips] for n, d in systems.items()})
    for name, paths in sources.items():
        ys, sr = [], None
        for path in paths:
            y, r = load_mono(path, sr)
            sr = r
            ys.append(normalise(y, a.target_db))
        track, starts = join(ys, gaps, sr, a.xfade)
        sf.write(a.out / f"{a.song}_{name}_timed.wav", track, sr)
    lines = [{"clip": c, "start_s": s, "gap_after_s": None if g is None else round(g, 2),
              "text": " ".join(w for w in items[c]["text"].split() if w != "<SP>")}
             for c, s, g in zip(clips, starts, list(gaps) + [None])]
    (a.out / f"{a.song}_timed_lines.json").write_text(json.dumps(
        {"song": a.song, "systems": systems, "max_gap": a.max_gap, "skip_gap": a.skip_gap, "lines": lines},
        ensure_ascii=False, indent=1))
    print(f"{len(clips)} lines, {len(track) / sr:.1f} s per track, gaps {[round(g, 2) for g in gaps]} -> {a.out}")


if __name__ == "__main__":
    main()
