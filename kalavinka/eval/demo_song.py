"""Stitch one song's rendered clips, in song order, into demo tracks (one per system).

Why: every evaluation render is a single 4-10 s lyric line.  For the talk we
want to hear a longer stretch of one held-out song, sung by each system and by
the real singer, with the same lines in the same order.

What it does: takes the clips of ``--song`` that **every** given render dir
has (plus the real clip from the export), orders them by clip number, loudness-
normalises each clip (same rule as the listening test), joins them with
``--gap`` seconds of silence and writes ``<out>/<song>_<name>.wav`` per system
and ``<out>/<song>_real.wav``.  ``<out>/<song>_lines.json`` lists the clips
used, their lyrics and their start times in the stitched tracks.  Lines missing
from any system are left out of all tracks, so the tracks stay aligned.

Usage::

    python -m kalavinka.eval.demo_song --export <PROJECT>/export/v1 --song 2021_bedardi_se_pyaar_ka__y_RHQ4mBDKA \\
        --out <PROJECT>/demo --sys score=<render dir> --sys i2=<render dir> [--gap 0.6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf

from kalavinka.eval.listening.build import normalise


def clip_number(index: str) -> int:
    """Clip number from an export index like ``<song>_c012``."""
    return int(index.rsplit("_c", 1)[1])


def common_clips(song: str, items: Dict[str, Dict], dirs: Sequence[Path]) -> List[str]:
    """Clip ids of ``song`` that every render dir has, in song order."""
    have = {k for k in items if k.rsplit("_c", 1)[0] == song}
    for d in dirs:
        have &= {w.stem for w in d.glob("*.wav")}
    return sorted(have, key=clip_number)


def stitch(clips: Sequence[np.ndarray], sr: int, gap_s: float) -> (np.ndarray, List[float]):
    """Join clips with ``gap_s`` of silence; return the track and each clip's start time (s)."""
    gap = np.zeros(int(gap_s * sr), dtype=np.float32)
    parts, starts, t = [], [], 0.0
    for y in clips:
        starts.append(round(t, 3))
        parts += [y, gap]
        t += (len(y) + len(gap)) / sr
    return (np.concatenate(parts[:-1]) if parts else np.zeros(0, np.float32)), starts


def load_mono(path: Path, sr: Optional[int] = None) -> (np.ndarray, int):
    """Mono float32 audio; checks the sample rate when ``sr`` is given."""
    y, r = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    if sr is not None and r != sr:
        raise ValueError(f"{path}: {r} Hz, expected {sr}")
    return y, r


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--song", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sys", action="append", required=True, help="name=render dir (repeat)")
    p.add_argument("--gap", type=float, default=0.6)
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
    a.out.mkdir(parents=True, exist_ok=True)
    sources = {"real": [a.export / items[c]["wav"] for c in clips]}
    sources.update({n: [Path(d) / f"{c}.wav" for c in clips] for n, d in systems.items()})
    sr = None
    for name, paths in sources.items():
        ys = []
        for path in paths:
            y, r = load_mono(path)
            if sr is None:
                sr = r
            if r != sr:
                raise SystemExit(f"{path}: {r} Hz, expected {sr}")
            ys.append(normalise(y, a.target_db))
        track, starts = stitch(ys, sr, a.gap)
        sf.write(a.out / f"{a.song}_{name}.wav", track, sr)
    lines = [{"clip": c, "start_s": s, "text": " ".join(w for w in items[c]["text"].split() if w != "<SP>")}
             for c, s in zip(clips, starts)]
    (a.out / f"{a.song}_lines.json").write_text(json.dumps({"song": a.song, "systems": systems, "gap_s": a.gap,
                                                            "lines": lines}, ensure_ascii=False, indent=1))
    print(f"{len(clips)} lines, {len(track) / sr:.1f} s per track -> {a.out}")


if __name__ == "__main__":
    main()
