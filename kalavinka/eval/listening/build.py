"""Build the blind listening-test page (naturalness ratings, MUSHRA-lite).

What it makes: for ``--n-clips`` test clips, one trial per clip with one
stimulus per condition (the stage-1 comparison: score mode, I1 contour,
I2 contour at tau*, ground-truth F0, plus the real clip as a hidden anchor).
Listeners rate each stimulus 1-5 for how natural the singing sounds.

Blinding: every stimulus file gets an opaque name (salted hash), the page only
knows trial -> stimulus ids, and the page shuffles both the trial order and
the order of stimuli inside a trial per listener.  The mapping id -> (clip,
condition) is written to ``key.json`` next to the page and is **never
published**.

Fairness: clips are chosen by a seeded random draw among test clips that all
conditions rendered and that last ``--min-s``..``--max-s`` seconds (no
cherry-picking); every stimulus is loudness-normalised (RMS of the frames
above -40 dB to ``--target-db``, peak at most -1 dBFS) and encoded to
128 kbps MP3.

Outputs in ``--out``: ``index.html`` (the page, from ``page.html`` with the
manifest embedded), ``stimuli/<id>.mp3``, ``key.json`` (private) and
``selection.json`` (the chosen clips).

Usage::

    python -m kalavinka.eval.listening.build --export <PROJECT>/export/v1 --out <dir> \\
        --cond score=<render dir> --cond i1=<dir> --cond i2=<dir> --cond gt=<dir> [--n-clips 15]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
FFMPEG = str(Path.home() / "bin" / "ffmpeg")


def stim_id(salt: str, clip: str, cond: str) -> str:
    """Opaque, stable stimulus id."""
    return hashlib.sha1(f"{salt}|{clip}|{cond}".encode()).hexdigest()[:12]


def normalise(y: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Scale so the RMS of the active frames is ``target_db`` dBFS, peak <= -1 dBFS."""
    y = y.astype(np.float64)
    fr = y[: len(y) // 480 * 480].reshape(-1, 480) if len(y) >= 480 else y[None]
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    act = rms[rms > 10 ** (-40 / 20) * max(rms.max(), 1e-9)]
    level = np.sqrt((act ** 2).mean()) if len(act) else 1e-9
    g = 10 ** (target_db / 20) / max(level, 1e-9)
    peak = np.abs(y).max() * g
    if peak > 10 ** (-1 / 20):
        g *= 10 ** (-1 / 20) / peak
    return (y * g).astype(np.float32)


def select_clips(candidates: Dict[str, float], n: int, min_s: float, max_s: float, seed: int) -> List[str]:
    """Seeded random draw of ``n`` clip ids whose duration is within [min_s, max_s]."""
    ok = sorted(k for k, d in candidates.items() if min_s <= d <= max_s)
    if len(ok) < n:
        raise ValueError(f"only {len(ok)} clips between {min_s} and {max_s} s, need {n}")
    return sorted(random.Random(seed).sample(ok, n))


def build_manifest(clips: Sequence[str], conds: Sequence[str], salt: str, texts: Dict[str, str]) -> Dict:
    """What the page is allowed to know: trials of opaque stimulus ids (+ the lyric line)."""
    return {"version": 1, "trials": [{"t": f"t{k:02d}", "text": texts.get(c, ""),
                                      "stimuli": [stim_id(salt, c, cond) for cond in conds]}
                                     for k, c in enumerate(clips)]}


def encode(y: np.ndarray, sr: int, dst: Path) -> None:
    """Write ``y`` as 128 kbps MP3 via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, y, sr)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", tmp.name, "-codec:a", "libmp3lame",
                        "-b:a", "128k", str(dst)], check=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Select clips, write stimuli, key and page."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cond", action="append", required=True, help="name=render dir (repeat)")
    p.add_argument("--no-real", action="store_true", help="leave out the real-clip anchor")
    p.add_argument("--n-clips", type=int, default=15)
    p.add_argument("--min-s", type=float, default=4.0)
    p.add_argument("--max-s", type=float, default=10.0)
    p.add_argument("--target-db", type=float, default=-20.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    conds = dict(c.split("=", 1) for c in a.cond)
    items = {}
    for line in (a.export / f"{a.split}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            items[it["index"]] = it
    have = set(items)
    for d in conds.values():
        have &= {w.stem for w in Path(d).glob("*.wav")}
    cand = {k: int(items[k]["n_frames"]) / 50 for k in have}
    clips = select_clips(cand, a.n_clips, a.min_s, a.max_s, a.seed)
    names = list(conds) + ([] if a.no_real else ["real"])
    salt = secrets.token_hex(8)
    texts = {c: " ".join(w for w in items[c]["text"].split() if w != "<SP>") for c in clips}
    (a.out / "stimuli").mkdir(parents=True, exist_ok=True)
    key = {"salt": salt, "conditions": names, "stimuli": {}}
    for c in clips:
        for name in names:
            src = (a.export / items[c]["wav"]) if name == "real" else Path(conds[name]) / f"{c}.wav"
            y, sr = sf.read(src, dtype="float32")
            if y.ndim > 1:
                y = y.mean(1)
            sid = stim_id(salt, c, name)
            encode(normalise(y, a.target_db), sr, a.out / "stimuli" / f"{sid}.mp3")
            key["stimuli"][sid] = {"clip": c, "cond": name, "src": str(src)}
    manifest = build_manifest(clips, names, salt, texts)
    page = (HERE / "page.html").read_text(encoding="utf-8")
    blob = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c")
    (a.out / "index.html").write_text(page.replace("/*__MANIFEST__*/null", blob), encoding="utf-8")
    (a.out / "key.json").write_text(json.dumps(key, indent=1))
    (a.out / "selection.json").write_text(json.dumps({"clips": clips, "conditions": conds, "seed": a.seed,
                                                      "min_s": a.min_s, "max_s": a.max_s}, indent=1))
    print(f"{len(clips)} trials x {len(names)} stimuli -> {a.out}")


if __name__ == "__main__":
    main()
