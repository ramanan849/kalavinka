"""S13: SoulX-Singer export of accepted P1 clips (global command ``p1-export``).

* **Acceptance** (``accept_band``): each planned clip's confidence features
  against the ``[qc]`` thresholds -> accept / review / reject.  Thresholds are
  provisional until calibrated on the gold set (PLAN.md §8, step C3).
* **Splits** are assigned per *identified* song (LRCLIB reference id, else the
  song id), so two uploads of the same performance never straddle train/test.
* **Items** use the repo's ``export.clip_item`` (SoulX metadata format + Indic
  extras); audio is the separated vocal, 24 kHz, exactly ``n_frames * 480``.
* **Check**: every item satisfies SoulX's frame arithmetic
  (``int(sum(durations) * 50) == n_frames == len(f0)``).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..audio import FPS, HOP, SR, load_audio, save_audio
from ..schema import NoteAnn, SongAnnotation
from .manifest import Song
from .state import atomic_bytes, atomic_json, read_json

DEFAULT_QC = {"min_good_src": 0.8, "max_asr1": 0.1, "min_mean_conf": -2.5, "min_f0_agree": 0.85,
              "max_unlabelled": 0.05, "review_margin": 0.1}


def accept_band(f: Dict, q: Dict) -> str:
    """'accept', 'review' or 'reject' for one clip's confidence features."""
    q = {**DEFAULT_QC, **(q or {})}
    m = q["review_margin"]
    checks = [
        (f["good_src_share"], q["min_good_src"], True),
        (f["asr1_share"], q["max_asr1"], False),
        (f["mean_conf"] if f["mean_conf"] is not None else -99, q["min_mean_conf"], True),
        (f["f0_agree"], q["min_f0_agree"], True),
        (f["unlabelled_voiced"], q["max_unlabelled"], False),
    ]
    ok = all((v >= t) if hi else (v <= t) for v, t, hi in checks)
    if ok:
        return "accept"
    near = all((v >= t - m * abs(t or 1)) if hi else (v <= t + m * abs(t or 1)) for v, t, hi in checks)
    return "review" if near else "reject"


def split_of(key: str, test: float = 0.03, val: float = 0.03) -> str:
    """Deterministic split from a hash of the grouping key."""
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else ("val" if h < test + val else "train")


def clip_notes(ann: SongAnnotation, a: int, b: int) -> List[NoteAnn]:
    """Notes of ``ann`` cut to frames [a, b); edges filled with rests so durations
    tile the clip exactly."""
    out: List[NoteAnn] = []
    t = a
    for n in ann.notes:
        if n.end <= a or n.start >= b:
            continue
        s, e = max(n.start, a), min(n.end, b)
        if s > t:
            out.append(NoteAnn(t, s, 1, -1))
        out.append(NoteAnn(**{**n.__dict__, "start": s, "end": e}))
        t = e
    if t < b:
        out.append(NoteAnn(t, b, 1, -1))
    return out


def check_item(item: Dict) -> Optional[str]:
    """None if the item meets SoulX's frame arithmetic, else the problem."""
    durs = [float(x) for x in item["duration"].split()]
    n = item["n_frames"]
    if int(sum(durs) * FPS) != n:
        return f"durations {sum(durs) * FPS:.3f} frames != {n}"
    if len(item["f0"].split()) != n:
        return "f0 length"
    k = len(durs)
    if not (len(item["text"].split()) == len(item["phoneme"].split()) == len(item["note_pitch"].split())
            == len(item["note_type"].split()) == k):
        return "per-note field lengths differ"
    return None


def export(songs: Sequence[Song], work: Path, out: Path, qc: Dict, include_review: bool = False) -> Dict:
    """Write the dataset to ``out`` (wavs/, {train,val,test}.jsonl/.json,
    excluded.jsonl, stats.json) and return the stats."""
    from ..export import Clip, clip_item
    out.mkdir(parents=True, exist_ok=True)
    (out / "wavs").mkdir(exist_ok=True)
    items: Dict[str, List[Dict]] = defaultdict(list)
    excluded, bands = [], Counter()
    hours = Counter()
    for song in songs:
        sdir = work / "songs" / song.song_id
        cj = read_json(sdir / "clips.json", None)
        if not cj or not (sdir / "annotation.json").exists():
            continue
        ann = SongAnnotation.load(sdir)
        ident = read_json(sdir / "identity.json", {}) or {}
        group = (ident.get("best") or {}).get("ref_id") or song.song_id
        split = split_of(f"ref:{group}")
        vocals = None
        for k, c in enumerate(cj["clips"]):
            band = accept_band(c["features"], qc)
            bands[band] += 1
            if band == "reject" or (band == "review" and not include_review):
                excluded.append({"song_id": song.song_id, "clip": k, "band": band, **c["features"]})
                continue
            a, b = c["start"], c["end"]
            ann.singer_id = f"{song.song_id}#{c['cluster']}"      # linked across songs later (H6)
            ann.status = "auto" if band == "accept" else "review"
            clip = Clip(ann, a, b, clip_notes(ann, a, b))
            cid = f"{song.song_id}_c{k:03d}"
            item = clip_item(clip, cid, f"wavs/{cid}.wav")
            item.update({"split_group": group, "band": band, "identity": ident.get("status"),
                         "features": c["features"]})
            problem = check_item(item)
            if problem:
                excluded.append({"song_id": song.song_id, "clip": k, "band": "invalid", "problem": problem})
                continue
            if vocals is None:
                vocals = load_audio(sdir / "vocals.flac", SR)
            seg = vocals[a * HOP: b * HOP]
            seg = np.pad(seg, (0, max(0, (b - a) * HOP - len(seg))))
            save_audio(out / "wavs" / f"{cid}.wav", seg, SR)
            items[split].append(item)
            hours[split] += (b - a) / FPS / 3600
    for split in ("train", "val", "test"):
        lst = items.get(split, [])
        atomic_bytes(out / f"{split}.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lst).encode())
        atomic_json(out / f"{split}.json", lst)
    atomic_bytes(out / "excluded.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in excluded).encode())
    stats = {"clips": {k: len(v) for k, v in items.items()}, "hours": {k: round(v, 3) for k, v in hours.items()},
             "bands": dict(bands), "excluded": len(excluded), "qc": {**DEFAULT_QC, **(qc or {})}}
    atomic_json(out / "stats.json", stats)
    return stats
