"""Dataset v1 export: the P1 SoulX export with calibrated bands and a per-frame confidence track.

Wraps ``saregama.p1.export`` (unchanged) and reuses its helpers
(``accept_band``, ``split_of``, ``clip_notes``, ``check_item``) and the repo's
``saregama.export.clip_item``.  Differences from ``p1-export``:

* thresholds come from the gold calibration (``--calibration``
  ``calibration.json`` -> ``qc``), passed straight to ``accept_band`` (the
  ``[qc]`` TOML path cannot override them);
* accepted **and review** clips are exported, each with its ``band``, because
  the confidence-aware fine-tune (``kalavinka.sx.train --weighting conf``)
  down-weights instead of dropping;
* every item gets ``conf``: one value per 50 fps frame -- the MMS_FA
  log-probability of the syllable covering it (``syllables.json``; a vocable's
  evidence share as ``log(evidence)``), ``0`` on rests (unvoiced, no syllable)
  and ``-99`` on voiced frames no syllable explains (unlabelled singing);
* ``year`` (from the song id) for the stage-1 era embedding.

Splits are the P1 ones: by identified reference (else song), 3 % test, 3 % val.

Usage::

    python -m kalavinka.annot.export_v1 --work <PROJECT>/work --out <PROJECT>/export/v1 \
        --calibration <PROJECT>/export/calibration/calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from saregama.audio import FPS, HOP, SR, load_audio, save_audio
from saregama.p1 import manifest
from saregama.p1.export import DEFAULT_QC, accept_band, check_item, clip_notes, split_of
from saregama.p1.state import atomic_bytes, atomic_json, read_json
from saregama.schema import SongAnnotation

UNLABELLED, REST = -99.0, 0.0


def conf_track(sylls: Sequence[Dict], f0: np.ndarray, a: int, b: int) -> np.ndarray:
    """Per-frame confidence for frames [a, b) (see the module doc)."""
    n = b - a
    out = np.where(f0[a:b] > 0, UNLABELLED, REST).astype(np.float32)
    for s in sylls:
        if s.get("kind") not in ("lyric", "vocable"):
            continue
        s0, s1 = max(int(s["start"]), a), min(int(s["end"]), b)
        if s1 <= s0:
            continue
        c = s.get("conf")
        if c is None:
            v = -20.0
        elif s["kind"] == "vocable":
            v = math.log(max(float(c), 1e-6))
        else:
            v = float(c)
        out[s0 - a: s1 - a] = v
    return out[:n]


def export_v1(work: Path, out: Path, qc: Dict, bands=("accept", "review"),
              songs: Optional[Sequence[manifest.Song]] = None) -> Dict:
    """Write wavs/, {train,val,test}.jsonl, excluded.jsonl and stats.json into ``out``."""
    from saregama.export import Clip, clip_item
    songs = songs if songs is not None else manifest.read(work / "songs.tsv")
    dup = (read_json(work / "dupes.json", {}) or {}).get("duplicate_of", {})
    out.mkdir(parents=True, exist_ok=True)
    (out / "wavs").mkdir(exist_ok=True)
    items: Dict[str, List[Dict]] = defaultdict(list)
    excluded, band_n, secs = [], Counter(), Counter()
    for song in songs:
        if song.song_id in dup:
            continue
        sdir = work / "songs" / song.song_id
        cj, sj = read_json(sdir / "clips.json", None), read_json(sdir / "syllables.json", None)
        if not cj or not sj or not (sdir / "annotation.json").exists():
            continue
        ann = SongAnnotation.load(sdir)
        ident = read_json(sdir / "identity.json", {}) or {}
        group = (ident.get("best") or {}).get("ref_id") or song.song_id
        split = split_of(f"ref:{group}")
        vocals = None
        for k, c in enumerate(cj["clips"]):
            band = accept_band(c["features"], qc)
            band_n[band] += 1
            if band not in bands:
                excluded.append({"song_id": song.song_id, "clip": k, "band": band, **c["features"]})
                continue
            a, b = int(c["start"]), int(c["end"])
            ann.singer_id = f"{song.song_id}#{c['cluster']}"
            ann.status = "auto" if band == "accept" else "review"
            cid = f"{song.song_id}_c{k:03d}"
            item = clip_item(Clip(ann, a, b, clip_notes(ann, a, b)), cid, f"wavs/{cid}.wav")
            conf = conf_track(sj["syllables"], ann.f0, a, b)
            item.update({"split_group": group, "band": band, "identity": ident.get("status"),
                         "features": c["features"], "year": song.year,
                         "conf": " ".join(f"{x:.2f}" for x in conf)})
            problem = check_item(item) or (None if len(conf) == item["n_frames"] else "conf length")
            if problem:
                excluded.append({"song_id": song.song_id, "clip": k, "band": "invalid", "problem": problem})
                continue
            if vocals is None:
                vocals = load_audio(sdir / "vocals.flac", SR)
            seg = vocals[a * HOP: b * HOP]
            save_audio(out / "wavs" / f"{cid}.wav", np.pad(seg, (0, max(0, (b - a) * HOP - len(seg)))), SR)
            items[split].append(item)
            secs[(split, band)] += (b - a) / FPS
    for split in ("train", "val", "test"):
        atomic_bytes(out / f"{split}.jsonl",
                     "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items.get(split, [])).encode())
    atomic_bytes(out / "excluded.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in excluded).encode())
    stats = {"clips": {k: len(v) for k, v in items.items()},
             "hours": {f"{s}/{b}": round(v / 3600, 3) for (s, b), v in sorted(secs.items())},
             "bands_all_planned": dict(band_n), "excluded": len(excluded), "qc": qc,
             "songs_with_clips": len({x["song_id"] for v in items.values() for x in v})}
    atomic_json(out / "stats.json", stats)
    return stats


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--calibration", type=Path, default=None, help="calibration.json (else provisional thresholds)")
    p.add_argument("--bands", default="accept,review")
    a = p.parse_args(argv)
    qc = json.loads(a.calibration.read_text())["qc"] if a.calibration else dict(DEFAULT_QC)
    print(json.dumps(export_v1(a.work, a.out, qc, tuple(a.bands.split(","))), indent=1))


if __name__ == "__main__":
    main()
