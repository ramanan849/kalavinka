"""The song list: ``<work>/songs.tsv``, one row per audio file.

Song ids are stable and readable: ``<year>_<title slug>__<youtube id>``, e.g.
``2013_tum_hi_ho__Umqb9KENgmk`` (the YouTube id keeps re-uploads of the same
title apart; files without one get a short hash of the file name).
"""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

AUDIO_EXTS = {".opus", ".m4a", ".mp3", ".wav", ".flac", ".webm", ".ogg"}
FIELDS = ["song_id", "audio", "year", "title", "video_id"]


@dataclass
class Song:
    """One row of songs.tsv."""
    song_id: str
    audio: str
    year: str
    title: str
    video_id: str


def parse_name(stem: str) -> Dict[str, str]:
    """Year, title and YouTube id from ``<year> - <title> [<id>]`` (best effort)."""
    m = re.match(r"^(\d{4}) - (.*?)\s*(?:\[([^\]]+)\])?$", stem)
    if not m:
        return {"year": "", "title": stem, "video_id": ""}
    return {"year": m.group(1), "title": m.group(2).strip(), "video_id": m.group(3) or ""}


def make_id(year: str, title: str, video_id: str, fname: str) -> str:
    """Stable song id (see the module docstring)."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40] or "untitled"
    tail = re.sub(r"[^A-Za-z0-9_-]", "", video_id) or hashlib.sha1(fname.encode()).hexdigest()[:8]
    return f"{year or '0000'}_{slug}__{tail}"


def build(audio_dir: str | Path) -> List[Song]:
    """Songs for every audio file in ``audio_dir`` (sorted by file name)."""
    out, seen = [], set()
    for p in sorted(Path(audio_dir).iterdir()):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        info = parse_name(p.stem)
        sid = make_id(info["year"], info["title"], info["video_id"], p.name)
        if sid in seen:                        # same id twice (same video, two formats)
            sid += "_" + p.suffix.lower().lstrip(".")
        seen.add(sid)
        out.append(Song(sid, str(p.resolve()), info["year"], info["title"], info["video_id"]))
    return out


def write(songs: List[Song], path: str | Path) -> None:
    """Write songs.tsv atomically."""
    from .state import atomic_bytes
    lines = ["\t".join(FIELDS)] + ["\t".join(str(getattr(s, k)) for k in FIELDS) for s in songs]
    atomic_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def read(path: str | Path) -> List[Song]:
    """Rows of songs.tsv."""
    with open(path, encoding="utf-8", newline="") as f:
        return [Song(**{k: r.get(k, "") or "" for k in FIELDS}) for r in csv.DictReader(f, delimiter="\t")]


def as_dict(s: Song) -> Dict[str, str]:
    """A song as a plain dict."""
    return asdict(s)
