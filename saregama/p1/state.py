"""Crash-safe files and per-song stage state.

Every output is written atomically (to ``<name>.tmp.<pid>`` then ``os.replace``),
so a job killed mid-write never leaves a file that looks complete.  Each song
folder holds ``stages.json``::

    {"separate": {"status": "done", "version": 2, "config": "a1b2c3d4",
                  "seconds": 41.2, "finished": "2026-11-15T10:00:00", "job": "386400"},
     "asr": {"status": "failed", "fails": 1, "error": "...", ...}}

A stage counts as done only if its status is ``done`` **and** its version and
config hash match the current code / config, so changing a stage re-runs it
(and, through ``depends``, everything after it) and nothing else.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

MAX_FAILS = 2          # a stage that failed this many times is parked, not retried


def _now() -> str:
    """Current local time with microseconds (stage stamps must differ between runs)."""
    from datetime import datetime
    return datetime.now().isoformat(timespec="microseconds")


def _tmp(path: Path) -> Path:
    """Temporary sibling of ``path`` unique to this process."""
    return path.with_name(f"{path.name}.tmp.{os.getpid()}")


def atomic_bytes(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (parents created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path: str | Path, obj: Any) -> None:
    """Write ``obj`` as UTF-8 JSON atomically."""
    atomic_bytes(path, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))


def atomic_npy(path: str | Path, arr: np.ndarray) -> None:
    """Save a numpy array (.npy) atomically."""
    buf = io.BytesIO()
    np.save(buf, arr)
    atomic_bytes(path, buf.getvalue())


def atomic_npz(path: str | Path, **arrays) -> None:
    """Save compressed numpy arrays (.npz) atomically."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    atomic_bytes(path, buf.getvalue())


def atomic_flac(path: str | Path, x: np.ndarray, sr: int) -> None:
    """Save mono / multichannel float audio as 16-bit FLAC atomically
    (lossless at 16 bit, about 4x smaller than float WAV)."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(x, dtype=np.float32), -1, 1), sr, format="FLAC", subtype="PCM_16")
    atomic_bytes(path, buf.getvalue())


def read_json(path: str | Path, default: Any = None) -> Any:
    """JSON content of ``path``, or ``default`` if it does not exist."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def config_hash(obj: Any) -> str:
    """Short stable hash of a JSON-able config (keys sorted)."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


class SongState:
    """Stage markers of one song (``<song_dir>/stages.json``)."""

    def __init__(self, song_dir: str | Path):
        """Bind to ``song_dir`` (created if missing); markers load lazily."""
        self.dir = Path(song_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "stages.json"

    def load(self) -> Dict[str, Dict]:
        """All stage markers (re-read from disk every time: other jobs may write)."""
        return read_json(self.path, {}) or {}

    def get(self, stage: str) -> Optional[Dict]:
        """Marker of ``stage`` or None."""
        return self.load().get(stage)

    def is_done(self, stage: str, version: int, cfg: str) -> bool:
        """Done with the current code version and config hash."""
        m = self.get(stage)
        return bool(m and m.get("status") == "done" and m.get("version") == version
                    and m.get("config") == cfg)

    def is_parked(self, stage: str) -> bool:
        """Failed ``MAX_FAILS`` times (needs a person or a code fix; ``--retry-failed`` resets)."""
        m = self.get(stage)
        return bool(m and m.get("status") == "failed" and m.get("fails", 0) >= MAX_FAILS)

    def _update(self, stage: str, marker: Dict) -> None:
        """Replace one stage's marker atomically (songs are claimed by one job at
        a time, so read-modify-write is safe)."""
        allm = self.load()
        allm[stage] = marker
        atomic_json(self.path, allm)

    def mark_done(self, stage: str, version: int, cfg: str, seconds: float, **extra) -> None:
        """Record a successful run of ``stage``."""
        self._update(stage, {"status": "done", "version": version, "config": cfg,
                             "seconds": round(seconds, 2), "finished": _now(),
                             "job": os.environ.get("SLURM_JOB_ID", "local"), **extra})

    def mark_failed(self, stage: str, version: int, cfg: str, error: str) -> int:
        """Record a failure; returns the failure count for this stage."""
        old = self.get(stage) or {}
        fails = (old.get("fails", 0) if old.get("status") == "failed" else 0) + 1
        self._update(stage, {"status": "failed", "version": version, "config": cfg, "fails": fails,
                             "error": error[-2000:], "finished": _now(),
                             "job": os.environ.get("SLURM_JOB_ID", "local")})
        return fails

    def mark_skipped(self, stage: str, version: int, cfg: str, reason: str) -> None:
        """Record that ``stage`` deliberately does not apply (counts as done, e.g.
        a duplicate song or one without singing)."""
        self._update(stage, {"status": "done", "skipped": reason, "version": version, "config": cfg,
                             "seconds": 0.0, "finished": _now(),
                             "job": os.environ.get("SLURM_JOB_ID", "local")})

    def reset(self, stage: str) -> None:
        """Forget ``stage`` (``--force`` / ``--retry-failed``)."""
        allm = self.load()
        if allm.pop(stage, None) is not None:
            atomic_json(self.path, allm)
