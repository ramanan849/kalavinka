"""History and cache of generated takes: nothing is ever sung twice, and every take can be found again.

Every take (from the command line or the web page) is saved under ``outputs/`` as ``<time>_<name>.wav`` plus a
``.json`` with what was sung, and gets one line in ``outputs/history.jsonl``:

    {"id": 12, "time": "2026-11-25 18:02:11", "key": "<sha1 of the request>", "kind": "sing|resing|sargam|compose",
     "model": "kalavinka|base", "request": {...}, "wav": "<file name>", "info": {...}}

**Cache.** The key is a hash of the full request (kind, text, tune, voice, model, pitch mode, seed, steps, tempo,
tonic, ...) plus ``CACHE_VERSION``. If the same request was made before and its wav is still there, the saved take is
returned at once instead of running the model again (``fresh=True`` forces a new run). Change the seed to get a
genuinely new take. Deleting a wav simply makes that request run again; deleting ``history.jsonl`` clears the history.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

CACHE_VERSION = "v4l-l4_v2/base-v0/2"      # bump when the models or the rendering change


def request_key(request: Dict) -> str:
    """Stable hash of a request (key order does not matter)."""
    blob = json.dumps({"v": CACHE_VERSION, **request}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def slug(text: str, n: int = 24) -> str:
    """Filesystem-safe short name (keeps Devanagari)."""
    s = re.sub(r"[\s|/\\:*?\"<>'\n]+", "_", str(text).strip())
    return s[:n].strip("_") or "take"


class History:
    """The ``outputs/history.jsonl`` log plus the cache lookup."""

    def __init__(self, out_dir: Path):
        """Use (and create) ``out_dir``."""
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "history.jsonl"

    def records(self) -> List[Dict]:
        """All records, oldest first (unreadable lines are skipped)."""
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def recent(self, n: int = 20) -> List[Dict]:
        """The last ``n`` records whose wav still exists, newest first."""
        return [r for r in reversed(self.records()) if (self.dir / r["wav"]).exists()][:n]

    def find(self, key: str) -> Optional[Dict]:
        """Newest record with this key whose wav still exists."""
        for r in reversed(self.records()):
            if r.get("key") == key and (self.dir / r["wav"]).exists():
                return r
        return None

    def get(self, rid: int) -> Optional[Dict]:
        """Record by id."""
        for r in self.records():
            if r.get("id") == rid:
                return r
        return None

    def add(self, kind: str, request: Dict, key: str, wav: np.ndarray, sr: int, info: Dict, name: str) -> Dict:
        """Save a new take (wav + json) and log it; returns the record."""
        import soundfile as sf
        recs = self.records()
        rid = (max((r.get("id", 0) for r in recs), default=0) + 1)
        fname = f"{time.strftime('%Y%m%d-%H%M%S')}_{rid:04d}_{slug(name)}.wav"
        sf.write(self.dir / fname, wav, sr)
        rec = {"id": rid, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "key": key, "kind": kind,
               "model": request.get("model", "kalavinka"), "request": request, "wav": fname, "info": info}
        (self.dir / fname).with_suffix(".json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def cached(self, kind: str, request: Dict, make: Callable[[], Tuple[np.ndarray, int, Dict]], name: str,
               fresh: bool = False) -> Tuple[np.ndarray, int, Dict, Path, bool]:
        """Return the saved take for ``request`` if there is one, else ``make()`` it and save it.

        Returns (wav, sr, info, wav path, was_cached)."""
        import soundfile as sf
        key = request_key({"kind": kind, **request})
        if not fresh:
            hit = self.find(key)
            if hit:
                wav, sr = sf.read(self.dir / hit["wav"], dtype="float32")
                return wav, sr, hit["info"], self.dir / hit["wav"], True
        wav, sr, info = make()
        rec = self.add(kind, request, key, wav, sr, info, name)
        return wav, sr, info, self.dir / rec["wav"], False


def describe(r: Dict) -> str:
    """One line about a record, for the history list."""
    q = r["request"]
    what = {"sing": lambda: q.get("lyrics", ""), "resing": lambda: f"tune {q.get('tune')} (original words)",
            "sargam": lambda: q.get("title") or q.get("sargam", "")[:40],
            "compose": lambda: "compose: " + str((q.get("comp") or {}).get("title", ""))}.get(r["kind"], lambda: "")()
    extra = f"{q.get('mode', 'score')}"
    if r["kind"] == "sargam" and "pitch" in r.get("info", {}):
        p = r["info"]["pitch"]
        extra = f"{int(p['within_50c'] * 100)}% notes on the swara, {p['mean_abs_cents']} c off"
    if r["kind"] == "compose" and "follow" in r.get("info", {}):
        f = r["info"]["follow"]
        extra = f"{int(f['within_50c'] * 100)}% of the curve within 50 c"
    return f"#{r['id']:<4} {r['time']}  {r['model']:<9} {r['kind']:<6} {what[:44]:<44}  {extra}"
