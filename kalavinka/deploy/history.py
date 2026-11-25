"""History, cache and logs of generated takes: nothing is sung twice, every take can be found, labelled and recovered.

Files under ``outputs/``
------------------------
* ``<time>_<id>_<name>.wav`` + ``.json`` per take: the sidecar json is the **full record** (request, info, curves).
* ``history.jsonl``: one line per take (the record without heavy curves such as the sung pitch), append-only::

      {"id": 12, "time": "2026-11-25 18:02:11", "key": "<sha1 of the request>", "kind": "sing|resing|sargam|compose",
       "model": "kalavinka|base", "request": {...}, "wav": "<file name>", "info": {...}}

* ``annotations.jsonl``: append-only edits ``{"id", "time", "field", "value"}`` for ``title``, ``notes``, ``tags``,
  ``star`` and ``deleted``. The latest value of each field wins. The take log is never rewritten, so a crash while
  editing a label can never lose a take.
* ``log.jsonl``: an event log of everything the app does (sing requested, cache hit or new take, seconds taken,
  errors with a short traceback, labels, deletes, profile edits, saves), to answer "what happened?".
* ``trash/``: deleted takes (wav + json), restorable; ``history.jsonl.bak-<time>`` after a ``rebuild``.

**Cache.** The key is a hash of the full request (kind, text, tune, voice, model, pitch mode, join, gaps, seed,
steps, tempo, tonic, ...) plus ``CACHE_VERSION``. If the same request was made before and its wav is still there (and
not deleted), the saved take is returned at once instead of running the model again (``fresh=True`` forces a new run).
Change the seed for a genuinely new take.

**Safety.** Writes go to a temporary file first and are renamed into place; appends and id allocation happen under a
lock (a thread lock plus ``fcntl`` file lock where available, so the web page and the command line can share one
``outputs/``). If ``history.jsonl`` is lost or damaged, ``History.rebuild()`` re-creates it from the sidecars.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import threading
import time
import traceback
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

CACHE_VERSION = "v4l-l4_v2/base-v0/3"      # bump when the models or the rendering change (3: auto tonic folded)
HEAVY_INFO = ("sung",)                     # info fields kept only in the sidecar json
FIELDS = ("title", "notes", "tags", "star", "deleted")
_THREAD_LOCK = threading.RLock()


def request_key(request: Dict) -> str:
    """Stable hash of a request (key order does not matter)."""
    blob = json.dumps({"v": CACHE_VERSION, **request}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def slug(text: str, n: int = 24) -> str:
    """Filesystem-safe short name (keeps Devanagari)."""
    s = re.sub(r"[\s|/\\:*?\"<>'\n\r  \x85]+", "_", str(text).strip())
    return s[:n].strip("_.") or "take"


def now() -> str:
    """Local time as 'YYYY-MM-DD HH:MM:SS'."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_jsonl(path: Path) -> List[Dict]:
    """Records of a jsonl file; unreadable lines are skipped. Splits on '\\n' only (lyrics may contain U+2028)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file + rename (never leaves a half-written file)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class History:
    """The take log, its annotations, the event log and the cache lookup, all under one ``outputs/`` folder."""

    def __init__(self, out_dir: Path):
        """Use (and create) ``out_dir``."""
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "history.jsonl"
        self.ann_path = self.dir / "annotations.jsonl"
        self.log_path = self.dir / "log.jsonl"
        self.trash = self.dir / "trash"

    # ------------------------------------------------------------------ locking
    @contextlib.contextmanager
    def lock(self):
        """Exclusive lock across threads and (where fcntl exists) processes sharing this folder."""
        with _THREAD_LOCK:
            fh = open(self.dir / ".lock", "a+")
            try:
                try:
                    import fcntl
                    fcntl.flock(fh, fcntl.LOCK_EX)
                except ImportError:                 # Windows: thread lock only
                    pass
                yield
            finally:
                fh.close()

    def _append(self, path: Path, rec: Dict) -> None:
        """Append one json line (call under ``lock``)."""
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ------------------------------------------------------------------ event log
    def log(self, event: str, **fields) -> None:
        """Add an event to ``log.jsonl`` (never raises: logging must not break singing)."""
        try:
            with self.lock():
                self._append(self.log_path, {"time": now(), "event": event, **fields})
        except Exception:
            pass

    def log_error(self, event: str, err: BaseException, **fields) -> None:
        """Log an error with the last lines of its traceback."""
        tb = traceback.format_exception(type(err), err, err.__traceback__)
        self.log(event, error=f"{type(err).__name__}: {err}", trace="".join(tb[-3:])[-1500:], **fields)

    def events(self, n: int = 50) -> List[Dict]:
        """The last ``n`` events, newest first."""
        return list(reversed(read_jsonl(self.log_path)))[:n]

    # ------------------------------------------------------------------ reading
    def annotations(self) -> Dict[int, Dict]:
        """Latest value of every annotated field, per take id."""
        out: Dict[int, Dict] = {}
        for a in read_jsonl(self.ann_path):
            if a.get("field") in FIELDS and isinstance(a.get("id"), int):
                out.setdefault(a["id"], {})[a["field"]] = a.get("value")
        return out

    def wav_path(self, r: Dict) -> Path:
        """Where a take's wav is now (outputs/ or trash/)."""
        p = self.dir / r["wav"]
        return p if p.exists() else self.trash / r["wav"]

    def records(self, with_annotations: bool = True) -> List[Dict]:
        """All records, oldest first, with their annotations merged in and ``missing`` set when the wav is gone."""
        recs = read_jsonl(self.path)
        ann = self.annotations() if with_annotations else {}
        for r in recs:
            r.update(ann.get(r.get("id"), {}))
            r["missing"] = not self.wav_path(r).exists()
        return recs

    def recent(self, n: int = 20, include_deleted: bool = False) -> List[Dict]:
        """The last ``n`` playable records, newest first (deleted ones only if asked)."""
        out = [r for r in reversed(self.records()) if not r["missing"] and (include_deleted or not r.get("deleted"))]
        return out[:n]

    def search(self, text: str = "", kind: str = "", model: str = "", starred: bool = False,
               deleted: bool = False, n: int = 200) -> List[Dict]:
        """Newest-first records matching all filters (``text`` searches the request, title, notes and tags)."""
        t = (text or "").strip().lower()
        out = []
        for r in reversed(self.records()):
            if bool(r.get("deleted")) != bool(deleted) or r["missing"]:
                continue
            if kind and r.get("kind") != kind or model and r.get("model") != model or starred and not r.get("star"):
                continue
            if t:
                blob = json.dumps([r.get("request"), r.get("title"), r.get("notes"), r.get("tags")],
                                  ensure_ascii=False).lower()
                if t not in blob:
                    continue
            out.append(r)
            if len(out) >= n:
                break
        return out

    def find(self, key: str) -> Optional[Dict]:
        """Newest playable, not deleted record with this key."""
        for r in reversed(self.records()):
            if r.get("key") == key and not r["missing"] and not r.get("deleted"):
                return r
        return None

    def get(self, rid: int, full: bool = False) -> Optional[Dict]:
        """Record by id (``full``: from its sidecar json, with the heavy info fields)."""
        for r in self.records():
            if r.get("id") == rid:
                if full:
                    side = self.wav_path(r).with_suffix(".json")
                    if side.exists():
                        try:
                            rec = json.loads(side.read_text(encoding="utf-8"))
                            rec.update({k: r[k] for k in FIELDS + ("missing",) if k in r})
                            return rec
                        except json.JSONDecodeError:
                            pass
                return r
        return None

    # ------------------------------------------------------------------ writing
    def add(self, kind: str, request: Dict, key: str, wav: np.ndarray, sr: int, info: Dict, name: str) -> Dict:
        """Save a new take (wav + sidecar json, atomically) and log it; returns the full record."""
        import io
        import soundfile as sf
        with self.lock():
            ids = [r.get("id", 0) for r in read_jsonl(self.path)]
            rid = max(ids, default=0) + 1
            stem = f"{time.strftime('%Y%m%d-%H%M%S')}_{rid:04d}_{slug(name)}"
            buf = io.BytesIO()
            sf.write(buf, np.asarray(wav, np.float32), sr, format="WAV")
            atomic_write(self.dir / f"{stem}.wav", buf.getvalue())
            rec = {"id": rid, "time": now(), "key": key, "kind": kind, "model": request.get("model", "kalavinka"),
                   "request": request, "wav": f"{stem}.wav", "info": info}
            atomic_write(self.dir / f"{stem}.json", json.dumps(rec, ensure_ascii=False, indent=1).encode("utf-8"))
            light = dict(rec, info={k: v for k, v in info.items() if k not in HEAVY_INFO})
            self._append(self.path, light)
        self.log("take_saved", id=rid, kind=kind, model=rec["model"], took_s=info.get("took_s"), wav=rec["wav"])
        return rec

    def annotate(self, rid: int, field: str, value) -> None:
        """Set ``title`` / ``notes`` (text), ``tags`` (list of words), ``star`` / ``deleted`` (bool) on a take."""
        if field not in FIELDS:
            raise ValueError(f"field must be one of {FIELDS}")
        if field == "tags" and isinstance(value, str):
            value = [t for t in re.split(r"[,\s]+", value) if t]
        with self.lock():
            self._append(self.ann_path, {"id": int(rid), "time": now(), "field": field, "value": value})
        self.log("annotate", id=int(rid), field=field, value=value)

    def delete(self, rid: int) -> None:
        """Soft delete: hide the take and move its files to ``trash/`` (``restore`` brings it back)."""
        r = self.get(rid)
        if r is None:
            raise KeyError(f"no take {rid}")
        self.trash.mkdir(exist_ok=True)
        with self.lock():
            for p in (self.dir / r["wav"], (self.dir / r["wav"]).with_suffix(".json")):
                if p.exists():
                    shutil.move(str(p), str(self.trash / p.name))
        self.annotate(rid, "deleted", True)

    def restore(self, rid: int) -> None:
        """Undo ``delete``."""
        r = self.get(rid)
        if r is None:
            raise KeyError(f"no take {rid}")
        with self.lock():
            for p in (self.trash / r["wav"], (self.trash / r["wav"]).with_suffix(".json")):
                if p.exists():
                    shutil.move(str(p), str(self.dir / p.name))
        self.annotate(rid, "deleted", False)

    def rebuild(self) -> int:
        """Re-create ``history.jsonl`` from the sidecar jsons in outputs/ and trash/ (keeps a backup of the old file).
        Returns the number of takes found."""
        recs = {}
        for folder in (self.dir, self.trash):
            if not folder.exists():
                continue
            for side in folder.glob("*.json"):
                try:
                    r = json.loads(side.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(r, dict) and {"id", "kind", "wav", "request"} <= set(r):
                    recs[r["id"]] = dict(r, info={k: v for k, v in r.get("info", {}).items() if k not in HEAVY_INFO})
        with self.lock():
            if self.path.exists():
                shutil.copy2(self.path, self.dir / f"history.jsonl.bak-{time.strftime('%Y%m%d-%H%M%S')}")
            body = "".join(json.dumps(recs[k], ensure_ascii=False) + "\n" for k in sorted(recs))
            atomic_write(self.path, body.encode("utf-8"))
        self.log("rebuild", takes=len(recs))
        return len(recs)

    def export(self, ids: Iterable[int], dest: Optional[Path] = None) -> Path:
        """Zip the chosen takes (wav + json each) and return the zip path."""
        dest = dest or self.dir / "exports" / f"takes_{time.strftime('%Y%m%d-%H%M%S')}.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            for rid in ids:
                r = self.get(int(rid))
                if r is None:
                    continue
                w = self.wav_path(r)
                for p in (w, w.with_suffix(".json")):
                    if p.exists():
                        z.write(p, p.name)
        self.log("export", ids=list(map(int, ids)), zip=str(dest))
        return dest

    def cached(self, kind: str, request: Dict, make: Callable[[], Tuple[np.ndarray, int, Dict]], name: str,
               fresh: bool = False) -> Tuple[np.ndarray, int, Dict, Path, bool]:
        """Return the saved take for ``request`` if there is one, else ``make()`` it and save it.

        Returns (wav, sr, info, wav path, was_cached). Logs the request, the outcome, and any error."""
        import soundfile as sf
        key = request_key({"kind": kind, **request})
        if not fresh:
            hit = self.find(key)
            if hit:
                full = self.get(hit["id"], full=True) or hit
                wav, sr = sf.read(self.wav_path(hit), dtype="float32")
                self.log("cache_hit", id=hit["id"], kind=kind, model=request.get("model"))
                return wav, sr, full.get("info", hit["info"]), self.wav_path(hit), True
        self.log("sing", kind=kind, request=request)
        try:
            wav, sr, info = make()
        except Exception as e:
            self.log_error("sing_failed", e, kind=kind, model=request.get("model"))
            raise
        rec = self.add(kind, request, key, wav, sr, info, name)
        return wav, sr, info, self.dir / rec["wav"], False


def what(r: Dict) -> str:
    """Short description of what a take sang."""
    q = r.get("request", {})
    return {"sing": lambda: q.get("lyrics", ""), "resing": lambda: f"tune {q.get('tune')} (original words)",
            "sargam": lambda: q.get("title") or q.get("sargam", "")[:40],
            "compose": lambda: "compose: " + str((q.get("comp") or {}).get("title", ""))}.get(r.get("kind"), lambda: "")()


def quality(r: Dict) -> str:
    """The take's quality number, if it has one (sargam pitch check, compose follow check)."""
    info = r.get("info", {})
    if r.get("kind") == "sargam" and "pitch" in info:
        p = info["pitch"]
        return f"{int(p['within_50c'] * 100)}% notes on the swara, {p['mean_abs_cents']} c off"
    if r.get("kind") == "compose" and "follow" in info:
        return f"{int(info['follow']['within_50c'] * 100)}% of the curve within 50 c"
    return str(q.get("mode", "score")) if (q := r.get("request", {})) else ""


def describe(r: Dict) -> str:
    """One line about a record, for the history list and the command line."""
    star = "★" if r.get("star") else " "
    title = f" [{r['title']}]" if r.get("title") else ""
    return (f"{star}#{r['id']:<4} {r['time']}  {r['model']:<9} {r['kind']:<7} {(what(r) + title)[:52]:<52}  "
            f"{quality(r)}")
