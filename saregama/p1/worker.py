"""Claim-based workers: any number of Slurm jobs process songs without overlap.

A worker walks the song list; for each song with pending stages it **claims**
the song (atomic ``mkdir <work>/claims/<song_id>`` holding the job id), runs
the pending stages in order, and releases the claim.  Claims of jobs that are
no longer alive (checked with ``squeue``) are stale and taken over.

A worker stops starting new songs when its time budget (``--budget-min``, set a
little under the Slurm ``--time``) would be exceeded by one more average song,
so it exits cleanly and nothing is half done; resubmitting simply continues.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .manifest import Song
from .state import SongState, config_hash


class Skip(Exception):
    """Raised by a stage that does not apply to a song (recorded as done + reason)."""


@dataclass
class Stage:
    """One pipeline stage: ``fn(ctx, song, song_dir) -> dict of stats``."""
    name: str
    version: int
    fn: Callable
    sections: Sequence[str] = ()          # config sections the stage reads (hashed)
    depends: Sequence[str] = ()           # stages that must be done first
    gpu: bool = False

    def cfg_hash(self, cfg: Dict) -> str:
        """Hash of the config sections this stage reads."""
        return config_hash({s: cfg.get(s) for s in self.sections})


@dataclass
class Context:
    """Shared by all stages of one worker: config, work root, and lazily
    loaded models (kept across songs, so each model loads once per job)."""
    cfg: Dict
    work: Path
    models: Dict[str, object] = field(default_factory=dict)

    def model(self, key: str, build: Callable[[], object]):
        """The model cached under ``key`` (built on first use)."""
        if key not in self.models:
            t = time.time()
            self.models[key] = build()
            print(f"  [model] {key} loaded in {time.time() - t:.0f}s", flush=True)
        return self.models[key]


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------
def _me() -> str:
    """Owner tag of this process: Slurm job id, or host:pid outside Slurm."""
    return os.environ.get("SLURM_JOB_ID") or f"{socket.gethostname()}:{os.getpid()}"


def _alive(owner: str) -> bool:
    """Is the claim owner still running?  Unknown -> assume alive (safe side)."""
    if owner.isdigit():
        try:
            r = subprocess.run(["squeue", "-h", "-j", owner, "-o", "%T"], capture_output=True,
                               text=True, timeout=30)
        except Exception:
            return True
        if r.returncode != 0 and "Invalid job id" in (r.stderr or ""):
            return False
        return r.returncode != 0 or bool(r.stdout.strip())
    host, _, pid = owner.partition(":")
    if host == socket.gethostname() and pid.isdigit():
        return Path(f"/proc/{pid}").exists()
    return True


def claim(work: Path, song_id: str) -> bool:
    """Try to claim ``song_id``; takes over a stale claim.  True if ours."""
    d = work / "claims" / song_id
    d.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            d.mkdir()
            (d / "owner").write_text(_me())
            return True
        except FileExistsError:
            owner_f = d / "owner"
            owner = owner_f.read_text().strip() if owner_f.exists() else ""
            if owner == _me():
                return True
            if owner and _alive(owner):
                return False
            if not owner and time.time() - d.stat().st_mtime < 60:
                return False                       # being created right now
            shutil.rmtree(d, ignore_errors=True)   # stale: owner gone
    return False


def release(work: Path, song_id: str) -> None:
    """Drop our claim on ``song_id``."""
    d = work / "claims" / song_id
    owner = (d / "owner").read_text().strip() if (d / "owner").exists() else ""
    if owner == _me():
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def is_fresh(marks: Dict[str, Dict], st: Stage, cfg: Dict) -> bool:
    """Done with the current version and config, and no dependency re-ran since
    (each done marker stores its dependencies' ``finished`` stamps in ``deps``;
    markers written before stamps existed are trusted)."""
    m = marks.get(st.name)
    if not (m and m.get("status") == "done" and m.get("version") == st.version
            and m.get("config") == st.cfg_hash(cfg)):
        return False
    for d, stamp in (m.get("deps") or {}).items():
        if (marks.get(d) or {}).get("finished") != stamp:
            return False
    return True


def pending(state: SongState, stages: Sequence[Stage], all_stages: Dict[str, Stage], cfg: Dict
            ) -> List[Stage]:
    """Stages of ``stages`` still to run for a song, in order, stopping at the first
    one that is parked or whose dependencies are not done (those need another
    worker type or a fix first).  A stage whose dependency is re-run in this pass
    is re-run too."""
    out, done_now = [], set()
    marks = state.load()
    for st in stages:
        if is_fresh(marks, st, cfg) and not any(d in {s.name for s in out} for d in st.depends):
            done_now.add(st.name)
            continue
        if state.is_parked(st.name):
            break
        ok = all(d in done_now or d in {s.name for s in out} or
                 state.is_done(d, all_stages[d].version, all_stages[d].cfg_hash(cfg)) for d in st.depends)
        if not ok:
            break
        out.append(st)
    return out


def run_worker(ctx: Context, songs: Sequence[Song], stages: Sequence[Stage], all_stages: Dict[str, Stage],
               budget_s: float, max_songs: Optional[int] = None) -> Dict[str, int]:
    """Process songs until done, out of songs, or out of time.  Returns counts
    of stage runs: done, skipped, failed."""
    t0 = time.time()
    counts = {"done": 0, "skipped": 0, "failed": 0, "songs": 0}
    durations: List[float] = []
    for song in songs:
        if max_songs is not None and counts["songs"] >= max_songs:
            break
        sdir = ctx.work / "songs" / song.song_id
        state = SongState(sdir)
        todo = pending(state, stages, all_stages, ctx.cfg)
        if not todo:
            continue
        est = (sum(durations) / len(durations)) * 1.5 if durations else 0.0
        if time.time() - t0 + est > budget_s:
            print(f"time budget reached after {counts['songs']} songs", flush=True)
            break
        if not claim(ctx.work, song.song_id):
            continue
        ts = time.time()
        try:
            todo = pending(state, stages, all_stages, ctx.cfg)    # re-check under the claim
            for st in todo:
                cfgh = st.cfg_hash(ctx.cfg)
                t = time.time()
                try:
                    stats = st.fn(ctx, song, sdir) or {}
                    marks = state.load()
                    deps = {d: (marks.get(d) or {}).get("finished") for d in st.depends}
                    state.mark_done(st.name, st.version, cfgh, time.time() - t, deps=deps, **stats)
                    counts["done"] += 1
                    print(f"{song.song_id} {st.name} ok {time.time() - t:.1f}s", flush=True)
                except Skip as e:
                    state.mark_skipped(st.name, st.version, cfgh, str(e))
                    counts["skipped"] += 1
                    print(f"{song.song_id} {st.name} skipped: {e}", flush=True)
                except Exception:
                    tb = traceback.format_exc()
                    n = state.mark_failed(st.name, st.version, cfgh, tb)
                    (sdir / "errors").mkdir(exist_ok=True)
                    (sdir / "errors" / f"{st.name}.txt").write_text(tb, encoding="utf-8")
                    counts["failed"] += 1
                    print(f"{song.song_id} {st.name} FAILED ({n}x): {tb.splitlines()[-1]}", flush=True)
                    break                                # later stages depend on it
        finally:
            release(ctx.work, song.song_id)
        counts["songs"] += 1
        durations.append(time.time() - ts)
    print(f"worker finished: {counts} in {time.time() - t0:.0f}s", flush=True)
    return counts


def status_table(songs: Sequence[Song], stages: Sequence[Stage], work: Path, cfg: Dict
                 ) -> Dict[str, Dict[str, int]]:
    """Per stage: how many songs are done / skipped / failed / parked / pending."""
    out = {st.name: {"done": 0, "skipped": 0, "failed": 0, "parked": 0, "pending": 0} for st in stages}
    for song in songs:
        p = work / "songs" / song.song_id / "stages.json"
        marks = SongState(p.parent).load() if p.exists() else {}
        for st in stages:
            m = marks.get(st.name)
            if is_fresh(marks, st, cfg):
                out[st.name]["skipped" if m.get("skipped") else "done"] += 1
            elif m and m.get("status") == "failed":
                out[st.name]["parked" if m.get("fails", 0) >= 2 else "failed"] += 1
            else:
                out[st.name]["pending"] += 1
    return out
