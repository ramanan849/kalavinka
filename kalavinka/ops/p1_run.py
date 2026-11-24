"""Run ``python -m saregama ...`` with a fix for the stage-marker ``status`` clobber.

Why: ``saregama.p1.state.SongState.mark_done(..., **extra)`` merges a stage's
returned stats into its marker *after* ``"status": "done"``.  The ``identify``
stage returns ``{"status": "claimed" | "relabeled" | "unidentified", ...}``, so
its marker never reads ``done``, ``is_fresh`` is never true, and every text
worker re-runs identify (and, through the dependency stamps, everything after
it) on every song again and again.

This wrapper leaves the existing code untouched: it patches ``mark_done`` at
runtime so a stat called ``status`` is stored as ``outcome`` instead, then
hands over to the normal ``saregama`` CLI.  It also guards ``worker.claim``:
with dozens of local workers, two can race on a stale claim (one reads the
owner file while another removes it) and the exception killed the worker.  ``--migrate WORK`` rewrites
existing markers the same way (``status`` not in done/failed/skipped ->
``outcome``) so finished songs are not re-run.

Usage::

    python -m kalavinka.ops.p1_run p1-run --p1-config work/p1.toml --stages text
    python -m kalavinka.ops.p1_run --migrate /workspace/vis-enc/624/work
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KNOWN = {"done", "failed", "skipped"}


def patch() -> None:
    """Make ``SongState.mark_done`` keep ``status == "done"`` whatever the stats hold."""
    from saregama.p1 import state

    orig = state.SongState.mark_done
    if getattr(orig, "_kalavinka_fixed", False):
        return

    def mark_done(self, stage, version, cfg, seconds, **extra):
        """``mark_done`` with a stat named ``status`` renamed to ``outcome``."""
        if "status" in extra:
            extra["outcome"] = extra.pop("status")
        return orig(self, stage, version, cfg, seconds, **extra)

    mark_done._kalavinka_fixed = True
    state.SongState.mark_done = mark_done

    from saregama.p1 import worker
    orig_claim = worker.claim

    def claim(work, song_id):
        """``claim`` that treats a race with another worker (owner file vanishing or
        the dir reappearing mid-takeover) as "not ours" instead of crashing."""
        try:
            return orig_claim(work, song_id)
        except (FileNotFoundError, FileExistsError):
            return False

    worker.claim = claim


def fix_markers(marks: dict) -> int:
    """Rewrite clobbered markers in one ``stages.json`` dict in place; returns how many."""
    n = 0
    for m in marks.values():
        if isinstance(m, dict) and m.get("status") not in KNOWN and "version" in m and "config" in m:
            m["outcome"], m["status"] = m["status"], "done"
            n += 1
    return n


def migrate(work: Path) -> int:
    """Apply :func:`fix_markers` to every ``<work>/songs/*/stages.json`` atomically."""
    from saregama.p1.state import atomic_json
    total = 0
    for f in sorted((work / "songs").glob("*/stages.json")):
        marks = json.loads(f.read_text())
        n = fix_markers(marks)
        if n:
            atomic_json(f, marks)
            total += n
    return total


def main(argv=None) -> None:
    """Entry point: ``--migrate WORK`` or any ``saregama`` CLI arguments."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--migrate"]:
        print(f"fixed {migrate(Path(argv[1]))} markers")
        return
    patch()
    from saregama.cli import main as sg_main
    sg_main(argv)


if __name__ == "__main__":
    main()
