"""A shared-disk GPU job queue, so every node's idle GPUs get used without SSH between nodes.

Why: the dev container cannot log in to the other node's container (no sshd there), but both
mount the same ``/workspace``.  So jobs are files in a queue directory on that disk, and one
worker per node takes the next job whenever that node has a free GPU.  Whichever node is free
runs the job; nobody has to pick.

Layout (``--root``, default ``<PROJECT>/runs/jobq``)::

    pending/<ts>_<name>.json     waiting (oldest first)
    running/<ts>_<name>.json     claimed: node, gpu, pid, start added
    done/ failed/                finished, with the exit code
    logs/<ts>_<name>.log         the job's stdout + stderr

A job is ``{"name", "cmd", "needs", "min_free_gb", "after"}``: ``cmd`` runs with ``bash -c`` in
the repo, with ``CUDA_VISIBLE_DEVICES`` and ``JOB_GPU`` set to the GPU it got; ``needs`` is
``sx`` (SoulX venv only: training, rendering) or ``p1`` (also Whisper / RMVPE / WavLM, whose
model caches exist only in the dev container); ``after`` is a path that must exist first (e.g.
a run's ``DONE``), so evaluation jobs can be queued before their training ends.

Claiming is an atomic ``rename`` on the shared file system, so two workers never take the same
job.  A worker runs at most one job per GPU and only on GPUs with ``min_free_gb`` free (other
users share these cards).  ``--delay S`` makes a worker take only jobs that have waited S seconds:
give the busier node a delay and the idle node gets first pick.  Jobs whose worker died stay in
``running/``; ``status`` shows them.

Usage::

    python kalavinka/ops/jobq.py submit --name H1d --needs sx --after <path> -- <command ...>
    python kalavinka/ops/jobq.py worker --node gokul-bench --can sx [--min-free-gb 16] [--max-jobs 8]
    python kalavinka/ops/jobq.py status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
DIRS = ("pending", "running", "done", "failed", "logs")


def ensure(root: Path) -> None:
    """Create the queue directories."""
    for d in DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)


def write_atomic(path: Path, obj: Dict) -> None:
    """Write JSON so readers never see half a file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def submit(root: Path, name: str, cmd: str, needs: str = "sx", min_free_gb: float = 16.0,
           after: Optional[str] = None) -> Path:
    """Queue one job; returns its pending file."""
    ensure(root)
    if needs not in ("sx", "p1"):
        raise ValueError("needs must be sx or p1")
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
    path = root / "pending" / f"{stem}.json"
    write_atomic(path, {"name": name, "cmd": cmd, "needs": needs, "min_free_gb": min_free_gb, "after": after,
                        "submitted": time.time()})
    return path


def eligible(job: Dict, can: Sequence[str], delay: float = 0.0) -> bool:
    """A worker may take a job it has the environment for, once its ``after`` path exists and it has
    waited ``delay`` seconds (the wait counts from submission, or from ``after`` appearing)."""
    if job["needs"] not in can:
        return False
    after = job.get("after")
    if after and not Path(after).exists():
        return False
    since = max(job.get("submitted", 0.0), Path(after).stat().st_mtime if after else 0.0)
    return time.time() - since >= delay


def next_job(root: Path, can: Sequence[str], delay: float = 0.0) -> Optional[Path]:
    """Oldest pending job this worker can run now."""
    for f in sorted((root / "pending").glob("*.json")):
        try:
            if eligible(json.loads(f.read_text()), can, delay):
                return f
        except (FileNotFoundError, json.JSONDecodeError):
            continue          # claimed or being written by someone else
    return None


def claim(root: Path, f: Path) -> Optional[Path]:
    """Move a pending job to running/ atomically; None if another worker got it first."""
    dst = root / "running" / f.name
    try:
        os.rename(f, dst)
    except FileNotFoundError:
        return None
    return dst


def free_gpus(min_free_gb: float) -> List[int]:
    """GPUs with at least ``min_free_gb`` free, most free first (nvidia-smi)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    rows = [(int(i), float(m) / 1024) for i, m in (l.split(",") for l in out.strip().splitlines())]
    return [i for i, m in sorted(rows, key=lambda r: -r[1]) if m >= min_free_gb]


def worker(root: Path, node: str, can: Sequence[str], min_free_gb: float, max_jobs: int, poll: float,
           delay: float = 0.0) -> None:
    """Run jobs on this node's free GPUs until killed (one job per GPU)."""
    ensure(root)
    mine: Dict[int, tuple] = {}            # gpu -> (Popen, running file)
    log(f"worker {node} can={','.join(can)} min_free={min_free_gb} GiB max_jobs={max_jobs} delay={delay}s root={root}")
    while True:
        for g, (proc, f) in list(mine.items()):
            rc = proc.poll()
            if rc is None:
                continue
            job = json.loads(f.read_text())
            job.update({"rc": rc, "end": time.time()})
            dst = root / ("done" if rc == 0 else "failed") / f.name
            write_atomic(dst, job)
            f.unlink()
            del mine[g]
            log(f"{job['name']} on gpu{g} -> {'done' if rc == 0 else f'FAILED rc={rc}'}")
        if len(mine) < max_jobs:
            try:
                gpus = [g for g in free_gpus(min_free_gb) if g not in mine]
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
                log(f"nvidia-smi failed: {e}"); gpus = []
            for g in gpus[: max_jobs - len(mine)]:
                f = next_job(root, can, delay)
                if f is None:
                    break
                run = claim(root, f)
                if run is None:
                    continue
                job = json.loads(run.read_text())
                job.update({"node": node, "gpu": g, "start": time.time()})
                logf = open(root / "logs" / f"{run.stem}.log", "ab")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), JOB_GPU=str(g),
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                proc = subprocess.Popen(["bash", "-c", job["cmd"]], cwd=REPO, env=env, stdout=logf,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                job["pid"] = proc.pid
                write_atomic(run, job)
                mine[g] = (proc, run)
                log(f"{job['name']} -> gpu{g} (pid {proc.pid})")
        time.sleep(poll)


def status(root: Path) -> str:
    """One line per job, grouped by state."""
    ensure(root)
    lines = []
    for d in ("running", "pending", "failed", "done"):
        for f in sorted((root / d).glob("*.json")):
            try:
                j = json.loads(f.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            where = f" {j.get('node')} gpu{j.get('gpu')}" if "node" in j else ""
            wait = f" after {j['after']}" if d == "pending" and j.get("after") else ""
            rc = f" rc={j['rc']}" if "rc" in j else ""
            lines.append(f"{d:8s} {j['name']:24s} [{j['needs']}]{where}{rc}{wait}")
    return "\n".join(lines) or "queue empty"


def log(msg: str) -> None:
    """Timestamped line to stdout."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=PROJECT / "runs" / "jobq")
    sub = p.add_subparsers(dest="op", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--name", required=True)
    s.add_argument("--needs", default="sx", choices=["sx", "p1"])
    s.add_argument("--min-free-gb", type=float, default=16.0)
    s.add_argument("--after", default=None)
    s.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the command")
    w = sub.add_parser("worker")
    w.add_argument("--node", required=True)
    w.add_argument("--can", default="sx", help="comma list of sx,p1")
    w.add_argument("--min-free-gb", type=float, default=16.0)
    w.add_argument("--max-jobs", type=int, default=8)
    w.add_argument("--poll", type=float, default=30.0)
    w.add_argument("--delay", type=float, default=0.0, help="take only jobs waiting this long (s)")
    sub.add_parser("status")
    a = p.parse_args(argv)
    if a.op == "submit":
        cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
        if not cmd:
            sys.exit("give the command after --")
        print(submit(a.root, a.name, " ".join(cmd), a.needs, a.min_free_gb, a.after))
    elif a.op == "worker":
        worker(a.root, a.node, a.can.split(","), a.min_free_gb, a.max_jobs, a.poll, a.delay)
    else:
        print(status(a.root))


if __name__ == "__main__":
    main()
