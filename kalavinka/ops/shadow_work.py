"""Make a *shadow* copy of a P1 work directory, so text stages can be re-run without touching the original.

Why: the dataset behind the reported results (``work/`` -> ``export/v1``) must stay reproducible. When a rule
changes (for example the G2P fix of 25 Nov: nasalised medial schwa, final schwa after r/v clusters, numbers and
symbols), the cheap CPU text stages (``align``, ``vocables``, ``notes_qc``) have to run again. This tool builds
``<dst>/songs/<id>/`` where every big cached file (audio, ``.npz``, ``.npy``) is a **symlink** to the original and
every small ``.json`` is a **copy**. Stages write their JSON with ``atomic_json`` (temp file + rename), which
replaces a symlink or copy in the shadow tree and never writes through to the original. Files a stage rewrites
*in place* (the annotation curves ``f0.npy`` etc., saved with a plain ``np.save``) are copied as well, and ``errors/``
is a real directory, so nothing is ever written through a link. ``--fix`` converts an existing shadow tree.

It also writes ``<dst>/p1.toml`` (the source config with ``work_dir`` pointed at ``<dst>``) and links the work-level
files (``songs.tsv``, ``refpool.json``, ``dupes.json``, ``gold_songs.txt``, ``models``).

Usage::

    python -m kalavinka.ops.shadow_work --src <PROJECT>/work --dst <PROJECT>/work_g2p2
    python -m kalavinka.ops.p1_run p1-run --p1-config <PROJECT>/work_g2p2/p1.toml \\
        --stages align,vocables,notes_qc --force --budget-min 100000
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Sequence

COPY_SUFFIXES = {".json", ".jsonl", ".txt", ".tsv", ".toml"}
# written with a plain np.save (not atomic) by saregama.schema.SongAnnotation.save: a symlink would be written
# THROUGH to the original, so these are copied too
COPY_NAMES = {"f0.npy", "f0_agree.npy", "energy_db.npy", "hnr_db.npy", "h1h2_db.npy"}
REAL_DIRS = {"errors"}          # stages write tracebacks into it


def shadow_song(src: Path, dst: Path) -> int:
    """Shadow one song directory; returns the number of copied files."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.iterdir():
        t = dst / f.name
        if t.exists() or t.is_symlink():
            continue
        if f.is_dir() and f.name in REAL_DIRS:
            t.mkdir()
        elif f.is_file() and (f.suffix in COPY_SUFFIXES or f.name in COPY_NAMES):
            shutil.copy2(f, t)
            n += 1
        else:
            os.symlink(f.resolve(), t)
    return n


def fix_links(dst: Path) -> int:
    """Turn links to in-place-written files (``COPY_NAMES``, ``REAL_DIRS``) of an existing shadow tree into copies."""
    n = 0
    for sd in (dst / "songs").iterdir():
        for name in COPY_NAMES | REAL_DIRS:
            t = sd / name
            if t.is_symlink():
                src = t.resolve()
                t.unlink()
                if name in REAL_DIRS:
                    t.mkdir()
                else:
                    shutil.copy2(src, t)
                n += 1
    return n


def shadow(src: Path, dst: Path) -> None:
    """Shadow a whole work directory (see the module doc)."""
    if dst.resolve() == src.resolve():
        raise SystemExit("dst must differ from src")
    (dst / "songs").mkdir(parents=True, exist_ok=True)
    songs = sorted(p for p in (src / "songs").iterdir() if p.is_dir())
    copied = sum(shadow_song(p, dst / "songs" / p.name) for p in songs)
    for f in src.iterdir():
        if f.name in ("songs", "p1.toml", "claims"):
            continue
        t = dst / f.name
        if not (t.exists() or t.is_symlink()):
            os.symlink(f.resolve(), t)
    (dst / "claims").mkdir(exist_ok=True)
    cfg = (src / "p1.toml").read_text(encoding="utf-8")
    cfg = re.sub(r'^(work_dir\s*=\s*).*$', lambda m: f'{m.group(1)}"{dst.resolve()}"', cfg, flags=re.M)
    (dst / "p1.toml").write_text(cfg, encoding="utf-8")
    print(f"{len(songs)} songs shadowed into {dst} ({copied} small files copied, the rest linked)")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--fix", action="store_true", help="only convert in-place-written links of an existing shadow")
    a = p.parse_args(argv)
    if a.fix:
        print(f"{fix_links(a.dst)} links replaced by copies")
    else:
        shadow(a.src, a.dst)


if __name__ == "__main__":
    main()
