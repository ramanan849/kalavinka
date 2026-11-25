"""Mix two exports of the same songs: all clips of a *primary* export, plus the clips of a *secondary* export whose
span of the song no primary clip covers.

Why: re-planning clips with a longer cap (``qc.clip_max_s`` = 28 s, ``export/v3_long``) gives long continuous
stretches, but a long clip fails the per-clip QC more often than the short ones it replaces, so the long export has
less audio (8.99 h vs 10.21 h for ``export/v2``). Training on that alone would mix two effects (longer clips, less
data). The mix keeps every long clip and adds back the short clips from the stretches the long export lost, so the
long-clip model sees about the same audio as the short-clip one, with no stretch of a song twice.

Secondary clips are renamed ``<index>_s`` (clip numbering differs between exports) and their wavs are linked into
``<out>/wavs/``. ``val`` and ``test`` are taken from the primary export only (the long clips, for evaluation).

Usage::

    python -m kalavinka.annot.mix_exports --primary <PROJECT>/export/v3_long --secondary <PROJECT>/export/v2 \\
        --out <PROJECT>/export/v3_mix
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def song_of(index: str) -> str:
    """Song id of a clip index."""
    return index.rsplit("_c", 1)[0]


def mix(primary: List[Dict], secondary: List[Dict], margin_ms: int = 0) -> List[Dict]:
    """Primary clips + secondary clips that overlap no primary clip of their song (renamed ``<index>_s``)."""
    spans: Dict[str, List] = {}
    for it in primary:
        spans.setdefault(song_of(it["index"]), []).append(it["time"])
    out = list(primary)
    for it in secondary:
        a, b = it["time"]
        if any(a < y + margin_ms and b > x - margin_ms for x, y in spans.get(song_of(it["index"]), [])):
            continue
        new = dict(it, index=it["index"] + "_s", wav=str(Path(it["wav"]).with_name(Path(it["wav"]).stem + "_s.wav")))
        out.append(new)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--secondary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    (a.out / "wavs").mkdir(parents=True, exist_ok=True)
    read = lambda f: [json.loads(l) for l in f.read_text(encoding="utf-8").split("\n") if l.strip()]
    for split in ("train", "val", "test"):
        prim = read(a.primary / f"{split}.jsonl")
        items = mix(prim, read(a.secondary / f"{split}.jsonl")) if split == "train" else prim
        for it in items:
            src = (a.secondary / it["wav"].replace("_s.wav", ".wav")) if it["index"].endswith("_s") else a.primary / it["wav"]
            dst = a.out / it["wav"]
            if not dst.exists():
                os.symlink(src.resolve(), dst)
        (a.out / f"{split}.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items),
                                              encoding="utf-8")
        hours = sum(it["n_frames"] for it in items) / 50 / 3600
        print(f"{split}: {len(items)} clips ({len(items) - len(prim)} from the secondary), {hours:.2f} h")


if __name__ == "__main__":
    main()
