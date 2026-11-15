"""Dataset cleaning stages (CPU): S0 inventory, S2 fingerprints + duplicates.

``inventory``    ffprobe facts, sha256 and a full decode check per file.
``fingerprint``  Chromaprint raw fingerprint (``fpcalc -raw``, first 180 s).
``dedupe`` (global command, not a per-song stage): finds re-uploads of the
same recording through an inverted index over sub-fingerprint values, verifies
each candidate pair by bit error rate at the best offset, groups duplicates and
keeps one canonical file per group (``<work>/dupes.json``).  Later stages skip
the other members (``duplicate_of``).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .manifest import Song
from .state import atomic_json, atomic_npy, read_json

FP_SECONDS = 180
DUP_BER = 0.20          # bit error rate under which two fingerprints are the same recording
MIN_SHARED = 40         # shared sub-fingerprint values that make a pair a candidate


def _run(cmd: List[str], timeout: int = 300) -> str:
    """stdout of ``cmd`` (raises on failure)."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr.strip()[-500:]}")
    return r.stdout


def inventory(ctx, song: Song, sdir: Path) -> Dict:
    """ffprobe facts, sha256 and a full decode check -> ``inventory.json``."""
    p = Path(song.audio)
    info = json.loads(_run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration,bit_rate:stream=codec_name,sample_rate,channels",
                            "-of", "json", str(p)]))
    st = (info.get("streams") or [{}])[0]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    dec = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "null", "-"],
                         capture_output=True, text=True, timeout=600)
    out = {"duration_s": float(info.get("format", {}).get("duration", 0) or 0),
           "bit_rate": int(info.get("format", {}).get("bit_rate", 0) or 0),
           "codec": st.get("codec_name", ""), "sample_rate": int(st.get("sample_rate", 0) or 0),
           "channels": int(st.get("channels", 0) or 0), "bytes": p.stat().st_size,
           "sha256": h.hexdigest(), "decode_errors": dec.stderr.strip()[-500:]}
    atomic_json(sdir / "inventory.json", out)
    return {"duration_s": round(out["duration_s"], 1), "decode_ok": not out["decode_errors"]}


def fingerprint(ctx, song: Song, sdir: Path) -> Dict:
    """Chromaprint raw fingerprint of the first ``FP_SECONDS`` -> ``fingerprint.npy``."""
    txt = _run(["fpcalc", "-raw", "-length", str(FP_SECONDS), song.audio])
    m = re.search(r"FINGERPRINT=([-\d,]+)", txt)
    fp = np.array([int(v) & 0xFFFFFFFF for v in m.group(1).split(",")], dtype=np.uint32) if m \
        else np.zeros(0, dtype=np.uint32)
    atomic_npy(sdir / "fingerprint.npy", fp)
    return {"values": int(len(fp))}


# ---------------------------------------------------------------------------
# global duplicate detection
# ---------------------------------------------------------------------------
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def offset_votes(a: np.ndarray, b: np.ndarray) -> int:
    """Most likely shift of ``b`` against ``a`` (in fingerprint values): the most
    common position difference of identical (top-28-bit) values.  Works for any
    shift, e.g. a music video with a 40 s intro against the audio release."""
    pos: Dict[int, List[int]] = defaultdict(list)
    for i, v in enumerate((a >> 4).tolist()):
        pos[v].append(i)
    diffs = Counter()
    for j, v in enumerate((b >> 4).tolist()):
        for i in pos.get(v, ())[:5]:
            diffs[i - j] += 1
    return diffs.most_common(1)[0][0] if diffs else 0


def ber(a: np.ndarray, b: np.ndarray, max_shift: int = 40, center: int | None = None) -> Tuple[float, int]:
    """Lowest bit error rate between fingerprints over shifts of +-``max_shift``
    values around ``center`` (default: the vote of ``offset_votes``), and that shift."""
    best, best_s = 1.0, 0
    c = offset_votes(a, b) if center is None else center
    for s in range(c - max_shift, c + max_shift + 1):
        x, y = (a[s:], b) if s >= 0 else (a, b[-s:])
        n = min(len(x), len(y))
        if n < 50:
            continue
        xor = np.bitwise_xor(x[:n], y[:n]).view(np.uint8)
        e = _POP[xor].sum() / (n * 32)
        if e < best:
            best, best_s = float(e), s
    return best, best_s


_PREFER_LESS = re.compile(r"alternate upload|lyrical|lyric video|full video|audio|8d|slowed|reverb|"
                          r"remix|cover|teaser", re.I)


def _rank(song: Song, inv: Dict) -> Tuple:
    """Sort key for choosing the canonical copy: plain titles first, then higher
    bit rate, then longer audio, then id (deterministic)."""
    return (len(_PREFER_LESS.findall(song.title)), -inv.get("bit_rate", 0),
            -inv.get("duration_s", 0), song.song_id)


def find_duplicates(songs: Sequence[Song], work: Path) -> Dict:
    """Group re-uploads of the same recording; write and return ``dupes.json``:
    ``{"groups": [[canonical, dup, ...], ...], "duplicate_of": {dup: canonical},
    "pairs": [[a, b, ber, shift_s], ...]}``."""
    fps, inv = {}, {}
    for s in songs:
        f = work / "songs" / s.song_id / "fingerprint.npy"
        if f.exists():
            fps[s.song_id] = np.load(f)
            inv[s.song_id] = read_json(work / "songs" / s.song_id / "inventory.json", {}) or {}
    index: Dict[int, List[str]] = defaultdict(list)
    for sid, fp in fps.items():
        for v in set((fp >> 4).tolist()):            # drop the 4 noisiest low bits
            index[v].append(sid)
    shared: Counter = Counter()
    for v, ids in index.items():
        if 1 < len(ids) <= 50:                       # very common values carry no information
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    shared[(ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])] += 1
    pairs = []
    for (a, b), n in shared.items():
        if n < MIN_SHARED:
            continue
        e, sh = ber(fps[a], fps[b])
        if e <= DUP_BER:
            pairs.append([a, b, round(e, 3), round(sh * 0.1238, 2)])
    parent = {sid: sid for sid in fps}

    def find(x):
        """Union-find root of ``x``."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _, _ in pairs:
        parent[find(a)] = find(b)
    groups: Dict[str, List[str]] = defaultdict(list)
    for sid in fps:
        groups[find(sid)].append(sid)
    by_id = {s.song_id: s for s in songs}
    out_groups, dup_of = [], {}
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda m: _rank(by_id[m], inv[m]))
        out_groups.append(members)
        for m in members[1:]:
            dup_of[m] = members[0]
    def norm_title(t: str) -> str:
        """Title without upload decorations, letters only (for conflict checks)."""
        t = _PREFER_LESS.sub(" ", re.sub(r"\(.*?\)|\[.*?\]|with lyrics|title track|hindi", " ", t.lower()))
        return re.sub(r"[^a-z]", "", t)

    conflicts = [g for g in out_groups if len({norm_title(by_id[m].title) for m in g}) > 1]
    res = {"groups": sorted(out_groups), "duplicate_of": dup_of, "pairs": sorted(pairs),
           "title_conflicts": sorted(conflicts), "songs_fingerprinted": len(fps)}
    atomic_json(work / "dupes.json", res)
    return res
