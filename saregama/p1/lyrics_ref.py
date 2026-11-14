"""Web lyrics: fetch lists, the reference pool, local word alignment, S7 identify.

**Fetching** happens on the login node (compute nodes are offline):
``p1-fetchlist`` writes ``work/raw/fetch_title.tsv`` (LRCLIB search by the
claimed title, YouTube oEmbed for the real video title / channel) and
``ops/fetch.sh`` downloads it with curl.

**Reference pool.** LRCLIB search covers titles and artists, not lyric text, so
a mislabeled file can't be looked up by its words.  Instead every lyric fetched
for every claimed title goes into one pool (``work/refpool.json``), and each
song's ASR transcript is matched against the whole pool: a file labeled X that
holds song Y is found whenever Y's lyrics are in the pool (popular songs
usually are, under their own title).

**Matching** works on the loose phonetic spelling (``lyrics_eval.loose_word``),
script-independent, with Smith-Waterman local alignment per ASR phrase, so
repeats, partial lines and any order map correctly.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .manifest import Song
from .state import atomic_bytes, atomic_json, read_json

LRCLIB = "https://lrclib.net/api/search?track_name={}"
OEMBED = "https://www.youtube.com/oembed?url=https%3A//www.youtube.com/watch%3Fv%3D{}&format=json"
_DECOR = re.compile(r"\((?:[^)]*(?:lyric|video|audio|upload|version|song|from)[^)]*)\)|\[[^\]]*\]|"
                    r"\b(?:lyrical|lyrics?|with lyrics|full song|full video|video song|audio song|"
                    r"alternate upload|official|hd|4k|title track)\b", re.I)


def clean_title(title: str) -> str:
    """Song title without upload decorations ("(Lyrical)", "alternate upload", ...)."""
    t = _DECOR.sub(" ", title)
    return re.sub(r"\s+", " ", re.sub(r"[|_]+", " ", t)).strip(" -") or title


def write_fetchlist(songs: Sequence[Song], work: Path, dup: Dict[str, str]) -> Path:
    """``work/raw/fetch_title.tsv``: LRCLIB by title + oEmbed per (non-duplicate) song."""
    rows = []
    for s in songs:
        if s.song_id in dup:
            continue
        rows.append(f"lrclib_title/{s.song_id}.json\t{LRCLIB.format(urllib.parse.quote(clean_title(s.title)))}")
        if s.video_id:
            rows.append(f"oembed/{s.song_id}.json\t{OEMBED.format(urllib.parse.quote(s.video_id))}")
    p = work / "raw" / "fetch_title.tsv"
    atomic_bytes(p, ("\n".join(rows) + "\n").encode())
    return p


# ---------------------------------------------------------------------------
# reference pool
# ---------------------------------------------------------------------------
_LRC = re.compile(r"^\s*\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$")


def ref_lines(rec: Dict) -> List[Tuple[Optional[float], str]]:
    """(time or None, text) lines of an LRCLIB record (synced preferred)."""
    if rec.get("syncedLyrics"):
        out = []
        for line in rec["syncedLyrics"].splitlines():
            m = _LRC.match(line)
            if m and m.group(3).strip():
                out.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
        if out:
            return out
    return [(None, l.strip()) for l in (rec.get("plainLyrics") or "").splitlines() if l.strip()]


def build_pool(work: Path) -> Dict:
    """All fetched LRCLIB records with lyrics, deduplicated by id, with the songs
    that fetched them -> ``work/refpool.json``."""
    refs: Dict[str, Dict] = {}
    by_song: Dict[str, List[str]] = defaultdict(list)
    for f in sorted((work / "raw" / "lrclib_title").glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for rec in data[:15]:
            if rec.get("instrumental") or not (rec.get("syncedLyrics") or rec.get("plainLyrics")):
                continue
            rid = str(rec["id"])
            by_song[f.stem].append(rid)
            if rid not in refs:
                refs[rid] = {"id": rid, "track": rec.get("trackName", ""), "artist": rec.get("artistName", ""),
                             "album": rec.get("albumName", ""), "duration": rec.get("duration"),
                             "synced": bool(rec.get("syncedLyrics")), "lines": ref_lines(rec)}
    pool = {"refs": refs, "by_song": by_song}
    atomic_json(work / "refpool.json", pool)
    return pool


# ---------------------------------------------------------------------------
# loose words and local alignment
# ---------------------------------------------------------------------------
@lru_cache(maxsize=200_000)
def loose(word: str) -> str:
    """Cached loose phonetic spelling of one word ("" if unreadable)."""
    from ..lyrics_eval import loose_word
    return loose_word(word)


def words_of(text: str) -> List[str]:
    """Surface words of a line (same splitting as the G2P)."""
    from ..g2p import split_words
    return [w for w in split_words(text.replace("।", " ").replace("॥", " ")) if loose(w)]


def sim_matrix(a: Sequence[str], b: Sequence[str]) -> np.ndarray:
    """Similarity 0..1 between loose words (normalised Indel similarity)."""
    from rapidfuzz.distance import Indel
    from rapidfuzz.process import cdist
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return cdist([loose(x) for x in a], [loose(y) for y in b], scorer=Indel.normalized_similarity,
                 dtype=np.float32, workers=1)


def local_align(S: np.ndarray, match: float = 0.75, gap: float = 0.6
                ) -> Tuple[float, List[Tuple[Optional[int], Optional[int]]]]:
    """Smith-Waterman over a similarity matrix S (query x reference).  Pair score
    is +2*s for s >= ``match``, else -(1 - s); gaps cost ``gap``.  Returns the best
    local score and its path as (query index or None, reference index or None)."""
    m, n = S.shape
    if m == 0 or n == 0:
        return 0.0, []
    P = np.where(S >= match, 2.0 * S, -(1.0 - S))
    H = np.zeros((m + 1, n + 1), dtype=np.float32)
    T = np.zeros((m + 1, n + 1), dtype=np.int8)     # 0 stop, 1 diag, 2 up (query gap), 3 left (ref gap)
    for i in range(1, m + 1):
        diag = H[i - 1, :-1] + P[i - 1]
        up = H[i - 1, 1:] - gap
        row = np.maximum(0, np.maximum(diag, up))
        t = np.where(row == 0, 0, np.where(diag >= up, 1, 2)).astype(np.int8)
        for j in range(1, n + 1):                   # left moves need a sequential pass
            left = row[j - 2] - gap if j >= 2 else -1.0
            if left > row[j - 1]:
                row[j - 1], t[j - 1] = left, 3
        H[i, 1:], T[i, 1:] = row, t
    i, j = np.unravel_index(int(np.argmax(H)), H.shape)
    best = float(H[i, j])
    path: List[Tuple[Optional[int], Optional[int]]] = []
    while i > 0 and j > 0 and T[i, j] != 0:
        if T[i, j] == 1:
            path.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif T[i, j] == 2:
            path.append((i - 1, None))
            i -= 1
        else:
            path.append((None, j - 1))
            j -= 1
    return best, path[::-1]


def ref_words(ref: Dict) -> Tuple[List[str], List[int]]:
    """All reference words in order and the line index of each."""
    ws, li = [], []
    for k, (_, text) in enumerate(ref["lines"]):
        for w in words_of(re.sub(r"\(([^)]*)\)", r" \1 ", text)):
            ws.append(w)
            li.append(k)
    return ws, li


def coverage(phrases: Sequence[List[str]], rws: Sequence[str], match: float = 0.75) -> float:
    """Share of ASR words matched (similarity >= ``match``) by per-phrase local
    alignment against the reference word list."""
    total = sum(len(p) for p in phrases)
    if not total or not rws:
        return 0.0
    hit = 0
    for p in phrases:
        if not p:
            continue
        S = sim_matrix(p, rws)
        _, path = local_align(S, match)
        hit += sum(1 for a, b in path if a is not None and b is not None and S[a, b] >= match)
    return hit / total


# ---------------------------------------------------------------------------
# S7 identify
# ---------------------------------------------------------------------------
def _pool(ctx) -> Tuple[Dict, Dict[str, List[str]], Dict[str, set]]:
    """(pool, words per ref, ref ids per loose-word trigram), cached per worker."""
    def build():
        """Load refpool.json and index it by loose-word trigrams."""
        pool = read_json(ctx.work / "refpool.json", None)
        if pool is None:
            raise RuntimeError("refpool.json missing: run p1-refpool after fetching")
        words, index = {}, defaultdict(set)
        for rid, ref in pool["refs"].items():
            ws, _ = ref_words(ref)
            words[rid] = ws
            lw = [loose(w) for w in ws]
            for k in range(len(lw) - 2):
                index[" ".join(lw[k:k + 3])].add(rid)
        return pool, words, index
    return ctx.model("refpool", build)


def asr_phrase_words(sdir: Path, system: str = "qwen") -> List[List[str]]:
    """Words of each kept ASR phrase of ``system``."""
    a = read_json(sdir / "asr.json", {}) or {}
    return [words_of(p[system]) for p in a.get("phrases", []) if p.get(system)]


def identify(ctx, song: Song, sdir: Path) -> Dict:
    """Which song is this?  Claimed-title candidates and the pool's best trigram
    matches are scored by ASR coverage -> ``identity.json``."""
    from .gpu_stages import _dup_check
    _dup_check(ctx, song)
    pool, words, index = _pool(ctx)
    c = ctx.cfg["lyrics"]
    phrases = asr_phrase_words(sdir)
    lw = [loose(w) for p in phrases for w in p]
    votes = Counter()
    for k in range(len(lw) - 2):
        for rid in index.get(" ".join(lw[k:k + 3]), ()):
            votes[rid] += 1
    claimed = list(dict.fromkeys(pool["by_song"].get(song.song_id, [])))
    cands = list(dict.fromkeys(claimed + [rid for rid, _ in votes.most_common(8)]))
    dur = (read_json(sdir / "inventory.json", {}) or {}).get("duration_s", 0)
    scored = []
    for rid in cands:
        cov = coverage(phrases, words.get(rid, []), c["match_threshold"])
        r = pool["refs"][rid]
        scored.append({"ref_id": rid, "track": r["track"], "artist": r["artist"], "coverage": round(cov, 3),
                       "claimed": rid in claimed, "duration": r.get("duration"), "synced": r["synced"]})
    scored.sort(key=lambda x: (-x["coverage"], not x["claimed"], -int(x["synced"])))
    best = scored[0] if scored else None
    best_claimed = next((x for x in scored if x["claimed"]), None)
    thr = c["min_version_coverage"]
    if best is None or best["coverage"] < thr:
        status = "unidentified"
    elif best["claimed"] or (best_claimed and best_claimed["coverage"] >= best["coverage"] - 0.05):
        status = "claimed"
        best = best_claimed if best_claimed and best_claimed["coverage"] >= best["coverage"] - 0.05 else best
    else:
        status = "relabeled"
    out = {"status": status, "best": best if status != "unidentified" else None,
           "claimed_title": song.title, "claimed_best": best_claimed, "audio_duration": dur,
           "asr_words": len(lw), "candidates": scored[:6]}
    atomic_json(sdir / "identity.json", out)
    return {"status": status, "coverage": best["coverage"] if best else 0.0,
            "track": (best or {}).get("track", "")}
