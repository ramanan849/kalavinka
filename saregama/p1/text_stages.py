"""CPU stages after the GPU caches (PLAN.md §4–§5, §8): S7 identify (in
``lyrics_ref``), S8 reconcile, S9 align, S10 vocables, S12 notes_qc.

All of them read only cached files (``asr.json``, ``identity.json``, ``mms.npz``,
``f0.npz``, ``poly.npz``, ``single_mask.npy``, ``voices.json``), so they run in
seconds per song and are re-run cheaply whenever thresholds or rules change.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..audio import FPS
from .lyrics_ref import identify, local_align, loose, ref_words, sim_matrix, words_of
from .manifest import Song
from .state import atomic_json, read_json
from .worker import Skip, Stage

_LATIN = re.compile(r"[A-Za-z]")
MATCH = 0.75


# ---------------------------------------------------------------------------
# English words: CMUdict -> Indic phones (one phone set for SoulX's in_ tokens)
# ---------------------------------------------------------------------------
ARPA = {"AA": ["aa"], "AE": ["ae"], "AH": ["a"], "AO": ["O"], "AW": ["au"], "AY": ["ai"], "EH": ["e"],
        "ER": ["a", "r"], "EY": ["ee"], "IH": ["i"], "IY": ["ii"], "OW": ["oo"], "OY": ["O", "i"],
        "UH": ["u"], "UW": ["uu"], "B": ["b"], "CH": ["c"], "D": ["D"], "DH": ["d"], "F": ["f"],
        "G": ["g"], "HH": ["h"], "JH": ["j"], "K": ["k"], "L": ["l"], "M": ["m"], "N": ["n"],
        "NG": ["ng"], "P": ["p"], "R": ["r"], "S": ["s"], "SH": ["sh"], "T": ["T"], "TH": ["th"],
        "V": ["v"], "W": ["v"], "Y": ["y"], "Z": ["z"], "ZH": ["z"]}


@lru_cache(maxsize=1)
def _cmu() -> Dict[str, List[List[str]]]:
    """CMU pronouncing dictionary (``cmudict`` package)."""
    import cmudict
    return cmudict.dict()


def is_english(word: str) -> bool:
    """A Latin-script word found in CMUdict."""
    return bool(_LATIN.search(word)) and word.lower().strip("'") in _cmu()


def english_syllables(word: str) -> List[Tuple[str, List[str]]]:
    """[(text, Indic phones)] syllables of an English word via CMUdict (Indian
    English reading: t/d retroflex, w as v)."""
    from ..g2p import syllabify
    arpa = _cmu()[word.lower().strip("'")][0]
    phones, srcs = [], []
    for k, p in enumerate(arpa):
        for q in ARPA[re.sub(r"\d", "", p)]:
            phones.append(q)
            srcs.append(k)
    groups = syllabify(phones, srcs)
    return [(word if k == 0 else "~", [phones[i] for i in g]) for k, g in enumerate(groups)]


def word_syllables(word: str, lang: str, lexicon=None) -> List[Tuple[str, List[str]]]:
    """[(text, phones)] of one word: English via CMUdict, else the Hindi G2P
    (Devanagari rules, Latin romanised-Hindi rules)."""
    from ..g2p import word_to_syllables
    if lang == "en":
        return english_syllables(word)
    return word_to_syllables(word, "hi", lexicon)


def word_lang(word: str) -> str:
    """'en' for English words in Latin script, else 'hi'."""
    return "en" if is_english(word) else "hi"


# ---------------------------------------------------------------------------
# S5b singers (CPU, from the cached embeddings)
# ---------------------------------------------------------------------------
def merge_small_clusters(lab: np.ndarray, emb: np.ndarray, dur: np.ndarray, min_share: float) -> np.ndarray:
    """Relabel clusters holding less than ``min_share`` of the single-voice time
    to the nearest large cluster (centroid cosine); labels renumbered by size."""
    if len(lab) == 0:
        return lab
    tot = dur.sum() or 1.0
    share = {k: dur[lab == k].sum() / tot for k in set(lab.tolist())}
    big = [k for k, s in share.items() if s >= min_share] or [max(share, key=share.get)]
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    cent = {k: e[lab == k].mean(axis=0) for k in big}
    out = lab.copy()
    for i, k in enumerate(lab):
        if k not in big:
            out[i] = max(big, key=lambda b: float(e[i] @ cent[b]))
    order = sorted(set(out.tolist()), key=lambda k: -dur[out == k].sum())
    ren = {k: r for r, k in enumerate(order)}
    return np.array([ren[k] for k in out])


def singers(ctx, song: Song, sdir: Path) -> Dict:
    """Singer clusters of the single-voice segments, from ``voices_emb.npy`` and
    the segments' median F0 -> ``singers.json``.  CPU-only, so thresholds can be
    tuned without touching the GPU stages."""
    from .gpu_stages import _dup_check, cluster_embeddings
    _dup_check(ctx, song)
    c = ctx.cfg["singers"]
    vj = read_json(sdir / "voices.json", {}) or {}
    segs = vj.get("segments", [])
    emb = np.load(sdir / "voices_emb.npy") if segs else np.zeros((0, 192), np.float32)
    med = np.array([s.get("median_hz", 0.0) for s in segs])
    dur = np.array([s["end"] - s["start"] for s in segs])
    lab = cluster_embeddings(emb, c["threshold"], med, c["register_weight"]) if len(segs) else np.zeros(0, int)
    lab = merge_small_clusters(lab, emb, dur, c["min_cluster_share"])
    out_segs = [{"start": s["start"], "end": s["end"], "median_hz": s.get("median_hz", 0.0), "cluster": int(k)}
                for s, k in zip(segs, lab)]
    clusters = {str(k): {"seconds": round(float(dur[lab == k].sum()), 1),
                         "median_hz": round(float(np.median(med[lab == k])), 1)} for k in sorted(set(lab.tolist()))}
    atomic_json(sdir / "singers.json", {"segments": out_segs, "clusters": clusters})
    return {"clusters": len(clusters)}


def cluster_track(sdir: Path, n: int) -> np.ndarray:
    """Singer cluster per 50 fps frame (-1 outside single-voice segments)."""
    cl = np.full(n, -1)
    for s in (read_json(sdir / "singers.json", {}) or {}).get("segments", []):
        cl[int(s["start"] * FPS): int(s["end"] * FPS)] = s["cluster"]
    return cl


# ---------------------------------------------------------------------------
# S8 reconcile
# ---------------------------------------------------------------------------
def _agrees(word: str, others: Sequence[str]) -> bool:
    """Does ``word`` loosely match any word of ``others``?"""
    if not others:
        return False
    return bool((sim_matrix([word], list(others)) >= MATCH).any())


def _surface(ref_w: str, asr_w: str) -> Tuple[str, str]:
    """(spelling, lang) for a matched ref/ASR word pair: English reference words
    keep their Latin spelling; romanised-Hindi references give way to the ASR's
    Devanagari; Devanagari references win."""
    if _LATIN.search(ref_w):
        if is_english(ref_w):
            return ref_w.lower(), "en"
        return (asr_w, word_lang(asr_w)) if not _LATIN.search(asr_w) else (ref_w.lower(), "hi")
    return ref_w, "hi"


def _w(text: str, src: str, lang: Optional[str] = None, optional: bool = False,
       alt: Optional[str] = None) -> Dict:
    """One word record of lyrics.final.json."""
    return {"text": text, "lang": lang or word_lang(text), "src": src, "optional": optional,
            **({"alt": alt} if alt else {})}


def reconcile_phrase(q: List[str], w: List[str], rws: Optional[List[str]]) -> Tuple[List[Dict], float]:
    """Word decisions for one phrase (PLAN.md §5 table).  ``q`` = Qwen words,
    ``w`` = Whisper words, ``rws`` = reference words (None: no reference).
    Returns the word records and the share of ``q`` matched to the reference."""
    if not rws:
        return [_w(x, "asr+whisper" if _agrees(x, w) else "asr1") for x in q], 0.0
    S = sim_matrix(q, rws)
    _, path = local_align(S, MATCH)
    matched = sum(1 for a, b in path if a is not None and b is not None and S[a, b] >= MATCH)
    ratio = matched / max(len(q), 1)
    if ratio < 0.5:
        return [_w(x, "asr+whisper" if _agrees(x, w) else "asr1") for x in q], ratio
    first = next(a for a, _ in path if a is not None)
    last = max(a for a, _ in path if a is not None)
    out = [_w(x, "asr+whisper" if _agrees(x, w) else "asr1") for x in q[:first]]
    for a, b in path:
        if a is not None and b is not None:
            if S[a, b] >= MATCH:
                text, lang = _surface(rws[b], q[a])
                out.append(_w(text, "ref+asr", lang))
            elif _agrees(rws[b], w):
                text, lang = _surface(rws[b], rws[b])
                out.append(_w(text, "ref+whisper", lang))
            elif _agrees(q[a], w):
                out.append(_w(q[a], "asr+whisper"))
            else:
                out.append(_w(q[a], "disputed", alt=rws[b]))
        elif a is not None:
            out.append(_w(q[a], "asr+whisper" if _agrees(q[a], w) else "asr1"))
        else:
            text, lang = _surface(rws[b], rws[b])
            if _agrees(rws[b], w):
                out.append(_w(text, "ref+whisper", lang))
            else:
                out.append(_w(text, "ref_only", lang, optional=True))
    out += [_w(x, "asr+whisper" if _agrees(x, w) else "asr1") for x in q[last + 1:]]
    return out, ratio


def reconcile(ctx, song: Song, sdir: Path) -> Dict:
    """ASR order + reference spelling, per phrase -> ``lyrics.final.json``."""
    from .gpu_stages import _dup_check
    _dup_check(ctx, song)
    ident = read_json(sdir / "identity.json", {}) or {}
    ref = None
    if ident.get("best"):
        from .lyrics_ref import _pool
        pool, words, _ = _pool(ctx)
        ref = pool["refs"].get(ident["best"]["ref_id"])
    rws = ref_words(ref)[0] if ref else None
    asr = read_json(sdir / "asr.json", {}) or {}
    hz = np.load(sdir / "f0.npz")["f0"]
    single = np.load(sdir / "single_mask.npy")
    seg_cl = cluster_track(sdir, len(single))
    phrases = []
    for p in asr.get("phrases", []):
        fa, fb = int(p["start"] * FPS), int(p["end"] * FPS)
        voiced = hz[fa:fb] > 0
        cl = seg_cl[fa:fb][voiced[: len(seg_cl[fa:fb])]]
        cl = cl[cl >= 0]
        p = {**p, "single_share": round(float(single[fa:fb][voiced].mean()) if voiced.any() else 0.0, 3),
             "cluster": int(np.bincount(cl).argmax()) if len(cl) else -1}
        q, w = words_of(p.get("qwen", "")), words_of(p.get("whisper", ""))
        if not q and w:                                  # Qwen dropped it, Whisper did not
            words, ratio = [_w(x, "whisper1") for x in w], 0.0
        else:
            words, ratio = reconcile_phrase(q, w, rws)
        if not words:
            continue
        phrases.append({"start": p["start"], "end": p["end"], "cluster": p.get("cluster", -1),
                        "single_share": p.get("single_share", 0.0), "ref_match": round(ratio, 3),
                        "words": words})
    atomic_json(sdir / "lyrics.final.json", {"ref_id": ref["id"] if ref else None,
                                              "identity": ident.get("status"), "phrases": phrases})
    srcs = [w["src"] for p in phrases for w in p["words"]]
    return {"phrases": len(phrases), "words": len(srcs),
            "ref_backed": round(sum(s.startswith("ref") for s in srcs) / max(len(srcs), 1), 3)}


# ---------------------------------------------------------------------------
# S9 align (acoustic choice + phrase-window alignment)
# ---------------------------------------------------------------------------
class _CachedEmitter:
    """Stand-in emitter for the cached MMS_FA emissions (``mms.npz``)."""
    unit = "roman"

    def __init__(self, vocab: Dict[str, int], blank: int, star: int):
        """Vocabulary, blank and star ids of the cache."""
        self.vocab, self.blank = vocab, blank
        self.star_id = None if star < 0 else star


def _syllables_for(words: List[Dict], lexicon=None):
    """g2p.Syllable list for a phrase's chosen words (unreadable words dropped)."""
    from ..g2p import Syllable
    out, kept = [], []
    for wi, w in enumerate(words):
        try:
            ss = word_syllables(w["text"], w["lang"], lexicon)
        except (ValueError, KeyError):
            continue
        if not ss:
            continue
        kept.append(wi)
        for k, (t, ph) in enumerate(ss):
            out.append(Syllable(text=t, phones=list(ph), word_index=wi, word_final=k == len(ss) - 1))
    return out, kept


def _variant_score(logp: np.ndarray, sylls, a: int, b: int, em) -> float:
    """CTC path log-probability of one phrase variant inside frames [a, b)."""
    from ..stages.align import build_tokens, ctc_viterbi
    seq, _ = build_tokens([sylls], em.unit, em.vocab, em.star_id)
    if len(seq.ids) + 2 > b - a:
        return -np.inf
    _, score = ctc_viterbi(logp[a:b], seq.ids, em.blank)
    return float(score)


def choose_variant(logp: np.ndarray, words: List[Dict], a: int, b: int, em) -> Tuple[List[Dict], Dict]:
    """Pick the phrase's wording by acoustic score: disputed words as ASR vs as
    reference, optional (reference-only) words in vs out (up to 4 variants)."""
    disputed = any("alt" in w for w in words)
    optional = any(w.get("optional") for w in words)
    variants = []
    for use_ref in ([False, True] if disputed else [False]):
        for with_opt in ([True, False] if optional else [True]):
            ws = []
            for w in words:
                if w.get("optional") and not with_opt:
                    continue
                if "alt" in w and use_ref:
                    t, lang = _surface(w["alt"], w["text"])
                    ws.append({**w, "text": t, "lang": lang, "src": "disputed->ref"})
                elif "alt" in w:
                    ws.append({**w, "src": "disputed->asr"})
                else:
                    ws.append(w)
            variants.append(((use_ref, with_opt), ws))
    if len(variants) == 1:
        return variants[0][1], {}
    scored = []
    for key, ws in variants:
        sy, _ = _syllables_for(ws)
        scored.append((_variant_score(logp, sy, a, b, em), key, ws))
    scored.sort(key=lambda x: -x[0])
    best = scored[0]
    return best[2], {"use_ref": best[1][0], "with_optional": best[1][1],
                     "margin": round(best[0] - scored[1][0], 2) if np.isfinite(scored[1][0]) else None}


def align(ctx, song: Song, sdir: Path) -> Dict:
    """Choose wordings acoustically, then align syllables inside each phrase
    window with the cached emissions -> ``syllables.json``."""
    from .gpu_stages import _dup_check
    _dup_check(ctx, song)
    from ..config import AlignCfg
    from ..stages.align import Aligner
    lf = read_json(sdir / "lyrics.final.json", {}) or {}
    if not lf.get("phrases"):
        raise Skip("no lyrics")
    z = np.load(sdir / "mms.npz")
    logp = z["logp"].astype(np.float32)
    em = _CachedEmitter(json.loads(str(z["vocab"])), int(z["blank"]), int(z["star"]))
    fz = np.load(sdir / "f0.npz")
    hz = fz["f0"]
    energy = np.load(sdir / "poly.npz")["vocal_db"]
    n = min(len(hz), len(logp), len(energy))
    hz, logp, energy = hz[:n], logp[:n], energy[:n]
    pad = ctx.cfg["align"]["window_pad_s"]
    lines, windows, meta, choices = [], [], [], []
    for pi, p in enumerate(lf["phrases"]):
        a, b = max(0, int((p["start"] - pad) * FPS)), min(n, int((p["end"] + pad) * FPS))
        words, ch = choose_variant(logp, p["words"], a, b, em)
        sy, kept = _syllables_for(words)
        if not sy:
            continue
        lines.append(sy)
        windows.append((p["start"], p["end"]))
        meta.append((pi, words))
        if ch:
            choices.append({"phrase": pi, **ch})
    acfg = AlignCfg(backend="external", use_star=em.star_id is not None)
    al = Aligner(acfg)
    al.emitter = em
    aligned, info = al.align(None, lines, hz, energy, emission=logp, windows=windows, window_pad_s=pad)
    out = []
    for s in aligned:
        if s.kind == "vocalise":
            out.append({"kind": "vocalise", "start": s.start, "end": s.end, "line": s.line})
            continue
        syl = lines[s.line][s.index]
        pi, words = meta[s.line]
        w = words[syl.word_index]
        out.append({"kind": "lyric", "text": syl.text, "phones": syl.phones, "start": s.start, "end": s.end,
                    "line": s.line, "phrase": pi, "word": syl.word_index, "word_text": w["text"],
                    "lang": w["lang"], "src": w["src"],
                    "conf": round(float(s.conf), 3) if np.isfinite(s.conf) else None})
    atomic_json(sdir / "syllables.json", {"fps": FPS, "frames": n, "choices": choices,
                                          "info": {k: v for k, v in info.items() if not isinstance(v, np.ndarray)},
                                          "syllables": out})
    lyr = [x for x in out if x["kind"] == "lyric"]
    return {"syllables": len(lyr), "vocalise": len(out) - len(lyr), "choices": len(choices),
            "mean_conf": round(float(np.mean([x["conf"] for x in lyr if x["conf"] is not None])), 3) if lyr else None}


# ---------------------------------------------------------------------------
# S10 vocables (aakar)
# ---------------------------------------------------------------------------
VOWEL_TOKENS = {"a": ("आ", ["aa"]), "e": ("ए", ["ee"]), "i": ("ई", ["ii"]), "o": ("ओ", ["oo"]),
                "u": ("ऊ", ["uu"]), "m": ("म्म", ["m"])}


def classify_vocalise(seg: Dict, prev: Optional[Dict], hz: np.ndarray, logp: np.ndarray,
                      vocab: Dict[str, int]) -> Dict:
    """A voiced stretch no word explains: ``melisma`` if it continues the
    previous syllable (no gap, pitch continuous), else a standalone vocable with
    the vowel the MMS letters favour."""
    a, b = seg["start"], seg["end"]
    if prev is not None and a - prev["end"] <= 3:
        ha, hb = hz[max(prev["end"] - 1, 0)], hz[min(a, len(hz) - 1)]
        if ha > 0 and hb > 0 and abs(1200 * np.log2(hb / ha)) < 300:
            return {"kind": "melisma"}
    ids = {k: vocab[k] for k in VOWEL_TOKENS if k in vocab}
    if not ids:
        return {"kind": "vocable", "vowel": "a"}
    p = np.exp(logp[a:b][:, list(ids.values())]).sum(axis=0)
    vowel = list(ids)[int(np.argmax(p))]
    return {"kind": "vocable", "vowel": vowel, "evidence": round(float(p.max() / max(p.sum(), 1e-9)), 3)}


def vocables(ctx, song: Song, sdir: Path) -> Dict:
    """Label every 'vocalise' span: merge melismas into the previous syllable,
    turn standalone vocalising into vocable syllables (आ, ओ, ...) -> updates
    ``syllables.json`` (``vocables_done``)."""
    from .gpu_stages import _dup_check
    _dup_check(ctx, song)
    sj = read_json(sdir / "syllables.json", None)
    if not sj:
        raise Skip("no syllables")
    z = np.load(sdir / "mms.npz")
    logp, vocab = z["logp"].astype(np.float32), json.loads(str(z["vocab"]))
    hz = np.load(sdir / "f0.npz")["f0"]
    src = [s for s in sj["syllables"] if s["kind"] != "vocable"]   # idempotent re-runs
    out, n_mel, n_voc = [], 0, 0
    for s in src:
        if s["kind"] != "vocalise":
            out.append(s)
            continue
        prev = next((x for x in reversed(out) if x["kind"] == "lyric"), None)
        c = classify_vocalise(s, prev, hz, logp, vocab)
        if c["kind"] == "melisma" and prev is not None:
            prev["end"] = s["end"]
            prev["melisma"] = True
            n_mel += 1
        else:
            text, phones = VOWEL_TOKENS[c["vowel"]]
            out.append({"kind": "vocable", "text": text, "phones": phones, "start": s["start"], "end": s["end"],
                        "line": s.get("line", -1), "phrase": -1, "word": -1, "word_text": text, "lang": "hi",
                        "src": "vocable", "conf": c.get("evidence")})
            n_voc += 1
    sj["syllables"] = out
    sj["vocables_done"] = True
    atomic_json(sdir / "syllables.json", sj)
    return {"melisma": n_mel, "vocables": n_voc}


# ---------------------------------------------------------------------------
# S12 notes + clip plan + confidence features
# ---------------------------------------------------------------------------
GOOD_SRC = {"ref+asr", "ref+whisper", "asr+whisper", "disputed->ref", "vocable"}


def _units(sylls: List[Dict], gap_min: int, max_f: int) -> List[Tuple[int, int]]:
    """Syllable ranges separated by rests >= ``gap_min`` frames; ranges longer than
    ``max_f`` are split at their largest internal gap until they fit."""
    units, i = [], 0
    for k in range(1, len(sylls) + 1):
        if k == len(sylls) or sylls[k]["start"] - sylls[k - 1]["end"] >= gap_min:
            units.append((i, k))
            i = k
    out: List[Tuple[int, int]] = []
    stack = units[::-1]
    while stack:
        i, j = stack.pop()
        if sylls[j - 1]["end"] - sylls[i]["start"] <= max_f or j - i < 2:
            out.append((i, j))
            continue
        gaps = [sylls[k]["start"] - sylls[k - 1]["end"] for k in range(i + 1, j)]
        k = i + 1 + int(np.argmax(gaps))
        stack.extend([(k, j), (i, k)])
    return out


def plan_clips(sylls: List[Dict], single: np.ndarray, cluster: np.ndarray, hz: np.ndarray,
               min_s: float = 2.0, max_s: float = 16.0, split_rest_s: float = 0.3,
               min_single: float = 0.9, min_purity: float = 0.9,
               reasons: Optional[Dict[str, int]] = None) -> List[Tuple[int, int]]:
    """Clip spans (syllable index ranges [i, j)) sung by one voice and one singer.

    The song is cut into phrase units at rests; each unit passes if at least
    ``min_single`` of its voiced frames are single-voice and its dominant singer
    holds at least ``min_purity`` of the labelled frames.  Consecutive passing
    units of the same singer are merged up to ``max_s``; clips under ``min_s``
    are dropped.  ``reasons`` (if given) counts why units were rejected.
    """
    reasons = reasons if reasons is not None else {}
    units = _units(sylls, int(split_rest_s * FPS), int(max_s * FPS))
    good: List[Tuple[int, int, int]] = []          # (i, j, singer)
    for i, j in units:
        a, b = sylls[i]["start"], sylls[j - 1]["end"]
        voiced = hz[a:b] > 0
        if not voiced.any():
            reasons["unvoiced"] = reasons.get("unvoiced", 0) + 1
            continue
        if single[a:b][voiced].mean() < min_single:
            reasons["overlap"] = reasons.get("overlap", 0) + 1
            continue
        cl = cluster[a:b][voiced]
        cl = cl[cl >= 0]
        if not len(cl):
            reasons["no_singer"] = reasons.get("no_singer", 0) + 1
            continue
        counts = np.bincount(cl)
        if counts.max() / len(cl) < min_purity:
            reasons["mixed_singers"] = reasons.get("mixed_singers", 0) + 1
            continue
        good.append((i, j, int(counts.argmax())))
    out: List[Tuple[int, int]] = []
    cur = None
    for i, j, s in good:
        if cur and cur[2] == s and cur[1] == i and sylls[j - 1]["end"] - sylls[cur[0]]["start"] <= max_s * FPS:
            cur = (cur[0], j, s)
            continue
        if cur:
            out.append((cur[0], cur[1]))
        cur = (i, j, s)
    if cur:
        out.append((cur[0], cur[1]))
    keep = [(i, j) for i, j in out if sylls[j - 1]["end"] - sylls[i]["start"] >= min_s * FPS]
    reasons["too_short"] = reasons.get("too_short", 0) + len(out) - len(keep)
    return keep


def notes_qc(ctx, song: Song, sdir: Path) -> Dict:
    """Notes (50 fps), tonic, clip plan and per-clip confidence features ->
    ``annotation.json`` (+ arrays) and ``clips.json``."""
    from .gpu_stages import _dup_check
    _dup_check(ctx, song)
    from ..config import NotesCfg
    from ..schema import SongAnnotation, SyllableAnn
    from ..stages.notes import build_notes
    from ..stages.tonic import estimate_tonic
    sj = read_json(sdir / "syllables.json", None)
    if not sj or not sj.get("vocables_done"):
        raise Skip("no syllables")
    fz = np.load(sdir / "f0.npz")
    hz, agree = fz["f0"], fz["agree"]
    n = min(len(hz), sj["frames"])
    hz, agree = hz[:n], agree[:n]
    single = np.load(sdir / "single_mask.npy")[:n]
    cluster = cluster_track(sdir, n)
    sylls = sorted([s for s in sj["syllables"] if s["kind"] in ("lyric", "vocable")], key=lambda s: s["start"])
    tonic, tconf = estimate_tonic(hz)
    ann_s = [SyllableAnn(s["text"], s["phones"], int(s["start"]), int(min(s["end"], n)), int(s.get("phrase", -1)),
                         int(s.get("word", -1)), float(s["conf"]) if s.get("conf") is not None else 0.0,
                         [s["src"]] + (["melisma"] if s.get("melisma") else []) + (["vocable"] if s["kind"] == "vocable" else []))
             for s in sylls]
    notes = build_notes(ann_s, hz, tonic, n, NotesCfg())
    ident = (read_json(sdir / "identity.json", {}) or {}).get("status", "")
    covered = np.zeros(n, bool)
    for s in sylls:
        covered[s["start"]:s["end"]] = True
    clips = []
    q = ctx.cfg["qc"]
    reasons: Dict[str, int] = {}
    for i, j in plan_clips(sylls, single, cluster, hz, min_single=q.get("clip_min_single", 0.9),
                           min_purity=q.get("clip_min_purity", 0.9), reasons=reasons):
        a, b = sylls[i]["start"], sylls[j - 1]["end"]
        seg = sylls[i:j]
        voiced = hz[a:b] > 0
        confs = [s["conf"] for s in seg if s["kind"] == "lyric" and s.get("conf") is not None]
        cl = cluster[a:b][voiced]
        cl = cl[cl >= 0]
        clips.append({"start": a, "end": b, "syllables": [i, j],
                      "cluster": int(np.bincount(cl).argmax()) if len(cl) else -1,
                      "features": {
                          "dur_s": round((b - a) / FPS, 2),
                          "good_src_share": round(sum(s["src"] in GOOD_SRC for s in seg) / len(seg), 3),
                          "asr1_share": round(sum(s["src"] in ("asr1", "whisper1", "disputed->asr") for s in seg) / len(seg), 3),
                          "ref_backed_share": round(sum(s["src"].startswith("ref") or s["src"] == "disputed->ref" for s in seg) / len(seg), 3),
                          "mean_conf": round(float(np.mean(confs)), 3) if confs else None,
                          "min_conf": round(float(np.min(confs)), 3) if confs else None,
                          "f0_agree": round(float(agree[a:b][voiced].mean()), 3) if voiced.any() else 0.0,
                          "unlabelled_voiced": round(float((voiced & ~covered[a:b]).sum() / max(voiced.sum(), 1)), 3),
                          "vocables": sum(s["kind"] == "vocable" for s in seg),
                          "identity": ident}})
    ann = SongAnnotation(song_id=song.song_id, language="hi", style="film", audio_path=song.audio,
                         vocals_path=str(sdir / "vocals.flac"), n_frames=n, tonic_hz=round(float(tonic), 3),
                         tonic_conf=round(float(tconf), 3), lines=[], syllables=ann_s, notes=notes,
                         f0=hz.astype(np.float32), f0_agree=agree.astype(np.float32),
                         energy_db=np.load(sdir / "poly.npz")["vocal_db"][:n],
                         hnr_db=np.zeros(n, np.float32), h1h2_db=np.zeros(n, np.float32),
                         qc={"identity": ident}, history=["p1"])
    ann.save(sdir)
    atomic_json(sdir / "clips.json", {"fps": FPS, "clips": clips, "rejected_units": reasons})
    return {"clips": len(clips), "rejected_units": reasons,
            "clip_s": round(sum(c["features"]["dur_s"] for c in clips), 1),
            "notes": len(notes)}


STAGES = [
    Stage("singers", 1, singers, sections=("singers",), depends=("voices",)),
    Stage("identify", 1, identify, sections=("lyrics",), depends=("asr",)),
    Stage("reconcile", 1, reconcile, sections=("lyrics",), depends=("identify", "singers")),
    Stage("align", 1, align, sections=("align",), depends=("reconcile", "emissions")),
    Stage("vocables", 1, vocables, sections=(), depends=("align",)),
    Stage("notes_qc", 2, notes_qc, sections=("qc",), depends=("vocables", "singers")),
]
