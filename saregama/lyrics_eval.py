"""Scoring automatic lyrics against reference lyrics: words and timing.

Film-song references come in any script: Devanagari, Roman Hinglish, Gurmukhi
for Punjabi lines, or a mix (LRCLIB's synced lyrics are like that), and ASR
systems answer in Devanagari (Qwen3-ASR, Whisper) or Roman Hinglish (Hinglish
fine-tunes).  So both sides are reduced to one **loose phonetic spelling**
before scoring:

* every word goes through the pipeline's G2P (``g2p.word_to_syllables``:
  Devanagari with schwa deletion, Latin via ``g2p.latin_word``), Gurmukhi is
  transliterated to Devanagari first;
* the phones are folded where Roman spellings of Hindi are ambiguous: vowel
  length, aspiration, retroflex vs dental, nasalisation, v/w, z/j, f/ph, q/k;
  doubled letters collapse.

WER and CER on that spelling compare systems fairly across scripts; they are
not comparable to WERs computed on native script.

**Timing.** An LRC file gives the start time of every sung line.  The
hypothesis words (with onsets from the aligner, or from the ASR model) are
matched to the reference words by edit-distance alignment; each reference line
whose first word is matched gives one onset error.  The recording and the LRC
may be different releases (a longer intro, say), so one global offset, the
median error, is removed first and reported.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .g2p import split_words, word_to_syllables
from .phoneset import BY_NAME
from .stages.asr import gurmukhi_to_devanagari

# phone -> loose letter(s); phones not listed keep their roman spelling
_FOLD = {"aa": "a", "ii": "i", "uu": "u", "ee": "e", "oo": "o", "E": "e", "O": "o", "ae": "e",
         "ai": "e", "au": "o", "nas": "n", "kh": "k", "gh": "g", "c": "c", "ch": "c", "jh": "j",
         "T": "t", "Th": "t", "th": "t", "D": "d", "Dh": "d", "dh": "d", "N": "n", "ny": "n",
         "ng": "n", "ph": "f", "bh": "b", "v": "v", "Sh": "s", "sh": "s", "R": "r", "Rh": "r",
         "q": "k", "x": "k", "gg": "g", "z": "j", "f": "f", "L": "l", "nn": "n", "rr": "r",
         "zh": "l", "ts": "c", "dz": "j"}
_LRC_LINE = re.compile(r"^\s*\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)$")


def loose_word(word: str, lang: str = "hi") -> str:
    """One word in the loose phonetic spelling ("" if it has no letters we read)."""
    try:
        sylls = word_to_syllables(gurmukhi_to_devanagari(word), lang)
    except (ValueError, KeyError):
        return ""
    s = "".join(_FOLD.get(p, BY_NAME[p].roman) for _, phs in sylls for p in phs)
    return re.sub(r"(.)\1+", r"\1", s)


def loose_words(text: str, lang: str = "hi") -> List[str]:
    """Words of ``text`` in the loose spelling (unreadable words dropped)."""
    t = unicodedata.normalize("NFC", gurmukhi_to_devanagari(text))
    t = re.sub(r"[।॥]", " ", t)                   # dandas
    return [w for w in (loose_word(x, lang) for x in split_words(t)) if w]


def _edit_ops(ref: Sequence, hyp: Sequence) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """Levenshtein alignment as a list of (op, ref index, hyp index); op is one
    of ``ok``, ``sub``, ``del`` (ref word missing) or ``ins`` (extra hyp word)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def error_rates(ref_text: str, hyp_text: str, lang: str = "hi") -> Dict[str, float]:
    """WER and CER of ``hyp_text`` against ``ref_text`` in the loose spelling,
    plus substitution / deletion / insertion counts and word totals."""
    r, h = loose_words(ref_text, lang), loose_words(hyp_text, lang)
    ops = _edit_ops(r, h)
    cnt = {k: sum(1 for o in ops if o[0] == k) for k in ("sub", "del", "ins")}
    rc, hc = " ".join(r), " ".join(h)
    cer = sum(1 for o in _edit_ops(list(rc), list(hc)) if o[0] != "ok") / max(len(rc), 1)
    return {"wer": (cnt["sub"] + cnt["del"] + cnt["ins"]) / max(len(r), 1), "cer": cer,
            "ref_words": len(r), "hyp_words": len(h), **cnt}


def parse_lrc(text: str) -> List[Tuple[float, str]]:
    """(start s, text) of every non-empty line of an LRC file."""
    out = []
    for line in text.splitlines():
        m = _LRC_LINE.match(line)
        if m and m.group(3).strip():
            out.append((int(m.group(1)) * 60 + float(m.group(2)), m.group(3).strip()))
    return out


def _robust_line(x: np.ndarray, y: np.ndarray, iters: int = 4, keep_s: float = 2.0
                 ) -> Tuple[float, float]:
    """Slope and intercept of y ~ x by least squares, refitted on the pairs
    within ``keep_s`` of the previous fit (starting from slope 1, median offset)."""
    a, b = 1.0, float(np.median(y - x))
    for _ in range(iters):
        keep = np.abs(y - (a * x + b)) <= keep_s
        if keep.sum() < 3:
            break
        a, b = np.polyfit(x[keep], y[keep], 1)
    return float(a), float(b)


def line_onset_errors(ref_lines: Sequence[Tuple[float, str]], hyp_words: Sequence[Tuple[str, float]],
                      lang: str = "hi", fit: str = "offset") -> Dict:
    """Line-onset timing of hypothesis words against LRC lines.

    ``hyp_words`` are (word, onset s) in sung order.  Reference words come from
    ``ref_lines``; the first word of each line carries the line's time.  Returns
    the offset removed (median error), the errors after removing it (s) and
    summary statistics; lines whose first word is unmatched are not scored.

    ``fit="linear"`` removes an offset *and* a constant tempo drift (robust line
    fit), for a recording that is a slightly different release of the LRC's
    (re-mastered at another speed, say); ``drift`` is then the speed ratio.
    """
    rw, rt = [], []
    for t, txt in ref_lines:
        for k, w in enumerate(loose_words(txt, lang)):
            rw.append(w)
            rt.append(t if k == 0 else None)
    hw, ht = [], []
    for w, t in hyp_words:
        lw = loose_word(w, lang)
        if lw:
            hw.append(lw)
            ht.append(t)
    pairs = [(rt[i], ht[j]) for op, i, j in _edit_ops(rw, hw) if op == "ok" and rt[i] is not None]
    n_lines = sum(1 for t in rt if t is not None)
    if not pairs:
        return {"lines": n_lines, "scored": 0}
    x, y = np.array(pairs).T
    extra = {}
    if fit == "linear" and len(x) >= 3:
        a, off = _robust_line(x, y)
        err = np.abs(y - (a * x + off))
        extra["drift"] = round(a, 5)
    else:
        off = float(np.median(y - x))
        err = np.abs(y - x - off)
    return {"lines": n_lines, "scored": int(len(err)), "offset_s": round(off, 3), **extra,
            "pairs": [[round(float(p), 2), round(float(q), 2)] for p, q in zip(x, y)],
            "median_abs_s": round(float(np.median(err)), 3), "mean_abs_s": round(float(err.mean()), 3),
            "within_0.3s": round(float((err <= 0.3).mean()), 3),
            "within_0.5s": round(float((err <= 0.5).mean()), 3),
            "within_1s": round(float((err <= 1.0).mean()), 3), "errors_s": [round(float(e), 3) for e in err]}
