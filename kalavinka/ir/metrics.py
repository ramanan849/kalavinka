"""Contour metrics for the IR model (stage 1) and for F0 of rendered audio.

All on 50 fps F0 in Hz (0 = unvoiced), compared on frames voiced in both:

* ``cents_rmse``     RMS of the cents difference;
* ``orn_rmse``       the same on *ornament* frames, where the reference moves
                     faster than 300 cents/s (the svc-check / pre-report definition);
* ``voicing_f1``     F1 of the voiced/unvoiced decision;
* ``vib_extent`` / ``vib_rate``  per sustained note (>= 0.4 s voiced inside one note): the
                     depth (std of the detrended cents, i.e. vibrato extent) and
                     rate (zero crossings / 2 per second) -- their distributions
                     are compared with the 1-D Wasserstein distance
                     (``wasserstein``) between generated and real contours;
* ``flatness``       median |slope| in cents/s over voiced frames (expressiveness; flat MIDI = ~0).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

FPS = 50


def cents(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1200 log2(a / b) (inputs > 0)."""
    return 1200 * np.log2(a / b)


def pair_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    """cents RMSE, ornament RMSE and voicing F1 for one clip."""
    n = min(len(pred), len(ref))
    p, r = pred[:n], ref[:n]
    both = (p > 0) & (r > 0)
    out = {}
    if both.any():
        d = cents(p[both], r[both])
        d = d - 1200 * np.round(d / 1200) * (np.abs(d) > 900)            # forgive octave errors
        out["cents_rmse"] = float(np.sqrt(np.mean(d ** 2)))
        slope = np.zeros(n)
        rc = np.where(r > 0, 1200 * np.log2(np.maximum(r, 1) / 55.0), np.nan)
        slope[1:] = np.abs(np.diff(rc)) * FPS
        orn = both & (np.nan_to_num(slope) > 300)
        if orn.any():
            e = cents(p[orn], r[orn])
            out["orn_rmse"] = float(np.sqrt(np.mean(e ** 2)))
    tp = float(((p > 0) & (r > 0)).sum()); fp = float(((p > 0) & (r <= 0)).sum()); fn = float(((p <= 0) & (r > 0)).sum())
    out["voicing_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    return out


def runs(mask: np.ndarray, min_len: int) -> List[slice]:
    """Slices of True runs at least ``min_len`` long."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [slice(a, b) for a, b in zip(idx[::2], idx[1::2]) if b - a >= min_len]


def note_runs(f0: np.ndarray, flat: np.ndarray, min_len: int) -> List[slice]:
    """Voiced runs inside one note (constant flat pitch), at least ``min_len`` frames."""
    out = []
    for s in runs((f0 > 0) & (flat > 0), min_len):
        seg = flat[s]
        cut = np.flatnonzero(np.diff(seg) != 0) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(seg)]):
            if b - a >= min_len:
                out.append(slice(s.start + a, s.start + b))
    return out


def vibrato_stats(f0: np.ndarray, flat: np.ndarray) -> Dict[str, List[float]]:
    """Per sustained note (>= 0.4 s voiced inside one note): vibrato extent (cents std after
    removing a linear trend) and rate (Hz, zero crossings / 2)."""
    ext, rate = [], []
    for s in note_runs(f0, flat, int(0.4 * FPS)):
        c = 1200 * np.log2(f0[s] / 55.0)
        x = np.arange(len(c))
        c = c - np.polyval(np.polyfit(x, c, 1), x)
        ext.append(float(np.std(c)))
        zc = np.sum(np.diff(np.sign(c)) != 0)
        rate.append(float(zc / 2 / (len(c) / FPS)))
    return {"extent": ext, "rate": rate}


def flatness(f0: np.ndarray) -> float:
    """Mean |slope| (cents/s) over consecutive voiced frames."""
    v = (f0[1:] > 0) & (f0[:-1] > 0)
    if not v.any():
        return 0.0
    return float(np.median(np.abs(np.diff(1200 * np.log2(np.maximum(f0, 1)))[v])) * FPS)


def wasserstein(a: Sequence[float], b: Sequence[float]) -> float:
    """1-D Wasserstein-1 distance between two samples (quantile matching)."""
    if not len(a) or not len(b):
        return float("nan")
    q = np.linspace(0, 1, 101)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def corpus_metrics(preds: Sequence[np.ndarray], refs: Sequence[np.ndarray], flats: Sequence[np.ndarray]
                   ) -> Dict[str, float]:
    """Mean per-clip metrics plus vibrato-distribution distances (notes from ``flats``,
    the flat note pitch per frame) and flatness over a set of clips."""
    per = [pair_metrics(p, r) for p, r in zip(preds, refs)]
    out = {k: float(np.nanmean([m.get(k, np.nan) for m in per])) for k in ("cents_rmse", "orn_rmse", "voicing_f1")}
    vp = [vibrato_stats(p, f) for p, f in zip(preds, flats)]
    vr = [vibrato_stats(r, f) for r, f in zip(refs, flats)]
    for k in ("extent", "rate"):
        a = [x for v in vp for x in v[k]]
        b = [x for v in vr for x in v[k]]
        out[f"vib_{k}_pred"] = float(np.median(a)) if a else float("nan")
        out[f"vib_{k}_ref"] = float(np.median(b)) if b else float("nan")
        out[f"vib_{k}_w1"] = wasserstein(a, b)
    out["flatness_pred"] = float(np.mean([flatness(p) for p in preds]))
    out["flatness_ref"] = float(np.mean([flatness(r) for r in refs]))
    return out
