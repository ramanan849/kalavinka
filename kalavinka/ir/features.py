"""Frame features and targets for the expressive-contour IR (stage 1).

The **IR** is the 50 fps expressive F0 that SoulX-Singer's melody mode follows.
Stage 1 predicts it from what a composer writes -- the *note scaffold* -- so
that a user can give notes + lyrics and still get Bollywood-style singing
(vibrato, meend glides, murki turns, scoops into notes), which score mode
flattens and which SoulX itself blurs even when given the contour (our svc-check:
43 cents RMSE, ~40 c of gamaka depth lost).

Per frame (``scaffold``), from one export item:

* ``cont`` (8 floats): note pitch in semitones re the clip's tonic / 12, the
  same for the previous and next sung notes (0 at edges), position in the
  note (0..1), seconds since onset and until offset (clipped at 2 s), note
  duration (clipped at 4 s), a rest flag;
* ``cat`` (4 ints): note type (1 rest / 2 onset / 3 slur), pitch class re
  tonic (0-11, 12 = rest), first-phone class and last-phone class of the
  note's syllable (vowel / nasal / stop / fricative / liquid-glide / none);
* ``ctx`` (2 ints): decade bucket (1980s ... 2020s) and register bucket
  (median note pitch: low / mid / high), the style condition.

Targets: ``resid`` = cents of F0 above the note's flat pitch (clamped to
+-600, scaled /100) and ``voiced``; ``rmask`` marks frames where the
residual is defined (voiced and inside a sung note).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from saregama.phoneset import BY_NAME

FPS = 50
CLASSES = {"none": 0, "vowel": 1, "nasal": 2, "stop": 3, "fricative": 4, "liquid": 5}
_NASAL = {"m", "n", "N", "ng", "ny", "nn", "nas"}
_STOP = {"k", "kh", "g", "gh", "c", "ch", "j", "jh", "T", "Th", "D", "Dh", "t", "th", "d", "dh", "p", "ph", "b",
         "bh", "q", "ts", "dz"}
_FRIC = {"sh", "Sh", "s", "h", "x", "gg", "z", "f"}


def phone_class(p: str) -> int:
    """Coarse articulatory class of an Indic phone name."""
    if p in _NASAL:
        return CLASSES["nasal"]
    if p in _STOP:
        return CLASSES["stop"]
    if p in _FRIC:
        return CLASSES["fricative"]
    if p in BY_NAME and BY_NAME[p].kind == "vowel":
        return CLASSES["vowel"]
    return CLASSES["liquid"]


def midi_to_hz(m: np.ndarray) -> np.ndarray:
    """MIDI note numbers -> Hz."""
    return 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)


def note_frames(item: Dict) -> np.ndarray:
    """Integer frame count per note (sums to n_frames)."""
    fr = np.array([int(round(float(d) * FPS)) for d in item["duration"].split()])
    fr[-1] += int(item["n_frames"]) - fr.sum()
    return fr


def scaffold(item: Dict) -> Dict[str, np.ndarray]:
    """Frame features and targets of one export item (see the module doc)."""
    n = int(item["n_frames"])
    fr = note_frames(item)
    pitch = np.array([int(x) for x in item["note_pitch"].split()])
    ntype = np.array([int(x) for x in item["note_type"].split()])
    phon = item["phoneme"].split()
    tonic = float(item.get("tonic_hz") or 0) or float(np.median(midi_to_hz(pitch[pitch > 0]))) if (pitch > 0).any() else 200.0
    tonic_midi = 69 + 12 * np.log2(tonic / 440.0)
    sung = [k for k in range(len(pitch)) if pitch[k] > 0]
    rel = np.where(pitch > 0, (pitch - tonic_midi) / 12.0, 0.0)
    cont = np.zeros((n, 8), np.float32)
    cat = np.zeros((n, 4), np.int64)
    flat = np.zeros(n, np.float32)
    t = 0
    for k, f in enumerate(fr):
        if f <= 0:
            continue
        sl = slice(t, t + f)
        prev = [j for j in sung if j < k]
        nxt = [j for j in sung if j > k]
        pos = (np.arange(f) + 0.5) / f
        cont[sl, 0] = rel[k]
        cont[sl, 1] = rel[prev[-1]] if prev else 0.0
        cont[sl, 2] = rel[nxt[0]] if nxt else 0.0
        cont[sl, 3] = pos
        cont[sl, 4] = np.minimum(np.arange(f) / FPS, 2.0)
        cont[sl, 5] = np.minimum((f - np.arange(f)) / FPS, 2.0)
        cont[sl, 6] = min(f / FPS, 4.0)
        cont[sl, 7] = float(pitch[k] == 0)
        p = phon[k]
        phs = p[3:].split("-") if p.startswith("in_") else []
        cat[sl, 0] = ntype[k]
        cat[sl, 1] = int(round(pitch[k] - tonic_midi)) % 12 if pitch[k] > 0 else 12
        cat[sl, 2] = phone_class(phs[0]) if phs else 0
        cat[sl, 3] = phone_class(phs[-1]) if phs else 0
        if pitch[k] > 0:
            flat[sl] = midi_to_hz(pitch[k])
        t += f
    f0 = np.array([float(x) for x in item["f0"].split()], np.float32)[:n]
    f0 = np.pad(f0, (0, n - len(f0)))
    voiced = (f0 > 0).astype(np.float32)
    rmask = ((f0 > 0) & (flat > 0)).astype(np.float32)
    resid = np.where(rmask > 0, 1200 * np.log2(np.maximum(f0, 1) / np.maximum(flat, 1)), 0.0)
    resid = np.clip(resid, -600, 600) / 100.0
    year = int(str(item.get("year") or item.get("song_id", "2000")[:4]) or 2000)
    decade = int(np.clip((year - 1980) // 10, 0, 4))
    med = np.median(pitch[pitch > 0]) if (pitch > 0).any() else 60
    register = 0 if med < 55 else (1 if med < 64 else 2)
    return {"cont": cont, "cat": cat, "ctx": np.array([decade, register], np.int64), "flat_hz": flat,
            "f0": f0, "resid": resid.astype(np.float32), "voiced": voiced, "rmask": rmask}


def to_f0(flat_hz: np.ndarray, resid: np.ndarray, voiced_p: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Expressive F0 (Hz) from the flat note pitch, predicted residual (/100 cents) and voicing."""
    f0 = flat_hz * 2 ** (resid * 100 / 1200)
    return np.where((voiced_p >= thresh) & (flat_hz > 0), f0, 0.0).astype(np.float32)


def batchify(items: List[Dict[str, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Pad a list of scaffolds to the longest; returns (arrays with a leading batch dim, frame mask)."""
    n = max(len(x["f0"]) for x in items)
    out: Dict[str, np.ndarray] = {}
    for k in ("cont", "cat", "flat_hz", "f0", "resid", "voiced", "rmask"):
        shape = (len(items), n) + items[0][k].shape[1:]
        a = np.zeros(shape, items[0][k].dtype)
        for i, x in enumerate(items):
            a[i, : len(x[k])] = x[k]
        out[k] = a
    out["ctx"] = np.stack([x["ctx"] for x in items])
    mask = np.zeros((len(items), n), np.float32)
    for i, x in enumerate(items):
        mask[i, : len(x["f0"])] = 1
    return out, mask
