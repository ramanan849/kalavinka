"""Compose: write the melody yourself (swaras, timing, ornaments, a hand-drawn F0 curve) and have a model sing it.

Why
---
``sing`` puts new words on a bundled tune and ``sargam`` sings plain swaras. Neither lets you decide *how* a note is
sung. Here you control every syllable's pitch and length, add Hindustani/Carnatic ornaments (meend, gamak, kan, murki,
andolan, vibrato ...), and even draw the pitch curve (F0) by hand. The result is a 50 fps F0 curve plus one note per
syllable; SoulX (Kalavinka V4-L, or the base model) sings it in **melody mode**, i.e. it follows that curve.

The composition (JSON)
----------------------
One JSON object is the single source of truth (the web editor, the table, the command line and LLMs all write it)::

    {"title": "...", "voice": 4, "tonic": "auto" | "C#3" | 138.6, "tempo_bpm": 80,
     "raag": "Yaman", "taal": "teentaal", "smooth_ms": 30,
     "phrases": [
       {"notes": [
          {"syl": "रा", "swara": "Ga", "beats": 1,
           "orn": [{"type": "meend", "to": "Pa", "start": 0.5}]},
          {"syl": "-",  "swara": "Pa", "beats": 0.5},      # "-": hold the previous syllable (a slur)
          {"syl": "_",  "beats": 1},                       # "_": a rest
          {"syl": "",   "swara": "Dha", "beats": 1}        # empty: sing the swara's own name (ध)
        ],
        "f0_edits": [[[0.50, 720], [0.62, 760], [0.80, 700]]]}   # hand-drawn strokes: [time s, cents above Sa]
     ]}

Per note:

* **pitch**: ``swara`` in sargam notation (``Sa Re Ga ma Pa Dha Ni``; komal ``re(k)``, tivra ``Ma(t)``; ``Sa'`` upper
  octave, ``.Ni`` lower), or ``midi`` (a number, fractions allowed), or ``hz``; plus ``cents`` (a shruti offset, e.g.
  Darbari's ati-komal ga = ``{"swara": "ga(k)", "cents": -20}``).
* **length**: ``beats`` (at ``tempo_bpm``) or ``sec``.
* **lyric**: ``syl`` (one Devanagari syllable; ``-`` slur; ``_`` rest; empty = the swara name). The phones come from the
  same G2P that built the dataset (``phones`` may also be given explicitly).
* **ornaments**: ``orn``, a list applied in order (``ORNAMENTS`` below lists every type, its parameters and defaults).

``f0_edits`` are free-hand strokes (time from the phrase start in seconds, pitch in cents above Sa); inside a stroke
they replace the computed curve. ``tonic: auto`` puts Sa on the singing voice's own tonic.

How the curve is built (``phrase_curve``)
-----------------------------------------
1. a step curve from the notes (cents above Sa; rests unvoiced);
2. joins between consecutive sung notes smoothed over ``smooth_ms`` (default 30 ms), as a voice never jumps instantly;
3. each note's ornaments, in order;
4. the hand-drawn strokes;
5. cents -> Hz (``Sa`` = the tonic).

Long phrases are cut at rests (or note boundaries) into pieces of at most ``MAX_PHRASE_S`` seconds, the clip lengths
SoulX was trained on; each piece is one SoulX call. ``follow_report`` tracks the pitch of the take (pYIN) and says how
closely the model followed the designed curve.
"""
from __future__ import annotations

import copy
import json
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sargam import (DEV, EDGE_REST, GAP_S, NAMES, SYL, SargamError, alankar_lines, name_of, parse_sargam,
                     swara_semitones, tonic_midi)

SR, FPS = 24000, 50
MAX_PHRASE_S = 12.0
F0_MIN, F0_MAX = 55.0, 1100.0
SWARA_RE = re.compile(r"^(\.*)([A-Za-z]+)(\((?:k|t)\))?('*)(\.*)$")


class ComposeError(ValueError):
    """A composition that cannot be read or sung; the message says why."""


# ------------------------------------------------------------------------------------- raags and taals
# Semitones above Sa allowed in each thaat / raag (used to snap pitches in the editor and to flag foreign notes).
THAATS = {
    "Bilawal": [0, 2, 4, 5, 7, 9, 11], "Kalyan": [0, 2, 4, 6, 7, 9, 11], "Khamaj": [0, 2, 4, 5, 7, 9, 10],
    "Kafi": [0, 2, 3, 5, 7, 9, 10], "Asavari": [0, 2, 3, 5, 7, 8, 10], "Bhairavi": [0, 1, 3, 5, 7, 8, 10],
    "Bhairav": [0, 1, 4, 5, 7, 8, 11], "Marwa": [0, 1, 4, 6, 7, 9, 11], "Purvi": [0, 1, 4, 6, 7, 8, 11],
    "Todi": [0, 1, 3, 6, 7, 8, 11],
}
RAAGS = {
    "Chromatic (all 12)": list(range(12)),
    "Yaman": THAATS["Kalyan"], "Bhoopali": [0, 2, 4, 7, 9], "Durga": [0, 2, 5, 7, 9], "Bhimpalasi": THAATS["Kafi"],
    "Desh": [0, 2, 4, 5, 7, 9, 10, 11], "Malkauns": [0, 3, 5, 8, 10], "Darbari Kanada": THAATS["Asavari"],
    "Bageshri": THAATS["Kafi"], "Bhairavi": THAATS["Bhairavi"], "Ahir Bhairav": [0, 1, 4, 5, 7, 9, 10],
    "Pahadi": [0, 2, 4, 5, 7, 9, 11], "Shivranjani": [0, 2, 3, 7, 9], "Hamsadhwani": [0, 2, 4, 7, 11],
    "Mohanam (Carnatic)": [0, 2, 4, 7, 9], "Mayamalavagowla (Carnatic)": THAATS["Bhairav"],
    "Kharaharapriya (Carnatic)": THAATS["Kafi"], "Shankarabharanam (Carnatic)": THAATS["Bilawal"],
    **{f"{k} thaat": v for k, v in THAATS.items()},
}
TAALS = {"teentaal": 16, "ektaal": 12, "jhaptaal": 10, "rupak": 7, "keherwa": 8, "dadra": 6, "adi (Carnatic)": 8,
         "none": 0}


def raag_scale(raag: Optional[str]) -> List[int]:
    """Allowed semitones (0-11) of a raag/thaat name (unknown or empty = all twelve)."""
    return RAAGS.get(raag or "", list(range(12)))


# ------------------------------------------------------------------------------------------- pitch
def parse_swara(text: str) -> int:
    """Sargam name (``Ga``, ``re(k)``, ``Ma(t)``, ``Sa'``, ``.Ni``) -> semitones above Sa."""
    m = SWARA_RE.match(str(text).strip())
    if not m:
        raise ComposeError(f"cannot read swara {text!r}; examples: Sa, Re, ga(k), Ma(t), Sa', .Ni")
    low1, name, mark, ups, low2 = m.groups()
    try:
        return swara_semitones(name, mark or "", len(ups) - len(low1) - len(low2))
    except SargamError as e:
        raise ComposeError(str(e)) from e


def note_cents(n: Dict, sa_midi: float) -> Optional[float]:
    """A note's pitch in cents above Sa (None for a rest)."""
    if n.get("syl") == "_" or str(n.get("swara", "")).lower() in ("rest", "_"):
        return None
    off = float(n.get("cents", 0) or 0)
    if n.get("hz") not in (None, ""):
        return 1200 * np.log2(float(n["hz"]) / 440) + 100 * (69 - sa_midi) + off
    if n.get("midi") not in (None, ""):
        return 100 * (float(n["midi"]) - sa_midi) + off
    if n.get("swara") not in (None, ""):
        return 100 * parse_swara(n["swara"]) + off
    raise ComposeError(f"note {n} has no pitch: give swara, midi or hz")


def cents_to_name(c: float) -> Tuple[str, float]:
    """Cents above Sa -> (nearest swara name, remaining cents)."""
    k = int(round(c / 100))
    return name_of(k), round(float(c - 100 * k), 1)


def neighbour(cents: float, scale: Sequence[int], step: int) -> float:
    """The scale swara ``step`` positions above (+) or below (-) ``cents`` (in cents above Sa)."""
    allowed = sorted({s + 12 * o for s in scale for o in range(-3, 4)})
    here = int(round(cents / 100))
    if step > 0:
        ups = [s for s in allowed if s > here]
        return 100.0 * ups[min(step, len(ups)) - 1]
    downs = [s for s in allowed if s < here][::-1]
    return 100.0 * downs[min(-step, len(downs)) - 1]


def pitch_arg(v, base: float, sa_midi: float, scale: Sequence[int]) -> float:
    """Read an ornament's pitch argument: a swara name, ``up``/``down`` (neighbour swara in the raag),
    ``up2``/``down2``, a number of cents relative to the note (``+150``), or ``{"midi": 62}``."""
    if isinstance(v, (int, float)):
        return base + float(v)
    if isinstance(v, dict):
        return note_cents(v, sa_midi)
    s = str(v).strip()
    m = re.match(r"^(up|down)(\d?)$", s)
    if m:
        return neighbour(base, scale, (1 if m.group(1) == "up" else -1) * int(m.group(2) or 1))
    if re.match(r"^[+-]?\d+(\.\d+)?$", s):
        return base + float(s)
    return 100.0 * parse_swara(s)


# ------------------------------------------------------------------------------------------- ornaments
def _ease(x: np.ndarray, shape: str) -> np.ndarray:
    """Glide shape on 0..1: linear, ease (S-curve), late (stays then drops), early (moves at once)."""
    x = np.clip(x, 0, 1)
    return {"linear": x, "ease": 0.5 - 0.5 * np.cos(np.pi * x), "late": x ** 2.5,
            "early": 1 - (1 - x) ** 2.5}.get(shape, 0.5 - 0.5 * np.cos(np.pi * x))


def _t(n: int) -> np.ndarray:
    """Frame times (s) of a note of ``n`` frames."""
    return np.arange(n) / FPS


def orn_meend(c, ctx, p):
    """Meend (Carnatic jaru): a slow glide from the note to ``to`` (default: the next note), starting at ``start``
    (fraction of the note) and arriving at the end. ``shape``: ease | linear | late | early."""
    n = len(c)
    to = pitch_arg(p["to"], ctx["base"], ctx["sa"], ctx["scale"]) if p.get("to") is not None else ctx["next"]
    if to is None:
        return c
    s = int(round(float(p.get("start", 0.4)) * n))
    x = (np.arange(n) - s) / max(n - s, 1)
    return c + np.where(np.arange(n) >= s, _ease(x, p.get("shape", "ease")) * (to - ctx["base"]), 0.0)


def orn_glide_in(c, ctx, p):
    """Slide into the note from ``from`` (swara, up/down, or cents relative; default -100) over ``time`` s."""
    fr = pitch_arg(p.get("from", -100), ctx["base"], ctx["sa"], ctx["scale"])
    k = min(len(c), max(1, int(round(float(p.get("time", 0.12)) * FPS))))
    out = c.copy()
    out[:k] += (fr - ctx["base"]) * (1 - _ease(np.arange(k) / k, p.get("shape", "ease")))
    return out


def orn_glide_out(c, ctx, p):
    """Slide away at the end of the note to ``to`` (default -200 c, a falling release) over ``time`` s."""
    to = pitch_arg(p.get("to", -200), ctx["base"], ctx["sa"], ctx["scale"])
    k = min(len(c), max(1, int(round(float(p.get("time", 0.15)) * FPS))))
    out = c.copy()
    out[-k:] += (to - ctx["base"]) * _ease(np.arange(1, k + 1) / k, p.get("shape", "ease"))
    return out


def orn_vibrato(c, ctx, p):
    """Regular vibrato: ``rate`` Hz, ``depth`` cents (half the swing), after ``delay`` s, faded in over ``fade`` s."""
    t = _t(len(c))
    d0, fade = float(p.get("delay", 0.15)), max(float(p.get("fade", 0.2)), 1e-3)
    env = np.clip((t - d0) / fade, 0, 1)
    return c + float(p.get("depth", 30)) * env * np.sin(2 * np.pi * float(p.get("rate", 5.5)) * (t - d0))


def orn_andolan(c, ctx, p):
    """Andolan: a slow, wide sway (``rate`` ~1-2 Hz, ``depth`` ~30-50 c). ``side``: around | below | above
    (komal Re and Dha in Bhairav sway below the note)."""
    t = _t(len(c))
    ph = 2 * np.pi * float(p.get("rate", 1.5)) * t
    d = float(p.get("depth", 40))
    side = p.get("side", "below")
    w = {"around": np.sin(ph), "below": -(1 - np.cos(ph)) / 2, "above": (1 - np.cos(ph)) / 2}.get(side, np.sin(ph))
    return c + d * w


def orn_gamak(c, ctx, p):
    """Gamak (Carnatic kampita): a heavy oscillation between the note and ``to`` (default the upper neighbour in the
    raag) at ``rate`` Hz (4-8), for ``cycles`` (default: the whole note), always returning to the note."""
    n = len(c)
    to = pitch_arg(p.get("to", "up"), ctx["base"], ctx["sa"], ctx["scale"])
    rate = float(p.get("rate", 6.0))
    t = _t(n)
    span = n / FPS if p.get("cycles") in (None, "", 0) else float(p["cycles"]) / rate
    w = np.where(t < span, (1 - np.cos(2 * np.pi * rate * t)) / 2, 0.0)
    return c + w * (to - ctx["base"])


def _cluster(c, ctx, p, default: Sequence, speed: float, at: str = "start"):
    """Shared body of kan/murki/khatka/zamzama: a run of quick touches (``swaras``) then the note itself."""
    seq = p.get("swaras", default)
    pts = [pitch_arg(s, ctx["base"], ctx["sa"], ctx["scale"]) for s in seq]
    k = max(1, int(round(float(p.get("speed", speed)) * FPS)))
    out = c.copy()
    n = len(c)
    total = min(n - 1, k * len(pts))
    if total <= 0:
        return out
    off = 0 if p.get("at", at) == "start" else n - total
    for i, v in enumerate(pts):
        a, b = off + i * k, min(off + (i + 1) * k, off + total)
        if a >= b:
            break
        out[a:b] = v + (c[a:b] - ctx["base"])
    return out


def orn_kan(c, ctx, p):
    """Kan swar (Carnatic sphurita/nokku): a grace-note touch of ``from`` (default the upper neighbour) for ``time``
    s (40-100 ms) at the start of the note."""
    return _cluster(c, ctx, {"swaras": [p.get("from", "up")], "speed": p.get("time", 0.06)}, [], 0.06)


def orn_murki(c, ctx, p):
    """Murki: a quick, light turn (default up, note, down, note), ``speed`` s per touch, at the ``start`` or ``end``."""
    return _cluster(c, ctx, p, ["up", 0, "down", 0], 0.08)


def orn_khatka(c, ctx, p):
    """Khatka: a forceful cluster (default up2, up, 0, down) sung fast (``speed`` 0.06 s per touch)."""
    return _cluster(c, ctx, p, ["up2", "up", 0, "down"], 0.06)


def orn_zamzama(c, ctx, p):
    """Zamzama: a repeated fast run (default up, 0, up, 0, down, 0), ``speed`` 0.07 s per touch."""
    return _cluster(c, ctx, p, ["up", 0, "up", 0, "down", 0], 0.07)


def orn_custom(c, ctx, p):
    """Any shape: ``points`` = [[fraction of the note 0..1, cents relative to the note], ...], interpolated and
    added to the note (used to keep a real singer's curve when a tune is imported)."""
    pts = sorted(p.get("points", []), key=lambda x: x[0])
    if not pts:
        return c
    x = np.linspace(0, 1, len(c), endpoint=False) + 0.5 / max(len(c), 1)
    return c + np.interp(x, [q[0] for q in pts], [q[1] for q in pts])


ORNAMENTS = {
    "meend": (orn_meend, {"to": None, "start": 0.4, "shape": "ease"}, "glide to another swara (default the next note)"),
    "andolan": (orn_andolan, {"rate": 1.5, "depth": 40, "side": "below"}, "slow wide sway"),
    "gamak": (orn_gamak, {"to": "up", "rate": 6.0, "cycles": 0}, "heavy oscillation to a neighbour swara"),
    "kan": (orn_kan, {"from": "up", "time": 0.06}, "grace note touched before the note"),
    "murki": (orn_murki, {"swaras": ["up", 0, "down", 0], "speed": 0.08, "at": "start"}, "quick light turn"),
    "khatka": (orn_khatka, {"swaras": ["up2", "up", 0, "down"], "speed": 0.06, "at": "start"}, "forceful cluster"),
    "zamzama": (orn_zamzama, {"swaras": ["up", 0, "up", 0, "down", 0], "speed": 0.07, "at": "start"},
                "repeated fast run"),
    "vibrato": (orn_vibrato, {"rate": 5.5, "depth": 30, "delay": 0.15, "fade": 0.2}, "regular wobble"),
    "glide_in": (orn_glide_in, {"from": -100, "time": 0.12, "shape": "ease"}, "slide into the note"),
    "glide_out": (orn_glide_out, {"to": -200, "time": 0.15, "shape": "ease"}, "slide off at the end"),
    "custom": (orn_custom, {"points": [[0, 0], [1, 0]]}, "any shape: [[fraction, cents], ...]"),
}
ALIASES = {"jaru": "meend", "kampita": "gamak", "sphurita": "kan", "nokku": "kan", "portamento": "glide_in",
           "slide": "meend", "glide": "meend", "shake": "gamak", "mi": "meend", "meed": "meend"}


def ornament_doc() -> str:
    """One line per ornament type with its parameters and defaults (for help texts and the LLM prompt)."""
    rows = [f"- {k}: {d}. params {json.dumps(v, ensure_ascii=False)}" for k, (_, v, d) in ORNAMENTS.items()]
    rows.append("- aliases: " + ", ".join(f"{a} = {b}" for a, b in ALIASES.items()))
    return "\n".join(rows)


def apply_ornament(c: np.ndarray, ctx: Dict, o: Dict) -> np.ndarray:
    """Apply one ornament dict ``{"type": ..., params}`` to a note's cents curve."""
    typ = ALIASES.get(str(o.get("type", "")).lower(), str(o.get("type", "")).lower())
    if typ not in ORNAMENTS:
        raise ComposeError(f"unknown ornament {o.get('type')!r}; known: {', '.join(ORNAMENTS)} "
                           f"(also {', '.join(ALIASES)})")
    fn, defaults, _ = ORNAMENTS[typ]
    return fn(c, ctx, {**defaults, **{k: v for k, v in o.items() if k != "type"}})


# ------------------------------------------------------------------------------------- normalising
def normalise(comp: Dict | str) -> Dict:
    """Read a composition (dict or JSON text), check it, fill defaults. Accepts ``notes`` at the top level for a
    single phrase. Raises ComposeError with a readable message."""
    if isinstance(comp, str):
        try:
            comp = json.loads(comp)
        except json.JSONDecodeError as e:
            raise ComposeError(f"not valid JSON: {e}") from e
    if not isinstance(comp, dict):
        raise ComposeError("a composition is a JSON object with a \"phrases\" list")
    comp = copy.deepcopy(comp)
    if "phrases" not in comp and "notes" in comp:
        comp["phrases"] = [{"notes": comp.pop("notes"), "f0_edits": comp.pop("f0_edits", [])}]
    if not comp.get("phrases"):
        raise ComposeError('the composition needs a "phrases" list, each with a "notes" list')
    comp.setdefault("tonic", "auto")
    comp["tempo_bpm"] = _number(comp.get("tempo_bpm") or 80, "tempo_bpm")
    if not 5 <= comp["tempo_bpm"] <= 600:
        raise ComposeError(f"tempo {comp['tempo_bpm']:g} bpm: use 5-600 beats per minute")
    comp.setdefault("raag", "Chromatic (all 12)")
    comp.setdefault("taal", "none")
    comp["smooth_ms"] = _number(30 if comp.get("smooth_ms") in (None, "") else comp["smooth_ms"], "smooth_ms")
    comp["phrase_gap_s"] = _number(0.3 if comp.get("phrase_gap_s") in (None, "") else comp["phrase_gap_s"],
                                   "phrase_gap_s")
    if comp["smooth_ms"] < 0 or comp["phrase_gap_s"] < 0:
        raise ComposeError("smooth_ms and phrase_gap_s cannot be negative")
    if not isinstance(comp["phrases"], list):
        raise ComposeError('"phrases" must be a list')
    for i, ph in enumerate(comp["phrases"]):
        if isinstance(ph, list):
            ph = comp["phrases"][i] = {"notes": ph}
        ph.setdefault("f0_edits", [])
        if not ph.get("notes"):
            raise ComposeError(f"phrase {i + 1} has no notes")
        for j, n in enumerate(ph["notes"]):
            n["syl"] = str(n.get("syl", "") if n.get("syl") is not None else "").strip()
            if str(n.get("swara", "")).strip().lower() in ("rest", "_"):
                n["syl"] = "_"
            if n["syl"] == "-" and (j == 0 or ph["notes"][j - 1]["syl"] == "_"):
                raise ComposeError(f"phrase {i + 1} note {j + 1}: '-' (hold) needs a sung note before it")
            where = f"phrase {i + 1} note {j + 1}"
            if n.get("sec") in (None, ""):
                n.pop("sec", None)
                n["beats"] = _number(1 if n.get("beats") in (None, "") else n["beats"], f"{where}: beats")
            else:
                n["sec"] = _number(n["sec"], f"{where}: sec")
            if float(n.get("sec", n.get("beats", 1))) <= 0:
                raise ComposeError(f"{where}: length must be positive")
            if n.get("cents") not in (None, ""):
                n["cents"] = _number(n["cents"], f"{where}: cents")
            n.setdefault("orn", [])
            if isinstance(n["orn"], dict):
                n["orn"] = [n["orn"]]
            if n["syl"] != "_":
                try:
                    note_cents(n, 60.0)              # raises if the pitch is unreadable
                except (ValueError, TypeError, KeyError) as e:
                    raise ComposeError(f"{where}: {e}") from e
    return comp


def _number(v, what: str) -> float:
    """``float(v)`` or a readable ComposeError."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ComposeError(f"{what}: {v!r} is not a number") from None
    if not np.isfinite(x):
        raise ComposeError(f"{what}: {v!r} is not a finite number")
    return x


def note_seconds(n: Dict, tempo_bpm: float) -> float:
    """Length of a note in seconds (``sec``, else ``beats`` at the tempo)."""
    return float(n["sec"]) if n.get("sec") not in (None, "") else float(n.get("beats", 1)) * 60.0 / tempo_bpm


def phrase_frames(notes: Sequence[Dict], tempo_bpm: float) -> List[int]:
    """Frames per note, rounded on the running total so lengths never drift."""
    ends = np.round(np.cumsum([note_seconds(n, tempo_bpm) for n in notes]) * FPS).astype(int)
    fr = np.diff(np.concatenate([[0], ends]))
    return [max(1, int(f)) for f in fr]


# ------------------------------------------------------------------------------------------- the curve
def phrase_curve(ph: Dict, sa_midi: float, comp: Dict) -> Tuple[np.ndarray, List[int], List[Optional[float]]]:
    """Cents-above-Sa curve of one phrase (NaN = unvoiced), the frames per note, and each note's base pitch."""
    notes = ph["notes"]
    frames = phrase_frames(notes, comp["tempo_bpm"])
    bases = [note_cents(n, sa_midi) for n in notes]
    scale = raag_scale(comp.get("raag"))
    curve = np.full(sum(frames), np.nan)
    starts = np.concatenate([[0], np.cumsum(frames)])[:-1].astype(int)
    for i, (n, f, b) in enumerate(zip(notes, frames, bases)):
        if b is not None:
            curve[starts[i]:starts[i] + f] = b
    # smooth the joins between consecutive sung notes: a ramp over ``smooth_ms`` centred on the join (at least one
    # frame each side). Joins next to an ornament are left to the ornament, which shapes its own transition (a meend
    # already arrives at the next note; a kan or glide starts the note on purpose), so nothing is added twice.
    k = max(1, int(round(comp["smooth_ms"] / 1000 * FPS / 2))) if comp["smooth_ms"] > 0 else 0
    if k > 0:
        for i in range(1, len(notes)):
            if bases[i] is None or bases[i - 1] is None or notes[i].get("orn") or notes[i - 1].get("orn"):
                continue
            s = starts[i]
            a, z = max(starts[i - 1], s - k), min(s + k, s + frames[i])
            if z - a >= 2:
                curve[a:z] = np.linspace(bases[i - 1], bases[i], z - a + 2)[1:-1]
    # ornaments
    for i, (n, f, b) in enumerate(zip(notes, frames, bases)):
        if b is None or not n.get("orn"):
            continue
        nxt = bases[i + 1] if i + 1 < len(notes) else None
        prv = bases[i - 1] if i > 0 else None
        ctx = {"base": b, "next": nxt, "prev": prv, "sa": sa_midi, "scale": scale, "seconds": f / FPS}
        seg = curve[starts[i]:starts[i] + f].copy()
        for o in n["orn"]:
            seg = apply_ornament(seg, ctx, o)
        curve[starts[i]:starts[i] + f] = seg
    # hand-drawn strokes override
    t = np.arange(len(curve)) / FPS
    for stroke in ph.get("f0_edits") or []:
        pts = sorted((float(a), float(c)) for a, c in stroke if c is not None)
        if len(pts) < 2:
            continue
        sel = (t >= pts[0][0]) & (t <= pts[-1][0]) & ~np.isnan(curve)
        curve[sel] = np.interp(t[sel], [p[0] for p in pts], [p[1] for p in pts])
    return curve, frames, bases


def cents_to_hz(c: np.ndarray, sa_midi: float) -> np.ndarray:
    """Cents above Sa -> Hz (NaN -> 0 = unvoiced), clipped to a singable range."""
    hz = 440.0 * 2 ** ((sa_midi - 69) / 12 + np.nan_to_num(c, nan=0) / 1200)
    return np.where(np.isnan(c), 0.0, np.clip(hz, F0_MIN, F0_MAX))


# ------------------------------------------------------------------------------------------- lyrics -> phones
def syllable_phones(syl: str, swara: Optional[str], g2p=None) -> Tuple[str, List[str]]:
    """(display text, phones) for one sung note. Empty ``syl`` sings the swara name; Devanagari goes through ``g2p``
    (a function text -> list of {text, phones}); a lone consonant gets its inherent 'a' back (schwa deletion only
    makes sense inside words)."""
    if not syl:
        name = re.sub(r"[.'()kt]", "", str(swara or "sa")).lower() or "sa"
        name = name if name in SYL else "sa"
        return DEV[name], list(SYL[name])
    if g2p is None:
        raise ComposeError("lyrics need the singer's G2P")
    parts = g2p(syl)
    phones = [p for s in parts for p in s["phones"]]
    vowels = {"a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo", "E", "O", "ae", "ai", "au"}
    if phones and not any(p in vowels for p in phones):
        phones.append("a")
    return syl, phones


def phrase_tokens(notes: Sequence[Dict], g2p=None) -> Tuple[List[str], List[str]]:
    """SoulX tokens and texts of a phrase's notes: rests -> <SP>, '-' repeats the previous syllable (a slur)."""
    tokens, texts, last = [], [], None
    for n in notes:
        if n["syl"] == "_":
            tokens.append("<SP>"); texts.append("_"); last = None
        elif n["syl"] == "-":
            if last is None:
                raise ComposeError("'-' (hold) needs a sung note before it")
            tokens.append(last[0]); texts.append(last[1])
        else:
            phones = n.get("phones") or syllable_phones(n["syl"], n.get("swara"), g2p)[1]
            text = n["syl"] or syllable_phones("", n.get("swara"))[0]
            last = ("in_" + "-".join(phones), text)
            tokens.append(last[0]); texts.append(text)
    return tokens, texts


def note_pitches(notes: Sequence[Dict], idx: Sequence[int], curve: np.ndarray, starts: Sequence[int],
                 bases: Sequence[Optional[float]], sa_midi: float) -> Tuple[List[int], List[int]]:
    """(MIDI note pitch, note type) for notes ``idx``: the median of the curve over the middle half of each note."""
    pitch, typ = [], []
    for i in idx:
        if notes[i]["syl"] == "_":
            pitch.append(0); typ.append(1)
            continue
        c = curve[starts[i]:starts[i + 1]]
        mid = c[len(c) // 4: max(len(c) // 4 + 1, 3 * len(c) // 4)]
        v = np.nanmedian(mid) if np.isfinite(mid).any() else bases[i]
        pitch.append(int(round(sa_midi + v / 100)))
        typ.append(3 if notes[i]["syl"] == "-" else 2)
    return pitch, typ


def compose_timeline(comp: Dict, sa_midi: float, g2p=None):
    """The whole composition on one ``seamless.Timeline`` (melody mode): phrases joined by a ``phrase_gap_s`` rest.
    Returns (timeline, target cents over the whole timeline (NaN = unvoiced), start frame of every phrase)."""
    from .seamless import Timeline
    comp = normalise(comp)
    tl = Timeline(melody=True)
    edge = int(round(EDGE_REST * FPS))
    tl.append_rest(EDGE_REST)
    cents_parts, phrase_starts = [np.full(edge, np.nan)], []
    for pi, ph in enumerate(comp["phrases"]):
        if pi:
            g0 = tl.total
            tl.append_rest(comp["phrase_gap_s"])
            cents_parts.append(np.full(tl.total - g0, np.nan))
        curve, frames, bases = phrase_curve(ph, sa_midi, comp)
        notes = ph["notes"]
        tokens, texts = phrase_tokens(notes, g2p)
        starts = np.concatenate([[0], np.cumsum(frames)]).astype(int)
        pitch, typ = note_pitches(notes, range(len(notes)), curve, starts, bases, sa_midi)
        it = {"phoneme": " ".join(tokens), "text": " ".join(texts), "note_pitch": " ".join(map(str, pitch)),
              "note_type": " ".join(map(str, typ)), "duration": " ".join(f"{f / FPS:.6f}" for f in frames)}
        hz = cents_to_hz(curve, sa_midi)
        phrase_starts.append(tl.total)
        tl.append_item(it, hz)
        cents_parts.append(np.where(hz > 0, curve, np.nan))
    g0 = tl.total
    tl.finish()
    cents_parts.append(np.full(tl.total - g0, np.nan))
    cents = np.concatenate(cents_parts)
    assert len(cents) == tl.total
    return tl, cents, phrase_starts


# ------------------------------------------------------------------------------------------- render plan
def plan(comp: Dict, sa_midi: float, g2p=None, max_s: float = MAX_PHRASE_S) -> List[Dict]:
    """Composition -> SoulX items (one per piece of at most ``max_s`` s) with their target curves.

    Returns [{item, cents (np.ndarray, with the edge rests), phrase, start_note}]."""
    comp = normalise(comp)
    out = []
    edge = int(round(EDGE_REST * FPS))
    for pi, ph in enumerate(comp["phrases"]):
        curve, frames, bases = phrase_curve(ph, sa_midi, comp)
        notes = ph["notes"]
        tokens, texts = phrase_tokens(notes, g2p)
        # cut into pieces <= max_s, preferring rests, never inside a slur
        pieces, cur = [], []
        for i in range(len(frames)):
            cur.append(i)
            if sum(frames[j] for j in cur) > max_s * FPS and len(cur) > 1:
                # cut before the last note that starts a syllable (or a rest), so a slur stays in one piece
                cut = next((k for k in range(len(cur) - 1, 0, -1) if notes[cur[k]]["syl"] != "-"), None)
                if cut is not None:
                    pieces.append(cur[:cut]); cur = cur[cut:]
        pieces.append(cur)
        starts = np.concatenate([[0], np.cumsum(frames)]).astype(int)
        for idx in pieces:
            # drop rests at the piece edges (edge rests are added below)
            while idx and notes[idx[0]]["syl"] == "_":
                idx = idx[1:]
            while idx and notes[idx[-1]]["syl"] == "_":
                idx = idx[:-1]
            if not idx:
                continue
            seg = curve[starts[idx[0]]:starts[idx[-1] + 1]]
            pitch, typ = note_pitches(notes, idx, curve, starts, bases, sa_midi)
            fr = [edge] + [frames[i] for i in idx] + [edge]
            cents = np.concatenate([np.full(edge, np.nan), seg, np.full(edge, np.nan)])
            hz = cents_to_hz(cents, sa_midi)
            item = {"index": f"compose_p{pi + 1}", "language": "Hindi",
                    "text": " ".join(["_"] + [texts[i] for i in idx] + ["_"]),
                    "phoneme": " ".join(["<SP>"] + [tokens[i] for i in idx] + ["<SP>"]),
                    "note_pitch": " ".join(map(str, [0] + pitch + [0])),
                    "note_type": " ".join(map(str, [1] + typ + [1])),
                    "duration": " ".join(f"{f / FPS:.2f}" for f in fr), "n_frames": int(sum(fr)),
                    "f0": " ".join(f"{x:.1f}" for x in hz)}
            out.append({"item": item, "cents": cents, "phrase": pi, "start_note": idx[0],
                        "t0": starts[idx[0]] / FPS - EDGE_REST})
    return out


# ------------------------------------------------------------------------------------------- view (for the editor)
def view(comp: Dict | str, sa_midi: float) -> Dict:
    """What the editor draws: per phrase the note boxes (start, length, base cents, label, ornaments) and the
    computed curve (cents, 50 fps, null = unvoiced), plus the tonic and the raag's swaras."""
    comp = normalise(comp)
    phrases = []
    for ph in comp["phrases"]:
        curve, frames, bases = phrase_curve(ph, sa_midi, comp)
        t0, boxes = 0, []
        for n, f, b in zip(ph["notes"], frames, bases):
            boxes.append({"t": t0 / FPS, "d": f / FPS, "c": b, "syl": n["syl"],
                          "label": "_" if b is None else (n["syl"] or syllable_phones("", n.get("swara"))[0]),
                          "swara": None if b is None else cents_to_name(b)[0],
                          "orn": [ALIASES.get(o.get("type", ""), o.get("type", "")) for o in n.get("orn", [])]})
            t0 += f
        phrases.append({"notes": boxes, "curve": [None if np.isnan(x) else round(float(x), 1) for x in curve],
                        "seconds": t0 / FPS})
    return {"sa_midi": sa_midi, "sa_hz": round(440 * 2 ** ((sa_midi - 69) / 12), 2), "scale": raag_scale(comp["raag"]),
            "names": NAMES, "tempo_bpm": comp["tempo_bpm"], "taal": TAALS.get(comp.get("taal", "none"), 0),
            "ornaments": {k: {"defaults": v[1], "doc": v[2]} for k, v in ORNAMENTS.items()}, "phrases": phrases}


# ------------------------------------------------------------------------------------------- starting points
def _item_notes(item: Dict, sa_midi: float, keep_curve: bool) -> Tuple[List[Dict], float]:
    """Notes of a dataset/tune item, pitches as swaras relative to ``sa_midi``; with ``keep_curve`` each note gets a
    ``custom`` ornament holding the real singer's F0 around it (so moving the note moves its ornaments)."""
    from_tok = lambda t: t[3:].split("-") if t.startswith("in_") else None
    toks = item["phoneme"].split()
    texts = item["text"].split()
    pitch = [int(x) for x in item["note_pitch"].split()]
    types = [int(x) for x in item["note_type"].split()]
    durs = [float(x) for x in item["duration"].split()]
    f0 = np.array([float(x) for x in item["f0"].split()]) if keep_curve else None
    notes, pos = [], 0
    for tok, tx, p, ty, d in zip(toks, texts, pitch, types, durs):
        nf = int(round(d * FPS))
        if ty == 1:
            notes.append({"syl": "_", "sec": round(d, 2)})
        else:
            c = 100 * (p - sa_midi)
            name, rest = cents_to_name(c)
            n = {"syl": "-" if ty == 3 else tx, "swara": name, "sec": round(d, 2), "orn": []}
            if rest:
                n["cents"] = rest
            if ty != 3 and from_tok(tok):
                n["phones"] = from_tok(tok)
            if keep_curve and nf >= 2:
                seg = f0[pos:pos + nf]
                if (seg > 0).sum() >= 2:
                    cents = 1200 * np.log2(np.where(seg > 0, seg, np.nan) / 440) + 100 * (69 - sa_midi) - c
                    idx = np.where(np.isfinite(cents))[0]
                    step = max(1, len(seg) // 24)               # at most ~24 points per note, enough for vibrato
                    sel = idx[::step] if len(idx) else idx
                    pts = [[round(float((j + 0.5) / nf), 3), round(float(cents[j]), 1)] for j in sel]
                    n["orn"] = [{"type": "custom", "points": pts}]
            notes.append(n)
        pos += nf
    return notes, sa_midi


def from_tune(tune: Dict, keep_curve: bool = True) -> Dict:
    """A bundled tune as a composition, with its real singer's pitch curve (as ``custom`` ornaments) and syllables."""
    it = tune["item"]
    sa = 69 + 12 * np.log2(float(it.get("tonic_hz") or 261.63) / 440)
    notes, _ = _item_notes(it, sa, keep_curve)
    return {"title": f"tune {tune['no']}: {tune['song']} ({tune['year']})", "voice": tune.get("voice", 1),
            "tonic": round(float(it.get("tonic_hz") or 261.63), 2), "tempo_bpm": 60, "raag": "Chromatic (all 12)",
            "taal": "none", "smooth_ms": 0 if keep_curve else 30, "phrases": [{"notes": notes, "f0_edits": []}]}


def from_lyrics(lines: Sequence[Sequence[Dict]], swaras: str = "", beats: float = 1.0) -> Dict:
    """Lyrics (already split into syllables {text, phones}, one list per line) on swaras given as sargam text
    (one swara per syllable, the last one repeated; empty = all on Sa). Each line is one phrase."""
    seq = [n for ln in parse_sargam(swaras) for n in ln if n["semis"] is not None] if swaras.strip() else []
    phrases, k = [], 0
    for sylls in lines:
        notes = []
        for s in sylls:
            sw = name_of(seq[min(k, len(seq) - 1)]["semis"]) if seq else "Sa"
            b = seq[min(k, len(seq) - 1)]["beats"] * beats if seq else beats
            notes.append({"syl": s["text"], "phones": list(s["phones"]), "swara": sw, "beats": b, "orn": []})
            k += 1
        phrases.append({"notes": notes, "f0_edits": []})
    return {"title": "my song", "tonic": "auto", "tempo_bpm": 80, "raag": "Chromatic (all 12)", "taal": "none",
            "phrases": phrases}


def from_sargam(text: str = "", alankar: Optional[int] = None) -> Dict:
    """Sargam text (or built-in alankar ``alankar``) as a composition that sings the swara names."""
    lines = alankar_lines(alankar) if alankar else parse_sargam(text)
    phrases = [{"notes": [{"syl": "_", "beats": n["beats"]} if n["semis"] is None else
                          {"syl": "", "swara": name_of(n["semis"]), "beats": n["beats"], "orn": []} for n in ln],
                "f0_edits": []} for ln in lines]
    return {"title": f"alankar {alankar}" if alankar else "sargam", "tonic": "auto", "tempo_bpm": 120,
            "raag": "Bilawal thaat", "taal": "teentaal", "phrases": phrases}


EXAMPLE = {
    "title": "Yaman phrase with ornaments", "voice": 4, "tonic": "auto", "tempo_bpm": 70, "raag": "Yaman",
    "taal": "teentaal",
    "phrases": [{"notes": [
        {"syl": "", "swara": ".Ni", "beats": 1, "orn": [{"type": "kan", "from": "Re"}]},
        {"syl": "", "swara": "Re", "beats": 1},
        {"syl": "", "swara": "Ga", "beats": 2, "orn": [{"type": "meend", "to": "Ma(t)", "start": 0.6}]},
        {"syl": "-", "swara": "Ma(t)", "beats": 1},
        {"syl": "", "swara": "Pa", "beats": 2, "orn": [{"type": "vibrato", "rate": 5.5, "depth": 30}]},
        {"syl": "_", "beats": 1},
        {"syl": "", "swara": "Dha", "beats": 1, "orn": [{"type": "murki"}]},
        {"syl": "", "swara": "Pa", "beats": 1, "orn": [{"type": "gamak", "to": "Dha", "rate": 5}]},
        {"syl": "", "swara": "Ga", "beats": 2, "orn": [{"type": "andolan", "depth": 30, "side": "around"}]},
    ], "f0_edits": []}]}


def set_phrase_lyrics(comp: Dict, phrase: int, sylls: Sequence[Dict]) -> Dict:
    """Put a typed lyric line (syllables ``[{text, phones}]``, from the G2P) on phrase ``phrase``: one syllable per
    sung note in order (rests and holds are skipped). More syllables than notes: new notes are added after the last
    sung note (same swara and length). Fewer: the remaining sung notes hold the previous syllable (``-``), or become
    rests if they follow a rest."""
    comp = normalise(comp)
    if not 0 <= phrase < len(comp["phrases"]):
        raise ComposeError(f"there is no phrase {phrase + 1}")
    if not sylls:
        raise ComposeError("no syllables in the lyrics")
    notes = comp["phrases"][phrase]["notes"]
    slots = [i for i, n in enumerate(notes) if n["syl"] not in ("_", "-")]
    if not slots:
        raise ComposeError("this phrase has no sung notes to put lyrics on")
    for k, i in enumerate(slots):
        if k < len(sylls):
            notes[i]["syl"] = sylls[k]["text"]
            notes[i]["phones"] = list(sylls[k]["phones"])
        else:
            notes[i].pop("phones", None)
            if i > 0 and notes[i - 1]["syl"] != "_":
                notes[i]["syl"] = "-"
            else:
                notes[i]["syl"], notes[i]["orn"] = "_", []
                notes[i].pop("swara", None)
    last = slots[-1]
    while last + 1 < len(notes) and notes[last + 1]["syl"] == "-":
        last += 1
    tmpl = notes[slots[-1]]
    extra = [{"syl": s["text"], "phones": list(s["phones"]), "swara": tmpl.get("swara", "Sa"),
              **({"sec": tmpl["sec"]} if "sec" in tmpl else {"beats": tmpl.get("beats", 1)}), "orn": []}
             for s in sylls[len(slots):]]
    notes[last + 1:last + 1] = extra
    return normalise(comp)


# ------------------------------------------------------------------------------------------- table view
TABLE_COLS = ["#", "phrase", "syllable", "swara", "cents", "beats", "sec", "ornaments"]
# "#" is the note's id (its position when the table was made): it links a row back to its note, so data the table does
# not show (a real singer's curve, G2P phones, a free midi/hz pitch) stays with the right note when rows are added,
# deleted or moved. New rows leave it empty.
def orn_to_text(orn: Sequence[Dict]) -> str:
    """Ornament list -> compact text: ``meend(to=Pa,start=0.5); vibrato(rate=5)`` (custom shapes shown as ``custom``)."""
    out = []
    for o in orn:
        if o.get("type") == "custom":
            out.append("custom")
            continue
        args = ",".join(f"{k}={json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v}"
                        for k, v in o.items() if k != "type")
        out.append(f"{o.get('type')}({args})" if args else str(o.get("type")))
    return "; ".join(out)


def orn_from_text(text: str, old: Sequence[Dict] = ()) -> List[Dict]:
    """Inverse of ``orn_to_text`` (``custom`` keeps the note's existing custom shape)."""
    out = []
    olds = [o for o in old if o.get("type") == "custom"]
    for part in [p.strip() for p in str(text or "").split(";") if p.strip()]:
        m = re.match(r"^([A-Za-z_]+)\s*(?:\((.*)\))?$", part)
        if not m:
            raise ComposeError(f"cannot read ornament {part!r}; e.g. meend(to=Pa,start=0.5)")
        if m.group(1) == "custom" and not m.group(2):
            out += olds[:1]
            continue
        o = {"type": m.group(1)}
        for kv in re.findall(r"(\w+)\s*=\s*(\[[^\]]*\]|[^,]+)", m.group(2) or ""):
            k, v = kv[0], kv[1].strip()
            try:
                o[k] = json.loads(v)
            except json.JSONDecodeError:
                o[k] = v
        out.append(o)
    return out


def _swara_cell(n: Dict, sa_midi: float) -> Tuple[str, float]:
    """(swara name, cents offset) shown for a note: its own ``swara`` and ``cents``, or a free ``midi``/``hz`` pitch
    named against the real tonic."""
    if n.get("swara"):
        return str(n["swara"]), float(n.get("cents", 0) or 0)
    name, rest = cents_to_name(note_cents(n, sa_midi))
    return name, round(float(rest), 1)


def to_table(comp: Dict, sa_midi: float = 60.0) -> List[List]:
    """Composition -> rows for the table view (``sa_midi``: the tonic, to name free midi/hz pitches)."""
    comp = normalise(comp)
    rows, k = [], 0
    for i, ph in enumerate(comp["phrases"]):
        for n in ph["notes"]:
            k += 1
            sw, ce = ("", 0) if n["syl"] == "_" else _swara_cell(n, sa_midi)
            rows.append([k, i + 1, n["syl"], sw, ce, n.get("beats", ""), n.get("sec", ""),
                         orn_to_text(n.get("orn", []))])
    return rows


def from_table(rows: Sequence[Sequence], comp: Dict, sa_midi: float = 60.0) -> Dict:
    """Rows of the table view back into ``comp`` (settings and hand-drawn strokes are kept). A row whose "#" names
    an original note keeps that note's hidden data (phones, custom curve, a free pitch whose cell was not changed);
    a duration typed in "beats" wins over an unchanged "sec"."""
    comp = normalise(comp)
    old = [n for ph in comp["phrases"] for n in ph["notes"]]
    blank = lambda x: str(x).strip() in ("", "nan", "None")
    num = lambda x, what: _number(x, f"table row {what}")
    phrases: Dict[int, List[Dict]] = {}
    for k, r in enumerate(rows):
        r = list(r) + [""] * (len(TABLE_COLS) - len(r))
        if all(blank(x) for x in r):
            continue
        rid, ph, syl, sw, ce, beats, sec, orn = r[:8]
        src = None
        if not blank(rid):
            j = int(num(rid, k + 1)) - 1
            src = old[j] if 0 <= j < len(old) else None
        p = int(num(ph, k + 1)) if not blank(ph) else 1
        n = {"syl": "" if blank(syl) else str(syl).strip()}
        if n["syl"] != "_":
            sw = str(sw).strip() if not blank(sw) else "Sa"
            ce = 0.0 if blank(ce) else num(ce, k + 1)
            if src is not None and src["syl"] != "_" and (sw, ce) == _swara_cell(src, sa_midi):
                n.update({f: src[f] for f in ("swara", "cents", "midi", "hz") if f in src})   # unchanged pitch
            else:
                n["swara"] = sw
                if ce:
                    n["cents"] = ce
        b_new = None if blank(beats) else num(beats, k + 1)
        s_new = None if blank(sec) else num(sec, k + 1)
        b_old = src.get("beats") if src is not None else None
        if b_new is not None and (s_new is None or b_new != b_old):
            n["beats"] = b_new                     # typed (or changed) beats win
        elif s_new is not None:
            n["sec"] = s_new
        else:
            n["beats"] = 1.0
        n["orn"] = orn_from_text(orn, src.get("orn", []) if src is not None else [])
        if src is not None and src.get("phones") and src["syl"] == n["syl"]:
            n["phones"] = src["phones"]
        phrases.setdefault(p, []).append(n)
    if not phrases:
        raise ComposeError("the table is empty")
    edits = {i + 1: ph.get("f0_edits", []) for i, ph in enumerate(comp["phrases"])}
    comp["phrases"] = [{"notes": phrases[p], "f0_edits": edits.get(p, [])} for p in sorted(phrases)]
    return normalise(comp)


# ------------------------------------------------------------------------------------------- follow check
def follow_report(wav: np.ndarray, placed: Sequence[Dict], sr: int = SR) -> Dict:
    """How closely a take followed the designed curve. ``placed``: [{cents, sa_midi, start_s}] per piece.
    pYIN on the take; over frames that are voiced in both, the share within 50 / 25 cents and the mean |error|;
    plus the sung curve in cents above Sa (for the editor's overlay, null = unvoiced)."""
    import librosa
    f0, voiced, _ = librosa.pyin(np.asarray(wav, np.float32), fmin=60, fmax=1000, sr=sr,
                                 frame_length=1920, hop_length=480)
    errs, n_target, n_both, sung = [], 0, 0, []
    for p in placed:
        a = int(round(p["start_s"] * FPS))
        tgt = p["cents"]
        got = f0[a:a + len(tgt)]
        vv = voiced[a:a + len(tgt)]
        cents = np.where(vv & np.isfinite(got), 1200 * np.log2(np.where(got > 0, got, 1) / 440)
                         + 100 * (69 - p["sa_midi"]), np.nan)
        sung.append([None if not np.isfinite(x) else round(float(x), 1) for x in cents])
        m = np.isfinite(tgt[:len(cents)])
        n_target += int(m.sum())
        both = m & np.isfinite(cents)
        n_both += int(both.sum())
        errs += list(np.abs(cents[both] - tgt[:len(cents)][both]))
    e = np.array(errs) if errs else np.array([np.nan])
    return {"frames_voiced_share": round(n_both / max(n_target, 1), 3),
            "within_50c": round(float(np.mean(e <= 50)) if errs else 0.0, 3),
            "within_25c": round(float(np.mean(e <= 25)) if errs else 0.0, 3),
            "mean_abs_cents": round(float(np.nanmean(e)), 1), "sung_cents": sung}


# ------------------------------------------------------------------------------------------- singing
class ComposeSinger:
    """Sing a composition with ``kalavinka`` (V4-L) or ``base`` SoulX, using a loaded ``KalavinkaSinger``."""

    def __init__(self, singer):
        """Wrap a loaded ``KalavinkaSinger`` (its G2P, voices and ``_render``)."""
        self.s = singer

    def g2p(self, text: str) -> List[Dict]:
        """Devanagari -> syllables with phones (the singer's G2P, with its unknown-phone check)."""
        try:
            return self.s.syllables(text)
        except Exception as e:                      # SingError: re-raise as ours so callers catch one type
            raise ComposeError(str(e)) from e

    def sa_midi(self, comp: Dict, voice: Optional[int] = None) -> float:
        """The tonic (MIDI, fractional) for ``comp`` sung by ``voice``."""
        vo = self.s._voice(voice or comp.get("voice") or 1)
        try:
            return float(tonic_midi(comp.get("tonic", "auto"), vo["item"]))
        except SargamError as e:
            raise ComposeError(str(e)) from e

    def view(self, comp, voice: Optional[int] = None) -> Dict:
        """``view`` with the right tonic, plus a ``warn`` on every note whose syllable the model cannot sing (not
        Devanagari, or a sound it has no phone for), so the editor can underline it."""
        comp = normalise(comp)
        v = view(comp, self.sa_midi(comp, voice))
        for ph, vp in zip(comp["phrases"], v["phrases"]):
            for n, b in zip(ph["notes"], vp["notes"]):
                if n["syl"] not in ("", "-", "_") and not n.get("phones"):
                    err = self._syl_error(n["syl"])
                    if err:
                        b["warn"] = err
        return v

    def _syl_error(self, syl: str) -> Optional[str]:
        """Why a syllable cannot be sung (None if it can); cached per syllable."""
        cache = self.__dict__.setdefault("_syl_cache", {})
        if syl not in cache:
            try:
                self.g2p(syl)
                cache[syl] = None
            except ComposeError as e:
                cache[syl] = str(e)
        return cache[syl]

    def lyrics(self, comp, phrase: int, text: str) -> Dict:
        """Put a typed lyric line on a phrase (``set_phrase_lyrics`` with this singer's G2P)."""
        return set_phrase_lyrics(comp, int(phrase), self.g2p(text))

    def sing(self, comp, model: str = "kalavinka", voice: Optional[int] = None, steps: Optional[int] = None,
             seed: int = 0, check: bool = True, join: str = "seamless") -> Tuple[np.ndarray, int, Dict]:
        """Render the composition in melody mode. ``join="seamless"`` (default): one timeline, phrases joined by
        ``phrase_gap_s`` rests and cut only where ``seamless.plan_pieces`` says; ``"separate"``: the old way (each
        piece of ``plan`` rendered alone and joined with short gaps). Returns (wav, sr, info)."""
        from .seamless import JOINS, render
        from .singer import default_steps
        if join not in JOINS:
            raise ComposeError(f"join must be one of {JOINS}")
        comp = normalise(comp)
        try:
            vo = self.s._voice(voice or comp.get("voice") or 1)
        except Exception as e:                        # SingError for an unknown voice
            raise ComposeError(str(e)) from e
        sa = self.sa_midi(comp, vo["no"])
        self.s.steps = int(steps or default_steps(self.s.device))
        t0 = time.time()
        if join == "seamless":
            tl, cents, pstarts = compose_timeline(comp, sa, self.g2p)
            if all(n.typ == 1 for n in tl.notes):
                raise ComposeError("nothing to sing (only rests?)")
            wav, pieces = render(self.s, tl, vo, seed, model)
            placed = [{"cents": cents, "sa_midi": sa, "start_s": 0.0}]
            n_pieces, cuts = len(pieces), [p["cut"] for p in pieces[1:]]
        else:
            pieces = plan(comp, sa, self.g2p)
            if not pieces:
                raise ComposeError("nothing to sing (only rests?)")
            out, placed, pos = [], [], 0.0
            gap = np.zeros(int(GAP_S * SR), np.float32)
            for k, p in enumerate(pieces):
                hz = [float(x) for x in p["item"]["f0"].split()]
                w = self.s._render(p["item"], vo, hz, seed + k, shift=False, model=model)
                placed.append({"cents": p["cents"], "sa_midi": sa, "start_s": pos, "phrase": p["phrase"], "t0": p["t0"]})
                out += [w, gap]
                pos += (len(w) + len(gap)) / SR
            wav = np.concatenate(out[:-1]).astype(np.float32)
            n_pieces, cuts = len(pieces), ["gap"] * (len(pieces) - 1)
        info = {"model": model, "voice": vo["no"], "sa_midi": round(sa, 2), "steps": self.s.steps,
                "device": self.s.device, "pieces": n_pieces, "cuts": cuts, "join": join,
                "title": comp.get("title", ""), "took_s": round(time.time() - t0, 1),
                "seconds": round(len(wav) / SR, 2)}
        if check:
            rep = follow_report(wav, placed)
            info["follow"] = {k: v for k, v in rep.items() if k != "sung_cents"}
            if join == "seamless":
                sung = rep["sung_cents"][0]
                ends = pstarts[1:] + [len(sung)]
                info["sung"] = [{"phrase": i, "t0": 0.0, "start_s": round(a / FPS, 2), "cents": sung[a:b]}
                                for i, (a, b) in enumerate(zip(pstarts, ends))]
            else:
                info["sung"] = [{"phrase": p["phrase"], "t0": round(p["t0"], 2), "start_s": round(p["start_s"], 2),
                                 "cents": c} for p, c in zip(placed, rep["sung_cents"])]
        return wav, SR, info


COMPOSE_PROMPT = """You write melodies as JSON for a Hindi singing synthesiser (Indian classical / film style).
Answer with ONLY one JSON object, no other text:

{"title": "...", "tonic": "auto", "tempo_bpm": 80, "raag": "Yaman", "taal": "teentaal",
 "phrases": [{"notes": [
    {"syl": "सा", "swara": "Sa", "beats": 1},
    {"syl": "जन", "swara": "Re", "beats": 2, "orn": [{"type": "meend", "to": "Ga", "start": 0.5}]},
    {"syl": "-", "swara": "Ga", "beats": 1},
    {"syl": "_", "beats": 1}
 ]}]}

Rules:
- syl: ONE Devanagari syllable per note; "-" holds the previous syllable on a new note (slur); "_" is a rest;
  "" sings the swara's own name (sa re ga ...).
- swara: Sa Re Ga ma Pa Dha Ni. Komal: re(k) ga(k) dha(k) ni(k). Tivra: Ma(t). Upper octave: Sa' (two up: Sa'').
  Lower octave: .Ni .Dha. Optional "cents" for a microtonal shift (e.g. -20). Keep within .Pa to Pa'.
- beats: length at tempo_bpm (0.25 to 8). One phrase = one breath, at most ~10 seconds.
- raag (optional) is one of: """ + ", ".join(RAAGS) + """.
- orn: list of ornaments, each {"type": ..., params}. Types and default params:
""" + ornament_doc() + """
  Pitch params ("to", "from", items of "swaras") take a swara name, "up"/"down" (neighbour swara in the raag),
  "up2"/"down2", or cents relative to the note (e.g. -100); 0 means the note itself.
- Use ornaments the way a real singer of that raag would; do not put one on every note.

Now write: """
