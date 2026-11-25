"""Alankars and sargam: sing written swaras with the base SoulX model and with Kalavinka, and check the pitch.

Why
---
Our test clips are film-song lines. Alankars (sargam exercises such as ``Sa Re Ga, Re Ga ma, Ga ma Pa ...``) are a
cleaner probe: every note is a known swara, so we can measure how exactly each model lands on the written pitch,
and the syllables (sa re ga ma pa dha ni) test simple Hindi consonants, including the aspirated dh.

Input, two ways
---------------
1. **Sargam text** (``parse_sargam``): swaras separated by spaces, phrases by ``,`` or ``|``, lines by new lines.

   ================  =====================================================================
   ``Sa Re Ga ma``    shuddh swaras (names are case-insensitive: ``ma`` = ``Ma`` = shuddh Ma)
   ``re(k) ga(k)``    komal Re, Ga, Dha, Ni: add ``(k)``;  tivra Ma: ``Ma(t)``
   ``Sa'`` / ``.Ni``  upper octave (taar saptak) / lower octave (mandra saptak); ``''`` = two up
   ``Pa:2``           two beats long (default one beat); ``-`` holds the previous swara one beat more
   ``_``              a rest of one beat
   ================  =====================================================================

2. **JSON** (``from_json``), the "intermediate representation" any LLM can write (``LLM_PROMPT`` explains it)::

       {"title": "...", "tonic": "auto" | "C#3" | 138.6, "beat_s": 0.45, "syllables": "sargam" | "aa",
        "lines": [[{"swara": "Sa", "octave": 0, "beats": 1}, {"swara": "rest", "beats": 1}, ...], ...]}

Every line is rendered as one phrase (lines longer than ``MAX_NOTES`` notes are cut into chunks), and the phrases are
joined with short gaps. ``tonic: auto`` puts Sa on the chosen voice's own tonic, so the exercise sits in that singer's range.

Models
------
``kalavinka`` = our final model V4-L; ``base`` = the released SoulX-Singer with every Hindi phone mapped to its nearest
English phone (our zero-shot baseline V0). Both run in score mode (they get the note numbers and make their own pitch).

Pitch check (``pitch_report``)
-------------------------------
pYIN (librosa) tracks the F0 of the render; for every written note the median pitch over the middle half of the
note is compared with the written swara, in cents. Reported: mean |error|, the share of notes within 50 cents
(a quarter tone either way: "the right swara") and within 25 cents ("in tune"), and the share of notes voiced at all.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SR, FPS = 24000, 50
MAX_NOTES = 14
EDGE_REST = 0.2                     # seconds of rest before and after every phrase
GAP_S = 0.25

SWARA = {"sa": 0, "re": 2, "ga": 4, "ma": 5, "pa": 7, "dha": 9, "ni": 11}
KOMAL_OK = {"re", "ga", "dha", "ni"}
# sung syllable of each swara (phones of saregama.phoneset)
SYL = {"sa": ["s", "a"], "re": ["r", "ee"], "ga": ["g", "a"], "ma": ["m", "a"], "pa": ["p", "a"],
       "dha": ["dh", "a"], "ni": ["n", "ii"]}
DEV = {"sa": "सा", "re": "रे", "ga": "ग", "ma": "म", "pa": "प", "dha": "ध", "ni": "नि"}
NAMES = ["Sa", "re(k)", "Re", "ga(k)", "Ga", "ma", "Ma(t)", "Pa", "dha(k)", "Dha", "ni(k)", "Ni"]
TOKEN = re.compile(r"^(\.*)([A-Za-z]+)(\((?:k|t)\))?('*)(?::(\d+(?:\.\d+)?))?(\.*)$")


class SargamError(ValueError):
    """A sargam line or JSON score that cannot be read; the message says why."""


# --------------------------------------------------------------------------------------------- notation
def swara_semitones(name: str, mark: str = "", octave: int = 0) -> int:
    """Semitones above Sa for a swara name (+ '(k)' komal / '(t)' tivra) in an octave (-1, 0, +1 ...)."""
    n = name.lower()
    if n not in SWARA:
        raise SargamError(f"unknown swara {name!r}; use Sa Re Ga ma Pa Dha Ni (komal: re(k), tivra: Ma(t))")
    s = SWARA[n]
    if mark == "(k)":
        if n not in KOMAL_OK:
            raise SargamError(f"{name} has no komal form (only Re, Ga, Dha, Ni)")
        s -= 1
    elif mark == "(t)":
        if n != "ma":
            raise SargamError("only Ma has a tivra form: Ma(t)")
        s += 1
    return s + 12 * octave


def parse_sargam(text: str) -> List[List[Dict]]:
    """Sargam text -> lines of notes ``{swara, semis, beats}`` (rests have ``swara='rest'``)."""
    lines = []
    for raw in re.split(r"\n+", text.strip()):
        notes: List[Dict] = []
        for tok in re.split(r"[\s,|]+", raw.strip()):
            if not tok:
                continue
            if tok == "-":
                if not notes:
                    raise SargamError("a line cannot start with '-' (hold)")
                notes[-1]["beats"] += 1
                continue
            if tok == "_":
                notes.append({"swara": "rest", "semis": None, "beats": 1.0})
                continue
            m = TOKEN.match(tok)
            if not m:
                raise SargamError(f"cannot read {tok!r}; examples: Sa, Re, ga(k), Ma(t), Sa', .Ni, Pa:2")
            low1, name, mark, ups, beats, low2 = m.groups()
            octave = len(ups) - len(low1) - len(low2)
            semis = swara_semitones(name, mark or "", octave)
            if beats is not None and float(beats) <= 0:
                raise SargamError(f"{tok!r}: a note must last more than 0 beats")
            notes.append({"swara": name.lower(), "semis": semis, "beats": float(beats or 1)})
        if notes:
            lines.append(notes)
    if not lines:
        raise SargamError("no swaras found")
    return lines


def from_json(obj: Dict | str) -> Tuple[List[List[Dict]], Dict]:
    """JSON score (dict or text) -> (lines, settings {title, tonic, beat_s, syllables})."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError as e:
            raise SargamError(f"not valid JSON: {e}") from e
    if not isinstance(obj, dict) or not isinstance(obj.get("lines"), list):
        raise SargamError('the JSON needs a "lines" list (see LLM_PROMPT)')
    lines = []
    for ln in obj["lines"]:
        notes = []
        for n in ln if isinstance(ln, list) else []:
            if not isinstance(n, dict):
                raise SargamError(f"each note is an object like {{\"swara\": \"Sa\", \"beats\": 1}}, not {n!r}")
            sw = str(n.get("swara", "")).strip()
            try:
                beats = float(n.get("beats", 1))
            except (TypeError, ValueError):
                raise SargamError(f"beats {n.get('beats')!r} is not a number") from None
            if beats <= 0:
                raise SargamError(f"{sw or 'a note'}: beats must be more than 0")
            if sw.lower() in ("rest", "_", ""):
                notes.append({"swara": "rest", "semis": None, "beats": beats})
                continue
            m = re.match(r"^([A-Za-z]+)(\((?:k|t)\))?$", sw)
            if not m:
                raise SargamError(f"cannot read swara {sw!r}")
            notes.append({"swara": m.group(1).lower(),
                          "semis": swara_semitones(m.group(1), m.group(2) or "", int(n.get("octave", 0))),
                          "beats": beats})
        if notes:
            lines.append(notes)
    if not lines:
        raise SargamError("the JSON has no notes")
    settings = {k: obj[k] for k in ("title", "tonic", "beat_s", "syllables") if k in obj}
    if "beat_s" in settings:
        try:
            settings["beat_s"] = float(settings["beat_s"])
        except (TypeError, ValueError):
            raise SargamError(f"beat_s {settings['beat_s']!r} is not a number") from None
        if settings["beat_s"] <= 0:
            raise SargamError("beat_s must be more than 0")
    return lines, settings


# ------------------------------------------------------------------------------------------ built-ins
SCALE = [0, 2, 4, 5, 7, 9, 11, 12]            # Bilawal (all shuddh): Sa Re Ga ma Pa Dha Ni Sa'


def _pattern(offsets: Sequence[int], scale: Sequence[int] = SCALE) -> List[List[Dict]]:
    """Classic alankar from a pattern of scale steps: aaroh (up) until the top reaches Sa', then avroh (mirror)."""
    top = len(scale) - 1
    span = max(offsets)
    up = [[scale[i + o] for o in offsets] for i in range(0, top - span + 1)]
    down = [[scale[top - i - o] for o in offsets] for i in range(0, top - span + 1)]

    def notes(groups):
        return [{"swara": None, "semis": s, "beats": 1.0} for g in groups for s in g]
    return [notes(up), notes(down)]


ALANKARS: Dict[int, Dict] = {
    1: {"name": "Scale (aaroh-avroh)", "text": "Sa Re Ga ma Pa Dha Ni Sa'\nSa' Ni Dha Pa ma Ga Re Sa"},
    2: {"name": "Groups of three: Sa Re Ga, Re Ga ma, ...", "lines": _pattern([0, 1, 2])},
    3: {"name": "Groups of four: Sa Re Ga ma, Re Ga ma Pa, ...", "lines": _pattern([0, 1, 2, 3])},
    4: {"name": "Jumps of a third: Sa Ga, Re ma, Ga Pa, ...", "lines": _pattern([0, 2])},
    5: {"name": "Doubled notes: Sa Sa, Re Re, Ga Ga, ...", "lines": _pattern([0, 0])},
    6: {"name": "Zig-zag: Sa Ga Re ma, Re ma Ga Pa, ...", "lines": _pattern([0, 2, 1, 3])},
    7: {"name": "Bhairav thaat scale (komal Re and Dha)",
        "text": "Sa re(k) Ga ma Pa dha(k) Ni Sa'\nSa' Ni dha(k) Pa ma Ga re(k) Sa"},
}


def name_of(semis: Optional[int]) -> str:
    """Display name of a note (``Sa'``, ``.Ni``, ``re(k)``, or ``_`` for a rest)."""
    if semis is None:
        return "_"
    octave, k = divmod(semis, 12)
    return ("." * -octave if octave < 0 else "") + NAMES[k] + ("'" * octave if octave > 0 else "")


def alankar_lines(no: int) -> List[List[Dict]]:
    """Notes of built-in alankar ``no`` (1-based)."""
    if no not in ALANKARS:
        raise SargamError(f"there is no alankar {no}; pick 1 to {len(ALANKARS)}")
    a = ALANKARS[no]
    lines = parse_sargam(a["text"]) if "text" in a else a["lines"]
    for ln in lines:
        for n in ln:
            if n["semis"] is not None and n["swara"] is None:
                n["swara"] = name_of(n["semis"]).strip(".'").split("(")[0].lower()
    return lines


def chunk(lines: List[List[Dict]], n: int = MAX_NOTES) -> List[List[Dict]]:
    """Cut long lines into phrases of at most ``n`` notes."""
    return [ln[i:i + n] for ln in lines for i in range(0, len(ln), n)]


def to_text(lines: List[List[Dict]]) -> str:
    """Lines back to sargam text (for printing and the .json next to a take)."""
    return "\n".join(" ".join(name_of(n["semis"]) + (f":{n['beats']:g}" if n["beats"] != 1 else "") for n in ln)
                     for ln in lines)


# ------------------------------------------------------------------------------------------- items
def fold_tonic(tonic_hz: float, f0_median: float) -> float:
    """The octave of ``tonic_hz`` that suits a voice singing around ``f0_median`` Hz: Sa about a fourth below the
    median (so Sa..Sa' sits in the voice's range). The dataset's tonic estimate is sometimes an octave off for the
    singer (voice 1: Sa 335 Hz for a male voice singing around 196 Hz), which put exercises an octave too high."""
    if tonic_hz <= 0 or f0_median <= 0:
        return tonic_hz
    target = f0_median * 2 ** (-5 / 12)
    k = int(np.round(np.log2(target / tonic_hz)))
    return float(tonic_hz * 2.0 ** k)


def voice_f0_median(voice_item: Dict) -> float:
    """Median sung F0 (Hz) of a voice clip (0 if unknown)."""
    try:
        f0 = np.array(str(voice_item.get("f0", "")).split(), float)
    except ValueError:
        return 0.0
    v = f0[f0 > 0]
    return float(np.median(v)) if len(v) else 0.0


def tonic_midi(tonic, voice_item: Dict) -> float:
    """Tonic as a (fractional) MIDI number: 'auto' = the voice clip's own tonic, in the octave that suits the voice
    (``fold_tonic``); else a name like C#3 or a frequency in Hz (also as text, "138.6")."""
    if tonic in (None, "", "auto"):
        hz = fold_tonic(float(voice_item.get("tonic_hz") or 0), voice_f0_median(voice_item))
        if hz <= 0:
            return 60.0
        return 69 + 12 * np.log2(hz / 440)
    if isinstance(tonic, str) and re.match(r"^\s*\d+(\.\d+)?\s*$", tonic):
        tonic = float(tonic)                               # "138.6" typed in a box = Hz
    if isinstance(tonic, (int, float)):
        if not 30 <= float(tonic) <= 2000:
            raise SargamError(f"tonic {tonic} Hz: use a frequency between 30 and 2000 Hz")
        return 69 + 12 * np.log2(float(tonic) / 440)
    m = re.match(r"^([A-Ga-g])([#b]?)(-?\d)$", str(tonic).strip())
    if not m:
        raise SargamError(f"tonic {tonic!r}: use auto, a note like C#3 / Bb2, or a frequency in Hz")
    pc = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[m.group(1).lower()]
    pc += {"#": 1, "b": -1, "": 0}[m.group(2)]
    return float(12 * (int(m.group(3)) + 1) + pc)


def phrase_item(notes: Sequence[Dict], sa_midi: int, beat_s: float, syllables: str = "sargam") -> Dict:
    """One phrase as a SoulX/export item (score mode): edge rests, one note per swara, note type 2 (1 for rests)."""
    ph, pitch, typ, dur, text = ["<SP>"], [0], [1], [EDGE_REST], ["_"]
    for n in notes:
        if n["semis"] is None:
            ph.append("<SP>"); pitch.append(0); typ.append(1); text.append("_")
        else:
            phones = ["aa"] if syllables == "aa" else SYL[n["swara"]]
            ph.append("in_" + "-".join(phones)); pitch.append(int(sa_midi + n["semis"])); typ.append(2)
            text.append("आ" if syllables == "aa" else DEV[n["swara"]])
        dur.append(n["beats"] * beat_s)
    ph.append("<SP>"); pitch.append(0); typ.append(1); dur.append(EDGE_REST); text.append("_")
    if beat_s <= 0 or any(d <= 0 for d in dur):
        raise SargamError("note lengths must be more than 0")
    ends = np.round(np.cumsum(dur) * FPS).astype(int)          # rounded on the running total: no drift
    frames = [int(f) for f in np.diff(np.concatenate([[0], ends]))]
    n_frames = sum(frames)
    return {"index": "sargam", "language": "Hindi", "text": " ".join(text), "phoneme": " ".join(ph),
            "note_pitch": " ".join(map(str, pitch)), "note_type": " ".join(map(str, typ)),
            "duration": " ".join(f"{f / FPS:.2f}" for f in frames), "n_frames": n_frames,
            "f0": " ".join(["0.0"] * n_frames)}


# ------------------------------------------------------------------------------------------ pitch check
def pitch_report(wav: np.ndarray, phrases: Sequence[Dict], sr: int = SR) -> Dict:
    """Compare the sung pitch with the written swaras. ``phrases``: [{item, start_s}] as placed in ``wav``."""
    import librosa
    f0, voiced, _ = librosa.pyin(np.asarray(wav, np.float32), fmin=60, fmax=1000, sr=sr,
                                 frame_length=1920, hop_length=480)
    t = np.arange(len(f0)) * 480 / sr
    errs, n_notes, n_voiced, per_note = [], 0, 0, []
    for ph in phrases:
        it = ph["item"]
        durs = [float(x) for x in it["duration"].split()]
        pitches = [int(x) for x in it["note_pitch"].split()]
        types = [int(x) for x in it["note_type"].split()]
        start = ph["start_s"]
        for d, p, ty in zip(durs, pitches, types):
            if ty != 1:
                n_notes += 1
                sel = (t >= start + 0.25 * d) & (t < start + 0.75 * d) & voiced & np.isfinite(f0)
                if sel.sum() >= 2:
                    n_voiced += 1
                    sung = 69 + 12 * np.log2(np.median(f0[sel]) / 440)
                    e = 100 * (sung - p)
                    errs.append(e)
                    per_note.append(round(float(e), 1))
                else:
                    per_note.append(None)
            start += d
    a = np.abs(np.array(errs)) if errs else np.array([np.nan])
    return {"notes": n_notes, "voiced_share": round(n_voiced / max(n_notes, 1), 3),
            "mean_abs_cents": round(float(np.nanmean(a)), 1),
            "within_50c": round(float(np.mean(a <= 50)) if errs else 0.0, 3),
            "within_25c": round(float(np.mean(a <= 25)) if errs else 0.0, 3),
            "per_note_cents": per_note}


# ------------------------------------------------------------------------------------------ singing
class SargamSinger:
    """Sing sargam with ``kalavinka`` (V4-L) or ``base`` (zero-shot SoulX), using a loaded ``KalavinkaSinger``."""

    def __init__(self, singer):
        """Wrap a loaded ``KalavinkaSinger`` (it loads the base model on first use)."""
        self.s = singer

    def sing(self, lines: List[List[Dict]], model: str = "kalavinka", voice: Optional[int] = None,
             tonic="auto", beat_s: float = 0.45, syllables: str = "sargam", steps: Optional[int] = None,
             seed: int = 0, check: bool = True) -> Tuple[np.ndarray, int, Dict]:
        """Render all lines (each a phrase; long ones chunked) and join them. Returns (wav, 24000, info + pitch report)."""
        from .singer import default_steps
        vo = self.s._voice(voice if voice is not None else 1)
        sa = int(round(tonic_midi(tonic, vo["item"])))
        steps = int(steps or default_steps(self.s.device))
        self.s.steps = steps
        t0 = time.time()
        pieces, placed, pos = [], [], 0.0
        gap = np.zeros(int(GAP_S * SR), np.float32)
        for k, ph in enumerate(chunk(lines)):
            item = phrase_item(ph, sa, beat_s, syllables)
            w = self.s._render(item, vo, None, seed + k, shift=False, model=model)
            placed.append({"item": item, "start_s": pos})
            pieces += [w, gap]
            pos += (len(w) + len(gap)) / SR
        wav = np.concatenate(pieces[:-1]).astype(np.float32)
        info = {"model": model, "voice": vo["no"], "sa_midi": sa, "beat_s": beat_s, "syllables": syllables,
                "steps": steps, "device": self.s.device, "sargam": to_text(lines),
                "took_s": round(time.time() - t0, 1), "seconds": round(len(wav) / SR, 2)}
        if check:
            info["pitch"] = pitch_report(wav, placed)
        return wav, SR, info


LLM_PROMPT = """You write Hindustani sargam exercises (alankars) as JSON for a singing synthesiser.
Return ONLY a JSON object, no prose, in exactly this format:
{
  "title": "short name",
  "tonic": "auto",
  "beat_s": 0.45,
  "syllables": "sargam",
  "lines": [
    [{"swara": "Sa", "octave": 0, "beats": 1}, {"swara": "Re", "octave": 0, "beats": 1}, ...],
    [ ... next line ... ]
  ]
}
Rules:
- swara is one of Sa Re Ga Ma Pa Dha Ni (shuddh), re(k) ga(k) dha(k) ni(k) (komal), Ma(t) (tivra), or "rest".
- octave: 0 = middle (madhya), 1 = upper (taar, e.g. Sa' is {"swara": "Sa", "octave": 1}), -1 = lower (mandra).
- beats: length in beats (1 = one note; 2 = held twice as long).
- One line = one breath/phrase, at most 14 notes; put the aaroh (ascent) and avroh (descent) in separate lines.
- Keep everything between .Pa (octave -1) and Pa' (octave 1).
Now write: <DESCRIBE THE ALANKAR HERE, e.g. "the alankar Sa Re Ga, Re Ga Ma, ... up to Dha Ni Sa', and back down">"""
