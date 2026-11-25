"""Seamless singing: one continuous performance, with silence only where the score has a rest.

Why
---
SoulX renders one phrase of at most about 15 s per call. Before this module, every tab joined its separate renders
with silence of its own: 0.3 s between lyric lines, 0.2 s edge rests plus 0.25 s gaps between sargam chunks, and
about 0.7 s inside long Compose phrases. That is the "patching" a listener hears. Here every tab first writes
**one long score on one timeline** (``Timeline``), and ``render`` cuts it into model-sized pieces itself:

1. a rest of at least ``LONG_REST_S`` (1 s) is always cut: after a long silence a seam is inaudible;
2. otherwise, when the music left is longer than ``MAX_S`` (14 s), a rest of at least ``MIN_REST_S`` (0.12 s;
   0.06 s in score mode): the longest one in the last 6 s of the window, else the latest one at least 3 s in. The
   cut is at the middle of the rest, with a 40 ms crossfade inside the silence;
3. with no usable rest (continuous singing), an **overlap cut**: the longest sung note *j* in the last 4 s of the
   window is rendered by both pieces (A runs about 1 s past *j*, B starts about 1 s before it, never on a slur), and
   the two renders are crossfaded (equal power, ``min(120 ms, half of j)``; 200 ms in score mode) in the steady
   middle of *j*. In melody mode both renders follow the same F0 and phonemes, so the fade joins the same vowel at the
   same pitch. Two renders have an arbitrary waveform phase, so B is first shifted by at most 12 ms to the lag that
   best correlates with A on *j* (otherwise the fade can cancel). A gain match (RMS of A over RMS of B on *j*, clipped
   to [0.5, 2], ramped back to 1 over 2 s) removes level jumps.

SoulX's output length is exactly ``frames x 480`` samples, so every piece is placed sample-exact on the timeline.

Singers
-------
* ``SeamlessSinger`` (subclass of ``KalavinkaSinger``): the Sing tab. Lines go on one timeline; between two lines the
  rest is the **real gap** when the second tune is the next clip of the same song (``time`` in ms, at most 1.5 s), else
  ``line_gap``. A line whose tune singer differs from the song voice is transposed by SoulX's own ``auto_shift`` rule
  for the mode (``soulx_shift``), so all lines can share renders with ``auto_shift`` off.
* ``SeamlessSargamSinger`` (wraps ``SargamSinger``): every sargam line on one timeline, ``line_gap`` rests between.
* Compose uses ``compose.compose_timeline`` + ``render``.

``join="separate"`` everywhere gives the old behaviour (one render per line or phrase) for comparison.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SR, FPS, HOP = 24000, 50, 480
MAX_S = 14.0              # longest render (SoulX was trained on clips of 3-16 s)
LONG_REST_S = 1.0         # always cut at rests this long
MIN_REST_S = 0.12         # shortest rest used as a cut (melody mode)
MIN_REST_SCORE_S = 0.06   # ... in score mode (overlap seams are less exact there)
MIN_PIECE_S = 3.0         # a rest cut leaves at least this much in the piece
REST_WINDOW_S = 6.0       # prefer the longest rest in the last 6 s of the window
OVERLAP_WINDOW_S = 4.0    # overlap cuts look for a held note in the last 4 s
CONTEXT_S = 1.0           # context rendered on each side of an overlap note
FADE_S, FADE_SCORE_S, REST_FADE_S = 0.12, 0.20, 0.04
GAIN_RAMP_S = 2.0
MAX_LAG_S = 0.012         # phase alignment search (one period of an 83 Hz voice)
MAX_NOTE_S = 6.0          # longer sung notes are split into slurs so a cut is always possible
EDGE_REST_S = 0.2
JOINS = ("seamless", "separate")


# ------------------------------------------------------------------------------------------- the timeline
@dataclass
class Note:
    """One note of the timeline: SoulX token, MIDI pitch (0 = rest), type (1 rest, 2 syllable, 3 slur), frames."""
    tok: str
    pitch: int
    typ: int
    frames: int
    text: str = ""


@dataclass
class Timeline:
    """One long score. ``melody``: an F0 curve (Hz, 50 fps, 0 = unvoiced) goes with the notes; else score mode."""
    melody: bool
    notes: List[Note] = field(default_factory=list)
    f0: List[np.ndarray] = field(default_factory=list)       # one array per note (melody mode)

    # ---- building
    def _push(self, note: Note, f0: Optional[np.ndarray] = None) -> None:
        """Append a note (adjacent rests are merged)."""
        if note.frames <= 0:
            return
        f0 = np.zeros(note.frames) if f0 is None else np.asarray(f0, float)
        if note.typ == 1 and self.notes and self.notes[-1].typ == 1:
            self.notes[-1].frames += note.frames
            self.f0[-1] = np.concatenate([self.f0[-1], np.zeros(note.frames)])
            return
        if note.typ != 1 and note.frames > MAX_NOTE_S * FPS:       # split very long notes into slurs
            step = int(MAX_NOTE_S * FPS * 2 // 3)
            for k, a in enumerate(range(0, note.frames, step)):
                n = min(step, note.frames - a)
                self.notes.append(Note(note.tok, note.pitch, note.typ if k == 0 else 3, n, note.text if k == 0 else ""))
                self.f0.append(f0[a:a + n])
            return
        self.notes.append(note)
        self.f0.append(f0)

    def append_rest(self, sec: float) -> "Timeline":
        """A rest of ``sec`` seconds (merged with a rest before it)."""
        self._push(Note("<SP>", 0, 1, int(round(sec * FPS)), "_"))
        return self

    def append_item(self, item: Dict, f0: Optional[Sequence[float]] = None, shift: int = 0) -> "Timeline":
        """Append an export-style item. Durations become frames by cumulative rounding (exact total). In melody mode
        the F0 is ``f0`` (else the item's own ``f0``), padded or trimmed to the notes. ``shift`` transposes the note
        pitches and the F0 by that many semitones."""
        toks = item["phoneme"].split()
        pitch = [int(x) for x in item["note_pitch"].split()]
        typ = [int(x) for x in item["note_type"].split()]
        dur = [float(x) for x in item["duration"].split()]
        if not (len(toks) == len(pitch) == len(typ) == len(dur)):
            raise ValueError("item fields have different lengths")
        texts = item.get("text", "").split()
        if len(texts) != len(toks):
            texts = [""] * len(toks)
        ends = np.round(np.cumsum(dur) * FPS).astype(int)
        frames = np.diff(np.concatenate([[0], ends])).astype(int)
        n = int(ends[-1]) if len(ends) else 0
        hz = np.zeros(n)
        if self.melody:
            src = f0 if f0 is not None else [float(x) for x in item.get("f0", "").split()]
            src = np.asarray(src, float)[:n]
            hz[:len(src)] = src
            hz = np.where(hz > 0, hz * 2 ** (shift / 12), 0.0)
        a = 0
        for t, p, ty, fr, tx in zip(toks, pitch, typ, frames, texts):
            if ty == 1 or t == "<SP>":
                self._push(Note("<SP>", 0, 1, int(fr), "_"))
            else:
                self._push(Note(t, p + shift if p > 0 else p, ty, int(fr), tx), hz[a:a + fr])
            a += fr
        return self

    def finish(self, edge_s: float = EDGE_REST_S) -> "Timeline":
        """Make sure the whole timeline starts and ends with a rest of at least ``edge_s``."""
        e = int(round(edge_s * FPS))
        if not self.notes or self.notes[0].typ != 1:
            self.notes.insert(0, Note("<SP>", 0, 1, e, "_")); self.f0.insert(0, np.zeros(e))
        elif self.notes[0].frames < e:
            self.notes[0].frames = e; self.f0[0] = np.zeros(e)
        if self.notes[-1].typ != 1:
            self._push(Note("<SP>", 0, 1, e, "_"))
        elif self.notes[-1].frames < e:
            self.notes[-1].frames = e; self.f0[-1] = np.zeros(e)
        return self

    # ---- reading
    @property
    def frames(self) -> np.ndarray:
        """Frames per note."""
        return np.array([n.frames for n in self.notes], int)

    @property
    def starts(self) -> np.ndarray:
        """Start frame of every note, plus the total at the end."""
        return np.concatenate([[0], np.cumsum(self.frames)]).astype(int)

    @property
    def total(self) -> int:
        """Length in frames."""
        return int(self.frames.sum())

    def f0_array(self) -> np.ndarray:
        """The whole F0 curve (Hz, 0 = unvoiced)."""
        return np.concatenate(self.f0) if self.f0 else np.zeros(0)

    def to_item(self, a: int = 0, b: Optional[int] = None, index: str = "timeline") -> Dict:
        """SoulX item for frames ``[a, b)``. Only rests may be cut part-way (pieces are cut at note boundaries or
        inside rests); a sung note that would be cut raises an AssertionError."""
        b = self.total if b is None else b
        st = self.starts
        toks, pit, typ, fr, txt = [], [], [], [], []
        for i, n in enumerate(self.notes):
            lo, hi = max(st[i], a), min(st[i + 1], b)
            if hi <= lo:
                continue
            if n.typ != 1:
                assert lo == st[i] and hi == st[i + 1], f"sung note {i} cut at {a}-{b}"
            toks.append(n.tok); pit.append(n.pitch); typ.append(n.typ); fr.append(hi - lo); txt.append(n.text or "_")
        if typ and typ[0] == 3:               # a piece never starts on a slur (the cutter avoids it)
            typ[0] = 2
        f0 = self.f0_array()[a:b] if self.melody else np.zeros(b - a)
        return {"index": index, "language": "Hindi", "text": " ".join(txt), "phoneme": " ".join(toks),
                "note_pitch": " ".join(map(str, pit)), "note_type": " ".join(map(str, typ)),
                "duration": " ".join(f"{f / FPS:.2f}" for f in fr), "n_frames": int(b - a),
                "f0": " ".join(f"{x:.1f}" for x in f0)}


# ------------------------------------------------------------------------------------------- cutting
@dataclass
class Piece:
    """One render: timeline frames ``[ra, rb)`` rendered, ``[oa, ob)`` kept. ``cut_in`` describes the cut at its
    start: {kind: rest|overlap, c (frame), fade (frames), note (index of the overlap note)}."""
    ra: int
    rb: int
    oa: int
    ob: int
    cut_in: Optional[Dict] = None


def plan_pieces(tl: Timeline, max_s: float = MAX_S) -> List[Piece]:
    """Cut the timeline into pieces of at most ``max_s`` (see the module doc for the rules)."""
    notes, st = tl.notes, tl.starts
    T, maxF = tl.total, int(max_s * FPS)
    score = not tl.melody
    min_rest = int(round((MIN_REST_SCORE_S if score else MIN_REST_S) * FPS))
    long_rest = int(LONG_REST_S * FPS)
    min_piece, ctx = int(MIN_PIECE_S * FPS), int(CONTEXT_S * FPS)
    sung = [n.typ != 1 for n in notes]

    def sung_between(a: int, b: int) -> bool:
        return any(s and st[i] < b and st[i + 1] > a for i, s in enumerate(sung))

    pieces: List[Piece] = []
    ra, oa, cut_in = 0, 0, None
    while True:
        limit = ra + maxF
        # 1. long rests are always cut (if there is singing on both sides)
        cut = None
        for i, n in enumerate(notes):
            if n.typ == 1 and n.frames >= long_rest:
                m = int(st[i] + n.frames // 2)
                if oa < m <= limit - 1 and m - ra > 1 and sung_between(oa, m) and sung_between(m, T):
                    cut = ("rest", m, None)
                    break
        if cut is None and T - ra <= maxF:
            pieces.append(Piece(ra, T, oa, T, cut_in))
            return pieces
        # 2. a rest
        if cut is None:
            rests = [(i, int(st[i] + notes[i].frames // 2)) for i, n in enumerate(notes)
                     if n.typ == 1 and n.frames >= min_rest]
            ok = [(i, m) for i, m in rests if oa + min_piece <= m <= limit - 1 and sung_between(m, T)]
            late = [(i, m) for i, m in ok if m >= limit - REST_WINDOW_S * FPS]
            if late:
                i, m = max(late, key=lambda x: (notes[x[0]].frames, x[1]))
                cut = ("rest", m, None)
            elif ok:
                cut = ("rest", ok[-1][1], None)
        # 3. an overlap note
        if cut is None:
            cands = []
            for j, n in enumerate(notes):
                if n.typ == 1 or n.frames < 4:
                    continue
                c = int(st[j] + n.frames // 2)
                k = j + 1                                    # A: about 1 s of context after j, at a note boundary
                while k < len(notes) and st[k] < st[j + 1] + ctx:
                    k += 1
                rb = int(st[k])
                b = j                                        # B: about 1 s before j, never starting on a slur
                while b > 0 and st[j] - st[b] < ctx:
                    b -= 1
                while b > 0 and notes[b].typ == 3:
                    b -= 1
                rb_b = int(st[b])
                if rb <= limit and c >= oa + min_piece and rb_b > ra:
                    cands.append((j, c, rb, rb_b))
            if cands:
                late = [x for x in cands if x[1] >= limit - (OVERLAP_WINDOW_S + CONTEXT_S) * FPS]
                j, c, rb, rb_b = max(late or cands, key=lambda x: (notes[x[0]].frames, x[1]))
                fade = max(2, min(int((FADE_SCORE_S if score else FADE_S) * FPS), notes[j].frames // 2))
                pieces.append(Piece(ra, rb, oa, c, cut_in))
                cut_in = {"kind": "overlap", "c": c, "fade": fade, "note": j}
                ra, oa = rb_b, c
                continue
            # 4. last resort (should not happen: long notes are split): the note boundary nearest the limit
            bounds = [int(s) for s in st if ra + min_piece <= s <= limit]
            m = bounds[-1] if bounds else int(max(s for s in st if s <= limit))
            pieces.append(Piece(ra, m, oa, m, cut_in))
            cut_in = {"kind": "hard", "c": m, "fade": 0, "note": None}
            ra, oa = m, m
            continue
        _, m, _ = cut
        h = max(1, int(REST_FADE_S * FPS) // 2)
        pieces.append(Piece(ra, min(m + h, T), oa, m, cut_in))
        cut_in = {"kind": "rest", "c": m, "fade": 2 * h, "note": None}
        ra, oa = max(m - h, 0), m


# ------------------------------------------------------------------------------------------- rendering
def _ramp(n: int) -> np.ndarray:
    """Equal-power fade-in weights over ``n`` samples (fade-out = the reversed array)."""
    t = (np.arange(n) + 0.5) / max(n, 1)
    return np.sin(0.5 * np.pi * t)


def best_lag(ref: np.ndarray, new: np.ndarray, max_lag_s: float = MAX_LAG_S) -> int:
    """Shift (samples, |shift| <= ``max_lag_s``) that best lines ``new`` up with ``ref`` (normalised cross-correlation
    over the middle of the held note). Two renders of the same note have the same pitch but an arbitrary waveform
    phase; without this an equal-power crossfade can cancel at the seam."""
    L = int(max_lag_s * SR)
    n = min(len(ref), len(new))
    if n < 4 * L:
        return 0
    a = ref[L:n - L].astype(np.float64)
    if np.sqrt(np.mean(a ** 2)) < 1e-6:
        return 0
    best, best_c = 0, -np.inf
    for lag in range(-L, L + 1):
        bseg = new[L - lag:n - L - lag].astype(np.float64)
        c = float(np.dot(a, bseg) / (np.linalg.norm(a) * np.linalg.norm(bseg) + 1e-12))
        if c > best_c:
            best, best_c = lag, c
    return best


def assemble(tl: Timeline, pieces: Sequence[Piece], waves: Sequence[np.ndarray]) -> np.ndarray:
    """Overlap-add the rendered pieces into one waveform of exactly ``tl.total x 480`` samples."""
    out = np.zeros(tl.total * HOP, np.float32)
    st = tl.starts
    prev_audio, prev_piece = None, None
    for k, (p, w) in enumerate(zip(pieces, waves)):
        n = (p.rb - p.ra) * HOP
        w = np.asarray(w, np.float32).reshape(-1)
        w = np.pad(w, (0, max(0, n - len(w))))[:n].copy()
        cin = p.cut_in
        if cin and cin["kind"] == "overlap" and prev_audio is not None:
            j = cin["note"]
            a, b = int(st[j]), int(st[j + 1])
            ra_prev = prev_piece.ra
            ref = prev_audio[(a - ra_prev) * HOP:(b - ra_prev) * HOP]
            lag = best_lag(ref, w[(a - p.ra) * HOP:(b - p.ra) * HOP])
            if lag:                                        # line up the waveform phase with the previous render
                w = np.roll(w, lag)
                if lag > 0:
                    w[:lag] = 0
                else:
                    w[lag:] = 0
            new = w[(a - p.ra) * HOP:(b - p.ra) * HOP]
            r1, r2 = np.sqrt(np.mean(ref ** 2) + 1e-12), np.sqrt(np.mean(new ** 2) + 1e-12)
            g = float(np.clip(r1 / r2, 0.5, 2.0)) if r2 > 1e-6 and r1 > 1e-6 else 1.0
            if abs(g - 1) > 1e-3:
                env = np.ones(n, np.float32)
                c0 = (cin["c"] + cin["fade"] // 2 - p.ra) * HOP
                c1 = c0 + int(GAIN_RAMP_S * SR)
                env[:max(c0, 0)] = g
                if c1 > c0:
                    idx = np.arange(max(c0, 0), min(c1, n))
                    env[idx] = g + (1 - g) * (idx - c0) / (c1 - c0)
                w *= env
        wt = np.zeros(n, np.float32)
        lo, hi = (p.oa - p.ra) * HOP, (p.ob - p.ra) * HOP
        wt[max(lo, 0):max(hi, 0)] = 1.0
        if cin:                                            # fade in around the cut at its start
            f = cin["fade"] * HOP
            s0 = (cin["c"] - p.ra) * HOP - f // 2
            seg = np.arange(max(s0, 0), min(s0 + f, n))
            wt[seg] = _ramp(f)[seg - s0]
            wt[:max(s0, 0)] = 0
        nxt = pieces[k + 1].cut_in if k + 1 < len(pieces) else None
        if nxt:                                            # fade out around the cut at its end
            f = nxt["fade"] * HOP
            s0 = (nxt["c"] - p.ra) * HOP - f // 2
            seg = np.arange(max(s0, 0), min(s0 + f, n))
            wt[seg] = _ramp(f)[::-1][seg - s0]
            wt[min(s0 + f, n):] = 0
        out[p.ra * HOP:p.ra * HOP + n] += w * wt
        prev_audio, prev_piece = w, p
    return out


def render(singer, tl: Timeline, voice: Dict, seed: int = 0, model: str = "kalavinka",
           max_s: float = MAX_S) -> Tuple[np.ndarray, List[Dict]]:
    """Render a timeline with ``singer._render`` (melody mode if ``tl.melody``, else score mode; never auto-shift).
    Returns (waveform of exactly ``tl.total x 480`` samples, [{ra, rb, oa, ob, cut, start_s}] per piece)."""
    pieces = plan_pieces(tl, max_s)
    waves = []
    for k, p in enumerate(pieces):
        item = tl.to_item(p.ra, p.rb, index=f"piece{k}")
        if all(t == "1" for t in item["note_type"].split()):
            waves.append(np.zeros((p.rb - p.ra) * HOP, np.float32))
            continue
        f0 = [float(x) for x in item["f0"].split()] if tl.melody else None
        w = singer._render(item, voice, f0, seed + k, shift=False, model=model)
        waves.append(np.asarray(w, np.float32).reshape(-1))
    wav = assemble(tl, pieces, waves)
    info = [{"ra": p.ra, "rb": p.rb, "oa": p.oa, "ob": p.ob, "start_s": round(p.ra / FPS, 3),
             "cut": (p.cut_in or {}).get("kind")} for p in pieces]
    return wav, info


# ------------------------------------------------------------------------------------------- transposition
def lower_median(x: Sequence[float]) -> float:
    """Median as torch.median computes it (the lower of the two middle values)."""
    s = np.sort(np.asarray(x, float))
    return float(s[(len(s) - 1) // 2]) if len(s) else float("nan")


def soulx_shift(prompt_item: Dict, target_item: Dict, target_f0: Optional[Sequence[float]]) -> int:
    """The semitone shift SoulX's ``auto_shift`` would apply (``soulxsinger.infer``): in score mode the rounded
    difference of the median note pitches (prompt - target); in melody mode ``round(12 log2(prompt / target))`` of the
    median voiced F0."""
    if target_f0 is None:
        pt = [int(x) for x in prompt_item["note_pitch"].split() if int(x) >= 1]
        gt = [int(x) for x in target_item["note_pitch"].split() if int(x) >= 1]
        if not pt or not gt:
            return 0
        return int(np.round(lower_median(pt) - lower_median(gt)))
    pf = [float(x) for x in prompt_item["f0"].split() if float(x) > 0]
    gf = [float(x) for x in target_f0 if float(x) > 0]
    if not pf or not gf:
        return 0
    return int(np.round(np.log2(lower_median(pf) / lower_median(gf)) * 12))


# ------------------------------------------------------------------------------------------- the Sing tab
def real_gap(a: Dict, b: Dict, max_gap: float = 1.5) -> Optional[float]:
    """Seconds between two tune clips when ``b`` follows ``a`` in the same song within ``max_gap`` s, else None."""
    ia, ib = a.get("item", a), b.get("item", b)
    if ia.get("song_id") is None or ia.get("song_id") != ib.get("song_id"):
        return None
    ta, tb = ia.get("time"), ib.get("time")
    if not ta or not tb:
        return None
    g = (tb[0] - ta[1]) / 1000.0
    return max(0.0, g) if -0.05 <= g <= max_gap else None


def _make_seamless_singer():
    """Build the subclass lazily, so this module imports without the model code."""
    from .singer import KalavinkaSinger, SingError, default_steps, split_lines

    class SeamlessSinger(KalavinkaSinger):
        """``KalavinkaSinger`` whose ``sing`` puts every line on one timeline (see the module doc)."""

        def sing(self, lyrics: str, tune: Optional[int] = None, voice: Optional[int] = None, mode: str = "score",
                 steps: Optional[int] = None, seed: int = 0, model: str = "kalavinka", join: str = "seamless",
                 line_gap: float = 0.3) -> Tuple[np.ndarray, int, Dict]:
            """Like ``KalavinkaSinger.sing`` (one call at a time); ``join="separate"`` calls the original."""
            with self._lock:
                return self._sing(lyrics, tune, voice, mode, steps, seed, model, join, line_gap)

        def resing(self, *args, **kwargs) -> Dict:
            """``KalavinkaSinger.resing``, one call at a time."""
            with self._lock:
                return super().resing(*args, **kwargs)

        def _sing(self, lyrics, tune, voice, mode, steps, seed, model, join, line_gap):
            """The body of ``sing``."""
            if join == "separate":
                return super().sing(lyrics, tune, voice, mode, steps, seed, model)
            if join not in JOINS:
                raise SingError(f"join must be one of {JOINS}")
            if not 0 <= float(line_gap) <= 10:
                raise SingError("the gap between lines must be 0-10 s")
            from kalavinka.eval.sing_new import set_lyrics
            if mode not in ("score", "expressive"):
                raise SingError("mode must be 'score' (model's own pitch) or 'expressive' (two-stage)")
            lines = split_lines(lyrics)
            if not lines:
                raise SingError("type some lyrics first")
            if tune is not None and not 1 <= int(tune) <= len(self._tunes):
                raise SingError(f"there is no tune {tune} (1-{len(self._tunes)})")
            self.steps = int(steps or default_steps(self.device))
            t0 = time.time()
            tl = Timeline(melody=(mode == "expressive"))
            info = {"lines": [], "mode": mode, "model": model, "device": self.device, "join": join,
                    "line_gap": line_gap}
            prev_tune, song_voice, seed0 = None, None, None
            for k, line in enumerate(lines):
                sylls = self.syllables(line)
                if tune is not None:
                    tu = self._tune((int(tune) - 1 + k) % len(self._tunes) + 1)
                    if tu["syllables"] < len(sylls):
                        raise SingError(f"tune {tu['no']} has {tu['syllables']} syllables but \"{line}\" needs "
                                        f"{len(sylls)}. Pick a longer tune, leave the tune on auto, or shorten the line.")
                else:
                    tu = self.auto_tune(len(sylls), seed + k)
                if song_voice is None:
                    song_voice = self._voice(voice) if voice is not None else self._voice(tu["voice"])
                target = set_lyrics(tu["item"], sylls)
                seed_k = zlib.crc32(f"{line}|{tu['no']}|{seed}".encode()) % (2 ** 31)
                seed0 = seed_k if seed0 is None else seed0
                f0 = self.contour(target, seed_k) if mode == "expressive" else None
                shift = soulx_shift(song_voice["item"], target, f0) if song_voice["no"] != tu["voice"] else 0
                if prev_tune is not None:
                    g = real_gap(prev_tune, tu)
                    tl.append_rest(line_gap if g is None else g)
                a = tl.total
                tl.append_item(target, f0, shift)
                info["lines"].append({"line": line, "syllables": " ".join(s["text"] for s in sylls), "tune": tu["no"],
                                      "tune_song": f"{tu['song']} ({tu['year']})", "voice": song_voice["no"],
                                      "shift": shift, "seen_in_training": round(self.novelty(sylls), 2),
                                      "seconds": round((tl.total - a) / FPS, 2)})
                prev_tune = tu
            tl.finish()
            wav, pieces = render(self, tl, song_voice, seed0 or 0, model)
            info["pieces"] = len(pieces)
            info["cuts"] = [p["cut"] for p in pieces[1:]]
            info["seconds"] = round(len(wav) / SR, 2)
            info["took_s"] = round(time.time() - t0, 1)
            return wav, SR, info

    return SeamlessSinger


def __getattr__(name):
    """``SeamlessSinger`` is created on first access (it needs ``.singer``)."""
    if name == "SeamlessSinger":
        cls = _make_seamless_singer()
        globals()["SeamlessSinger"] = cls
        return cls
    raise AttributeError(name)


# ------------------------------------------------------------------------------------------- the Sargam tab
def strip_edges(item: Dict) -> Dict:
    """Drop leading and trailing rest notes of an item (the timeline adds its own rests)."""
    fields = ("phoneme", "note_pitch", "note_type", "duration", "text")
    cols = {f: item[f].split() for f in fields}
    typ = cols["note_type"]
    a, b = 0, len(typ)
    while a < b and typ[a] == "1":
        a += 1
    while b > a and typ[b - 1] == "1":
        b -= 1
    out = dict(item)
    for f in fields:
        out[f] = " ".join(cols[f][a:b]) if len(cols[f]) == len(typ) else item[f]
    fr = [int(round(float(d) * FPS)) for d in out["duration"].split()]
    out["n_frames"] = sum(fr)
    out["f0"] = " ".join(["0.0"] * out["n_frames"])
    return out


class SeamlessSargamSinger:
    """``SargamSinger`` with every line on one timeline (``join="separate"`` = the original)."""

    def __init__(self, singer):
        """Wrap a loaded singer (``KalavinkaSinger`` or ``SeamlessSinger``)."""
        from .sargam import SargamSinger
        self.s = singer
        self.old = SargamSinger(singer)

    def sing(self, lines, model: str = "kalavinka", voice: Optional[int] = None, tonic="auto", beat_s: float = 0.45,
             syllables: str = "sargam", steps: Optional[int] = None, seed: int = 0, check: bool = True,
             join: str = "seamless", line_gap: float = 0.3) -> Tuple[np.ndarray, int, Dict]:
        """Render all lines as one performance. Returns (wav, 24000, info + pitch report)."""
        from .sargam import SargamError, phrase_item, pitch_report, to_text, tonic_midi
        from .singer import default_steps
        if join == "separate":
            return self.old.sing(lines, model, voice, tonic, beat_s, syllables, steps, seed, check)
        lines = [ln for ln in lines if ln]
        if not lines or not any(n["semis"] is not None for ln in lines for n in ln):
            raise SargamError("nothing to sing (no swaras)")
        vo = self.s._voice(voice if voice is not None else 1)
        sa = int(round(tonic_midi(tonic, vo["item"])))
        self.s.steps = int(steps or default_steps(self.s.device))
        t0 = time.time()
        tl = Timeline(melody=False)
        for k, ln in enumerate(lines):
            if k:
                tl.append_rest(line_gap)
            tl.append_item(strip_edges(phrase_item(ln, sa, beat_s, syllables)))
        tl.finish()
        wav, pieces = render(self.s, tl, vo, seed, model)
        info = {"model": model, "voice": vo["no"], "sa_midi": sa, "beat_s": beat_s, "syllables": syllables,
                "steps": self.s.steps, "device": self.s.device, "sargam": to_text(lines), "join": join,
                "line_gap": line_gap, "pieces": len(pieces), "cuts": [p["cut"] for p in pieces[1:]],
                "took_s": round(time.time() - t0, 1), "seconds": round(len(wav) / SR, 2)}
        if check:
            info["pitch"] = pitch_report(wav, [{"item": tl.to_item(), "start_s": 0.0}])
        return wav, SR, info
