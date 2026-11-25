"""Tests for kalavinka.deploy.seamless (no model): timeline building, cut choice, exact placement, crossfades,
gain match, SoulX's transposition rule, real gaps, and the seamless Sing / Sargam / Compose paths."""
import numpy as np
import pytest

from kalavinka.deploy import seamless as S

FPS, HOP, SR = S.FPS, S.HOP, S.SR


def item(notes, f0=None):
    """notes: [(token, midi, type, seconds)] -> export-style item (melody F0 from the pitches unless given)."""
    toks, pit, typ, dur = zip(*notes)
    fr = np.diff(np.concatenate([[0], np.round(np.cumsum(dur) * FPS)])).astype(int)
    hz = []
    for p, t, f in zip(pit, typ, fr):
        hz += [0.0 if t == 1 else 440 * 2 ** ((p - 69) / 12)] * int(f)
    return {"phoneme": " ".join(toks), "note_pitch": " ".join(map(str, pit)), "note_type": " ".join(map(str, typ)),
            "duration": " ".join(f"{d:.2f}" for d in dur), "text": " ".join("x" for _ in toks),
            "f0": " ".join(f"{x:.1f}" for x in (hz if f0 is None else f0)), "n_frames": int(fr.sum())}


class Fake:
    """A 'singer' that renders a sine at the item's F0 (melody) or note pitch (score); length = frames x 480."""
    device, steps = "cpu", 16

    def __init__(self, gain_per_call=1.0):
        self.calls, self.g = [], gain_per_call

    def _render(self, it, voice, f0, seed, shift, model="kalavinka"):
        assert shift is False
        self.calls.append(it)
        if f0 is None:
            fr = np.diff(np.concatenate([[0], np.round(np.cumsum([float(d) for d in it["duration"].split()]) * FPS)])).astype(int)
            f0 = []
            for p, t, f in zip(it["note_pitch"].split(), it["note_type"].split(), fr):
                f0 += [0.0 if t == "1" else 440 * 2 ** ((int(p) - 69) / 12)] * int(f)
        hz = np.repeat(np.asarray(f0, float), HOP)
        ph = 2 * np.pi * np.cumsum(hz) / SR
        amp = 0.3 * self.g ** (len(self.calls) - 1)
        return (amp * np.sin(ph) * (hz > 0)).astype(np.float32)


def rms_frames(w):
    """RMS per 50 fps frame."""
    n = len(w) // HOP
    return np.sqrt(np.mean(w[:n * HOP].reshape(n, HOP) ** 2, axis=1))


def test_timeline_exact_frames_rest_merge_and_long_notes():
    """Cumulative rounding keeps the total; rests merge; a 10 s note becomes slurs; finish() adds edge rests."""
    tl = S.Timeline(melody=True)
    tl.append_item(item([("in_s-a", 60, 2, 0.45)] * 3 + [("<SP>", 0, 1, 0.3)]))
    tl.append_rest(0.5)
    tl.append_item(item([("in_aa", 62, 2, 10.0)]))
    tl.finish()
    rest = int(round(1.65 * FPS)) - int(round(1.35 * FPS)) + 25      # cumulative rounding, then 0.5 s merged
    assert tl.notes[4].typ == 1 and tl.notes[4].frames == rest
    long = [n for n in tl.notes if n.tok == "in_aa"]
    assert len(long) >= 2 and long[0].typ == 2 and all(n.typ == 3 for n in long[1:])
    assert sum(n.frames for n in long) == 500
    assert tl.notes[0].typ == 1 and tl.notes[-1].typ == 1
    assert len(tl.f0_array()) == tl.total


def test_rests_stay_silent_and_length_is_exact():
    """Lines joined by rests: output length = timeline x 480, rests silent for their length, no silence elsewhere."""
    tl = S.Timeline(melody=True)
    for k in range(6):                                   # 6 lines of 5 s, 0.3 s rests: 31.5 s, needs cuts
        if k:
            tl.append_rest(0.3)
        tl.append_item(item([("in_s-a", 60 + k, 2, 0.5)] * 10))
    tl.finish()
    wav, pieces = S.render(Fake(), tl, {}, 0)
    assert len(wav) == tl.total * HOP and len(pieces) >= 3
    assert all(p["rb"] - p["ra"] <= S.MAX_S * FPS for p in pieces)
    assert all(p["cut"] == "rest" for p in pieces[1:])
    r = rms_frames(wav)
    st = tl.starts
    for i, n in enumerate(tl.notes):
        seg = r[st[i] + 1:st[i + 1] - 1]
        if n.typ == 1:
            assert seg.max() < 1e-3
        else:
            assert seg.min() > 0.1                       # no dip inside sung notes (seams included)


def test_overlap_cut_inside_held_note_pitch_continuous():
    """40 s of rest-free singing is cut by overlaps: pieces <= 14 s, never starting on a slur, fades inside held notes,
    no level dip and no pitch jump at the seams."""
    notes = []
    for k in range(40):
        notes.append((f"in_t{k % 7}", 60 + k % 5, 2, 0.6))
        notes.append((f"in_t{k % 7}", 60 + k % 5, 3, 0.4))    # a slur on every syllable
    tl = S.Timeline(melody=True).append_item(item(notes)).finish()
    fake = Fake()
    wav, pieces = S.render(fake, tl, {}, 0)
    assert len(wav) == tl.total * HOP
    assert len(pieces) >= 3 and all(p["cut"] == "overlap" for p in pieces[1:])
    assert all(p["rb"] - p["ra"] <= S.MAX_S * FPS for p in pieces)
    assert all(c["note_type"].split()[0] in ("1", "2") for c in fake.calls)
    r = rms_frames(wav)
    st = tl.starts
    sung = np.concatenate([np.arange(st[i] + 1, st[i + 1] - 1) for i, n in enumerate(tl.notes) if n.typ != 1])
    assert r[sung].min() > 0.15                         # equal-power fade of identical renders: no dip
    for p in pieces[1:]:
        c = p["oa"]
        j = int(np.searchsorted(st, c, side="right") - 1)
        assert tl.notes[j].typ != 1 and st[j] < c < st[j + 1]


def test_gain_match_evens_out_level_steps():
    """If each render is half as loud as the one before, the seam still has no level jump."""
    notes = [(f"in_t{k % 5}", 62, 2, 0.8) for k in range(40)]
    tl = S.Timeline(melody=True).append_item(item(notes)).finish()
    wav, pieces = S.render(Fake(gain_per_call=0.5), tl, {}, 0)
    r = rms_frames(wav)
    for p in pieces[1:]:
        c = p["oa"]
        before, after = r[c - 8:c - 3].mean(), r[c + 3:c + 8].mean()
        assert 0.8 < after / before < 1.25


def test_rest_preferred_and_long_rests_always_cut():
    """A rest in the window beats an overlap; a >= 1 s rest is cut even when everything would fit in one render."""
    tl = S.Timeline(melody=True)
    tl.append_item(item([("in_aa", 60, 2, 0.5)] * 18)).append_rest(0.2)
    tl.append_item(item([("in_aa", 60, 2, 0.5)] * 14)).finish()
    p = S.plan_pieces(tl)
    assert p[1].cut_in["kind"] == "rest"
    tl = S.Timeline(melody=True)
    tl.append_item(item([("in_aa", 60, 2, 0.5)] * 4)).append_rest(1.5)
    tl.append_item(item([("in_aa", 60, 2, 0.5)] * 4)).finish()
    p = S.plan_pieces(tl)
    assert len(p) == 2 and p[1].cut_in["kind"] == "rest"


def test_score_mode_renders_with_note_pitches():
    """Score mode: no F0 passed, the timeline item carries zeros, cuts prefer even short rests."""
    tl = S.Timeline(melody=False)
    for k in range(5):
        if k:
            tl.append_rest(0.08)
        tl.append_item(item([("in_aa", 60, 2, 0.5)] * 8))
    tl.finish()
    wav, pieces = S.render(Fake(), tl, {}, 0)
    assert len(wav) == tl.total * HOP and all(p["cut"] == "rest" for p in pieces[1:])


def test_soulx_shift_matches_the_formula():
    """Score mode: round(lower median prompt pitch - lower median target pitch); melody mode: on the F0 medians."""
    prompt = {"note_pitch": "0 60 62 64 65", "f0": "0 220 220 233 0"}
    target = {"note_pitch": "0 55 57 59 60 0"}
    assert S.soulx_shift(prompt, target, None) == 62 - 57          # lower medians: 62 and 57
    assert S.soulx_shift(prompt, target, [110.0, 110.0, 0.0]) == 12
    assert S.lower_median([1, 2, 3, 4]) == 2


def test_real_gap_only_for_adjacent_clips_of_one_song():
    """The gap between two tune clips counts only when the second follows the first in the same song."""
    a = {"item": {"song_id": "s", "time": [1000, 4000]}}
    assert S.real_gap(a, {"item": {"song_id": "s", "time": [4300, 6000]}}) == pytest.approx(0.3)
    assert S.real_gap(a, {"item": {"song_id": "s", "time": [9000, 9900]}}) is None
    assert S.real_gap(a, {"item": {"song_id": "t", "time": [4300, 6000]}}) is None


def test_strip_edges():
    """Edge rests go, inner rests stay."""
    it = item([("<SP>", 0, 1, 0.2), ("in_aa", 60, 2, 0.5), ("<SP>", 0, 1, 0.3), ("in_aa", 60, 2, 0.5), ("<SP>", 0, 1, 0.2)])
    out = S.strip_edges(it)
    assert out["note_type"] == "2 1 2" and out["n_frames"] == 65
