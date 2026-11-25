"""Tests for kalavinka.deploy.compose (no GPU, no model): ornament shapes, composition -> SoulX items, the table
view, starting points, the editor view and the follow check."""
import json

import numpy as np
import pytest

from kalavinka.deploy import compose as C
from kalavinka.deploy.history import History

FPS = C.FPS


def g2p(text):
    """The repo's own G2P, as the singer uses it."""
    from saregama.g2p import line_to_syllables
    return [{"text": s.text, "phones": list(s.phones)} for s in line_to_syllables(text, "hi")]


def one(notes, **kw):
    """Curve of a single phrase at Sa = MIDI 60, no smoothing unless asked."""
    comp = C.normalise({"tempo_bpm": 60, "smooth_ms": kw.pop("smooth_ms", 0), "phrases": [{"notes": notes, **kw}]})
    return C.phrase_curve(comp["phrases"][0], 60.0, comp)


def test_plain_notes_step_curve_and_rests():
    """Swaras land on their cents; rests are unvoiced; one beat at 60 bpm = 50 frames."""
    c, frames, bases = one([{"swara": "Sa"}, {"syl": "_"}, {"swara": "Pa"}, {"swara": "ga(k)", "cents": -20}])
    assert frames == [50, 50, 50, 50] and bases[1] is None
    assert np.allclose(c[:50], 0) and np.isnan(c[50:100]).all() and np.allclose(c[100:150], 700)
    assert np.allclose(c[150:], 280)


def test_meend_reaches_target_and_gamak_rate():
    """A meend ends on its target; a gamak oscillates at its rate between the note and the neighbour."""
    c, _, _ = one([{"swara": "Ga", "beats": 2, "orn": [{"type": "meend", "to": "Pa", "start": 0.5}]}])
    assert abs(c[0] - 400) < 1 and abs(c[-1] - 700) < 15 and abs(c[49] - 400) < 1
    c, _, _ = one([{"swara": "Pa", "beats": 4, "orn": [{"type": "gamak", "to": "Dha", "rate": 5}]}])
    assert c.min() >= 699 and c.max() <= 901
    spec = np.abs(np.fft.rfft(c - c.mean()))
    assert abs(np.fft.rfftfreq(len(c), 1 / FPS)[spec.argmax()] - 5) < 0.3


def test_vibrato_kan_and_aliases():
    """Vibrato peaks at its rate and depth; kan touches its grace note first; Carnatic names map to ours."""
    c, _, _ = one([{"swara": "Sa", "beats": 4, "orn": [{"type": "vibrato", "rate": 6, "depth": 30, "delay": 0}]}])
    spec = np.abs(np.fft.rfft(c - c.mean()))
    assert abs(np.fft.rfftfreq(len(c), 1 / FPS)[spec.argmax()] - 6) < 0.3 and 25 < c.max() <= 30.01
    c, _, _ = one([{"swara": "Ga", "orn": [{"type": "kan", "from": "Pa", "time": 0.08}]}])
    assert np.allclose(c[:4], 700) and np.allclose(c[5:], 400)
    a, _, _ = one([{"swara": "Ga", "orn": [{"type": "kampita", "to": "Pa"}]}])
    b, _, _ = one([{"swara": "Ga", "orn": [{"type": "gamak", "to": "Pa"}]}])
    assert np.allclose(a, b)
    with pytest.raises(C.ComposeError):
        one([{"swara": "Ga", "orn": [{"type": "wobble"}]}])


def test_up_down_follow_the_raag():
    """'up' is the next swara of the raag (Yaman: Ga -> tivra Ma)."""
    assert C.neighbour(400, C.RAAGS["Yaman"], 1) == 600
    assert C.neighbour(400, C.RAAGS["Bhoopali"], 1) == 700
    assert C.neighbour(0, C.RAAGS["Bhoopali"], -1) == -300


def test_strokes_override_the_curve():
    """A hand-drawn stroke replaces the curve over its time span only."""
    c, _, _ = one([{"swara": "Sa", "beats": 2}], f0_edits=[[[0.5, 100], [1.0, 300]]])
    assert np.allclose(c[:24], 0) and abs(c[25] - 100) < 1 and abs(c[50] - 300) < 1 and np.allclose(c[51:], 0)


def test_plan_items_line_up():
    """Tokens, pitches, types and durations line up; '-' is a slur with the same phoneme; f0 length = frames."""
    p = C.plan(C.EXAMPLE, 60.0, g2p)
    assert len(p) == 1
    it = p[0]["item"]
    toks, pit, typ, dur = (it[k].split() for k in ("phoneme", "note_pitch", "note_type", "duration"))
    assert len(toks) == len(pit) == len(typ) == len(dur) == 11
    assert typ[4] == "3" and toks[4] == toks[3]
    assert typ[6] == "1" and toks[6] == "<SP>" and pit[6] == "0"
    assert len(it["f0"].split()) == it["n_frames"] == sum(round(float(d) * FPS) for d in dur)
    f0 = np.array(it["f0"].split(), float)
    assert (f0[:10] == 0).all() and (f0[-10:] == 0).all() and (f0[f0 > 0] >= C.F0_MIN).all()


def test_lyrics_get_g2p_phones_and_lone_consonants_keep_a():
    """A Devanagari syllable gets its phones from the G2P; empty syl sings the swara name."""
    assert C.syllable_phones("रा", "Sa", g2p) == ("रा", ["r", "aa"])
    assert C.syllable_phones("", "Dha")[1] == ["dh", "a"]
    assert C.syllable_phones("म", None, g2p)[1][-1] == "a"


def test_long_phrases_are_split():
    """A 30-beat phrase at 60 bpm is cut into pieces of at most 12 s, never before a slur."""
    notes = [{"swara": "Sa", "syl": "", "beats": 1} if i % 3 else {"swara": "Re", "syl": "-" if i else "", "beats": 1}
             for i in range(30)]
    comp = {"tempo_bpm": 60, "phrases": [{"notes": notes}]}
    p = C.plan(comp, 60.0, g2p)
    assert len(p) >= 3
    for q in p:
        assert q["item"]["n_frames"] <= (12 + 2 * C.EDGE_REST) * FPS + 1
        assert q["item"]["note_type"].split()[1] != "3"


def test_from_tune_keeps_the_real_curve():
    """Importing a tune with its real pitch reproduces that curve (within a cent) and its phones."""
    f0 = [0.0] * 5 + [220.0 * 2 ** (0.3 * np.sin(k / 3) / 12) for k in range(40)] + [0.0] * 5
    tune = {"no": 9, "song": "x", "year": 2000, "voice": 1, "item": {
        "phoneme": "in_m-aa <SP> in_r-ii", "text": "मा _ री", "note_pitch": "57 0 57", "note_type": "2 1 2",
        "duration": "0.30 0.20 0.50", "f0": " ".join(map(str, [220.0] * 15 + [0.0] * 10 + f0[:25])),
        "tonic_hz": 220.0}}
    comp = C.from_tune(tune)
    it = C.plan(comp, 57.0, g2p)
    assert comp["phrases"][0]["notes"][0]["phones"] == ["m", "aa"]
    got = np.array(it[0]["item"]["f0"].split(), float)[10:-10]
    want = np.array(tune["item"]["f0"].split(), float)[:len(got)]
    m = (got > 0) & (want > 0)
    assert np.abs(1200 * np.log2(got[m] / want[m])).max() < 2


def test_table_round_trip_and_errors():
    """The table view goes back to the same notes; bad input gives a readable error."""
    rows = C.to_table(C.EXAMPLE)
    back = C.from_table(rows, C.EXAMPLE)
    assert C.to_table(back) == rows
    assert back["phrases"][0]["notes"][2]["orn"] == [{"type": "meend", "to": "Ma(t)", "start": 0.6}]
    with pytest.raises(C.ComposeError):
        C.normalise('{"phrases": [{"notes": [{"syl": "-", "swara": "Sa"}]}]}')
    with pytest.raises(C.ComposeError):
        C.normalise({"phrases": [{"notes": [{"swara": "Xa"}]}]})
    with pytest.raises(C.ComposeError):
        C.normalise("not json")


def test_starting_points_and_view():
    """Sargam, alankars and lyrics become compositions; the view carries notes, curve and ornament defaults."""
    a = C.from_sargam(alankar=2)
    assert a["phrases"][0]["notes"][2]["swara"] == "Ga"
    s = C.from_sargam("Sa Re _ Ga:2")
    assert [n["syl"] for n in s["phrases"][0]["notes"]] == ["", "", "_", ""]
    ly = C.from_lyrics([g2p("गीत नया")], "Ga Pa")
    assert [n["swara"] for n in ly["phrases"][0]["notes"]] == ["Ga", "Pa", "Pa"]
    v = C.view(C.EXAMPLE, 60.0)
    assert len(v["phrases"][0]["notes"]) == 9 and "gamak" in v["ornaments"] and v["taal"] == 16
    assert len(v["phrases"][0]["curve"]) == sum(round(n["d"] * FPS) for n in v["phrases"][0]["notes"])
    json.dumps(v)


def test_prompt_lists_every_ornament():
    """The LLM prompt documents every ornament type."""
    for k in C.ORNAMENTS:
        assert k in C.COMPOSE_PROMPT


class FakeSinger:
    """Stands in for KalavinkaSinger: returns a sine at the requested F0 instead of running SoulX."""
    device = "cpu"
    steps = 16

    def _voice(self, no):
        return {"no": int(no), "item": {"tonic_hz": 220.0}}

    def syllables(self, line):
        return g2p(line)

    def _render(self, item, voice, f0, seed, shift, model="kalavinka"):
        hz = np.repeat(np.asarray(f0, float), C.SR // FPS)
        ph = 2 * np.pi * np.cumsum(hz) / C.SR
        return (0.3 * np.sin(ph) * (hz > 0)).astype(np.float32)


def test_compose_singer_follow_check_and_cache(tmp_path):
    """A perfect singer follows the curve (pYIN within 50 c almost everywhere); takes are cached in history."""
    cs = C.ComposeSinger(FakeSinger())
    comp = {"voice": 1, "tempo_bpm": 90, "phrases": [{"notes": [
        {"syl": "", "swara": "Sa"}, {"syl": "", "swara": "Pa", "beats": 2, "orn": [{"type": "vibrato"}]}]}]}
    wav, sr, info = cs.sing(comp, check=True)
    assert sr == C.SR and info["pieces"] == 1
    assert info["follow"]["within_50c"] > 0.9 and info["sung"][0]["phrase"] == 0
    h = History(tmp_path)
    req = {"comp": C.normalise(comp), "model": "kalavinka", "seed": 0, "steps": 16}
    calls = []
    make = lambda: (calls.append(1), cs.sing(comp, check=False))[1]
    h.cached("compose", req, make, "x")
    *_, hit = h.cached("compose", req, make, "x")
    assert hit and len(calls) == 1
