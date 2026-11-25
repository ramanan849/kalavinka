"""Regression tests for the deploy review fixes (25 Nov 2026) and the seamless Sing / Sargam / Compose paths, with a
fake model (a sine at the requested pitch)."""
import json
import threading

import numpy as np
import pytest

from kalavinka.deploy import compose as C
from kalavinka.deploy import sargam as SG
from kalavinka.deploy import seamless as S
from kalavinka.deploy.compose_ui import SOURCES, visibility
from kalavinka.tests.test_compose import FakeSinger, g2p, one

FPS, HOP = 50, 480


# ------------------------------------------------------------------ compose
def test_smoothing_works_and_meend_does_not_overshoot():
    """The default 30 ms smoothing really ramps a plain join; a meend into the next note never overshoots it."""
    c, _, _ = one([{"swara": "Sa"}, {"swara": "Pa"}], smooth_ms=30)
    j = 50
    assert 0 < c[j - 1] < 700 and 0 < c[j] < 700                       # a ramp, not a step
    c, _, _ = one([{"swara": "Ga", "orn": [{"type": "meend", "to": "Pa", "start": 0.3}]}, {"swara": "Pa"}],
                  smooth_ms=100)
    assert c.max() <= 700.5


def test_normalise_rejects_bad_values_readably():
    """Negative tempo, empty sec, non-numeric beats: ComposeError, never a raw ValueError / TypeError."""
    for bad in ({"tempo_bpm": -80, "phrases": [{"notes": [{"swara": "Sa"}]}]},
                {"phrases": [{"notes": [{"swara": "Sa", "beats": "x"}]}]},
                {"smooth_ms": None, "phrases": [{"notes": [{"swara": "Sa", "sec": None, "beats": "abc"}]}]},
                {"phrases": "notes"}):
        with pytest.raises(C.ComposeError):
            C.normalise(bad)
    assert C.normalise({"phrases": [{"notes": [{"swara": "Sa", "sec": ""}]}]})["phrases"][0]["notes"][0]["beats"] == 1


def test_table_keeps_free_pitches_and_matches_rows_by_id():
    """A midi note survives 'Apply table' at its real tonic; deleting a row keeps hidden data with the right note;
    a changed beats cell wins over the old sec."""
    comp = {"tempo_bpm": 60, "phrases": [{"notes": [
        {"syl": "मा", "phones": ["m", "aa"], "midi": 50, "sec": 0.5},
        {"syl": "री", "phones": ["r", "ii"], "swara": "Ga", "beats": 1,
         "orn": [{"type": "custom", "points": [[0, 0], [1, 30]]}]}]}]}
    sa = 49.0
    rows = C.to_table(comp, sa)
    assert rows[0][3] == "re(k)" and rows[0][4] == 0                    # 50 is one semitone above Sa=49
    back = C.from_table(rows, comp, sa)
    assert back["phrases"][0]["notes"][0].get("midi") == 50
    back = C.from_table([rows[1]], comp, sa)                            # delete row 1: row 2 keeps its own data
    n = back["phrases"][0]["notes"][0]
    assert n["phones"] == ["r", "ii"] and n["orn"][0]["type"] == "custom"
    rows[0][5] = 2                                                      # type beats over an existing sec
    n = C.from_table(rows, comp, sa)["phrases"][0]["notes"][0]
    assert n.get("beats") == 2 and "sec" not in n


def test_set_phrase_lyrics_more_and_fewer_syllables():
    """One syllable per sung note; extra syllables add notes; missing ones hold the last syllable."""
    base = {"phrases": [{"notes": [{"swara": "Sa"}, {"syl": "_"}, {"swara": "Re"}, {"swara": "Ga"}]}]}
    out = C.set_phrase_lyrics(base, 0, g2p("गीत नया है"))
    assert [n["syl"] for n in out["phrases"][0]["notes"]] == ["गीत", "_", "न", "या", "है"]
    out = C.set_phrase_lyrics(base, 0, g2p("गीत नया"))
    assert [n["syl"] for n in out["phrases"][0]["notes"]] == ["गीत", "_", "न", "या"]
    out = C.set_phrase_lyrics({"phrases": [{"notes": [{"swara": "Sa"}, {"swara": "Re"}, {"syl": "_"}, {"swara": "Ga"}]}]},
                              0, g2p("गीत"))
    assert [n["syl"] for n in out["phrases"][0]["notes"]] == ["गीत", "-", "_", "_"]
    with pytest.raises(C.ComposeError):
        C.set_phrase_lyrics(base, 3, g2p("गीत"))


def test_start_from_shows_only_its_own_controls():
    """Each source shows only its controls (the rest are hidden)."""
    assert visibility(SOURCES[1]) == {"tune": True, "lyrics": False, "alankar": False, "sargam": False, "json": False}
    assert sum(visibility(SOURCES[0]).values()) == 0
    assert all(sum(visibility(s).values()) <= 1 for s in SOURCES)


def test_compose_seamless_vs_separate():
    """A 30 s rest-free phrase: separate inserts gaps (longer output), seamless keeps the exact length and no
    silence inside the phrase."""
    comp = {"voice": 1, "tempo_bpm": 120, "phrases": [{"notes": [{"syl": "", "swara": ["Sa", "Re", "Ga"][k % 3]}
                                                                  for k in range(60)]}]}
    cs = C.ComposeSinger(FakeSinger())
    w1, sr, i1 = cs.sing(comp, check=False, join="seamless")
    w2, _, i2 = cs.sing(comp, check=False, join="separate")
    assert i1["pieces"] >= 3 and set(i1["cuts"]) == {"overlap"}
    assert len(w1) / sr == pytest.approx(30 + 2 * C.EDGE_REST, abs=0.05) and len(w2) > len(w1)
    r = np.sqrt(np.mean(w1[:len(w1) // HOP * HOP].reshape(-1, HOP) ** 2, axis=1))
    assert r[15:-15].min() > 0.05


# ------------------------------------------------------------------ sargam / tonic
def test_tonic_text_hz_and_bad_beats():
    """'138.6' typed in a box is Hz; 0 Hz and zero-length notes are refused with a SargamError."""
    assert SG.tonic_midi("138.6", {}) == pytest.approx(69 + 12 * np.log2(138.6 / 440))
    with pytest.raises(SG.SargamError):
        SG.tonic_midi(0.0, {})
    with pytest.raises(SG.SargamError):
        SG.parse_sargam("Sa Pa:0 Re")
    with pytest.raises(SG.SargamError):
        SG.from_json({"lines": [[{"swara": "Sa", "beats": -1}]]})
    it = SG.phrase_item(SG.parse_sargam("Sa Re Ga ma Pa Dha Ni Sa' Ni Dha Pa ma Ga Re")[0], 60, 0.45)
    assert it["n_frames"] == round((14 * 0.45 + 2 * SG.EDGE_REST) * FPS)   # no drift over 14 notes


# ------------------------------------------------------------------ seamless singers with a fake model
class FakeSeamless:
    """A SeamlessSinger without the model: real ``sing`` logic, fake G2P / tunes / render."""

    @staticmethod
    def make():
        cls = S.SeamlessSinger
        s = cls.__new__(cls)
        s.device, s.steps, s._lock, s._seen, s._cflow, s._base = "cpu", 16, threading.RLock(), set(), None, None
        item = lambda song, t0, t1, p: {"song_id": song, "time": [t0, t1], "phoneme": " ".join(["in_aa"] * 8),
                                         "note_pitch": " ".join([str(p)] * 8), "note_type": " ".join(["2"] * 8),
                                         "duration": " ".join(["0.4"] * 8), "text": " ".join(["आ"] * 8),
                                         "f0": " ".join(["220"] * 160), "n_frames": 160}
        s._tunes = [{"no": 1, "song": "A", "year": 2000, "syllables": 8, "seconds": 3.2, "voice": 1, "item": item("a", 0, 3200, 57)},
                    {"no": 2, "song": "A", "year": 2000, "syllables": 8, "seconds": 3.2, "voice": 1, "item": item("a", 3400, 6600, 57)},
                    {"no": 3, "song": "B", "year": 2010, "syllables": 8, "seconds": 3.2, "voice": 2, "item": item("b", 0, 3200, 64)}]
        s._voices = [{"no": 1, "item": {"note_pitch": "57 57", "f0": "220 220", "tonic_hz": 220.0}},
                     {"no": 2, "item": {"note_pitch": "64 64", "f0": "330 330", "tonic_hz": 330.0}}]
        s.syllables = lambda line: [{"text": "आ", "phones": ["aa"]} for _ in line.split()]
        s.calls = []
        fake = FakeSinger()

        def render(item, voice, f0, seed, shift, model="kalavinka"):
            s.calls.append(item)
            if f0 is None:
                durs = np.diff(np.concatenate([[0], np.round(np.cumsum([float(d) for d in item["duration"].split()]) * FPS)]))
                f0 = np.concatenate([[0.0 if t == "1" else 440 * 2 ** ((int(p) - 69) / 12)] * int(d) for p, t, d in
                                     zip(item["note_pitch"].split(), item["note_type"].split(), durs)])
            return fake._render(item, voice, f0, seed, shift, model)
        s._render = render
        return s


def test_seamless_sing_real_gap_line_gap_and_transpose():
    """Lines from adjacent clips of one song get the real gap (0.2 s); other lines the line gap; a line from another
    singer's tune is moved into the song voice's register by SoulX's own rule; one render for short songs."""
    s = FakeSeamless.make()
    wav, sr, info = s.sing("आ आ आ आ | आ आ आ आ | आ आ आ आ", tune=1, mode="score", line_gap=0.5)
    assert [l["tune"] for l in info["lines"]] == [1, 2, 3]
    assert [l["shift"] for l in info["lines"]] == [0, 0, -7]          # tune 3 (median 64) into voice 1 (57)
    assert info["pieces"] == 1 and len(s.calls) == 1
    expect = 0.2 + 3 * 1.6 + 0.2 + 0.5 + 0.2                           # edges, three lines, real gap 0.2, line gap 0.5
    assert len(wav) / sr == pytest.approx(expect, abs=0.03)


def test_seamless_sargam_one_timeline():
    """All alankar lines on one timeline: no chunking, line gaps as rests, pitch report over the whole take."""
    s = FakeSeamless.make()
    ss = S.SeamlessSargamSinger(s)
    lines = SG.alankar_lines(2)
    wav, sr, info = ss.sing(lines, voice=1, tonic="C4", beat_s=0.3, line_gap=0.25, check=True)
    n = sum(len(l) for l in lines)
    expect = 2 * SG.EDGE_REST + n * 0.3 + 0.25 * (len(lines) - 1)
    assert len(wav) / sr == pytest.approx(expect, abs=0.05)
    assert info["pitch"]["within_50c"] > 0.95 and info["pieces"] <= 2
