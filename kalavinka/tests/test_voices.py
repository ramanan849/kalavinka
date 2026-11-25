"""Tests for voice profiles (kalavinka.deploy.voices, voices_build): loading with and without profiles, local edits,
filters, name lookup, the credits-first gender rule, register terciles, balanced picking, and the auto-tonic fold."""
import json

import numpy as np
import pytest

from kalavinka.deploy import voices as V
from kalavinka.deploy import voices_build as VB
from kalavinka.deploy.sargam import fold_tonic


def bundle(tmp_path, profiles=True):
    """A tiny bundle: 3 voices (+ profiles)."""
    d = tmp_path / "b" / "data"
    d.mkdir(parents=True)
    rows = []
    for no, (song, year, tonic, f0) in enumerate([("Song A", 1982, 335.0, 196.0), ("Song B", 1993, 200.0, 400.0),
                                                  ("Song C", 2017, 150.0, 210.0)], 1):
        rows.append({"no": no, "song": song, "year": year, "register": "mid",
                     "item": {"wav": f"wavs/{no}.wav", "tonic_hz": tonic, "f0": " ".join([str(f0)] * 10 + ["0"] * 3)}})
    (d / "voices.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    if profiles:
        spec = {"voices": [{"no": 1, "gender": "male", "gender_conf": 0.95, "gender_from": "credits", "register": "low",
                            "decade": "1980s", "singer": {"name": "Kishore Kumar", "source": "credited", "confirmed": False}},
                           {"no": 2, "gender": "female", "gender_conf": 0.85, "register": "high", "decade": "1990s",
                            "singer": {"name": "Lata Mangeshkar", "source": "credited"}},
                           {"no": 3, "gender": "male", "register": "mid", "decade": "2010s", "singer": None}]}
        (d / "voice_profiles.json").write_text(json.dumps(spec), encoding="utf-8")
    return tmp_path / "b"


def test_load_filter_resolve_and_label(tmp_path):
    """Profiles merge in; filters and name lookup work; the label carries gender, register, era and name."""
    ps = V.load(bundle(tmp_path), tmp_path / "out")
    assert ps[0]["label"] == "Voice 1 · male · low · 1980s · Kishore Kumar"
    assert [p["no"] for p in V.filter_profiles(ps, gender="female")] == [2]
    assert [p["no"] for p in V.filter_profiles(ps, era="2010s")] == [3]
    assert V.resolve(ps, "lata") == 2 and V.resolve(ps, "7") == 7 and V.resolve(ps, "Voice 3 · male") == 3
    with pytest.raises(ValueError):
        V.resolve(ps, "nobody")
    assert "estimated" in V.card(ps[0]) and "Song A" in V.card(ps[0])


def test_old_bundle_without_profiles(tmp_path):
    """No voice_profiles.json: gender 'unsure', nothing breaks."""
    ps = V.load(bundle(tmp_path, profiles=False), None)
    assert ps[0]["gender"] == "unsure" and ps[0]["label"].startswith("Voice 1 · mid · 1980s")


def test_local_edits_win_and_survive(tmp_path):
    """Edits go to outputs/voice_profiles.local.json and override the bundle profile."""
    b, out = bundle(tmp_path), tmp_path / "out"
    V.save_edit(out, 3, name="Rahul Mishra", notes="clear tone")
    p = V.load(b, out)[2]
    assert p["name"] == "Rahul Mishra" and p["edited"] and p["name_source"] == "your edit"
    with pytest.raises(ValueError):
        V.save_edit(out, 3, gender="robot")
    with pytest.raises(ValueError):
        V.save_edit(out, 3, colour="red")


def test_auto_tonic_is_folded_into_the_voice_range(tmp_path):
    """Voice 1's dataset tonic (335 Hz) is an octave off for a voice singing around 196 Hz: Sa becomes 167.5 Hz."""
    assert fold_tonic(335.0, 196.0) == pytest.approx(167.5)
    assert fold_tonic(150.0, 210.0) == pytest.approx(150.0)
    assert V.load(bundle(tmp_path), None)[0]["sa_hz"] == pytest.approx(167.5)


def test_gender_from_credits():
    """Credits decide (the classifier is unreliable on singing); duets split by pitch; contradictions -> pitch."""
    solo = {"singers": [{"name": "Arijit Singh", "gender": "male"}], "confident": True}
    g = VB.gender_and_name(277.0, 0.998, solo)                  # classifier says female, wrongly
    assert g["gender"] == "male" and g["name"] == "Arijit Singh" and g["gender_from"] == "credits"
    duet = {"singers": [{"name": "Kishore Kumar", "gender": "male"}, {"name": "Asha Bhosle", "gender": "female"}]}
    assert VB.gender_and_name(196.0, 0.5, duet)["name"] == "Kishore Kumar"
    assert VB.gender_and_name(380.0, 0.5, duet)["name"] == "Asha Bhosle"
    many = {"singers": [{"name": "A", "gender": "male"}, {"name": "B", "gender": "male"}, {"name": "C", "gender": "female"}]}
    g = VB.gender_and_name(200.0, 0.1, many)
    assert g["name"] is None and g["name_options"] == ["A", "B"]
    assert VB.gender_and_name(408.0, 0.99, {"singers": [{"name": "X", "gender": "male"}]})["gender"] == "female"
    assert VB.gender_and_name(150.0, 0.01, None)["gender"] == "male"         # no credits: classifier + pitch


def test_registers_and_balanced_pick():
    """Registers are terciles within a gender; picking fills the emptiest decade x gender cell, one voice per song."""
    singers = {}
    for k, (y, g, f) in enumerate([(1985, "male", 150), (1985, "male", 200), (1995, "female", 300), (1995, "female", 450),
                                   (2015, "male", 250), (2015, "female", 380), (2015, "female", 390)]):
        singers[f"s{k}"] = {"singer_id": f"s{k}", "song_id": f"{y}_song{k}", "year": y, "gender": g, "f0_med": f,
                            "best": {"conf": "0", "f0": "200 0"}, "clips": [{}]}
    VB.registers(singers)
    assert [singers[k]["register"] for k in ("s0", "s1", "s4")] == ["low", "mid", "high"]
    chosen = VB.pick(singers, ["s0"], 4)
    assert chosen[0]["singer_id"] == "s0" and len(chosen) == 4
    cells = {(VB.decade(s["year"]), s["gender"]) for s in chosen}
    assert len(cells) == 4                                        # four different cells before any repeats
