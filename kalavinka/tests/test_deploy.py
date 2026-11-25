"""Tests for kalavinka.deploy (no GPU, no model weights needed).

Covers device choice, the Devanagari check, line splitting, the novelty hash, the command-line shortcut
(``sing "lyrics"`` without the word sing), tune selection rules, and, when a built bundle exists at
``<PROJECT>/bundles``, that both zips carry the files they need and nothing private.
"""
import json
import zipfile
from pathlib import Path

import pytest

from kalavinka.deploy import cli, history, sargam, singer
from kalavinka.deploy.build_bundle import PROJECT, pretty_song, register

BUNDLES = PROJECT / "bundles"


def test_devanagari_check_and_split():
    """Hindi passes, Hinglish and mixed text do not; | and new lines split phrases."""
    assert singer.is_devanagari("ठंडी हवा में गीत नया")
    assert not singer.is_devanagari("thandi hawa")
    assert not singer.is_devanagari("ठंडी hawa")
    assert singer.split_lines("एक | दो\n\nतीन ") == ["एक", "दो", "तीन"]


def test_pick_device_explicit_and_steps():
    """An explicit device is kept; CPU gets fewer flow steps."""
    assert singer.pick_device("cpu") == "cpu"
    assert singer.pick_device("auto") in ("cuda", "mps", "cpu")
    assert singer.default_steps("cpu") == 16 and singer.default_steps("cuda") == 32


def test_span_hash_is_order_sensitive():
    """The novelty hash depends on the syllables and their order."""
    a = singer.span_hash(["ते", "री", "या", "दें"])
    assert a == singer.span_hash(["ते", "री", "या", "दें"]) and len(a) == 8
    assert a != singer.span_hash(["री", "ते", "या", "दें"])


def test_auto_tune_rules():
    """Auto-tune picks a tune with n..n+4 syllables, else the next longer one, else explains the problem."""
    s = singer.KalavinkaSinger.__new__(singer.KalavinkaSinger)
    s._tunes = [{"no": 1, "syllables": 6}, {"no": 2, "syllables": 9}, {"no": 3, "syllables": 20}]
    assert s.auto_tune(5)["no"] in (1, 2)
    assert s.auto_tune(12)["no"] == 3
    with pytest.raises(singer.SingError):
        s.auto_tune(25)


def test_cli_bare_lyrics_become_sing(monkeypatch, capsys, tmp_path):
    """``tunes`` works without a model, and a bare lyric is routed to the sing command."""
    seen = {}

    class Fake:
        """Stands in for the singer so no model is loaded."""
        def __init__(self, bundle, device="auto"):
            seen["init"] = True
            self.device = "cpu"

        def sing(self, lyrics, tune, voice, mode, steps, seed, model="kalavinka", join="seamless", line_gap=0.3):
            seen["args"] = (lyrics, tune, voice, mode, join)
            raise singer.SingError("stop here")
    from kalavinka.deploy import seamless
    monkeypatch.setattr(seamless, "SeamlessSinger", Fake, raising=False)
    with pytest.raises(SystemExit):
        cli.main(["--device", "cpu", "--outputs", str(tmp_path), "ठंडी हवा", "--tune", "2"])
    assert seen["args"] == ("ठंडी हवा", 2, None, "score", "seamless")


def test_names():
    """Song ids become readable names; the register follows the median note."""
    assert pretty_song("1991_tu_pyaar_hai_kisi_aur_ka__98MGT9JStKk") == "Tu Pyaar Hai Kisi Aur Ka"
    assert register({"note_pitch": "50 52 0 53"}) == "low" and register({"note_pitch": "67 69 70"}) == "high"


@pytest.mark.parametrize("name,extra", [("kalavinka_colab", ["Kalavinka_Colab.ipynb"]),
                                        ("kalavinka_mac", ["setup_mac.sh", "sing", "MAC.md"])])
def test_built_zip_contents(name, extra):
    """A built zip has the core, its own extras, a consistent tune library, and no private files or base weights."""
    z = BUNDLES / f"{name}.zip"
    if not z.exists():
        pytest.skip("bundles not built")
    with zipfile.ZipFile(z) as zf:
        names = set(zf.namelist())
        need = ["kalavinka_app/singer.py", "kalavinka_app/sargam.py", "kalavinka_app/history.py", "kalavinka_app/__main__.py", "weights/kalavinka_v4l/best.pt",
                "weights/contour_cflow/best.pt", "data/tunes.jsonl", "data/voices.jsonl", "data/phone_set_indic.json",
                "vendor/kalavinka/eval/render.py", "vendor/saregama/g2p.py",
                "vendor/SoulX-Singer/soulxsinger/models/soulxsinger.py", "requirements.txt", "NOTICE.md"] + extra
        for f in need:
            assert f"{name}/{f}" in names, f
        assert not any(n.endswith(("key.json", "model.pt")) or "responses" in n for n in names)
        tunes = [json.loads(l) for l in zf.read(f"{name}/data/tunes.jsonl").decode().splitlines() if l.strip()]
        voices = {json.loads(l)["no"] for l in zf.read(f"{name}/data/voices.jsonl").decode().splitlines() if l.strip()}
        for t in tunes:
            assert t["voice"] in voices
            assert f"{name}/data/{t['item']['wav']}" in names


def test_sargam_notation():
    """Swara names, komal/tivra, octave marks, holds and rests parse to the right semitones and beats."""
    lines = sargam.parse_sargam("Sa re(k) Ga ma Ma(t) Pa:2 - _ dha(k) Ni Sa' .Ni\nSa'' ")
    semis = [n["semis"] for n in lines[0]]
    assert semis == [0, 1, 4, 5, 6, 7, None, 8, 11, 12, -1]
    assert lines[0][5]["beats"] == 3.0 and lines[1][0]["semis"] == 24
    for bad in ("Xa", "Sa(t)", "Pa(k)", "- Sa"):
        with pytest.raises(sargam.SargamError):
            sargam.parse_sargam(bad)


def test_builtin_alankars_and_json():
    """Built-in alankar 2 is Sa Re Ga, Re Ga ma, ... up to Dha Ni Sa' and back; JSON gives the same notes."""
    up, down = sargam.alankar_lines(2)
    assert [n["semis"] for n in up[:6]] == [0, 2, 4, 2, 4, 5] and [n["semis"] for n in up[-3:]] == [9, 11, 12]
    assert [n["semis"] for n in down[:3]] == [12, 11, 9] and down[-1]["semis"] == 0
    assert sargam.to_text([up[:3]]) == "Sa Re Ga"
    js = {"tonic": "C#3", "lines": [[{"swara": "Sa"}, {"swara": "rest"}, {"swara": "Ga", "octave": 1, "beats": 2}]]}
    lines, st = sargam.from_json(js)
    assert [n["semis"] for n in lines[0]] == [0, None, 16] and lines[0][2]["beats"] == 2 and st["tonic"] == "C#3"
    assert sargam.tonic_midi("C#3", {}) == 49 and abs(sargam.tonic_midi("auto", {"tonic_hz": 220.0}) - 57) < 1e-6
    assert all(len(c) <= sargam.MAX_NOTES for c in sargam.chunk(sargam.alankar_lines(3)))


def test_phrase_item_is_consistent():
    """A phrase item has edge rests, one note per swara, and frame counts that add up."""
    it = sargam.phrase_item(sargam.parse_sargam("Sa Re _ Dha")[0], 60, 0.5)
    assert it["phoneme"].split() == ["<SP>", "in_s-a", "in_r-ee", "<SP>", "in_dh-a", "<SP>"]
    assert it["note_pitch"].split() == ["0", "60", "62", "0", "69", "0"]
    assert it["note_type"].split() == ["1", "2", "2", "1", "2", "1"]
    frames = sum(round(float(d) * 50) for d in it["duration"].split())
    assert frames == it["n_frames"] == len(it["f0"].split())


def test_history_cache(tmp_path):
    """The same request is made once and then reused; another seed is a new take; history lists both."""
    import numpy as np
    h = history.History(tmp_path)
    calls = []

    def make():
        calls.append(1)
        return np.zeros(2400, np.float32), 24000, {"took_s": 0.1}
    req = {"lyrics": "ठंडी हवा", "model": "base", "seed": 0}
    *_, hit1 = h.cached("sing", req, make, "a")
    *_, hit2 = h.cached("sing", dict(req), make, "a")
    *_, hit3 = h.cached("sing", {**req, "seed": 1}, make, "a")
    *_, hit4 = h.cached("sing", req, make, "a", fresh=True)
    assert (hit1, hit2, hit3, hit4) == (False, True, False, False) and len(calls) == 3
    assert [r["id"] for r in h.recent()] == [3, 2, 1] and "base" in history.describe(h.recent()[0])
