"""Tests for kalavinka.deploy (no GPU, no model weights needed).

Covers device choice, the Devanagari check, line splitting, the novelty hash, the command-line shortcut
(``sing "lyrics"`` without the word sing), tune selection rules, and, when a built bundle exists at
``<PROJECT>/bundles``, that both zips carry the files they need and nothing private.
"""
import json
import zipfile
from pathlib import Path

import pytest

from kalavinka.deploy import cli, singer
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


def test_cli_bare_lyrics_become_sing(monkeypatch, capsys):
    """``tunes`` works without a model, and a bare lyric is routed to the sing command."""
    seen = {}

    class Fake:
        """Stands in for the singer so no model is loaded."""
        def __init__(self, bundle, device="auto"):
            seen["init"] = True

        def sing(self, lyrics, tune, voice, mode, steps, seed):
            seen["args"] = (lyrics, tune, voice, mode)
            raise singer.SingError("stop here")
    monkeypatch.setattr(singer, "KalavinkaSinger", Fake)
    with pytest.raises(SystemExit):
        cli.main(["--device", "cpu", "ठंडी हवा", "--tune", "2"])
    assert seen["args"] == ("ठंडी हवा", 2, None, "score")


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
        need = ["kalavinka_app/singer.py", "kalavinka_app/__main__.py", "weights/kalavinka_v4l/best.pt",
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
