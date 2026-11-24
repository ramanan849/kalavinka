"""Tests for kalavinka.eval.sing_new (putting new syllables on a borrowed tune)."""
import pytest

from kalavinka.eval.sing_new import n_syllables, phoneme_token, set_lyrics, syllable_groups

TUNE = {"index": "t_c001", "text": "री आ आ <SP> है ह", "phoneme": "in_r-ii in_aa in_aa <SP> in_h-E in_h-a",
        "note_type": "2 2 3 1 2 2", "note_pitch": "60 62 64 0 65 67", "duration": "0.20 0.30 0.10 0.40 0.20 0.20",
        "n_frames": 70, "f0": ""}


def test_groups_keep_slurs_with_their_syllable():
    """A type-3 note joins the syllable before it; a rest is its own group."""
    assert syllable_groups(TUNE) == [[0], [1, 2], [3], [4], [5]]
    assert n_syllables(TUNE) == 4


def test_set_lyrics_replaces_first_syllables_and_keeps_tune():
    """New syllables take the first groups' notes (slur repeated), inner rest kept, rest of tune dropped."""
    out = set_lyrics(TUNE, [{"text": "चाँ", "phones": ["c", "aa", "nas"]}, {"text": "द", "phones": ["d", "a"]},
                            {"text": "की", "phones": ["k", "ii"]}])
    assert out["text"] == "चाँ द द <SP> की"
    assert out["phoneme"].split()[:3] == [phoneme_token(["c", "aa", "nas"]), "in_d-a", "in_d-a"]
    assert out["note_pitch"] == "60 62 64 0 65" and out["note_type"] == "2 2 3 1 2"
    assert out["n_frames"] == 10 + 15 + 5 + 20 + 10 == len(out["f0"].split())


def test_tune_too_short_is_refused():
    """A line with more syllables than the tune is an error, not a silent truncation."""
    with pytest.raises(ValueError):
        set_lyrics(TUNE, [{"text": "x", "phones": ["k", "a"]}] * 5)
