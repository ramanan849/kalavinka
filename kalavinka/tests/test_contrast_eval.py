"""Tests for kalavinka.eval.contrast_eval (CPU, no model download)."""
import numpy as np

from kalavinka.eval.contrast_eval import lyric_chars, pooled, score_clip, contrast_tests


def test_contrast_buckets():
    """ठ is a retroflex aspirate: place + aspiration tests, both also counted as held-out."""
    t = dict(contrast_tests("ठ"))
    assert t["place_retro"] == "थ" and t["asp_asp"] == "ट"
    assert t["held_place"] == "थ" and t["held_asp"] == "ट"
    assert dict(contrast_tests("त")) == {"place_dental": "ट", "asp_plain": "थ"}
    assert contrast_tests("म") == []


def test_lyric_chars_nfd_and_cleanup():
    """<SP>, spaces and Latin are dropped; precomposed nukta letters are split."""
    s = lyric_chars("पढ़ <SP> ab ना")
    assert " " not in s and "a" not in s and "़" in s


def test_score_clip_ratio():
    """The score is p(target)/(p(target)+p(partner)) averaged over the aligned frames."""
    vocab = {"ठ": 0, "थ": 1, "ट": 2, "ा": 3}
    p = np.array([[0.6, 0.2, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
    c = score_clip(np.log(p), "ठा", vocab, [[0], [1]])
    assert c["place_retro_n"] == 1 and abs(c["place_retro_sum"] - 0.75) < 1e-6
    assert abs(c["asp_asp_sum"] - 0.6 / 0.7) < 1e-6 and c["held_place_win"] == 1
    s = pooled([c])
    assert s["place_retro"]["mean"] == 0.75 and s["place_dental"]["n"] == 0
