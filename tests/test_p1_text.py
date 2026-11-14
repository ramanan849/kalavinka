"""P1 text stages without models: local alignment, reconciliation rules,
English G2P, reference parsing, vocable classification and clip planning."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saregama.p1.lyrics_ref import clean_title, coverage, local_align, ref_lines, sim_matrix  # noqa: E402
from saregama.p1.text_stages import (classify_vocalise, english_syllables, plan_clips,  # noqa: E402
                                     reconcile_phrase, word_lang)


def test_clean_title():
    """Upload decorations are removed."""
    assert clean_title("Raataan Lambiyan Lyrical") == "Raataan Lambiyan"
    assert clean_title("Akh Lad Jaave (alternate upload)") == "Akh Lad Jaave"
    assert clean_title("Yeh Ishq Hai (Lyrical)") == "Yeh Ishq Hai"


def test_local_alignment_partial_and_cross_script():
    """A half line in Devanagari matches the romanised reference in the middle."""
    ref = "hum tere bin ab reh nahi sakte tere bina kya wajood mera".split()
    q = "तेरे बिना क्या वजूद".split()
    S = sim_matrix(q, ref)
    score, path = local_align(S)
    assert [b for a, b in path if a is not None and b is not None] == [7, 8, 9, 10]
    assert coverage([q, ["xyz"]], ref) == 4 / 5


def test_reconcile_rules():
    """Matched words take reference spelling, ASR repeats survive, English stays
    Latin, unmatched-but-agreed ASR words are kept, missed reference words are optional."""
    ref = ["कैसे", "तुझसे", "दिल", "ना", "लगाए", "कोई", "party", "शुरू"]
    q = ["कैसे", "तुझसे", "दिलना", "लगाए", "कोई"]          # ASR joins दिल ना
    words, ratio = reconcile_phrase(q, ["कैसे", "तुझसे", "दिल", "लगाए"], ref)
    texts = [w["text"] for w in words]
    assert texts[:2] == ["कैसे", "तुझसे"] and ratio >= 0.8
    assert all(w["src"] in ("ref+asr", "ref+whisper", "ref_only", "asr+whisper", "asr1", "disputed")
               for w in words)
    words, _ = reconcile_phrase(["पार्टी", "शुरू"], [], ref)
    assert words[0]["text"] == "party" and words[0]["lang"] == "en"
    words, ratio = reconcile_phrase(["ला", "ला", "ला"], ["ला", "ला"], ref)
    assert ratio == 0 and [w["src"] for w in words] == ["asr+whisper"] * 3
    words, _ = reconcile_phrase(["कैसे", "दिल", "ना"], [], ref)
    assert any(w["optional"] and w["text"] == "तुझसे" for w in words)


def test_english_g2p():
    """CMUdict words get Indic phones with Indian-English t/d, w as v."""
    assert word_lang("party") == "en" and word_lang("pyaar") == "hi" and word_lang("दिल") == "hi"
    ph = [p for _, s in english_syllables("party") for p in s]
    assert ph == ["p", "aa", "r", "T", "ii"]
    assert english_syllables("weekend")[0][1][0] == "v"


def test_ref_lines():
    """Synced lyrics give timed lines; plain lyrics untimed ones."""
    assert ref_lines({"syncedLyrics": "[00:01.50] a b\n[00:03.00]\n"}) == [(1.5, "a b")]
    assert ref_lines({"plainLyrics": "x\n\ny"}) == [(None, "x"), (None, "y")]


def test_vocable_classification():
    """Continuous pitch right after a syllable is melisma; a separate span gets
    the vowel the emissions favour."""
    vocab = {"-": 0, "a": 1, "e": 2, "i": 3, "o": 4, "u": 5, "m": 6}
    logp = np.full((200, 7), -9.0)
    logp[100:150, 4] = -0.1                          # 'o' dominates 100-150
    hz = np.full(200, 220.0)
    prev = {"start": 40, "end": 60}
    assert classify_vocalise({"start": 61, "end": 90}, prev, hz, logp, vocab)["kind"] == "melisma"
    c = classify_vocalise({"start": 100, "end": 150}, prev, hz, logp, vocab)
    assert c["kind"] == "vocable" and c["vowel"] == "o"


def test_plan_clips():
    """Clips cut at rests, respect one singer and the single-voice mask."""
    n = 2000
    hz = np.full(n, 200.0)
    single = np.ones(n, bool)
    cluster = np.zeros(n, int)
    sylls = [{"start": s, "end": s + 20} for s in range(0, 600, 25)]        # 0-12 s, no rests
    sylls += [{"start": s, "end": s + 20} for s in range(700, 1000, 25)]    # after a 2 s rest
    clips = plan_clips(sylls, single, cluster, hz)
    assert clips == [(0, 24), (24, len(sylls))]
    cluster[750:800] = 1                                                    # singer change inside clip 2
    assert plan_clips(sylls, single, cluster, hz) == [(0, 24)]
