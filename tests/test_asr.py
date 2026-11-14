"""Automatic lyrics (``stages.asr``), phrase-windowed alignment, Latin-script
G2P and lyric scoring (``lyrics_eval``) - offline, with a fake ASR model."""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from saregama.audio import FPS, save_audio  # noqa: E402
from saregama.config import AsrCfg, load_config  # noqa: E402
from saregama.g2p import latin_word, line_to_syllables  # noqa: E402
from saregama.lyrics_eval import error_rates, line_onset_errors, loose_words, parse_lrc  # noqa: E402
from saregama.pipeline import Pipeline, read_songs, safe_syllables  # noqa: E402
from saregama.stages.align import Aligner  # noqa: E402
from saregama.stages.asr import (AsrBackend, clean_text, collapse_loops,  # noqa: E402
                                 gurmukhi_to_devanagari, plan_phrases)
from synth import MMS_LIKE_VOCAB, oracle_emission, synth_song  # noqa: E402
from test_pipeline import OracleEmitter  # noqa: E402

LINES = ["मन मेरा गाए रे", "प्रभु तेरे चरणों में", "सुबह की धूप खिली"]


def test_plan_phrases_merges_and_splits():
    """Short blips join a neighbour, long phrases split at their longest pause."""
    v = np.zeros(40 * FPS, dtype=bool)
    v[50:150] = True                 # 2 s phrase
    v[175:190] = True                # 0.3 s blip 0.5 s later -> merged
    v[400:2000] = True               # 32 s phrase ...
    v[1400:1415] = False             # ... whose only pause (0.3 s, under min_gap) is at 28 s
    v[1700:1705] = False             # a shorter 0.1 s pause at 34 s
    cfg = AsrCfg(min_gap_s=0.4, min_phrase_s=1.0, max_phrase_s=25.0)
    ph = plan_phrases(v, FPS, cfg)
    assert ph[0] == (1.0, 3.8)
    assert len(ph) == 3 and ph[1][0] == 8.0 and ph[2][1] == 40.0
    assert 28.0 <= ph[1][1] <= 28.3            # cut inside the longest pause


def test_clean_text():
    """Loops collapse, boilerplate and impossible speeds are dropped, Gurmukhi converts."""
    assert collapse_loops("ho ho ho ho ho ho ho".split()) == ["ho"] * 4
    assert collapse_loops("tum hi ho tum hi ho".split()) == "tum hi ho tum hi ho".split()
    assert clean_text("Thanks for watching!", 2.0, 22)[1] == "boilerplate"
    assert clean_text("तेरा मेरा रिश्ता " * 10, 1.0, 22)[1].startswith("too_fast")
    t, why = clean_text("Hello, दिल्ली वाली Girlfriend! (music)", 3.0, 22)
    assert why == "" and t == "hello दिल्ली वाली girlfriend"
    assert gurmukhi_to_devanagari("ਤੇਰੀ ਅੱਖ") .startswith("ते")
    assert clean_text("تم ہی ہو", 2.0, 22) == ("", "perso_arabic_script")


def test_latin_g2p():
    """Hinglish words in Latin script get phones and syllables."""
    phs, _ = latin_word("pyaar")
    assert [p for p, _ in phs] == ["p", "y", "aa", "r"]
    assert [p for p, _ in latin_word("baby")[0]][-1] == "ii"
    assert [p for p, _ in latin_word("hello")[0]] == ["h", "ee", "l", "oo"]
    sylls = line_to_syllables("dil tera girlfriend", "hi")
    assert {s.word_index for s in sylls} == {0, 1, 2}
    out, kept, win, bad = safe_syllables(["दिल ★ tera", "★"], [(0, 1), (2, 3)], "hi", None)
    assert kept == ["दिल tera"] and win == [(0, 1)] and bad == ["★", "★"]


def test_phrase_window_alignment_matches_global():
    """Aligning each line inside its phrase window finds the true onsets."""
    s = synth_song("w", "hi", LINES, 150.0, seed=3)
    em = oracle_emission(s, MMS_LIKE_VOCAB, seed=1)
    cfg = load_config(None).align
    cfg.backend = "external"
    al = Aligner(cfg)
    al.emitter = OracleEmitter()
    sylls = [line_to_syllables(l, "hi") for l in LINES]
    f0 = s.gold.f0
    e = np.zeros_like(f0)
    # windows: from each line's first onset to the next line's first onset (s)
    firsts = np.cumsum([0] + [len(x) for x in sylls])[:-1]
    starts = [s.onsets[k] / FPS for k in firsts]
    win = [(a, b) for a, b in zip(starts, starts[1:] + [len(f0) / FPS])]
    out, info = al.align(None, sylls, f0, e, emission=em, windows=win)
    got = [x.start for x in out if x.kind == "lyric"]
    assert info["align_mode"] == "phrase"
    assert np.median(np.abs(np.array(got) - np.array(s.onsets))) <= 3


class FakeAsr(AsrBackend):
    """Returns the synthetic song's lines, one per call, in order."""
    name = "fake"
    model_id = "fake-asr"

    def __init__(self, lines):
        """Remember the lines to hand out."""
        self.lines = list(lines)

    def __call__(self, clips, lang, words=False):
        """One line per clip (cycling), no word times."""
        return [(self.lines[k % len(self.lines)], []) for k in range(len(clips))]


def test_pipeline_runs_asr_when_no_sheet():
    """A songs.csv row without lyrics is transcribed, aligned and flagged."""
    tmp = Path(tempfile.mkdtemp(prefix="asr_"))
    s = synth_song("hi_asr_0001", "hi", LINES, 150.0, seed=2)
    save_audio(tmp / "a.wav", s.audio)
    with open(tmp / "songs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["song_id", "audio", "language", "lyrics", "vocal_only"])
        w.writerow(["hi_asr_0001", "a.wav", "hi", "", "1"])
    cfg = load_config(None, work_dir=str(tmp / "work"))
    cfg.separation.backend = "none"
    cfg.f0.backends = ["yin"]
    cfg.align.backend = "external"
    cfg.g2p.lexicon = []
    cfg.asr.backend = "qwen3"                   # replaced by the fake below
    pipe = Pipeline(cfg)
    pipe._asr = FakeAsr(LINES)
    pipe.aligner.emitter = OracleEmitter()
    row = read_songs(tmp / "songs.csv")[0]
    ann = pipe.run_song(row, force=True, emission=oracle_emission(s, MMS_LIKE_VOCAB, seed=1))
    assert ann.qc["lyrics"]["source"] == "asr" and ann.qc["lyrics"]["model"] == "fake-asr"
    assert "lyrics_asr_unverified" in ann.qc["flags"] and ann.status in ("flagged", "rejected")
    assert ann.syllables and (tmp / "work" / "songs" / "hi_asr_0001" / "lyrics.asr.txt").exists()


def test_lyrics_eval_cross_script():
    """Devanagari and Roman spellings of the same words score as matches."""
    assert loose_words("तुम ही हो") == loose_words("tum hi ho")
    r = error_rates("तेरे बिना क्या वजूद मेरा", "tere bina kya wajood mera")
    assert r["wer"] == 0.0
    r = error_rates("tum hi ho ab tum hi ho", "tum hi ho")
    assert r["del"] == 4 and abs(r["wer"] - 4 / 7) < 1e-9


def test_line_onsets():
    """Line onsets are scored after removing one global offset."""
    lrc = parse_lrc("[00:10.00] tum hi ho\n[00:12.50]\n[00:15.00] ab tum hi ho\n")
    assert lrc == [(10.0, "tum hi ho"), (15.0, "ab tum hi ho")]
    hyp = [("tum", 12.1), ("hi", 12.4), ("ho", 12.6), ("ab", 17.0), ("tum", 17.3)]
    r = line_onset_errors(lrc, hyp)
    assert r["scored"] == 2 and abs(r["offset_s"] - 2.05) < 1e-6 and r["median_abs_s"] <= 0.06


def test_line_onsets_drift_fit():
    """A recording 1 % slower than the LRC scores near zero once drift is fitted."""
    lines = [(10.0 * k, w) for k, w in enumerate(["tum", "hi", "ho", "ab", "dil", "jaan", "mera", "tera"], 1)]
    hyp = [(w, 3.0 + 1.01 * t) for t, w in lines]
    off = line_onset_errors(lines, hyp, fit="offset")
    lin = line_onset_errors(lines, hyp, fit="linear")
    assert off["median_abs_s"] > 0.15 and lin["median_abs_s"] < 0.01 and abs(lin["drift"] - 1.01) < 1e-6
