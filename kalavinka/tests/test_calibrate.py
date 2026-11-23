"""Tests for kalavinka.annot.calibrate: per-clip gold scoring and the threshold search."""
import json
from pathlib import Path

from kalavinka.annot import calibrate as C
from saregama.review import write_textgrid


def _syl(text, start, end, phrase, word, word_text, kind="lyric"):
    """One syllables.json record."""
    return {"kind": kind, "text": text, "start": start, "end": end, "phrase": phrase, "word": word,
            "word_text": word_text, "conf": -1.0, "src": "ref+asr", "phones": ["a"]}


def _feat(mean_conf, good_src=1.0):
    """Clip features with the given confidence and source share."""
    return {"dur_s": 2.0, "good_src_share": good_src, "asr1_share": 0.0, "ref_backed_share": 1.0,
            "mean_conf": mean_conf, "min_conf": mean_conf, "f0_agree": 0.95, "unlabelled_voiced": 0.0,
            "vocables": 0, "identity": "claimed"}


def test_score_clip_counts_word_and_onset_errors():
    """A wrong word and a 100 ms-late onset are counted; overlap from AB singers."""
    sylls = [_syl("दिल", 0, 20, 0, 0, "दिल"), _syl("तू", 25, 40, 0, 1, "तू"), _syl("जान", 50, 70, 0, 2, "जान")]
    tiers = {"words": [(0.0, 0.4, "दिल"), (0.5, 0.8, "मेरा"), (1.0, 1.4, "जान")],
             "syllables": [(0.0, 0.4, "दिल"), (0.4, 0.8, "मे"), (1.0, 1.4, "जान")],
             "singers": [(0.0, 0.8, "A"), (0.8, 1.4, "AB")]}
    s = C.score_clip(tiers, 0, sylls, 0, 75)
    assert s["gold_words"] == 3 and s["word_errors"] == 1
    assert s["gold_onsets"] == 3 and s["onsets_50ms"] == 2          # 'मे' at 20 vs pred 25 is 100 ms off
    assert 0.35 < s["overlap_share"] < 0.45


def test_search_picks_threshold_that_meets_targets():
    """Clips with low confidence are wrong; the search must cut them and keep the good ones."""
    good = [C.ClipScore("s", k, 0, 100, _feat(-1.5), gold_words=10, word_errors=0, gold_onsets=10, onsets_50ms=10)
            for k in range(5)]
    bad = [C.ClipScore("s", 10 + k, 0, 100, _feat(-3.5), gold_words=10, word_errors=5, gold_onsets=10, onsets_50ms=3)
           for k in range(5)]
    q, best, _ = C.search(good + bad, 0.95, 0.9)
    assert best["ok"] and -3.5 < q["min_mean_conf"] <= -1.5
    assert best["clips"] == 5


def test_collect_reads_gold_layout(tmp_path: Path):
    """collect() finds corrected files in gold/pack/corrected too and honours reviewed flags."""
    gold, work = tmp_path / "gold", tmp_path / "work"
    sid = "2000_x__abc"
    (gold / "pack" / sid).mkdir(parents=True)
    (gold / "pack" / "corrected").mkdir(parents=True)
    (gold / "pack" / sid / "meta.json").write_text(json.dumps({"start_frame": 100, "end_frame": 200, "fps": 50}))
    write_textgrid(gold / "pack" / "corrected" / f"{sid}.TextGrid", 2.0,
                   [("words", [(0.0, 0.4, "दिल")]), ("syllables", [(0.0, 0.4, "दिल")]), ("vocables", []),
                    ("singers", [(0.0, 0.4, "A")])])
    (gold / ".annotator").mkdir()
    (gold / ".annotator" / "state.json").write_text(json.dumps({sid: {"listened_pct": 100}}))
    sd = work / "songs" / sid
    sd.mkdir(parents=True)
    (sd / "syllables.json").write_text(json.dumps({"syllables": [_syl("दिल", 100, 120, 0, 0, "दिल")]}))
    (sd / "clips.json").write_text(json.dumps({"clips": [{"start": 100, "end": 150, "features": _feat(-1.0)}]}))
    cl = C.collect(work, gold)
    assert len(cl) == 1 and cl[0].reviewed and cl[0].word_errors == 0 and cl[0].onsets_50ms == 1
    res = C.calibrate(work, gold, tmp_path / "out")
    assert (tmp_path / "out" / "report.md").exists() and res["gold_songs"] == [sid]


def test_conf_track_marks_rests_unlabelled_and_syllables():
    """conf_track: syllable log-probs on their frames, 0 on unvoiced gaps, -99 on unlabelled voiced frames."""
    import numpy as np
    from kalavinka.annot.export_v1 import REST, UNLABELLED, conf_track
    f0 = np.array([0, 0, 200, 200, 200, 200, 0, 0, 210, 210], dtype=np.float32)
    sylls = [{"kind": "lyric", "start": 2, "end": 4, "conf": -1.5},
             {"kind": "vocable", "start": 8, "end": 10, "conf": 0.5}]
    c = conf_track(sylls, f0, 0, 10)
    assert list(c[:2]) == [REST, REST] and list(c[2:4]) == [-1.5, -1.5]
    assert list(c[4:6]) == [UNLABELLED, UNLABELLED] and abs(c[8] - np.log(0.5)) < 1e-5
