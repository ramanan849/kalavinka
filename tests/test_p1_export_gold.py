"""P1 export (bands, splits, note cutting, SoulX frame check) and gold scoring."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saregama.p1.export import accept_band, check_item, clip_notes, split_of  # noqa: E402
from saregama.p1.gold import best_window, score_song  # noqa: E402
from saregama.review import write_textgrid  # noqa: E402
from saregama.schema import NoteAnn, SongAnnotation  # noqa: E402


def test_bands():
    """All checks pass -> accept; slightly off -> review; far off -> reject."""
    f = {"good_src_share": 0.9, "asr1_share": 0.05, "mean_conf": -1.5, "f0_agree": 0.9,
         "unlabelled_voiced": 0.0}
    assert accept_band(f, {}) == "accept"
    assert accept_band({**f, "good_src_share": 0.75}, {}) == "review"
    assert accept_band({**f, "good_src_share": 0.3}, {}) == "reject"
    assert accept_band({**f, "mean_conf": None}, {}) == "reject"


def test_splits_deterministic():
    """Same key, same split; roughly 3 % test."""
    assert split_of("ref:1") == split_of("ref:1")
    s = [split_of(f"ref:{i}") for i in range(4000)]
    assert 0.01 < s.count("test") / 4000 < 0.05 and s.count("train") > 3500


def test_clip_notes_tile_exactly():
    """Notes cut to the clip with rests filling the edges sum to the clip length."""
    ann = SongAnnotation(song_id="x", language="hi", notes=[NoteAnn(10, 30, 2, 0, 60), NoteAnn(30, 50, 3, 0, 62),
                                                           NoteAnn(60, 90, 2, 1, 64)])
    ns = clip_notes(ann, 20, 80)
    assert ns[0].start == 20 and ns[-1].end == 80
    assert sum(n.end - n.start for n in ns) == 60
    assert [n.type for n in ns] == [2, 3, 1, 2]


def test_check_item():
    """SoulX frame arithmetic is enforced."""
    good = {"duration": "0.20 0.3001", "n_frames": 25, "f0": " ".join(["0"] * 25), "text": "a b",
            "phoneme": "in_a in_b", "note_pitch": "60 62", "note_type": "2 2"}
    assert check_item(good) is None
    assert "durations" in check_item({**good, "n_frames": 26, "f0": " ".join(["0"] * 26)})


def test_best_window():
    """The window with the most syllables is chosen."""
    sylls = [{"start": s} for s in range(5000, 6000, 10)]
    assert 1500 <= best_window(sylls, 10000, 3750) <= 5000


def test_score_song():
    """A perfect prediction scores 0 WER and 100 % onsets; edits are counted."""
    d = Path(tempfile.mkdtemp())
    tiers = [("words", [(0.0, 0.5, "तुम"), (0.5, 1.0, "ही"), (1.0, 1.6, "हो")]),
             ("syllables", [(0.0, 0.5, "तुम"), (0.5, 1.0, "ही"), (1.0, 1.6, "हो")]),
             ("vocables", [(2.0, 3.0, "आ")]),
             ("singers", [(0.0, 3.0, "A")])]
    write_textgrid(d / "g.TextGrid", 4.0, tiers)
    write_textgrid(d / "p.TextGrid", 4.0, tiers)
    r = score_song(d / "g.TextGrid", d / "p.TextGrid")
    assert r["wer"] == 0 and r["onset_within_50ms"] == 1.0 and r["vocable_recall"] == 1.0
    assert r["single_voice_precision"] == 1.0 and r["singer_purity"] == 1.0
    pred = [("words", [(0.0, 0.5, "tum"), (0.5, 1.6, "हो")]), ("syllables", [(0.08, 0.5, "x")]),
            ("vocables", []), ("singers", [(0.0, 3.0, "A")])]
    write_textgrid(d / "p2.TextGrid", 4.0, pred)
    r = score_song(d / "g.TextGrid", d / "p2.TextGrid")
    assert abs(r["wer"] - 1 / 3) < 1e-3 and r["vocable_recall"] == 0.0
