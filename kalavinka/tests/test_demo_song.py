"""Tests for kalavinka.eval.demo_song (ordering, common clips, stitching)."""
from pathlib import Path

import numpy as np

from kalavinka.eval.demo_song import clip_number, common_clips, stitch


def test_common_clips_in_song_order(tmp_path: Path):
    """Only clips every system has, ordered by clip number (not string order)."""
    items = {f"song_c{n:03d}": {} for n in (2, 10, 3)}
    items["other_c001"] = {}
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    for n in (2, 3, 10):
        (a / f"song_c{n:03d}.wav").touch()
    for n in (2, 10):
        (b / f"song_c{n:03d}.wav").touch()
    assert common_clips("song", items, [a, b]) == ["song_c002", "song_c010"]
    assert clip_number("x_y_c017") == 17


def test_stitch_gaps_and_starts():
    """Clips are joined with the gap between them (not after the last) and start times are right."""
    sr = 10
    track, starts = stitch([np.ones(10, np.float32), np.ones(5, np.float32)], sr, 0.5)
    assert len(track) == 10 + 5 + 5 and starts == [0.0, 1.5]
    assert track[10:15].sum() == 0
