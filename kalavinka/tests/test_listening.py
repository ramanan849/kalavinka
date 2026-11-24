"""Tests for the listening-test builder and analysis (CPU)."""
import json
import random

import numpy as np

from kalavinka.eval.listening.analyze import bootstrap, screen, table
from kalavinka.eval.listening.build import build_manifest, normalise, select_clips, stim_id


def test_manifest_is_blind():
    """The page manifest holds opaque ids only: no condition or clip names."""
    m = build_manifest(["clipA", "clipB"], ["score", "i1", "real"], "salt", {"clipA": "x"})
    blob = json.dumps(m)
    for word in ("score", "i1", "real", "clipA", "clipB", "salt"):
        assert word not in blob
    assert all(len(t["stimuli"]) == 3 for t in m["trials"])
    assert stim_id("s", "c", "a") != stim_id("s", "c", "b") and stim_id("s", "c", "a") != stim_id("t", "c", "a")


def test_select_clips_seeded_and_bounded():
    """Selection respects the duration window and the seed."""
    cand = {f"c{i}": 2 + i * 0.5 for i in range(30)}
    a = select_clips(cand, 5, 4, 10, 0)
    assert a == select_clips(cand, 5, 4, 10, 0) and all(4 <= cand[c] <= 10 for c in a)


def test_normalise_level_and_peak():
    """Active RMS goes to the target; the peak never exceeds -1 dBFS."""
    y = np.concatenate([np.zeros(4800), 0.01 * np.sin(np.arange(48000) / 5)]).astype(np.float32)
    z = normalise(y, -20)
    act = z[4800:]
    assert abs(20 * np.log10(np.sqrt((act ** 2).mean())) + 20) < 0.5
    assert np.abs(normalise(np.r_[np.zeros(9600), np.ones(480)], -3)).max() <= 10 ** (-1 / 20) + 1e-6


def test_analysis_recovers_condition_gap_and_screens():
    """A built-in +1 gap between conditions is found; a listener who rates the real clip low is flagged."""
    key = {"conditions": ["a", "b", "real"], "stimuli": {}}
    for c in range(8):
        for cond in key["conditions"]:
            key["stimuli"][f"{c}{cond}"] = {"clip": f"c{c}", "cond": cond}
    rng = random.Random(0)
    resp = []
    for li in range(6):
        r = {}
        for c in range(8):
            base = rng.choice([2, 3])
            r[f"{c}a"], r[f"{c}b"] = base, base + 1
            r[f"{c}real"] = 1 if li == 5 else 5
        resp.append({"ratings": r})
    rows = table(key, resp)
    flagged, _ = screen(rows, 3.0)
    assert flagged == [5]
    res = bootstrap(rows, [("a", "b")], n=300)
    assert res["pairs"]["b-a"]["mean"] == 1.0 and res["pairs"]["b-a"]["significant"]
    assert res["listeners"] == 6 and res["clips"] == 8
    assert table(key, [{"ratings": {"0a": 3}}]) == []          # incomplete responses are skipped


def test_build_id_changes_with_salt():
    """A rebuild (new salt) gets a new build id, so the page starts a fresh draft."""
    a = build_manifest(["clipA"], ["score", "real"], "s1", {})
    b = build_manifest(["clipA"], ["score", "real"], "s2", {})
    assert a["build"] != b["build"] and a["build"] == build_manifest(["clipA"], ["score", "real"], "s1", {})["build"]


def test_comments_resolve_to_clip_not_condition():
    """Per-page comments map to the clip of that page; the overall comment has no clip."""
    from kalavinka.eval.listening.analyze import comments
    key = {"stimuli": {"s1": {"clip": "c1", "cond": "i1"}, "s2": {"clip": "c1", "cond": "real"}}}
    resp = [{"initials": "AB", "stimOrder": {"t00": ["s2", "s1"]},
             "comments": {"t00": " C sounds robotic ", "_overall": "nice", "t01": "  "}}]
    out = comments(key, resp)
    assert {"listener": 0, "initials": "AB", "clip": "c1", "text": "C sounds robotic"} in out
    assert any(c["clip"] is None and c["text"] == "nice" for c in out) and len(out) == 2
