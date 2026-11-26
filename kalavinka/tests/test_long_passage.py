"""``eval.long_passage``: passage grouping and sample-exact clip offsets on the passage timeline."""
from kalavinka.eval.long_passage import EDGE_S, FPS, group_passages, passage_timeline, pick_prompt


def clip(i, t0_ms, sec):
    """A one-note export item of ``sec`` seconds starting at ``t0_ms`` in the song."""
    return {"index": f"s_c{i:03d}", "time": [t0_ms, t0_ms + int(sec * 1000)], "n_frames": int(sec * FPS),
            "phoneme": "<SP> k_a", "note_pitch": "0 60", "note_type": "1 2", "duration": f"0.20 {sec - 0.2:.2f}",
            "text": "_ क"}


def test_grouping_respects_gap_and_length():
    """A gap over max_gap or a passage over max_s starts a new passage; short passages are dropped."""
    cs = [clip(0, 0, 8), clip(1, 9000, 8), clip(2, 18000, 8), clip(3, 40000, 8), clip(4, 49000, 4)]
    ps = group_passages(cs, max_gap=3, min_s=18, max_s=28)
    assert [[c["index"] for c in p] for p in ps] == [["s_c000", "s_c001", "s_c002"]]
    # 3 x 8 s + 2 x 1 s gaps + edges = 26.4 s fits; a fourth clip would not
    ps = group_passages(cs[:3], max_gap=3, min_s=10, max_s=20)
    assert [len(p) for p in ps] == [2]


def test_offsets_are_exact():
    """Each clip's frames sit at its recorded offset; the gap between clips is the real one."""
    cs = [clip(0, 0, 4), clip(1, 5500, 3)]
    tl, starts = passage_timeline(cs)
    assert starts[0] == int(EDGE_S * FPS)
    assert starts[1] == starts[0] + 4 * FPS + int(1.5 * FPS)
    assert tl.total >= starts[1] + 3 * FPS + int(EDGE_S * FPS)
    item = tl.to_item(starts[1], starts[1] + 3 * FPS)
    assert item["phoneme"].split()[-1] == "k_a"


def test_prompt_not_inside_passage():
    """The prompt is taken from a clip whose evaluation prompt lies outside the passage."""
    cs = [clip(0, 0, 4), clip(1, 5000, 4)]
    assert pick_prompt(cs, {"s_c000": "s_c001", "s_c001": "s_c009"}) == "s_c009"
