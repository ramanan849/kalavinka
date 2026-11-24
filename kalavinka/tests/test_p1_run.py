"""Tests for kalavinka.ops.p1_run: the stage-marker ``status`` clobber fix."""
from kalavinka.ops.p1_run import fix_markers, patch


def test_fix_markers_renames_outcome():
    """A marker whose status is a stage outcome becomes done + outcome; others untouched."""
    marks = {"identify": {"status": "claimed", "version": 1, "config": "x"},
             "align": {"status": "done", "version": 1, "config": "y"},
             "asr": {"status": "failed", "version": 1, "config": "z", "fails": 1}}
    assert fix_markers(marks) == 1
    assert marks["identify"]["status"] == "done" and marks["identify"]["outcome"] == "claimed"
    assert marks["align"] == {"status": "done", "version": 1, "config": "y"}
    assert marks["asr"]["status"] == "failed"


def test_patched_mark_done_keeps_done(tmp_path):
    """After patch(), a stat named status no longer overwrites the done marker."""
    patch()
    from saregama.p1.state import SongState
    st = SongState(tmp_path)
    st.mark_done("identify", 1, "cfg", 1.0, status="relabeled", coverage=0.9)
    assert st.is_done("identify", 1, "cfg")
    assert st.get("identify")["outcome"] == "relabeled"
