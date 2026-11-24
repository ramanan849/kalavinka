"""Tests for kalavinka.ops.jobq (submit, eligibility, atomic claim, status)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("jobq", Path(__file__).resolve().parents[1] / "ops" / "jobq.py")
jobq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobq)


def test_submit_and_claim_once(tmp_path: Path):
    """A job is claimed by exactly one worker; the second claim gets None."""
    f = jobq.submit(tmp_path, "a", "true")
    assert jobq.next_job(tmp_path, ["sx"]) == f
    assert jobq.claim(tmp_path, f) is not None
    assert jobq.claim(tmp_path, f) is None
    assert jobq.next_job(tmp_path, ["sx"]) is None


def test_needs_and_after_gate_jobs(tmp_path: Path):
    """A p1 job is invisible to an sx-only worker; an 'after' job waits for its path."""
    jobq.submit(tmp_path, "p1job", "true", needs="p1")
    gate = tmp_path / "DONE"
    f = jobq.submit(tmp_path, "later", "true", after=str(gate))
    assert jobq.next_job(tmp_path, ["sx"]) is None
    gate.touch()
    assert jobq.next_job(tmp_path, ["sx"]) == f
    assert "p1job" in jobq.status(tmp_path)


def test_delay_gives_other_node_first_pick(tmp_path: Path):
    """A delayed worker skips a fresh job; an undelayed one takes it."""
    f = jobq.submit(tmp_path, "fresh", "true")
    assert jobq.next_job(tmp_path, ["sx"], delay=300) is None
    assert jobq.next_job(tmp_path, ["sx"], delay=0) == f
