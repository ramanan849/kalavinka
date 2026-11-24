"""Tests for kalavinka.eval.compare."""
from kalavinka.eval.compare import paired_bootstrap, stat


def test_stat_specs():
    """Mean, capped mean and ratio of sums."""
    rows = [{"cer": 0.2, "x_sum": 1.0, "x_n": 2}, {"cer": 3.0, "x_sum": 3.0, "x_n": 2}]
    assert abs(stat(rows, "cer") - 1.6) < 1e-9
    assert abs(stat(rows, "cer@1") - 0.6) < 1e-9
    assert stat(rows, "x_sum/x_n") == 1.0


def test_paired_bootstrap_detects_shift_and_not_noise():
    """A constant shift is significant; identical systems give a CI around 0."""
    a = {str(i): {"index": str(i), "v": float(i % 7)} for i in range(60)}
    b = {k: dict(r, v=r["v"] + 0.5) for k, r in a.items()}
    va, vb, d, lo, hi, n = paired_bootstrap(a, b, "v", n=500)
    assert abs(d - 0.5) < 1e-9 and lo > 0 and n == 60
    _, _, d0, lo0, hi0, _ = paired_bootstrap(a, a, "v", n=500)
    assert d0 == 0 and lo0 <= 0 <= hi0
