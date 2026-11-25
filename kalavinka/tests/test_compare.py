"""Tests for kalavinka.eval.compare."""
from kalavinka.eval.compare import holm, paired_bootstrap, stat


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
    va, vb, d, lo, hi, n, p, g = paired_bootstrap(a, b, "v", n=500, unit="clip")
    assert abs(d - 0.5) < 1e-9 and lo > 0 and n == 60 and p < 0.01 and g == 60
    _, _, d0, lo0, hi0, _, p0, _ = paired_bootstrap(a, a, "v", n=500)
    assert d0 == 0 and lo0 <= 0 <= hi0 and p0 == 1.0


def test_song_bootstrap_is_wider_when_songs_differ():
    """When the effect lives in a few songs, resampling songs gives a wider interval than resampling clips."""
    a, b = {}, {}
    for s in range(6):
        for c in range(10):
            i = f"song{s}_c{c:03d}"
            a[i] = {"index": i, "v": 0.0}
            b[i] = {"index": i, "v": 3.0 if s == 0 else 0.1 * (c % 3)}
    clip = paired_bootstrap(a, b, "v", n=2000, unit="clip")
    song = paired_bootstrap(a, b, "v", n=2000, unit="song")
    assert song[7] == 6 and (song[4] - song[3]) > 2 * (clip[4] - clip[3])


def test_holm():
    """Holm adjusts the smallest p most and keeps the order monotone."""
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
