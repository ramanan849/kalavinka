"""Model-free helpers of the P1 GPU stages: second-pitch salience, runs,
single-voice mask and singer clustering."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saregama.p1.gpu_stages import cluster_embeddings, runs, second_peak, single_voice_mask  # noqa: E402


def test_second_peak_ignores_octaves():
    """A peak an octave above the main one is not a second voice; a fifth is."""
    sal = np.zeros((3, 360))
    sal[:, 100] = 1.0
    sal[0, 160] = 0.8          # octave (60 bins) -> ignored
    sal[1, 135] = 0.6          # a fifth (35 bins) -> second voice
    sal[2, 103] = 0.9          # 60 cents away -> same voice
    sp = second_peak(sal)
    assert sp[0] == 0 and abs(sp[1] - 0.6) < 1e-9 and sp[2] == 0


def test_runs_and_mask():
    """Runs bridge short gaps and drop short islands; the mask needs all cues."""
    m = np.zeros(100, bool)
    m[10:40] = m[42:60] = m[80:83] = True
    assert runs(m, min_len=10, max_gap=3) == [(10, 60)]
    hz = np.full(4, 200.0)
    hz[3] = 0
    poly = {"resid_db": np.array([-40, -10, -40, -40.0]), "vocal_db": np.zeros(4),
            "second_peak": np.array([0.1, 0.1, 0.1, 0.1])}
    mask = single_voice_mask(hz, np.array([1, 1, 0.3, 1.0]), poly,
                             {"residual_db": -12, "second_peak": 0.35})
    assert mask.tolist() == [True, False, False, False]


def test_clustering_two_singers():
    """Two well-separated embedding groups give two clusters, labelled in order."""
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=192), rng.normal(size=192)
    emb = np.array([a + 0.1 * rng.normal(size=192) for _ in range(4)] +
                   [b + 0.1 * rng.normal(size=192) for _ in range(3)] + [a + 0.1 * rng.normal(size=192)])
    assert cluster_embeddings(emb, 0.5).tolist() == [0, 0, 0, 0, 1, 1, 1, 0]
    assert cluster_embeddings(emb[:1], 0.5).tolist() == [0]
