"""Tests for the stage-1 contour IR: scaffold features, F0 round trip, metrics, model shapes."""
import numpy as np
import pytest

from kalavinka.ir import features as Fe
from kalavinka.ir import metrics as M


def _item():
    """Two sung notes (MIDI 60, 62) around a rest; F0 = the notes with a +30 cent offset."""
    n = 50
    f0 = np.zeros(n)
    f0[:20] = Fe.midi_to_hz(60) * 2 ** (30 / 1200)
    f0[30:] = Fe.midi_to_hz(62) * 2 ** (30 / 1200)
    return {"n_frames": n, "duration": "0.40 0.20 0.4001", "note_pitch": "60 0 62", "note_type": "2 1 2",
            "phoneme": "in_k-aa <SP> in_m-ii", "f0": " ".join(f"{x:.1f}" for x in f0), "tonic_hz": 261.63,
            "year": "1995"}


def test_scaffold_targets_and_roundtrip():
    """Residual is +0.3 (30 cents) on sung voiced frames; to_f0 rebuilds the F0."""
    s = Fe.scaffold(_item())
    assert s["cont"].shape == (50, 8) and s["cat"].shape == (50, 4)
    assert np.allclose(s["resid"][s["rmask"] > 0], 0.3, atol=0.01)
    assert s["rmask"][20:30].sum() == 0 and tuple(s["ctx"]) == (1, 1)
    f0 = Fe.to_f0(s["flat_hz"], s["resid"], s["voiced"])
    ref = s["f0"]
    assert np.allclose(f0[ref > 0], ref[ref > 0], rtol=1e-3)


def test_metrics_zero_on_identity_and_vibrato_detected():
    """A contour against itself has 0 RMSE; a 6 Hz +-50 c vibrato is measured on a long note."""
    t = np.arange(100) / 50
    flat = np.full(100, 220.0)
    vib = 220 * 2 ** (50 * np.sin(2 * np.pi * 6 * t) / 1200)
    assert M.pair_metrics(vib, vib)["cents_rmse"] < 1e-6
    st = M.vibrato_stats(vib, flat)
    assert 30 < st["extent"][0] < 40 and 5 < st["rate"][0] < 7
    out = M.corpus_metrics([flat], [vib], [flat])
    assert out["flatness_pred"] == 0.0 and out["flatness_ref"] > 0


def test_model_shapes_and_losses():
    """Both modes run forward / loss / generate on a tiny batch."""
    torch = pytest.importorskip("torch")
    from kalavinka.ir.model import ContourNet
    arrs, mask = Fe.batchify([Fe.scaffold(_item()), Fe.scaffold(_item())])
    b = {k: torch.from_numpy(v) for k, v in arrs.items()}
    m = torch.from_numpy(mask)
    for mode in ("regress", "flow"):
        net = ContourNet(mode, d=32, layers=1, heads=2)
        loss, parts = net.loss(b, m)
        loss.backward()
        r, v = net.generate(b, m, steps=2)
        assert r.shape == (2, 50) and float(v.min()) >= 0 and np.isfinite(float(loss))
