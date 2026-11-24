"""Tests for the SoulX ablation switches: phone holdout, subset, reject-band weight
(run in the SoulX venv: ``.venv-sx/bin/python -m pytest kalavinka/tests/test_sx_ablation.py``)."""
import pytest

torch = pytest.importorskip("torch")
from kalavinka.sx.data import PairDataset, has_tokens, seeded_subset  # noqa: E402
from kalavinka.sx.train import frame_weights  # noqa: E402


def _clip(i, toks, singer="s#0", band="accept", t0=0):
    """A tiny synthetic cached clip."""
    T = 10
    return {"index": f"c{i}", "phoneme": torch.tensor(toks), "note_pitch": torch.full((len(toks),), 60),
            "note_type": torch.full((len(toks),), 2), "mel2note": torch.zeros(T, dtype=torch.long),
            "f0": torch.zeros(T), "mel": torch.zeros(T, 128), "conf": torch.zeros(T), "f0_agree": torch.ones(T),
            "band": band, "singer_id": singer, "time": [t0, t0 + 100]}


def test_has_tokens():
    """Membership of phone ids in a clip."""
    assert has_tokens(torch.tensor([5, 7, 9]), [7])
    assert not has_tokens(torch.tensor([5, 9]), [7])
    assert not has_tokens(torch.tensor([5, 9]), [])


def test_seeded_subset_size_and_determinism():
    """The subset reaches the requested duration and is the same for the same seed."""
    idx, secs = list(range(100)), [10.0] * 100
    a, b = seeded_subset(idx, secs, 200, 0), seeded_subset(idx, secs, 200, 0)
    assert a == b and len(a) == 20 and a == sorted(a)
    assert seeded_subset(idx, secs, 200, 1) != a


def test_exclusion_removes_targets_and_prompts(tmp_path):
    """Clips with a held-out phone are neither targets nor prompts of others."""
    clips = [_clip(0, [3, 4], t0=0), _clip(1, [3, 99], t0=200), _clip(2, [4, 5], t0=400)]
    torch.save({"clips": clips}, tmp_path / "train.pt")
    ds = PairDataset(tmp_path / "train.pt", exclude_tokens=[99], gate=None)
    assert ds.n_excluded == 1 and [ds.all[i]["index"] for i in ds.idx] == ["c0", "c2"]
    assert ds.prompt[2] == 0                      # clip 1 sits closer before clip 2 but is excluded
    full = PairDataset(tmp_path / "train.pt", gate=None)
    assert full.prompt[2] == 1


def test_reject_weight():
    """Reject clips get their own clip weight; accept stays 1."""
    B, T = 3, 8
    batch = {"x_mask": torch.ones(B, T), "conf": torch.zeros(B, T), "f0_agree": torch.ones(B, T),
             "band": ["accept", "review", "reject"]}
    w = frame_weights(batch, "conf", tau=-1.0, s=0.5, floor=0.0, review_w=0.4, reject_w=0.2)
    r = w.mean(1) / w[0].mean()
    assert torch.allclose(r, torch.tensor([1.0, 0.4, 0.2]), atol=1e-5)
    w2 = frame_weights(batch, "conf", tau=-1.0, s=0.5, floor=0.0, review_w=0.4)
    assert torch.allclose(w2[2], w2[1])           # default: reject weighted like review
