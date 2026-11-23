"""Tests for kalavinka.sx.adapter and kalavinka.sx.phones (SoulX venv; skipped elsewhere)."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("soulxsinger")

from kalavinka.sx import adapter, phones  # noqa: E402
from saregama.phoneset import PHONES, PREFIX  # noqa: E402


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    """Extended phone table (upstream + Indic) and its token -> row map."""
    p = tmp_path_factory.mktemp("ph") / "phones.json"
    adapter.build_phone_table(p)
    return p, {t: i for i, t in enumerate(json.loads(p.read_text()))}


def _item(n_frames=60):
    """A small export item: two notes and a rest, durations with float noise."""
    return {"index": "x", "duration": "0.40 0.30 0.5001", "phoneme": "in_k-aa <SP> in_m-ii-nas",
            "note_pitch": "60 0 62", "note_type": "2 1 2", "n_frames": n_frames,
            "f0": " ".join(["200.0"] * n_frames), "text": "का <SP> मीं"}


def test_indic_processor_expands_in_notes(table):
    """in_ notes expand into <BOW> phones <SEP> <EOW> and frame counts match n_frames."""
    path, p2i = table
    proc = adapter.IndicDataProcessor(hop_size=480, sample_rate=24000, phoneset_path=str(path), device="cpu")
    out = proc.process(adapter.normalise_item(_item()))
    assert out["mel2note"].shape[1] == 60
    toks = out["phoneme"][0].tolist()
    assert p2i["in_k"] in toks and p2i["in_aa"] in toks and p2i["in_nas"] in toks and p2i["<SEP>"] in toks


def test_exact_durations_sum_to_frames():
    """Rebuilt durations always give int(sum*50) == n_frames."""
    it = _item(61)
    d = [float(x) for x in adapter.exact_durations(it).split()]
    assert int(sum(d) * 50) == 61


def test_nearest_english_remap_uses_existing_tokens(table):
    """R0 remap only produces tokens that exist in the upstream table."""
    _, p2i = table
    s = adapter.nearest_english("in_kh-aa-T")
    assert s.startswith("en_") and all(("en_" + x) in p2i for x in s[3:].split("-"))


def test_articulatory_starts_equal_to_nearest_and_shares_updates(table):
    """Anchored articulatory rows equal the baseline at init; a dV update moves all phones with that feature."""
    _, p2i = table
    base = torch.nn.Embedding(3000, 16)
    near = phones.NearestTextEmbedding(base, p2i)
    art = phones.ArticulatoryTextEmbedding(base, p2i)
    assert torch.allclose(near.new_rows(), art.new_rows(), atol=1e-6)
    with torch.no_grad():
        art.dV[phones.FIDX["aspirated"]] += 1.0
    moved = (art.new_rows() - near.new_rows()).abs().sum(1) > 0
    names = [p.name for p in PHONES]
    assert moved[names.index("kh")] and moved[names.index("bh")] and not moved[names.index("k")]
    tok = torch.tensor([[p2i[PREFIX + "kh"], p2i["en_K"]]])
    assert torch.allclose(art(tok)[0, 1], base.weight[p2i["en_K"]])      # English rows untouched
    assert art.bake().weight.shape == (3000, 16)


def test_fit_features_recovers_linear_embeddings():
    """fit_features is exact on embeddings that really are linear in the features."""
    A = (torch.rand(40, 8) > 0.5).float()
    V, b = torch.randn(8, 5), torch.randn(5)
    Vh, bh = phones.fit_features(A, b + A @ V, ridge=1e-6)
    assert torch.allclose(bh + A @ Vh, b + A @ V, atol=1e-3)


def test_every_indic_phone_has_features():
    """All 62 Indic phones and every English token have a feature description."""
    for p in PHONES:
        assert phones.in_features(p.name)
    assert phones.en_features("en_AA1") and phones.en_features("en_UW")
