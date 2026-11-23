"""Hindi phone embeddings for SoulX-Singer: nearest-copy baseline and articulatory composition (B).

SoulX-Singer never saw Hindi, so the 62 ``in_*`` tokens (rows 2820-2881 of the
3000-row ``note_text_encoder``; ``kalavinka.sx.adapter``) need embeddings.

* ``NearestTextEmbedding`` (baseline): each new row starts as the mean of the
  English rows named by ``saregama.phoneset`` ``init`` (e.g. ``kh`` = mean of
  ``en_K`` and ``en_HH``) and is then trained freely.
* ``ArticulatoryTextEmbedding`` (novel): new rows adapt through **shared
  articulatory feature vectors**, ``E[p] = E0[p] + sum_f A[p, f] dV[f] + R[p]``.
  ``A`` is a binary phone x feature matrix (place, manner, voicing,
  aspiration, vowel height / backness / rounding / length, stress ...), ``dV``
  one 512-d correction per feature, ``R`` a per-phone residual (L2-penalised by
  ``residual_penalty``, so most of the adaptation has to go through the shared
  features).  A gradient on ``kh`` therefore also moves ``ph``, ``th``, ``ch``
  ... through ``dV[aspirated]`` -- the point for a 5-10 h corpus in which some
  phones are rare.  ``anchor`` picks ``E0``:

  - ``nearest`` (default): ``E0`` = the baseline's nearest-copy rows and
    ``dV = R = 0``, so both arms start from the *same* table and differ only in
    how they adapt (a clean ablation of the parameterisation);
  - ``composed``: ``E0 = b + A V`` with ``V, b`` fitted by ridge least squares
    to SoulX's 70 English rows (features English lacks start from an
    analogue: retroflex := alveolar, aspiration := half the ``HH`` direction).
    On the released checkpoint this linear model reproduces the English rows
    only loosely (mean cosine ~0.59, ``english_fit``), which is why
    ``nearest`` is the default.

Both wrap the pretrained embedding with the same call signature (so upstream
``SoulXSinger.infer`` keeps working), keep every pretrained row frozen, and can
``bake()`` the current table into a plain ``nn.Embedding`` for export.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from saregama.phoneset import PHONES, PREFIX

# ---------------------------------------------------------------------------
# articulatory features
# ---------------------------------------------------------------------------
FEATURES: List[str] = [
    "vowel", "consonant", "mark_nasal",
    "bilabial", "labiodental", "dental", "alveolar", "postalveolar", "retroflex", "palatal", "velar",
    "uvular", "glottal",
    "stop", "nasal", "fricative", "affricate", "approximant", "flap", "trill", "lateral",
    "voiced", "aspirated",
    "high", "mid", "low", "front", "central", "back", "round", "long", "diphthong", "rhotic",
    "stress0", "stress1", "stress2",
]
FIDX = {f: i for i, f in enumerate(FEATURES)}

# English (ARPAbet, without stress digit)
_EN: Dict[str, Tuple[str, ...]] = {
    "AA": ("vowel", "low", "back", "long"), "AE": ("vowel", "low", "front"), "AH": ("vowel", "mid", "central"),
    "AO": ("vowel", "mid", "back", "round", "long"), "AW": ("vowel", "low", "central", "diphthong"),
    "AY": ("vowel", "low", "central", "diphthong"), "EH": ("vowel", "mid", "front"),
    "ER": ("vowel", "mid", "central", "rhotic"), "EY": ("vowel", "mid", "front", "long", "diphthong"),
    "IH": ("vowel", "high", "front"), "IY": ("vowel", "high", "front", "long"),
    "OW": ("vowel", "mid", "back", "round", "long", "diphthong"), "OY": ("vowel", "mid", "back", "round", "diphthong"),
    "UH": ("vowel", "high", "back", "round"), "UW": ("vowel", "high", "back", "round", "long"),
    "B": ("consonant", "bilabial", "stop", "voiced"), "P": ("consonant", "bilabial", "stop"),
    "M": ("consonant", "bilabial", "nasal", "voiced"), "F": ("consonant", "labiodental", "fricative"),
    "V": ("consonant", "labiodental", "fricative", "voiced"), "W": ("consonant", "bilabial", "approximant", "voiced", "round"),
    "TH": ("consonant", "dental", "fricative"), "DH": ("consonant", "dental", "fricative", "voiced"),
    "T": ("consonant", "alveolar", "stop"), "D": ("consonant", "alveolar", "stop", "voiced"),
    "N": ("consonant", "alveolar", "nasal", "voiced"), "S": ("consonant", "alveolar", "fricative"),
    "Z": ("consonant", "alveolar", "fricative", "voiced"), "L": ("consonant", "alveolar", "lateral", "voiced"),
    "R": ("consonant", "alveolar", "approximant", "voiced", "rhotic"),
    "SH": ("consonant", "postalveolar", "fricative"), "ZH": ("consonant", "postalveolar", "fricative", "voiced"),
    "CH": ("consonant", "postalveolar", "affricate"), "JH": ("consonant", "postalveolar", "affricate", "voiced"),
    "Y": ("consonant", "palatal", "approximant", "voiced"), "K": ("consonant", "velar", "stop"),
    "G": ("consonant", "velar", "stop", "voiced"), "NG": ("consonant", "velar", "nasal", "voiced"),
    "HH": ("consonant", "glottal", "fricative"),
}
_C = "consonant"
# Indic phones (names from saregama.phoneset); vowels carry stress1 like their init tokens
_IN: Dict[str, Tuple[str, ...]] = {
    "a": ("vowel", "mid", "central"), "aa": ("vowel", "low", "central", "long"), "i": ("vowel", "high", "front"),
    "ii": ("vowel", "high", "front", "long"), "u": ("vowel", "high", "back", "round"),
    "uu": ("vowel", "high", "back", "round", "long"), "e": ("vowel", "mid", "front"),
    "ee": ("vowel", "mid", "front", "long"), "o": ("vowel", "mid", "back", "round"),
    "oo": ("vowel", "mid", "back", "round", "long"), "E": ("vowel", "low", "front", "long"),
    "O": ("vowel", "low", "back", "round", "long"), "ae": ("vowel", "low", "front"),
    "ai": ("vowel", "low", "central", "diphthong"), "au": ("vowel", "low", "central", "diphthong"),
    "nas": ("mark_nasal", "nasal"),
    "k": (_C, "velar", "stop"), "kh": (_C, "velar", "stop", "aspirated"), "g": (_C, "velar", "stop", "voiced"),
    "gh": (_C, "velar", "stop", "voiced", "aspirated"), "ng": (_C, "velar", "nasal", "voiced"),
    "c": (_C, "palatal", "affricate"), "ch": (_C, "palatal", "affricate", "aspirated"),
    "j": (_C, "palatal", "affricate", "voiced"), "jh": (_C, "palatal", "affricate", "voiced", "aspirated"),
    "ny": (_C, "palatal", "nasal", "voiced"),
    "T": (_C, "retroflex", "stop"), "Th": (_C, "retroflex", "stop", "aspirated"),
    "D": (_C, "retroflex", "stop", "voiced"), "Dh": (_C, "retroflex", "stop", "voiced", "aspirated"),
    "N": (_C, "retroflex", "nasal", "voiced"),
    "t": (_C, "dental", "stop"), "th": (_C, "dental", "stop", "aspirated"), "d": (_C, "dental", "stop", "voiced"),
    "dh": (_C, "dental", "stop", "voiced", "aspirated"), "n": (_C, "dental", "nasal", "voiced"),
    "p": (_C, "bilabial", "stop"), "ph": (_C, "bilabial", "stop", "aspirated"), "b": (_C, "bilabial", "stop", "voiced"),
    "bh": (_C, "bilabial", "stop", "voiced", "aspirated"), "m": (_C, "bilabial", "nasal", "voiced"),
    "y": (_C, "palatal", "approximant", "voiced"), "r": (_C, "alveolar", "flap", "voiced", "rhotic"),
    "l": (_C, "alveolar", "lateral", "voiced"), "v": (_C, "labiodental", "approximant", "voiced"),
    "sh": (_C, "postalveolar", "fricative"), "Sh": (_C, "retroflex", "fricative"), "s": (_C, "alveolar", "fricative"),
    "h": (_C, "glottal", "fricative", "voiced"),
    "R": (_C, "retroflex", "flap", "voiced"), "Rh": (_C, "retroflex", "flap", "voiced", "aspirated"),
    "L": (_C, "retroflex", "lateral", "voiced"), "zh": (_C, "retroflex", "approximant", "voiced"),
    "rr": (_C, "alveolar", "trill", "voiced", "rhotic"), "nn": (_C, "alveolar", "nasal", "voiced"),
    "q": (_C, "uvular", "stop"), "x": (_C, "velar", "fricative"), "gg": (_C, "velar", "fricative", "voiced"),
    "z": (_C, "alveolar", "fricative", "voiced"), "f": (_C, "labiodental", "fricative"),
    "ts": (_C, "alveolar", "affricate"), "dz": (_C, "alveolar", "affricate", "voiced"),
}
# features absent from English: start from the feature vector of an analogue
ANALOGUE = {"retroflex": "alveolar", "flap": "approximant", "trill": "approximant", "uvular": "velar",
            "mark_nasal": "consonant"}


def en_features(token: str) -> Tuple[str, ...]:
    """Feature names of an English token like ``en_AA1`` or ``en_K``."""
    base = token[3:]
    stress = base[-1] if base[-1].isdigit() else None
    core = base[:-1] if stress else base
    feats = _EN[core]
    return feats + ((f"stress{stress}",) if stress else (("stress1",) if "vowel" in feats else ()))


def in_features(name: str) -> Tuple[str, ...]:
    """Feature names of an Indic phone (vowels get ``stress1``)."""
    f = _IN[name]
    return f + (("stress1",) if "vowel" in f else ())


def feature_matrix(feats: Sequence[Sequence[str]]) -> torch.Tensor:
    """Binary (n, len(FEATURES)) matrix from feature-name lists."""
    A = torch.zeros(len(feats), len(FEATURES))
    for i, fs in enumerate(feats):
        for f in fs:
            A[i, FIDX[f]] = 1.0
    return A


def fit_features(A: torch.Tensor, E: torch.Tensor, ridge: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ridge least squares ``E ~ b + A V``; returns (V [F, D], b [D]).  Columns that
    never occur in ``A`` get V = 0 (filled from analogues by the caller)."""
    b = E.mean(0)
    Ec = E - b
    Ac = A - A.mean(0, keepdim=True)
    G = Ac.T @ Ac + ridge * torch.eye(A.shape[1])
    V = torch.linalg.solve(G, Ac.T @ Ec)
    b = b - (A.mean(0) @ V)
    V[A.sum(0) == 0] = 0.0
    return V, b


class _Wrapped(nn.Module):
    """Common part: frozen pretrained table + a block of new rows starting at ``first``."""

    def __init__(self, base: nn.Embedding, first: int, n_new: int):
        """Keep a frozen copy of the pretrained weights."""
        super().__init__()
        self.register_buffer("base_weight", base.weight.detach().clone())
        self.first, self.n_new = first, n_new
        self.num_embeddings, self.embedding_dim = base.weight.shape

    def new_rows(self) -> torch.Tensor:
        """The (n_new, D) block of new rows (subclasses)."""
        raise NotImplementedError

    def table(self) -> torch.Tensor:
        """The full (V, D) table: pretrained rows with the new block spliced in."""
        w = self.base_weight
        new = self.new_rows().to(w.dtype)
        return torch.cat([w[: self.first], new, w[self.first + self.n_new:]], 0)

    @property
    def weight(self) -> torch.Tensor:
        """Same attribute as nn.Embedding (read-only view of the current table)."""
        return self.table()

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        """Embedding lookup in the current table."""
        return F.embedding(tok, self.table())

    def bake(self) -> nn.Embedding:
        """A plain nn.Embedding holding the current table (for export / inference)."""
        e = nn.Embedding(self.num_embeddings, self.embedding_dim)
        e.weight.data.copy_(self.table().detach())
        return e

    def penalty(self) -> torch.Tensor:
        """Extra regulariser added to the loss (0 by default)."""
        return self.base_weight.new_zeros(())


class NearestTextEmbedding(_Wrapped):
    """Baseline: new rows = mean of their ``init`` English rows, then trained freely."""

    def __init__(self, base: nn.Embedding, phone2idx: Dict[str, int]):
        """``phone2idx``: the extended table (token -> row)."""
        idx = [phone2idx[PREFIX + p.name] for p in PHONES]
        first, n = min(idx), len(idx)
        assert idx == list(range(first, first + n)), "Indic rows must be contiguous"
        super().__init__(base, first, n)
        W = base.weight.detach()
        init = torch.stack([W[[phone2idx[t] for t in p.init]].mean(0) for p in PHONES])
        self.rows = nn.Parameter(init.clone())

    def new_rows(self) -> torch.Tensor:
        """The trainable block."""
        return self.rows


class ArticulatoryTextEmbedding(_Wrapped):
    """Novel: new rows adapt through shared articulatory feature corrections (see module doc)."""

    def __init__(self, base: nn.Embedding, phone2idx: Dict[str, int], anchor: str = "nearest",
                 ridge: float = 1e-2, residual_penalty: float = 1e-2):
        """Build the anchor rows E0, the feature matrix and zero-initialised dV, R."""
        idx = [phone2idx[PREFIX + p.name] for p in PHONES]
        first, n = min(idx), len(idx)
        assert idx == list(range(first, first + n)), "Indic rows must be contiguous"
        super().__init__(base, first, n)
        W = base.weight.detach().float()
        en = sorted([t for t in phone2idx if t.startswith("en_")], key=phone2idx.get)
        A_en = feature_matrix([en_features(t) for t in en])
        E_en = W[[phone2idx[t] for t in en]]
        V, b = fit_features(A_en, E_en, ridge)
        for f, src in ANALOGUE.items():
            V[FIDX[f]] = V[FIDX[src]]
        stops = [phone2idx[t] for t in ("en_P", "en_T", "en_K")]
        V[FIDX["aspirated"]] = 0.5 * (W[phone2idx["en_HH"]] - W[stops].mean(0))
        A_new = feature_matrix([in_features(p.name) for p in PHONES])
        if anchor == "nearest":
            E0 = torch.stack([W[[phone2idx[t] for t in p.init]].mean(0) for p in PHONES])
        elif anchor == "composed":
            E0 = b + A_new @ V
        else:
            raise ValueError(f"anchor must be nearest | composed, not {anchor!r}")
        self.anchor = anchor
        self.register_buffer("E0", E0)
        self.register_buffer("A_new", A_new)
        self.register_buffer("A_en", A_en)
        self.register_buffer("E_en", E_en)
        self.register_buffer("V_fit", V)
        self.register_buffer("b_fit", b)
        self.dV = nn.Parameter(torch.zeros(len(FEATURES), W.shape[1]))
        self.R = nn.Parameter(torch.zeros(n, W.shape[1]))
        self.residual_penalty = residual_penalty

    def new_rows(self) -> torch.Tensor:
        """E0 + A_new dV + R."""
        return self.E0 + self.A_new @ self.dV + self.R

    def penalty(self) -> torch.Tensor:
        """L2 on the per-phone residuals (mean squared, scaled)."""
        return self.residual_penalty * self.R.pow(2).mean()

    @torch.no_grad()
    def english_fit(self) -> Dict[str, float]:
        """How well the fitted linear feature model reproduces the English rows."""
        P = self.b_fit + self.A_en @ self.V_fit
        cos = F.cosine_similarity(P, self.E_en, dim=1)
        rel = (P - self.E_en).norm(dim=1) / self.E_en.norm(dim=1)
        return {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "rel_err": float(rel.mean())}

    @torch.no_grad()
    def shared_share(self) -> float:
        """Share of the total update norm carried by the shared features (vs the residuals)."""
        s, r = (self.A_new @ self.dV).norm(), self.R.norm()
        return float(s / (s + r + 1e-12))


def wrap_text_encoder(model: nn.Module, phone2idx: Dict[str, int], kind: str, **kw) -> nn.Module:
    """Replace ``model.note_text_encoder`` by the chosen wrapper (``nearest`` | ``articulatory``)."""
    cls = {"nearest": NearestTextEmbedding, "articulatory": ArticulatoryTextEmbedding}[kind]
    model.note_text_encoder = cls(model.note_text_encoder, phone2idx, **kw)
    return model.note_text_encoder
