"""Expressive-contour IR networks (stage 1): a conv-transformer over the note scaffold.

``ContourNet`` embeds the frame features of ``kalavinka.ir.features`` (8
continuous + 4 categorical per frame, plus the clip's decade / register
context) and runs ``layers`` pre-norm transformer blocks whose feed-forward is
a 1-D convolution (kernel 5, so the network sees local shape) and whose
attention is bidirectional over the clip (masked padding).  Two uses:

* ``mode="regress"`` (I1, baseline): heads predict the residual (cents/100)
  and a voicing logit; trained with L1 + BCE.  Regression to the mean
  smooths ornaments.
* ``mode="flow"`` (I2): conditional flow matching on the residual.  The
  noisy residual ``x_t`` enters as an extra input channel and the time
  ``t`` modulates every block (AdaLN: per-block scale and shift from a
  sinusoidal time embedding).  Sampling integrates the learned velocity from
  noise with ``steps`` Euler steps, so the output is a *sample* of a plausible
  ornamented contour instead of the average one.  The voicing head is
  trained alongside with BCE.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of flow time t in [0, 1] -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000 * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    """Pre-norm self-attention + conv feed-forward, with optional AdaLN time modulation."""

    def __init__(self, d: int, heads: int, kernel: int = 5, adaln: bool = False):
        """Layers of one block."""
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d, elementwise_affine=not adaln), nn.LayerNorm(d, elementwise_affine=not adaln)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
        self.ff = nn.Sequential(nn.Conv1d(d, 4 * d, kernel, padding=kernel // 2), nn.GELU(),
                                nn.Conv1d(4 * d, d, kernel, padding=kernel // 2))
        self.drop = nn.Dropout(0.1)
        self.ada = nn.Linear(d, 4 * d) if adaln else None
        if self.ada is not None:
            nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, pad: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, d); pad: (B, T) True on padding; temb: (B, d) or None."""
        if self.ada is not None and temb is not None:
            s1, b1, s2, b2 = self.ada(temb)[:, None].chunk(4, -1)
        else:
            s1 = b1 = s2 = b2 = 0.0
        h = self.n1(x) * (1 + s1) + b1
        h, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + self.drop(h)
        h = self.n2(x) * (1 + s2) + b2
        h = self.ff(h.masked_fill(pad[..., None], 0).transpose(1, 2)).transpose(1, 2)
        return x + self.drop(h)


class ContourNet(nn.Module):
    """Note scaffold -> residual contour (+ voicing); ``mode`` = regress | flow (module doc)."""

    def __init__(self, mode: str = "regress", d: int = 384, layers: int = 6, heads: int = 6):
        """Embeddings, positional encoding, blocks and heads."""
        super().__init__()
        assert mode in ("regress", "flow")
        self.mode, self.d = mode, d
        self.cont = nn.Linear(8 + (1 if mode == "flow" else 0), d)
        self.ntype = nn.Embedding(4, d)
        self.pclass = nn.Embedding(13, d)
        self.ph_first = nn.Embedding(6, d)
        self.ph_last = nn.Embedding(6, d)
        self.decade = nn.Embedding(5, d)
        self.register_ = nn.Embedding(3, d)
        self.tproj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)) if mode == "flow" else None
        self.blocks = nn.ModuleList([Block(d, heads, adaln=mode == "flow") for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.resid_head = nn.Linear(d, 1)
        self.voice_head = nn.Linear(d, 1)

    def _pos(self, T: int, device) -> torch.Tensor:
        """Sinusoidal positions (T, d)."""
        pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.d, 2, device=device) * (-math.log(10000.0) / self.d))
        pe = torch.zeros(T, self.d, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe

    def forward(self, cont, cat, ctx, mask, xt=None, t=None):
        """Returns (residual or velocity (B, T), voicing logit (B, T))."""
        if self.mode == "flow":
            cont = torch.cat([cont, xt[..., None]], -1)
        h = self.cont(cont) + self.ntype(cat[..., 0]) + self.pclass(cat[..., 1]) + self.ph_first(cat[..., 2]) \
            + self.ph_last(cat[..., 3]) + (self.decade(ctx[:, 0]) + self.register_(ctx[:, 1]))[:, None]
        h = h + self._pos(h.shape[1], h.device)[None]
        pad = mask < 0.5
        temb = self.tproj(time_embedding(t, self.d)) if self.mode == "flow" else None
        for b in self.blocks:
            h = b(h, pad, temb)
        h = self.norm(h)
        return self.resid_head(h)[..., 0], self.voice_head(h)[..., 0]

    def loss(self, batch, mask, gen=None):
        """Training loss: L1 (regress) or flow-matching MSE (flow) on rmask frames + voicing BCE."""
        rm = batch["rmask"] * mask
        y = batch["resid"]
        if self.mode == "regress":
            r, v = self(batch["cont"], batch["cat"], batch["ctx"], mask)
            lr = ((r - y).abs() * rm).sum() / rm.sum().clamp(min=1)
        else:
            B = y.shape[0]
            t = torch.rand(B, device=y.device, generator=gen)
            z = torch.randn(y.shape, device=y.device, generator=gen)
            xt = (1 - t[:, None]) * z + t[:, None] * y
            u, v = self(batch["cont"], batch["cat"], batch["ctx"], mask, xt=xt, t=t)
            lr = (((u - (y - z)) ** 2) * rm).sum() / rm.sum().clamp(min=1)
        lv = (F.binary_cross_entropy_with_logits(v, batch["voiced"], reduction="none") * mask).sum() / mask.sum()
        return lr + 0.5 * lv, {"resid": float(lr), "voicing": float(lv)}

    @torch.no_grad()
    def generate(self, batch, mask, steps: int = 10, gen=None, temperature: float = 1.0):
        """(residual (B, T), voicing prob (B, T)); flow mode integrates from noise with Euler steps."""
        if self.mode == "regress":
            r, v = self(batch["cont"], batch["cat"], batch["ctx"], mask)
            return r, torch.sigmoid(v)
        B, T = mask.shape
        x = torch.randn(B, T, device=mask.device, generator=gen) * temperature
        v = None
        for k in range(steps):
            t = torch.full((B,), (k + 0.5) / steps, device=mask.device)
            u, v = self(batch["cont"], batch["cat"], batch["ctx"], mask, xt=x, t=t)
            x = x + u / steps
        return x, torch.sigmoid(v)
