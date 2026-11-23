"""Fine-tune SoulX-Singer on Hindi: LoRA + new phone rows, with confidence-aware flow matching (C).

Model (``build_model``)
-----------------------
The released checkpoint loads with ``strict=True``; the vocoder is dropped
(training needs only mels).  Every pretrained weight is frozen.  Trainable:

* LoRA (peft, rank ``--lora-r``) on the DiT's attention and MLP projections
  (``diff_estimator.layers.*.{q,k,v,o,gate,up,down}_proj``);
* the 62 Hindi phone rows through ``kalavinka.sx.phones`` (``--phones nearest``
  baseline or ``articulatory``, novel piece B);
* optionally the F0 bin embedding (``--train-f0-emb``).

Loss
----
Upstream's conditional flow matching, re-implemented (the released code has
no training entry point; ``flow_matching.FlowMatchingTransformer`` holds the
recipe): ``t ~ U(1e-5, 1)`` with the yaml's cosine schedule, ``xt = (1-(1-s)t)
z + t x`` on non-prompt frames (prompt frames stay clean), target ``x - (1-s)
z``, L1 averaged over the 128 mel bins.  Classifier-free guidance dropout is
per sample (``--cfg-drop``): the whole sequence is noised and the condition
zeroed, as upstream does for the whole batch.

**Confidence-aware weighting (``--weighting conf``, novel piece C).**  The
annotation pipeline knows how sure it is about every frame (MMS alignment
log-probability per syllable, unlabelled voiced frames, F0-ensemble agreement,
the clip's accept/review band).  Instead of throwing uncertain clips away we
weight the loss per frame::

    c_f  = sigmoid((conf_f - tau) / s)     tau, s from the train-set quantiles
    c_f  = 0 if c_f < --conf-floor          (hard mask on clearly wrong frames)
    w_f  = smooth5(c_f) * clip_w * clamp(mean f0_agree, 0.5, 1)
    clip_w = 1 (accept) | --review-weight (review)
    loss = sum(l_f * m_f * w_f) / sum(m_f * w_f)

``--weighting none`` is the uniform baseline.  Validation is always the
*unweighted* loss on accepted val clips with fixed noise and times, so arms
are comparable.

Runs are resumable (``<out>/last.pt``), log to ``<out>/train_<ts>.log`` and
``metrics.jsonl``, keep ``best.pt`` (lowest val loss) and stop early after
``--patience`` validations without improvement.

Usage::

    python -m kalavinka.sx.train --cache <dir with train.pt val.pt> --phones-table <phone_set_indic.json> \
        --out <runs/R4> --phones articulatory --bands accept,review --weighting conf [--steps 5000]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from saregama.paths import SOULX_REPO
from .data import UNLABELLED_CONF, FrameBudgetSampler, GateProbs, PairDataset, collate
from .phones import wrap_text_encoder

CKPT = SOULX_REPO / "pretrained_models" / "SoulX-Singer" / "model.pt"
CONFIG = SOULX_REPO / "soulxsinger" / "config" / "soulxsinger.yaml"
LORA_TARGETS = r".*layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_model(phone_table: Path, phones: str, lora_r: int, lora_alpha: int, train_f0_emb: bool = False,
                device: str = "cuda"):
    """Pretrained SoulXSinger (strict load, no vocoder) + phone wrapper + LoRA; returns (model, cfg)."""
    from peft import LoraConfig, inject_adapter_in_model
    from soulxsinger.models.soulxsinger import SoulXSinger
    from soulxsinger.utils.file_utils import load_config
    cfg = load_config(str(CONFIG))
    model = SoulXSinger(cfg)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.vocoder = None
    for p in model.parameters():
        p.requires_grad_(False)
    p2i = {t: i for i, t in enumerate(json.loads(Path(phone_table).read_text(encoding="utf-8")))}
    if phones != "none":
        wrap_text_encoder(model, p2i, phones)
    est = model.cfm_decoder.model.diff_estimator
    if lora_r > 0:
        inject_adapter_in_model(LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                                           target_modules=LORA_TARGETS), est)
        for n, p in est.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
    if train_f0_emb:
        model.f0_encoder.weight.requires_grad_(True)
    enable_checkpointing(est)
    return model.to(device), cfg


def enable_checkpointing(est) -> None:
    """Activation checkpointing per DiT layer.  Upstream's own branch raises
    NotImplementedError, so each layer's ``forward`` is wrapped on the instance
    (parameter names, hence checkpoints, are unchanged)."""
    from torch.utils.checkpoint import checkpoint
    for layer in est.layers:
        orig = layer.forward

        def fwd(hidden_states, _orig=orig, **kw):
            """Checkpointed call while training with grad, plain call otherwise."""
            if torch.is_grad_enabled() and hidden_states.requires_grad:
                return checkpoint(lambda h: _orig(h, **kw), hidden_states, use_reentrant=False)
            return _orig(hidden_states, **kw)
        layer.forward = fwd


def trainable_state(model) -> Dict[str, torch.Tensor]:
    """Only the trainable tensors (LoRA, phone rows, ...) -- what checkpoints store."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def conf_stats(conf: torch.Tensor) -> Dict[str, float]:
    """tau (median - 0.5 IQR) and s (IQR / 4) of the labelled syllable frames' log-probabilities."""
    v = conf[(conf < 0) & (conf > UNLABELLED_CONF + 1)]
    if v.numel() < 10:
        return {"tau": -3.0, "s": 1.0}
    q1, med, q3 = (torch.quantile(v[torch.randperm(v.numel())[:200000]].float(), q).item() for q in (0.25, 0.5, 0.75))
    iqr = max(q3 - q1, 1e-3)
    return {"tau": med - 0.5 * iqr, "s": iqr / 4}


def frame_weights(batch: Dict, mode: str, tau: float, s: float, floor: float, review_w: float) -> torch.Tensor:
    """Per-frame loss weights (B, T): ones for ``mode='none'``, else the confidence weights (module doc)."""
    x_mask = batch["x_mask"]
    if mode == "none":
        return torch.ones_like(x_mask)
    c = torch.sigmoid((batch["conf"] - tau) / s)
    c = torch.where(c < floor, torch.zeros_like(c), c)
    c = F.avg_pool1d(c[:, None], 5, stride=1, padding=2, count_include_pad=False)[:, 0]
    agree = (batch["f0_agree"] * x_mask).sum(1) / x_mask.sum(1).clamp(min=1)
    clip_w = torch.tensor([1.0 if b == "accept" else review_w for b in batch["band"]], device=c.device)
    return c * (clip_w * agree.clamp(0.5, 1.0))[:, None]


def cond_features(model, batch: Dict) -> torch.Tensor:
    """Frame-level decoder condition, exactly SoulXSinger.infer's feature path."""
    feat = model.note_pitch_encoder(batch["note_pitch"]) + model.note_type_encoder(batch["note_type"]) \
        + model.note_text_encoder(batch["tokens"])
    feat = model.preflow(feat)
    feat = model.expand_states(feat, batch["mel2note"])
    feat = feat + model.f0_encoder(model.f0_to_coarse(batch["f0"]))
    return model.cfm_decoder.model.cond_emb(feat)


def fm_loss(model, batch: Dict, weights: torch.Tensor, cfg_drop: float, gen: Optional[torch.Generator] = None,
            t_fixed: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Weighted conditional flow-matching L1 loss (prompt frames unnoised and unlossed)."""
    fm = model.cfm_decoder.model
    x, x_mask = batch["mel"], batch["x_mask"]
    B = x.shape[0]
    cond = cond_features(model, batch)
    is_prompt = batch["is_prompt"].clone()
    if cfg_drop > 0:
        drop = torch.rand(B, device=x.device, generator=gen) < cfg_drop
        is_prompt[drop] = 0
        cond = cond * (~drop).float()[:, None, None]
    if t_fixed is None:
        t = torch.rand(B, device=x.device, generator=gen).clamp(1e-5, 1.0)
        if fm.time_scheduler == "cos":
            t = 1 - torch.cos(t * math.pi * 0.5)
    else:
        t = t_fixed
    z = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype)
    m = (1 - is_prompt)[..., None]
    tt = t[:, None, None]
    xt = ((1 - (1 - fm.sigma) * tt) * z + tt * x) * m + x * (1 - m)
    pred = fm.diff_estimator(xt, t, cond, x_mask)
    target = x - (1 - fm.sigma) * z
    l = (pred.float() - target.float()).abs().mean(-1)
    w = (1 - is_prompt) * x_mask * weights
    return (l * w).sum() / w.sum().clamp(min=1e-6)


def to_device(batch: Dict, device: str) -> Dict:
    """Move the tensor fields of a batch to ``device``."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def validate(model, loader, device: str) -> float:
    """Unweighted val loss with fixed noise and a fixed grid of times (comparable across arms)."""
    model.eval()
    gen = torch.Generator(device=device)
    tot, n = 0.0, 0
    for k, batch in enumerate(loader):
        batch = to_device(batch, device)
        ones = torch.ones_like(batch["x_mask"])
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            gen.manual_seed(1000 * k + int(t * 10))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tt = torch.full((batch["mel"].shape[0],), t, device=device)
                tot += float(fm_loss(model, batch, ones, 0.0, gen, tt))
            n += 1
    model.train()
    return tot / max(n, 1)


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse args, build data and model, train with validation, early stopping and resume."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True, help="dir with train.pt and val.pt")
    p.add_argument("--phones-table", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phones", choices=["nearest", "articulatory", "none"], default="nearest")
    p.add_argument("--bands", default="accept", help="comma list of bands to train on")
    p.add_argument("--weighting", choices=["none", "conf"], default="none")
    p.add_argument("--review-weight", type=float, default=0.4)
    p.add_argument("--conf-floor", type=float, default=0.2)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--train-f0-emb", action="store_true")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-phones", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--max-frames", type=int, default=3600, help="padded frames per batch")
    p.add_argument("--cfg-drop", type=float, default=0.2)
    p.add_argument("--p-score", type=float, default=0.4)
    p.add_argument("--p-melody", type=float, default=0.4)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-steps-smoke", type=int, default=0, help="stop after N steps (smoke test)")
    p.add_argument("--no-preflight", action="store_true", help="skip collating every batch once before training")
    a = p.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    log_f = open(a.out / f"train_{datetime.now():%Y%m%d-%H%M%S}.log", "a", buffering=1)

    def log(msg: str) -> None:
        """Print and append to the run log with a timestamp."""
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    device = "cuda"
    log(f"args {json.dumps({k: str(v) for k, v in vars(a).items()})}")
    bands = tuple(a.bands.split(","))
    ds = PairDataset(a.cache / "train.pt", bands=bands, gate=GateProbs(a.p_score, a.p_melody), seed=a.seed)
    vds = PairDataset(a.cache / "val.pt", bands=("accept",), gate=None, seed=0)
    stats = conf_stats(torch.cat([ds.all[i]["conf"] for i in ds.idx]))
    log(f"train targets {len(ds)} ({bands}), val {len(vds)}; conf weighting stats {stats}")
    loader = torch.utils.data.DataLoader(ds, batch_sampler=FrameBudgetSampler(ds, a.max_frames, a.seed),
                                         collate_fn=collate, num_workers=a.workers, persistent_workers=a.workers > 0)
    vloader = torch.utils.data.DataLoader(vds, batch_sampler=FrameBudgetSampler(vds, a.max_frames, 0, shuffle=False),
                                          collate_fn=collate, num_workers=0)
    if not a.no_preflight and not (a.out / "last.pt").exists():
        n_b, n_f = 0, 0
        for batch in torch.utils.data.DataLoader(ds, batch_sampler=FrameBudgetSampler(ds, a.max_frames, a.seed),
                                                 collate_fn=collate, num_workers=a.workers):
            assert torch.isfinite(batch["mel"]).all() and batch["mel2note"].max() < batch["tokens"].shape[1]
            n_b += 1; n_f += int(batch["x_mask"].sum())
        log(f"preflight ok: {n_b} batches, {n_f / 50 / 3600:.2f} h of frames per epoch")
    model, _ = build_model(a.phones_table, a.phones, a.lora_r, a.lora_alpha, a.train_f0_emb, device)
    phone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("note_text_encoder")]
    other = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("note_text_encoder")]
    log(f"trainable: phones {sum(p.numel() for p in phone_params)}, lora/other {sum(p.numel() for p in other)}")
    opt = torch.optim.AdamW([{"params": other, "lr": a.lr}, {"params": phone_params, "lr": a.lr_phones}],
                            weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.5 * (1 + math.cos(math.pi * min(s / a.steps, 1.0)))))
    step, best, bad = 0, float("inf"), 0
    last = a.out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu")
        model.load_state_dict(ck["trainable"], strict=False)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best, bad = ck["step"], ck["best"], ck["bad"]
        log(f"resumed from step {step} (best {best:.4f}, bad {bad})")
    else:
        v0 = validate(model, vloader, device)
        log(f"step 0 val {v0:.4f}")
        with open(a.out / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": 0, "val": v0}) + "\n")
    model.train()
    metrics = open(a.out / "metrics.jsonl", "a", buffering=1)
    t0, run_loss, run_n = time.time(), 0.0, 0
    done = False
    while not done:
        for batch in loader:
            batch = to_device(batch, device)
            w = frame_weights(batch, a.weighting, stats["tau"], stats["s"], a.conf_floor, a.review_weight)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = fm_loss(model, batch, w, a.cfg_drop)
            loss = loss + model.note_text_encoder.penalty() if hasattr(model.note_text_encoder, "penalty") else loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step(); sched.step(); step += 1
            run_loss += float(loss); run_n += 1
            if step % 25 == 0:
                rec = {"step": step, "loss": run_loss / run_n, "gnorm": float(gn), "lr": sched.get_last_lr()[0],
                       "frames": int(batch["x_mask"].sum()), "sec": round(time.time() - t0, 1),
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
                if hasattr(model.note_text_encoder, "shared_share"):
                    rec["shared_share"] = model.note_text_encoder.shared_share()
                metrics.write(json.dumps(rec) + "\n")
                log(f"step {step} loss {rec['loss']:.4f} gn {rec['gnorm']:.2f} lr {rec['lr']:.2e} peak {rec['peak_gb']} GB")
                run_loss, run_n = 0.0, 0
            if step % a.val_every == 0 or step >= a.steps:
                v = validate(model, vloader, device)
                improved = v < best - 1e-4
                best, bad = (v, 0) if improved else (best, bad + 1)
                metrics.write(json.dumps({"step": step, "val": v, "best": best}) + "\n")
                log(f"step {step} val {v:.4f} best {best:.4f}{' *' if improved else ''} (bad {bad})")
                if improved:
                    torch.save({"trainable": trainable_state(model), "step": step, "val": v, "args": vars(a)},
                               a.out / "best.pt")
                torch.save({"trainable": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "best": best, "bad": bad}, str(last) + ".tmp")
                os.replace(str(last) + ".tmp", last)
                if bad >= a.patience:
                    log(f"early stop: {a.patience} validations without improvement")
                    done = True
            if step >= a.steps or (a.max_steps_smoke and step >= a.max_steps_smoke):
                done = True
            if done:
                break
    log(f"finished at step {step}, best val {best:.4f}")
    (a.out / "DONE").write_text(json.dumps({"step": step, "best": best}))


if __name__ == "__main__":
    main()
