"""Train and evaluate the stage-1 expressive-contour IR model.

Reads the v1 export (``train/val/test.jsonl``), builds frame scaffolds
(``kalavinka.ir.features``) once, trains ``ContourNet`` (``--mode regress`` =
I1 baseline, ``flow`` = I2), validates every ``--val-every`` steps with early
stopping (``--patience``), and at the end evaluates the best checkpoint on the
test split against two references: the real contour and the **flat MIDI
baseline** (residual 0 = what score mode gives).  It also writes the
generated F0 of every test clip (``<out>/test_f0.json``: index -> 50 fps Hz
list) so ``kalavinka.eval`` can render them with SoulX in melody mode.

Usage::

    python -m kalavinka.ir.train --export <PROJECT>/export/v1 --out <PROJECT>/runs/I2_flow --mode flow
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .features import batchify, scaffold, to_f0
from .metrics import corpus_metrics
from .model import ContourNet


def load_split(path: Path, bands=("accept",)) -> List[Dict]:
    """Export items of the chosen bands (non-empty scaffolds only)."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            if it.get("band", "accept") in bands:
                out.append(it)
    return out


def to_torch(arrs: Dict[str, np.ndarray], mask: np.ndarray, device: str):
    """numpy batch -> torch on ``device``."""
    b = {k: torch.from_numpy(v).to(device) for k, v in arrs.items()}
    return b, torch.from_numpy(mask).to(device)


def batches(scs: Sequence[Dict], max_frames: int, shuffle: bool, rng: random.Random):
    """Length-bucketed batches of scaffolds under a padded-frame budget."""
    order = sorted(range(len(scs)), key=lambda i: len(scs[i]["f0"]) + (rng.random() if shuffle else 0))
    out, cur, longest = [], [], 0
    for i in order:
        L = max(longest, len(scs[i]["f0"]))
        if cur and L * (len(cur) + 1) > max_frames:
            out.append(cur); cur, L = [], len(scs[i]["f0"])
        cur.append(i); longest = L
    if cur:
        out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


@torch.no_grad()
def evaluate(model, scs, items, device, max_frames, steps=10, seed=0, temperature=1.0, t_label="left") -> Dict:
    """Generated F0 for every clip + corpus metrics vs the real F0 and three flat-MIDI baselines
    (``temperature`` scales the flow model's starting noise; ignored in regress mode; ``t_label``: see
    ``ContourNet.generate``).

    Baselines (all the flat note pitch, no ornaments), differing only in where they are voiced:

    * ``flat_midi``: voiced on every sung note (its own voicing, like the model's), the fair baseline;
    * ``flat_midi_same_frames``: voiced exactly where the model is, so cents / ornament errors are on the same
      frames as the model's;
    * ``flat_midi_ref_voicing``: voiced where the *real* singer is (the definition before 25 Nov 2026; it borrows the
      reference voicing, so its voicing F1 is ~1 and its cents error is on a different frame set)."""
    model.eval()
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    preds, refs, notes, out_f0 = [], [], [], {}
    flats = {"flat_midi": [], "flat_midi_same_frames": [], "flat_midi_ref_voicing": []}
    for idx in batches(scs, max_frames, False, random.Random(0)):
        arrs, mask = batchify([scs[i] for i in idx])
        b, m = to_torch(arrs, mask, device)
        r, v = model.generate(b, m, steps=steps, gen=gen, temperature=temperature, t_label=t_label)
        r, v = r.float().cpu().numpy(), v.float().cpu().numpy()
        for j, i in enumerate(idx):
            n = len(scs[i]["f0"])
            f0 = to_f0(scs[i]["flat_hz"], r[j, :n], v[j, :n])
            flat = scs[i]["flat_hz"].astype(np.float32)
            preds.append(f0); refs.append(scs[i]["f0"])
            flats["flat_midi"].append(np.where(flat > 0, flat, 0).astype(np.float32))
            flats["flat_midi_same_frames"].append(np.where(f0 > 0, flat, 0).astype(np.float32))
            flats["flat_midi_ref_voicing"].append(np.where(scs[i]["f0"] > 0, flat, 0).astype(np.float32))
            notes.append(scs[i]["flat_hz"])
            out_f0[items[i]["index"]] = [round(float(x), 1) for x in f0]
    model.train()
    res = {"model": corpus_metrics(preds, refs, notes)}
    res.update({k: corpus_metrics(v, refs, notes) for k, v in flats.items()})
    return res, out_f0


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Train, early-stop, evaluate on test and write the generated contours."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["regress", "flow"], default="regress")
    p.add_argument("--bands", default="accept")
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--max-frames", type=int, default=12000)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    logf = open(a.out / f"train_{datetime.now():%Y%m%d-%H%M%S}.log", "a", buffering=1)

    def log(msg):
        """Print + timestamped run log."""
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True); logf.write(line + "\n")

    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bands = tuple(a.bands.split(","))
    data = {s: load_split(a.export / f"{s}.jsonl", bands if s == "train" else ("accept",)) for s in ("train", "val", "test")}
    scs = {s: [scaffold(it) for it in v] for s, v in data.items()}
    log(f"args {vars(a)}; clips " + ", ".join(f"{s} {len(v)}" for s, v in scs.items()))
    model = ContourNet(a.mode, a.d, a.layers, a.heads).to(device)
    log(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))
    metrics = open(a.out / "metrics.jsonl", "a", buffering=1)
    step, best, bad, done = 0, float("inf"), 0, False
    vgen = torch.Generator(device=device)
    while not done:
        for idx in batches(scs["train"], a.max_frames, True, rng):
            arrs, mask = batchify([scs["train"][i] for i in idx])
            b, m = to_torch(arrs, mask, device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, parts = model.loss(b, m)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                metrics.write(json.dumps({"step": step, "loss": float(loss), **parts}) + "\n")
            if step % a.val_every == 0:
                model.eval(); tot, n = 0.0, 0
                with torch.no_grad():
                    for k, vidx in enumerate(batches(scs["val"], a.max_frames, False, random.Random(0))):
                        arrs, mask = batchify([scs["val"][i] for i in vidx])
                        vb, vm = to_torch(arrs, mask, device)
                        vgen.manual_seed(k)
                        tot += float(model.loss(vb, vm, gen=vgen)[0]); n += 1
                model.train()
                v = tot / max(n, 1)
                improved = v < best - 1e-4
                best, bad = (v, 0) if improved else (best, bad + 1)
                metrics.write(json.dumps({"step": step, "val": v, "best": best}) + "\n")
                log(f"step {step} loss {float(loss):.4f} val {v:.4f} best {best:.4f}{' *' if improved else ''}")
                if improved:
                    torch.save({"model": model.state_dict(), "args": vars(a), "step": step}, a.out / "best.pt")
                if bad >= a.patience:
                    log("early stop"); done = True
            if step >= a.steps:
                done = True
            if done:
                break
    model.load_state_dict(torch.load(a.out / "best.pt", map_location=device)["model"])
    res, f0s = evaluate(model, scs["test"], data["test"], device, a.max_frames, a.flow_steps)
    (a.out / "test_metrics.json").write_text(json.dumps(res, indent=1))
    (a.out / "test_f0.json").write_text(json.dumps(f0s))
    log(f"test: {json.dumps(res)}")
    (a.out / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
