"""Render held-out clips with SoulX-Singer (zero-shot or a fine-tuned run) -- SoulX venv.

For every test clip that has a prompt clip of the same singer id (same song +
cluster; ``kalavinka.sx.data.prompt_map``), build the inference meta exactly
like upstream ``cli/inference.py`` (prompt + target through
``IndicDataProcessor``) and call ``SoulXSinger.infer``.  Choices:

* ``--run``: a training run dir (its ``best.pt`` trainable weights -- LoRA and
  phone rows -- are loaded onto the rebuilt model) or ``zero-shot`` (R0: the
  released model with Hindi phones remapped to the nearest English tokens);
* ``--control``: ``score`` (MIDI notes only), ``melody`` (the clip's real F0) or
  ``melody-ir`` with ``--f0-json`` (a stage-1 IR model's generated F0,
  ``kalavinka.ir.train`` writes ``test_f0.json``);
* fixed seed per clip, so every system renders from the same noise.

Writes ``<out>/<index>.wav`` (24 kHz) and ``<out>/render.jsonl`` (index,
prompt, text, control, the F0 that was given).

Usage::

    python -m kalavinka.eval.render --export <PROJECT>/export/v1 --run <PROJECT>/runs/R4 \
        --out <PROJECT>/runs/eval_v1/R4_melody --control melody [--limit 100]
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import soundfile as sf
import torch

from kalavinka.sx.adapter import UPSTREAM_PHONES, IndicDataProcessor, normalise_item
from kalavinka.sx.data import prompt_map
from kalavinka.sx.train import CKPT, CONFIG, build_model

SR, HOP = 24000, 480


def load_items(path: Path, bands=("accept",)):
    """Export items of the chosen bands with the fields prompt_map needs."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            if it.get("band", "accept") in bands:
                out.append(it)
    return out


def load_system(run: str, phone_table: Path, device: str):
    """(model with vocoder, remap_english) for ``zero-shot`` or a run dir's best.pt."""
    from soulxsinger.models.modules.vocoder import Vocoder
    if run == "zero-shot":
        from soulxsinger.models.soulxsinger import SoulXSinger
        from soulxsinger.utils.file_utils import load_config
        model = SoulXSinger(load_config(str(CONFIG)))
        model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"], strict=True)
        return model.to(device).eval(), True
    ck = torch.load(Path(run) / "best.pt", map_location="cpu")
    a = ck["args"]
    model, _ = build_model(phone_table, a["phones"], int(a["lora_r"]), int(a["lora_alpha"]),
                           bool(a.get("train_f0_emb")), device)
    missing = model.load_state_dict(ck["trainable"], strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    voc = Vocoder()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    voc.load_state_dict({k[len("vocoder."):]: v for k, v in sd.items() if k.startswith("vocoder.")})
    model.vocoder = voc.to(device)
    return model.eval(), False


def side(proc, item: Dict, wav: Optional[Path], remap: bool, f0: Optional[Sequence[float]], device: str) -> Dict:
    """Processor output for one clip (optionally with its waveform and a replacement F0)."""
    it = normalise_item(item, remap_english=remap)
    if f0 is not None:
        it["f0"] = " ".join(f"{x:.1f}" for x in f0)
    out = proc.process(it, str(wav) if wav else None)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Render the test clips for one system and control mode."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--run", required=True, help="run dir or 'zero-shot'")
    p.add_argument("--phones-table", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--control", choices=["score", "melody", "melody-ir"], default="melody")
    p.add_argument("--f0-json", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--cfg", type=float, default=3.0)
    a = p.parse_args(argv)
    device = "cuda"
    a.out.mkdir(parents=True, exist_ok=True)
    table = a.phones_table or (a.export.parent / "soulx" / "phone_set_indic.json")
    model, remap = load_system(a.run, table, device)
    proc = IndicDataProcessor(hop_size=HOP, sample_rate=SR, device=device,
                              phoneset_path=str(UPSTREAM_PHONES if remap else table))
    items = load_items(a.export / f"{a.split}.jsonl")
    pm = prompt_map([{"band": "accept", "singer_id": it["singer_id"], "time": it["time"]} for it in items])
    ir_f0 = json.loads(a.f0_json.read_text()) if a.f0_json else {}
    todo = [(i, j) for i, j in enumerate(pm) if j is not None and (a.control != "melody-ir" or items[i]["index"] in ir_f0)]
    if a.limit:
        todo = todo[: a.limit]
    log = open(a.out / "render.jsonl", "w")
    for i, j in todo:
        t, pr = items[i], items[j]
        f0 = ir_f0.get(t["index"]) if a.control == "melody-ir" else None
        meta = {"prompt": side(proc, pr, a.export / pr["wav"], remap, None, device),
                "target": side(proc, t, None, remap, f0, device)}
        torch.manual_seed(zlib.crc32(t["index"].encode()) % (2 ** 31))   # stable across processes (hash() is salted)
        with torch.no_grad():
            wav = model.infer(meta, auto_shift=False, pitch_shift=0, n_steps=a.n_steps, cfg=a.cfg,
                              control="score" if a.control == "score" else "melody")
        y = wav[0].float().cpu().numpy().reshape(-1)
        sf.write(a.out / f"{t['index']}.wav", y, SR)
        given = meta["target"]["f0"][0].float().cpu().numpy().tolist() if a.control != "score" else None
        log.write(json.dumps({"index": t["index"], "prompt": pr["index"], "prompt_wav": pr["wav"], "ref_wav": t["wav"],
                              "text": t["text"], "control": a.control, "f0_given": given}, ensure_ascii=False) + "\n")
        log.flush()
    print(f"rendered {len(todo)} clips -> {a.out}")


if __name__ == "__main__":
    main()
