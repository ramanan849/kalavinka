"""Re-render a demo song as one continuous performance (``deploy.seamless``) instead of stitching separate clips.

Why: ``demo_song`` joins the per-clip evaluation renders with a fixed 0.6 s gap and ``demo_song_timed`` with the
song's real gaps, but each clip is still a separate SoulX call, so a listener hears a seam (breath, level, timbre)
at every clip boundary, in every system. Here the song's test clips are put on **one timeline** (their own notes,
the song's real gaps between adjacent clips; gaps longer than ``--max-gap`` become ``--skip-gap`` of silence, as in
``demo_song_timed``) and rendered by ``seamless.render``: pieces of at most 14 s cut at rests (or inside a held note,
crossfaded), so silence appears only where the song has it.

Systems (``--sys``): ``zeroshot`` (released SoulX, Hindi phones mapped to English, score mode), ``score`` (our model,
score mode), ``twostage_i2`` (our model, melody mode on the stage-1 contours of ``--f0-json``, as the evaluation's
``L4_irI2t085``), ``realpitch`` (our model on the real singer's F0). One voice prompt for the whole song: the prompt
the evaluation used for the song's first clip. Writes ``<out>/<song>_<sys>_seamless.wav`` and
``<song>_seamless.json`` (pieces, cut kinds, timeline).

Usage::

    python -m kalavinka.eval.demo_song_seamless --export <PROJECT>/export/v1 --song 2017_tu_hi_hai__KyMNLIYRsR4 \\
        --out <PROJECT>/demo --run <PROJECT>/runs/L4_v2 --prompts <PROJECT>/runs/eval_v1/L4_score/render.jsonl \\
        --f0-json <PROJECT>/runs/I2_v1/sweep_left/test_f0_tau0.85.json --sys zeroshot --sys twostage_i2
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf

SR = 24000


class EvalSinger:
    """Gives ``seamless.render`` the ``_render`` it needs, with the evaluation's exact model loading and meta."""

    def __init__(self, run: str, table: Path, export: Path, device: str = "cuda", steps: int = 32, cfg: float = 3.0):
        """Load the system (``run`` = a run dir or ``zero-shot``)."""
        from kalavinka.eval.render import HOP, UPSTREAM_PHONES, load_system
        from kalavinka.sx.adapter import IndicDataProcessor
        self.model, self.remap = load_system(run, table, device)
        self.proc = IndicDataProcessor(hop_size=HOP, sample_rate=SR, device=device,
                                       phoneset_path=str(UPSTREAM_PHONES if self.remap else table))
        self.export, self.device, self.steps, self.cfg = export, device, steps, cfg

    def _render(self, item: Dict, voice: Dict, f0, seed: int, shift: bool, model: str = "") -> np.ndarray:
        """One piece (melody mode when ``f0`` is given)."""
        import torch
        from kalavinka.eval.render import side
        meta = {"prompt": side(self.proc, voice["item"], self.export / voice["item"]["wav"], self.remap, None, self.device),
                "target": side(self.proc, item, None, self.remap, f0, self.device)}
        torch.manual_seed(seed)
        with torch.no_grad():
            wav = self.model.infer(meta, auto_shift=False, pitch_shift=0, n_steps=self.steps, cfg=self.cfg,
                                   control="score" if f0 is None else "melody")
        return wav[0].float().cpu().numpy().reshape(-1)


def song_timeline(clips: Sequence[Dict], melody: bool, f0s: Optional[Dict[str, List[float]]],
                  max_gap: float, skip_gap: float):
    """One ``Timeline`` for the song's clips in order, with the real (or skip) gaps; returns (timeline, gaps)."""
    from kalavinka.deploy.seamless import Timeline
    from kalavinka.eval.demo_song_timed import gaps_from_times
    gaps = gaps_from_times([c["time"] for c in clips], max_gap, skip_gap)
    tl = Timeline(melody=melody)
    for k, c in enumerate(clips):
        if k:
            tl.append_rest(gaps[k - 1])
        f0 = None
        if melody:
            f0 = f0s[c["index"]] if f0s is not None else [float(x) for x in c["f0"].split()]
        tl.append_item(c, f0)
    return tl.finish(), gaps


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    from kalavinka.deploy.seamless import render
    from kalavinka.eval.demo_song import clip_number
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--song", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run", required=True, help="our model's run dir")
    p.add_argument("--prompts", type=Path, required=True, help="a render.jsonl of the evaluation (index -> prompt)")
    p.add_argument("--f0-json", type=Path, default=None, help="stage-1 contours (index -> Hz) for twostage_i2")
    p.add_argument("--sys", action="append", required=True,
                   choices=["zeroshot", "score", "twostage_i2", "realpitch"])
    p.add_argument("--phones-table", type=Path, default=None)
    p.add_argument("--max-gap", type=float, default=1.5)
    p.add_argument("--skip-gap", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-s", type=float, default=14.0, help="longest render (28 for a model trained on long clips)")
    p.add_argument("--tag", default="seamless", help="file name tag: <song>_<sys>_<tag>.wav")
    a = p.parse_args(argv)
    items = {}
    for line in (a.export / f"{a.split}.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            it = json.loads(line)
            items[it["index"]] = it
    prompts = {}
    for line in a.prompts.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            r = json.loads(line)
            prompts[r["index"]] = r["prompt"]
    clips = sorted((k for k in items if k.rsplit("_c", 1)[0] == a.song and k in prompts), key=clip_number)
    if not clips:
        raise SystemExit(f"no rendered clips of {a.song}")
    all_items = dict(items)
    for split in ("train", "val", "test"):                  # the prompt can be any clip of the same song
        f = a.export / f"{split}.jsonl"
        if f.exists():
            for line in f.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    it = json.loads(line)
                    all_items.setdefault(it["index"], it)
    voice = {"item": all_items[prompts[clips[0]]]}
    table = a.phones_table or (a.export.parent / "soulx" / "phone_set_indic.json")
    f0s = json.loads(a.f0_json.read_text()) if a.f0_json else None
    a.out.mkdir(parents=True, exist_ok=True)
    seed = zlib.crc32(a.song.encode()) % (2 ** 31)
    report = {"song": a.song, "clips": clips, "prompt": voice["item"]["index"], "systems": {}}
    singers: Dict[str, EvalSinger] = {}
    for name in a.sys:
        run = "zero-shot" if name == "zeroshot" else a.run
        if run not in singers:
            singers[run] = EvalSinger(run, table, a.export, a.device)
        melody = name in ("twostage_i2", "realpitch")
        if name == "twostage_i2" and f0s is None:
            raise SystemExit("twostage_i2 needs --f0-json")
        tl, gaps = song_timeline([items[c] for c in clips], melody, f0s if name == "twostage_i2" else None,
                                 a.max_gap, a.skip_gap)
        wav, pieces = render(singers[run], tl, voice, seed, max_s=a.max_s)
        sf.write(a.out / f"{a.song}_{name}_{a.tag}.wav", wav, SR)
        report["systems"][name] = {"seconds": round(len(wav) / SR, 2), "pieces": pieces, "gaps": gaps}
        print(f"{name}: {len(wav) / SR:.1f} s, {len(pieces)} renders, cuts {[p['cut'] for p in pieces[1:]]}", flush=True)
    (a.out / f"{a.song}_{a.tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
