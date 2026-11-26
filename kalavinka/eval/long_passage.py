"""Does a model trained on short clips struggle on long passages? Render the same 18-28 s stretches of the test songs
three ways and score every clip inside them against the real singer.

Why: the evaluation renders each test clip alone (median 4 s, at most 16 s), and the long-clip export
(``export/v3_long``) still has only 9 test clips over 16 s, because the test songs mostly break at long rests. So
neither says what happens when the model sings 20-28 s in one go, which is what a user asking for "big lyrics" gets.

Passages: consecutive evaluated (``--all-clips``: all) test clips of one song (the ``--prompts`` render's clips, in song order) whose real
gap is at most ``--max-gap`` s, grouped greedily into stretches of ``--min-s`` to ``--max-s`` seconds (real gaps kept
as rests). One voice prompt per passage (the evaluation's prompt of its first clip, or of a later clip if that prompt
lies inside the passage), one seed per passage.

Conditions (``--cond``, all with the same prompt and seed):

* ``alone`` -- every clip rendered on its own (the evaluation's setting);
* ``seam14`` -- the passage on one timeline, ``seamless.render`` in pieces of at most 14 s (the app's default);
* ``single`` -- the whole passage in **one** SoulX call (up to 28 s target + the prompt).

The renders are cut back into the passage's clips (sample-exact: the timeline keeps every clip's frames), and each
condition becomes a render directory (``<out>/<cond>/<index>.wav`` + ``render.jsonl``) that ``kalavinka.eval.score``
and ``kalavinka.eval.compare`` read as usual. ``passages.json`` lists each clip's passage and its offset in it, for a
by-position breakdown (does a clip sung 20 s into a render come out worse?).

Usage::

    python -m kalavinka.eval.long_passage --export <PROJECT>/export/v2 --run <PROJECT>/runs/L4_v2 \\
        --prompts <PROJECT>/runs/eval_v2/L4_score/render.jsonl --out <PROJECT>/runs/eval_passage/L4
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

SR, FPS, HOP = 24000, 50, 480
EDGE_S = 0.2


def read_jsonl(path: Path) -> List[Dict]:
    """Lines of a jsonl file."""
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def song_of(index: str) -> str:
    """Song id of a clip index."""
    return index.rsplit("_c", 1)[0]


def group_passages(clips: Sequence[Dict], max_gap: float, min_s: float, max_s: float) -> List[List[Dict]]:
    """Greedy passages of consecutive clips (already in song order, one song). A clip joins the current passage if
    the real gap before it is at most ``max_gap`` and the passage stays within ``max_s`` (clips + gaps + two edge
    rests); passages shorter than ``min_s`` are dropped."""
    out, cur, length = [], [], 0.0
    for c in clips:
        d = c["n_frames"] / FPS
        if cur:
            gap = max(0.0, (c["time"][0] - cur[-1]["time"][1]) / 1000.0)
            if gap <= max_gap and length + gap + d + 2 * EDGE_S <= max_s:
                cur.append(c); length += gap + d
                continue
            if length + 2 * EDGE_S >= min_s:
                out.append(cur)
            cur, length = [], 0.0
        if d + 2 * EDGE_S <= max_s:
            cur, length = [c], d
    if cur and length + 2 * EDGE_S >= min_s:
        out.append(cur)
    return out


def passage_timeline(clips: Sequence[Dict]):
    """Score-mode ``Timeline`` of the clips with their real gaps; returns (timeline, start frame of each clip)."""
    from kalavinka.deploy.seamless import Timeline
    tl = Timeline(melody=False)
    tl.append_rest(EDGE_S)                      # the leading edge rest first, so clip offsets are known exactly
    starts = []
    for k, c in enumerate(clips):
        if k:
            tl.append_rest(max(0.0, (c["time"][0] - clips[k - 1]["time"][1]) / 1000.0))
        starts.append(tl.total)
        tl.append_item(c)
    return tl.finish(EDGE_S), starts


def pick_prompt(clips: Sequence[Dict], prompts: Dict[str, str]) -> str:
    """The evaluation prompt of the first clip whose prompt is not inside the passage."""
    inside = {c["index"] for c in clips}
    for c in clips:
        if prompts[c["index"]] not in inside:
            return prompts[c["index"]]
    return prompts[clips[0]["index"]]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI."""
    import soundfile as sf
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--run", required=True)
    p.add_argument("--prompts", type=Path, required=True, help="render.jsonl of the evaluation (clip set + prompts)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cond", action="append", choices=["alone", "seam14", "single"], default=None)
    p.add_argument("--max-gap", type=float, default=3.0)
    p.add_argument("--min-s", type=float, default=18.0)
    p.add_argument("--max-s", type=float, default=28.0)
    p.add_argument("--phones-table", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--all-clips", action="store_true",
                   help="use every clip of the split (any band), not only the evaluated ones; a clip outside the "
                        "evaluation gets the prompt of an evaluated clip of its song")
    p.add_argument("--plan-only", action="store_true", help="print the passages and stop")
    a = p.parse_args(argv)
    conds = a.cond or ["alone", "seam14", "single"]
    items = {it["index"]: it for it in read_jsonl(a.export / f"{a.split}.jsonl")}
    rows = {r["index"]: r for r in read_jsonl(a.prompts)}
    if a.all_clips:
        song_prompt = {}
        for k, r in rows.items():
            song_prompt.setdefault(song_of(k), r["prompt"])
        for k, it in items.items():
            if k not in rows and song_of(k) in song_prompt:
                rows[k] = {"index": k, "prompt": song_prompt[song_of(k)], "ref_wav": it["wav"], "text": it["text"],
                           "control": "score", "f0_given": None}
    prompts = {k: r["prompt"] for k, r in rows.items()}
    from kalavinka.eval.demo_song import clip_number
    songs: Dict[str, List[Dict]] = {}
    for k in sorted(rows, key=lambda k: (song_of(k), clip_number(k))):
        songs.setdefault(song_of(k), []).append(items[k])
    passages = [ps for s in songs.values() for ps in group_passages(s, a.max_gap, a.min_s, a.max_s)]
    plan = []
    for n, ps in enumerate(passages):
        tl, starts = passage_timeline(ps)
        plan.append({"passage": n, "song": song_of(ps[0]["index"]), "seconds": round(tl.total / FPS, 2),
                     "prompt": pick_prompt(ps, prompts),
                     "clips": [{"index": c["index"], "offset_s": round(s / FPS, 2), "seconds": c["n_frames"] / FPS}
                               for c, s in zip(ps, starts)]})
    n_clips = sum(len(x["clips"]) for x in plan)
    print(f"{len(plan)} passages, {n_clips} clips, {len({x['song'] for x in plan})} songs, "
          f"{sum(x['seconds'] for x in plan) / 60:.1f} min; lengths {sorted(x['seconds'] for x in plan)}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "passages.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    if a.plan_only:
        return
    from kalavinka.deploy.seamless import render
    from kalavinka.eval.demo_song_seamless import EvalSinger
    all_items = dict(items)
    for split in ("train", "val", "test"):
        f = a.export / f"{split}.jsonl"
        if f.exists():
            for it in read_jsonl(f):
                all_items.setdefault(it["index"], it)
    table = a.phones_table or (a.export.parent / "soulx" / "phone_set_indic.json")
    singer = EvalSinger(a.run, table, a.export, a.device)
    for cond in conds:
        (a.out / cond).mkdir(exist_ok=True)
    out_rows: Dict[str, List[Dict]] = {c: [] for c in conds}
    for x, ps in zip(plan, passages):
        voice = {"item": all_items[x["prompt"]]}
        seed = zlib.crc32(f"{x['song']}/{x['passage']}".encode()) % (2 ** 31)
        tl, starts = passage_timeline(ps)
        for cond in conds:
            if cond == "alone":
                waves = [np.asarray(singer._render(c, voice, None, seed, shift=False), np.float32).reshape(-1)
                         for c in ps]
            else:
                if cond == "single":
                    wav = np.asarray(singer._render(tl.to_item(), voice, None, seed, shift=False), np.float32).reshape(-1)
                else:
                    wav, _ = render(singer, tl, voice, seed, max_s=14.0)
                assert len(wav) == tl.total * HOP, (len(wav), tl.total * HOP)
                waves = [wav[s * HOP:(s + c["n_frames"]) * HOP] for c, s in zip(ps, starts)]
            for c, w in zip(ps, waves):
                sf.write(a.out / cond / f"{c['index']}.wav", w, SR)
                r = rows[c["index"]]
                out_rows[cond].append(dict(r, prompt=x["prompt"], prompt_wav=all_items[x["prompt"]]["wav"],
                                           passage=x["passage"]))
        print(f"passage {x['passage']}: {x['seconds']} s, {len(ps)} clips", flush=True)
    for cond in conds:
        (a.out / cond / "render.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows[cond]), encoding="utf-8")
        print(f"rendered {len(out_rows[cond])} clips -> {a.out / cond}")


if __name__ == "__main__":
    main()
