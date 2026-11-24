"""Sing new lyrics (never seen in training) to the tune of a held-out song line.

Why: every evaluation render re-sings a real lyric line.  To show the model
singing words it has never heard, we write short new Hindi lines, turn them
into syllables and phones with ``saregama.g2p`` (the same G2P that built the
dataset), and set them to the note pattern of a held-out (test) line.

How the tune is borrowed: the tune clip's notes are grouped into syllables (a
``note_type`` 2 onset plus its type-3 slur notes; type 1 is a rest).  The new
syllables replace the first N syllable groups in order, keeping their pitches,
durations, slurs and the rests between them; anything after the N-th syllable
is dropped.  So the tune needs at least as many syllables as the new line
(``pick_tunes`` finds such test clips).  The F0 field is a flat contour from
the note pitches; in ``score`` mode SoulX makes its own contour anyway.

Voice: the prompt is another clip of the tune's singer (``prompt_map``, as in
the evaluation renders), so the timbre is that singer's.

Novelty check: every new line is compared with the training lyrics (spaces
and syllable splits removed); lines that occur in training are refused.

Outputs in ``--out``: ``<k>_<tune>.wav`` per (line, tune) and ``sing_new.jsonl``
(line, syllables, tune, prompt, notes used).

Usage::

    .venv-sx/bin/python -m kalavinka.eval.sing_new --export <PROJECT>/export/v1 --run <PROJECT>/runs/L4_v2 \\
        --out <PROJECT>/demo/new_lyrics --line "ठंडी हवा में गीत नया" [--line ...] [--tunes 3] [--control score]
"""
from __future__ import annotations

import argparse
import json
import random
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PREFIX = "in_"
FPS = 50


def phoneme_token(phones: Sequence[str]) -> str:
    """Export-style phoneme token for one syllable, e.g. ``in_r-ii``."""
    return PREFIX + "-".join(phones)


def syllable_groups(item: Dict) -> List[List[int]]:
    """Note indices of an item grouped per syllable (onset + slurs); rests are their own group."""
    types = [int(t) for t in item["note_type"].split()]
    groups: List[List[int]] = []
    for i, t in enumerate(types):
        if t == 3 and groups and types[groups[-1][0]] != 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def n_syllables(item: Dict) -> int:
    """Sung syllables (non-rest groups) in an item."""
    types = [int(t) for t in item["note_type"].split()]
    return sum(1 for g in syllable_groups(item) if types[g[0]] != 1)


def set_lyrics(tune: Dict, sylls: Sequence[Dict]) -> Dict:
    """Copy of ``tune`` singing ``sylls`` ({text, phones}) on its first len(sylls) syllables."""
    types = [int(t) for t in tune["note_type"].split()]
    durs = tune["duration"].split()
    pitch = tune["note_pitch"].split()
    keep, text, phon = [], [], []
    k = 0
    for g in syllable_groups(tune):
        if types[g[0]] == 1:
            if k < len(sylls):          # a rest inside the new line
                keep += g; text += ["<SP>"] * len(g); phon += ["<SP>"] * len(g)
            continue
        if k == len(sylls):
            break
        keep += g
        text += [sylls[k]["text"]] * len(g)
        phon += [phoneme_token(sylls[k]["phones"])] * len(g)
        k += 1
    if k < len(sylls):
        raise ValueError(f"tune has {k} syllables, the line needs {len(sylls)}")
    while keep and types[keep[-1]] == 1:       # no trailing rest
        keep.pop(); text.pop(); phon.pop()
    frames = [int(round(float(durs[i]) * FPS)) for i in keep]
    f0 = []
    for i, f in zip(keep, frames):
        hz = 0.0 if types[i] == 1 else 440.0 * 2 ** ((int(pitch[i]) - 69) / 12)
        f0 += [hz] * f
    out = dict(tune)
    out.update({"text": " ".join(text), "phoneme": " ".join(phon), "note_type": " ".join(str(types[i]) for i in keep),
                "note_pitch": " ".join(pitch[i] for i in keep), "duration": " ".join(f"{f / FPS:.2f}" for f in frames),
                "n_frames": sum(frames), "f0": " ".join(f"{x:.1f}" for x in f0)})
    return out


def squash(s: str) -> str:
    """Lyric text without spaces, syllable splits or rest marks (for the novelty check)."""
    return "".join(w for w in s.split() if w != "<SP>")


def pick_tunes(items: Sequence[Dict], n_syll: int, k: int, seed: int) -> List[Dict]:
    """``k`` test items (seeded draw) that suit a short demo line: ``n_syll``..``n_syll + 4`` syllables, no rest
    first, a singable pace (0.3-0.7 s per syllable, so no rap lines) and at most 2 slur notes (no long melismas)."""
    def singable(it: Dict) -> bool:
        types = it["note_type"].split()
        sung = sum(float(d) for d, t in zip(it["duration"].split(), types) if t != "1")
        return 0.3 <= sung / max(n_syllables(it), 1) <= 0.7 and types.count("3") <= 2
    ok = [it for it in items if n_syll <= n_syllables(it) <= n_syll + 4 and int(it["note_type"].split()[0]) != 1
          and singable(it)]
    ok.sort(key=lambda it: it["index"])
    return random.Random(seed).sample(ok, min(k, len(ok)))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI: G2P each line, check it is new, borrow tunes, render."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--run", required=True, help="run dir or 'zero-shot'")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--line", action="append", required=True, help="a new lyric line in Devanagari (repeat)")
    p.add_argument("--tune", action="append", default=[], help="test clip index to borrow the tune from (repeat)")
    p.add_argument("--tunes", type=int, default=3, help="tunes per line when --tune is not given")
    p.add_argument("--control", choices=["score", "melody"], default="score")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    import soundfile as sf
    import torch
    from saregama.g2p import line_to_syllables
    from kalavinka.eval.render import HOP, SR, load_items, load_system, side
    from kalavinka.sx.adapter import UPSTREAM_PHONES, IndicDataProcessor
    from kalavinka.sx.data import prompt_map

    train_text = {squash(it["text"]) for it in load_items(a.export / "train.jsonl", bands=("accept", "review", "reject"))}
    test = load_items(a.export / "test.jsonl")
    pm = prompt_map([{"band": "accept", "singer_id": it["singer_id"], "time": it["time"]} for it in test])
    by_index = {it["index"]: i for i, it in enumerate(test)}

    table = a.export.parent / "soulx" / "phone_set_indic.json"
    known = {x[len(PREFIX):] for x in json.loads(table.read_text()) if x.startswith(PREFIX)}
    device = "cuda"
    model, remap = load_system(a.run, table, device)
    proc = IndicDataProcessor(hop_size=HOP, sample_rate=SR, device=device,
                              phoneset_path=str(UPSTREAM_PHONES if remap else table))
    a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "sing_new.jsonl", "a", encoding="utf-8")
    for k, line in enumerate(a.line):
        if any(squash(line) in t for t in train_text):
            raise SystemExit(f"line {k} occurs in the training lyrics: {line}")
        sylls = [{"text": s.text, "phones": s.phones} for s in line_to_syllables(line, "hi")]
        unknown = {ph for s in sylls for ph in s["phones"] if ph not in known}
        if unknown:
            raise SystemExit(f"line {k}: phones outside the table {sorted(unknown)}")
        tunes = [test[by_index[t]] for t in a.tune] or pick_tunes(test, len(sylls), a.tunes, a.seed + k)
        for tune in tunes:
            j = pm[by_index[tune["index"]]]
            if j is None:
                print(f"skip tune {tune['index']}: no prompt clip of that singer")
                continue
            pr = test[j]
            target = set_lyrics(tune, sylls)
            meta = {"prompt": side(proc, pr, a.export / pr["wav"], remap, None, device),
                    "target": side(proc, target, None, remap, None, device)}
            torch.manual_seed(zlib.crc32(f"{line}|{tune['index']}".encode()) % (2 ** 31))
            with torch.no_grad():
                wav = model.infer(meta, auto_shift=False, pitch_shift=0, n_steps=32, cfg=3.0, control=a.control)
            name = f"{k:02d}_{tune['index']}"
            sf.write(a.out / f"{name}.wav", wav[0].float().cpu().numpy().reshape(-1), SR)
            log.write(json.dumps({"file": f"{name}.wav", "line": line, "syllables": target["text"],
                                  "phoneme": target["phoneme"], "tune": tune["index"], "tune_text": tune["text"],
                                  "prompt": pr["index"], "run": a.run, "control": a.control,
                                  "seconds": target["n_frames"] / FPS}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"{name}: {target['text']}  ({target['n_frames'] / FPS:.1f} s)")


if __name__ == "__main__":
    main()
