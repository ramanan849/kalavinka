"""Command line for the Kalavinka singer (``python -m kalavinka_app ...`` inside a bundle; ``./sing ...`` on the Mac).

Commands::

    sing "ठंडी हवा में गीत नया" [--tune 3] [--voice 5] [--expressive] [--out song.wav]
                                sing new lyrics (several lines: separate them with | or new lines)
    "ठंडी हवा में गीत नया"      same as sing (the word "sing" can be left out)
    resing 3 [--mode score|expressive|real-pitch]
                                sing bundled tune 3 with its original words, and save the real singer next to it
    tunes                       list the tunes (song, syllables, seconds, original words)
    voices                      list the voices
    demo                        sing three example lines (both pitch modes) into outputs/demo/
    ui [--share] [--port 7860]  open the web page (type lyrics, pick tune and voice, listen)
    fetch                       download SoulX's base weights now (otherwise done on first use)

Common options (before the command): ``--device auto|cuda|mps|cpu`` (default auto), ``--steps N`` (default 32 on a
GPU, 16 on CPU; fewer = faster, rougher), ``--play`` (play each take),
``--seed 0`` (another seed = another take), ``--bundle DIR`` (default: the folder this package sits in).
Every take is written to ``outputs/`` with the lyrics and settings in a matching ``.json``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

EXAMPLES = ["ठंडी हवा में गीत नया", "चाँद की रोशनी में तेरा नाम", "बारिश की बूँदों ने कुछ कहा"]


def default_bundle() -> Path:
    """The bundle root: the folder that holds this package."""
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path):
    """Rows of a jsonl file."""
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def slug(text: str, n: int = 24) -> str:
    """Filesystem-safe short name (keeps Devanagari)."""
    s = re.sub(r"[\s|/\\:*?\"<>]+", "_", text.strip())
    return s[:n].strip("_") or "take"


def print_tunes(bundle: Path) -> None:
    """Tune table (no model needed)."""
    print(f"{'no':>3}  {'song':34s} {'year':>4}  {'syll':>4}  {'sec':>4}  original words")
    for t in read_jsonl(bundle / "data" / "tunes.jsonl"):
        print(f"{t['no']:>3}  {t['song'][:34]:34s} {t['year']:>4}  {t['syllables']:>4}  {t['seconds']:>4}  {t['lyric']}")
    print("\nA new line needs a tune with at least as many syllables (one syllable is roughly one vowel).")


def print_voices(bundle: Path) -> None:
    """Voice table (no model needed)."""
    for v in read_jsonl(bundle / "data" / "voices.jsonl"):
        print(f"{v['no']:>3}  {v['name']}")


def play(path: Path, on: bool) -> None:
    """Play a wav if asked (afplay on a Mac, aplay/paplay on Linux); silently skip if no player."""
    if not on:
        return
    import shutil
    import subprocess
    for player in ("afplay", "paplay", "aplay"):
        if shutil.which(player):
            subprocess.run([player, str(path)], check=False)
            return


def take_path(out_dir: Path, name: str) -> Path:
    """outputs/<time>_<name>.wav"""
    return out_dir / f"{time.strftime('%H%M%S')}_{name}.wav"


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the command and run it."""
    p = argparse.ArgumentParser(prog="sing", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, default=default_bundle())
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=None, help="flow steps (default 32 on a GPU, 16 on CPU)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outputs", type=Path, default=None, help="folder for the takes (default: <bundle>/outputs)")
    p.add_argument("--play", action="store_true", help="play each take when it is ready (Mac: afplay)")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("sing", help="sing new lyrics")
    s.add_argument("lyrics")
    s.add_argument("--tune", type=int, default=None)
    s.add_argument("--voice", type=int, default=None)
    s.add_argument("--expressive", action="store_true", help="two-stage: stage-1 contour model draws the pitch")
    s.add_argument("--out", type=Path, default=None)
    r = sub.add_parser("resing", help="re-sing a bundled tune with its original words")
    r.add_argument("tune", type=int)
    r.add_argument("--mode", choices=["score", "expressive", "real-pitch"], default="score")
    sub.add_parser("tunes"); sub.add_parser("voices"); sub.add_parser("demo")
    sub.add_parser("fetch", help="download SoulX's base weights (2.7 GB) now")
    u = sub.add_parser("ui", help="open the web page")
    u.add_argument("--share", action="store_true", help="public link (needed on Colab)")
    u.add_argument("--port", type=int, default=7860)
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"sing", "resing", "tunes", "voices", "demo", "ui", "fetch", "-h", "--help"}
    first = next((i for i, x in enumerate(argv) if not x.startswith("-") and (i == 0 or argv[i - 1] not in
                  ("--bundle", "--device", "--steps", "--seed", "--outputs"))), None)
    if first is not None and argv[first] not in cmds:
        argv.insert(first, "sing")                       # ./sing "lyrics"  ==  ./sing sing "lyrics"
    a = p.parse_args(argv)
    bundle = a.bundle.resolve()
    outs = (a.outputs or bundle / "outputs").resolve()
    if a.cmd in (None, "tunes"):
        if a.cmd is None:
            p.print_help(); print()
        print_tunes(bundle); return
    if a.cmd == "voices":
        print_voices(bundle); return
    if a.cmd == "fetch":
        from .singer import fetch_base_weights
        print(fetch_base_weights(bundle)); return

    from .singer import KalavinkaSinger, SingError, save_wav
    if a.cmd == "ui":
        from .app import launch
        launch(bundle, device=a.device, share=a.share, port=a.port)
        return
    singer = KalavinkaSinger(bundle, device=a.device)
    try:
        if a.cmd == "sing":
            mode = "expressive" if a.expressive else "score"
            wav, sr, info = singer.sing(a.lyrics, a.tune, a.voice, mode, a.steps, a.seed)
            path = save_wav(a.out or take_path(outs, f"{slug(a.lyrics)}_{mode}"), wav, sr)
            path.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
            for ln in info["lines"]:
                print(f"  \"{ln['line']}\" on tune {ln['tune']} ({ln['tune_song']}), voice {ln['voice']}; "
                      f"{int(ln['seen_in_training'] * 100)}% of its phrases occur in the training lyrics")
            print(f"saved {path}  ({info['seconds']} s of singing, took {info['took_s']} s on {info['device']})")
            play(path, a.play)
        elif a.cmd == "resing":
            out = singer.resing(a.tune, a.mode, a.steps, a.seed)
            m = save_wav(take_path(outs, f"tune{a.tune}_{a.mode}"), *out["model"])
            real = save_wav(m.with_name(m.stem + "_REAL.wav"), *out["real"])
            print(f"{out['song']}: {out['lyric']}\n  model: {m}\n  real singer: {real}\n  took {out['took_s']} s")
            play(m, a.play); play(real, a.play)
        elif a.cmd == "demo":
            d = outs / "demo"
            for k, line in enumerate(EXAMPLES):
                for mode in ("score", "expressive"):
                    wav, sr, info = singer.sing(line, None, None, mode, a.steps, a.seed)
                    path = save_wav(d / f"{k + 1}_{mode}.wav", wav, sr)
                    print(f"{path.name}: {line} (tune {info['lines'][0]['tune']}, {info['took_s']} s)")
            print(f"demo takes in {d}")
    except SingError as e:
        print(f"\nCannot sing that: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
