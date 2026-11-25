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
    alankar 2 | all | list      sing a built-in alankar (sargam exercise) and check its pitch note by note
    sargam "Sa Re Ga | Re Ga ma" [--json score.json]
                                sing your own swaras (notation: see kalavinka_app/sargam.py), or an LLM-written JSON
    prompt                      print the prompt to give ChatGPT (or any LLM) to write a sargam JSON for you
    history [--n 20]            list earlier takes (every take is kept in outputs/ and logged in history.jsonl)

Compare with the base model: add ``--model both`` (or ``base``) to sing, resing, alankar and sargam. ``base`` is the
released SoulX-Singer with Hindi phones mapped to English ones (our zero-shot baseline); ``kalavinka`` is ours.
Cache: a request you made before (same words, tune, voice, model, mode, seed, steps ...) is not sung again; the saved
take is reused. ``--fresh`` forces a new run; ``--seed N`` gives a new take.

Common options (before the command): ``--device auto|cuda|mps|cpu`` (default auto), ``--steps N`` (default 32 on a
GPU, 16 on CPU; fewer = faster, rougher), ``--play`` (play each take),
``--seed 0`` (another seed = another take), ``--bundle DIR`` (default: the folder this package sits in).
Every take is written to ``outputs/`` with the request and settings in a matching ``.json``.
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


MODELS = ["kalavinka", "base", "both"]


def models_of(choice: str):
    """'both' -> [kalavinka, base]."""
    return ["kalavinka", "base"] if choice == "both" else [choice]


def run_sargam(a, singer, hist, steps: int) -> None:
    """alankar / sargam commands: sing with each model (cached), then print the pitch check side by side."""
    from .sargam import ALANKARS, SargamError, SargamSinger, alankar_lines, from_json, parse_sargam, to_text
    jobs = []
    settings = {}
    if a.cmd == "alankar":
        nos = list(ALANKARS) if a.which == "all" else [int(a.which)]
        jobs = [(f"alankar {n}: {ALANKARS[n]['name']}", alankar_lines(n)) for n in nos]
    else:
        if a.json:
            lines, settings = from_json(a.json.read_text(encoding="utf-8"))
        elif a.text:
            lines = parse_sargam(a.text)
        else:
            raise SargamError('give swaras, e.g. ./sing sargam "Sa Re Ga | Re Ga ma", or --json score.json')
        jobs = [(settings.get("title") or "my sargam", lines)]
    tonic = a.tonic or settings.get("tonic", "auto")
    beat = float(a.beat or settings.get("beat_s", 0.45))
    vowel = a.vowel or settings.get("syllables", "sargam")
    ss = SargamSinger(singer)
    voice = a.voice or 1
    for title, lines in jobs:
        print(f"\n== {title}\n" + to_text(lines))
        rows = []
        for model in models_of(a.model):
            req = {"title": title, "sargam": to_text(lines), "model": model, "voice": voice, "tonic": tonic,
                   "beat_s": beat, "syllables": vowel, "seed": a.seed, "steps": steps}
            wav, sr, info, path, hit = hist.cached(
                "sargam", req, lambda: ss.sing(lines, model, voice, tonic, beat, vowel, steps, a.seed),
                f"{model}_{title.split(':')[0]}", a.fresh)
            p = info["pitch"]
            rows.append((model, p, path, hit))
            play(path, a.play)
        print(f"  {'model':10s} {'on the swara (50 c)':>20s} {'in tune (25 c)':>15s} {'mean error':>11s} {'voiced':>7s}")
        for model, p, path, hit in rows:
            print(f"  {model:10s} {p['within_50c'] * 100:19.0f}% {p['within_25c'] * 100:14.0f}% "
                  f"{p['mean_abs_cents']:9.1f} c {p['voiced_share'] * 100:6.0f}%   {'(reused) ' if hit else ''}{path.name}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the command and run it."""
    p = argparse.ArgumentParser(prog="sing", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, default=default_bundle())
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=None, help="flow steps (default 32 on a GPU, 16 on CPU)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outputs", type=Path, default=None, help="folder for the takes (default: <bundle>/outputs)")
    p.add_argument("--play", action="store_true", help="play each take when it is ready (Mac: afplay)")
    p.add_argument("--fresh", action="store_true", help="sing again even if this exact request is in the history")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("sing", help="sing new lyrics")
    s.add_argument("lyrics")
    s.add_argument("--tune", type=int, default=None)
    s.add_argument("--voice", type=int, default=None)
    s.add_argument("--expressive", action="store_true", help="two-stage: stage-1 contour model draws the pitch")
    s.add_argument("--out", type=Path, default=None, help="also copy the take here")
    s.add_argument("--model", choices=MODELS, default="kalavinka")
    r = sub.add_parser("resing", help="re-sing a bundled tune with its original words")
    r.add_argument("tune", type=int)
    r.add_argument("--mode", choices=["score", "expressive", "real-pitch"], default="score")
    r.add_argument("--model", choices=MODELS, default="kalavinka")
    for name, hlp in (("alankar", "sing a built-in alankar"), ("sargam", "sing your own swaras")):
        g = sub.add_parser(name, help=hlp)
        if name == "alankar":
            g.add_argument("which", nargs="?", default="list", help="1-7, all, or list")
        else:
            g.add_argument("text", nargs="?", default=None, help='e.g. "Sa Re Ga | Re Ga ma"')
            g.add_argument("--json", type=Path, default=None, help="a JSON score (see: ./sing prompt)")
        g.add_argument("--model", choices=MODELS, default="both")
        g.add_argument("--voice", type=int, default=None, help="default: voice 1")
        g.add_argument("--tonic", default=None, help="auto (the voice's own Sa), a note like C#3, or Hz")
        g.add_argument("--beat", type=float, default=None, help="seconds per note (default 0.45)")
        g.add_argument("--vowel", choices=["sargam", "aa"], default=None, help="sing the swara names, or 'aa' (akaar)")
    h = sub.add_parser("history", help="list earlier takes")
    h.add_argument("--n", type=int, default=20)
    sub.add_parser("prompt", help="the LLM prompt for writing a sargam JSON")
    sub.add_parser("tunes"); sub.add_parser("voices"); sub.add_parser("demo")
    sub.add_parser("fetch", help="download SoulX's base weights (2.7 GB) now")
    u = sub.add_parser("ui", help="open the web page")
    u.add_argument("--share", action="store_true", help="public link (needed on Colab)")
    u.add_argument("--port", type=int, default=7860)
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"sing", "resing", "tunes", "voices", "demo", "ui", "fetch", "alankar", "sargam", "history", "prompt",
            "-h", "--help"}
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
    from .history import History, describe
    hist = History(outs)
    if a.cmd == "history":
        recs = hist.recent(a.n)
        print("\n".join(describe(r) + f"\n      {outs / r['wav']}" for r in recs) or "no takes yet")
        return
    if a.cmd == "prompt":
        from .sargam import LLM_PROMPT
        print(LLM_PROMPT); return
    if a.cmd == "alankar" and a.which == "list":
        from .sargam import ALANKARS, alankar_lines, to_text
        for no, al in ALANKARS.items():
            print(f"{no}. {al['name']}\n   " + to_text(alankar_lines(no)).replace("\n", "\n   "))
        return

    from .sargam import SargamError
    from .singer import KalavinkaSinger, SingError, default_steps, save_wav
    if a.cmd == "ui":
        from .app import launch
        launch(bundle, device=a.device, share=a.share, port=a.port)
        return
    singer = KalavinkaSinger(bundle, device=a.device)
    steps = int(a.steps or default_steps(singer.device))
    try:
        if a.cmd == "sing":
            mode = "expressive" if a.expressive else "score"
            for model in models_of(a.model):
                req = {"lyrics": a.lyrics, "tune": a.tune, "voice": a.voice, "mode": mode, "model": model,
                       "seed": a.seed, "steps": steps}
                wav, sr, info, path, hit = hist.cached(
                    "sing", req, lambda: singer.sing(a.lyrics, a.tune, a.voice, mode, steps, a.seed, model=model),
                    f"{model}_{mode}_{a.lyrics}", a.fresh)
                for ln in info["lines"]:
                    print(f"  [{model}] \"{ln['line']}\" on tune {ln['tune']} ({ln['tune_song']}), voice {ln['voice']}; "
                          f"{int(ln['seen_in_training'] * 100)}% of its phrases occur in the training lyrics")
                print(f"{'reused' if hit else 'saved'} {path}  ({info['seconds']} s of singing"
                      + ("" if hit else f", took {info['took_s']} s on {info['device']}") + ")")
                if a.out:
                    out = a.out if len(models_of(a.model)) == 1 else a.out.with_name(f"{a.out.stem}_{model}{a.out.suffix}")
                    save_wav(out, wav, sr)
                play(path, a.play)
        elif a.cmd == "resing":
            real = None
            for model in models_of(a.model):
                req = {"tune": a.tune, "mode": a.mode, "model": model, "seed": a.seed, "steps": steps}

                def make(model=model):
                    """Re-sing and keep the real clip for printing."""
                    nonlocal real
                    out = singer.resing(a.tune, a.mode, steps, a.seed, model=model)
                    real = out["real"]
                    w, sr = out["model"]
                    return w, sr, {k: out[k] for k in ("lyric", "song", "mode", "model", "took_s")}
                wav, sr, info, path, hit = hist.cached("resing", req, make, f"{model}_{a.mode}_tune{a.tune}", a.fresh)
                print(f"[{model}] {info['song']}: {info['lyric']}\n  {'reused' if hit else 'saved'} {path}")
                play(path, a.play)
            import soundfile as sf
            from .history import slug
            tu = next(t for t in read_jsonl(bundle / "data" / "tunes.jsonl") if t["no"] == a.tune)
            rpath = outs / f"REAL_tune{a.tune}_{slug(tu['song'])}.wav"
            if not rpath.exists():
                w, sr = sf.read(bundle / "data" / tu["item"]["wav"], dtype="float32")
                save_wav(rpath, w, sr)
            print(f"  real singer: {rpath}")
            play(rpath, a.play)
        elif a.cmd in ("alankar", "sargam"):
            run_sargam(a, singer, hist, steps)
        elif a.cmd == "demo":
            for k, line in enumerate(EXAMPLES):
                for mode in ("score", "expressive"):
                    req = {"lyrics": line, "tune": None, "voice": None, "mode": mode, "model": "kalavinka",
                           "seed": a.seed, "steps": steps}
                    wav, sr, info, path, hit = hist.cached(
                        "sing", req, lambda: singer.sing(line, None, None, mode, steps, a.seed), f"demo{k + 1}_{mode}",
                        a.fresh)
                    print(f"{path.name}: {line} (tune {info['lines'][0]['tune']}{', reused' if hit else ''})")
            print(f"demo takes in {outs}")
    except (SingError, SargamError) as e:
        print(f"\nCannot sing that: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
