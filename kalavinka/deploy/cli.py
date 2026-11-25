"""Command line for the Kalavinka singer (``python -m kalavinka_app ...`` inside a bundle; ``./sing ...`` on the Mac).

Commands::

    sing "ठंडी हवा में गीत नया" [--tune 3] [--voice 5 | --voice "Lata"] [--expressive] [--out song.wav]
                                sing new lyrics (several lines: separate them with | or new lines; they are sung
                                as one performance: --join separate for the old line-by-line way, --line-gap 0.3)
    "ठंडी हवा में गीत नया"      same as sing (the word "sing" can be left out)
    resing 3 [--mode score|expressive|real-pitch]
                                sing bundled tune 3 with its original words, and save the real singer next to it
    tunes                       list the tunes (song, syllables, seconds, original words)
    voices [--gender female] [--era 1990s]
                                list the voices with their profiles (estimated gender, register, era, credited singer)
    voice-edit 7 --name "..." [--gender male] [--notes "..."]
                                correct or complete a voice's profile (kept in outputs/voice_profiles.local.json)
    demo                        sing three example lines (both pitch modes) into outputs/demo/
    ui [--share] [--port 7860]  open the web page (type lyrics, pick tune and voice, listen)
    fetch                       download SoulX's base weights now (otherwise done on first use)
    alankar 2 | all | list      sing a built-in alankar (sargam exercise) and check its pitch note by note
    sargam "Sa Re Ga | Re Ga ma" [--json score.json]
                                sing your own swaras (notation: see kalavinka_app/sargam.py), or an LLM-written JSON
    prompt                      print the prompt to give ChatGPT (or any LLM) to write a sargam JSON for you
    compose score.json [--model both] [--voice 4]
                                sing your own composition (swaras, lengths, lyrics, ornaments, hand-drawn pitch;
                                format: kalavinka_app/compose.py); the web page's Compose tab edits it on the F0 curve
    compose --example | --from-tune 3 | --sargam "Sa Re Ga" --save x.json
                                write a starting composition to edit (by hand, in the web page, or with an LLM)
    compose-prompt              print the prompt to give ChatGPT (or any LLM) to write a composition JSON
    history [--n 20] [--search WORDS] [--star] [--kind sing|sargam|compose|resing] [--deleted]
                                list earlier takes (every take is kept in outputs/ and logged in history.jsonl)
    note 12 "text" | title 12 "text" | tag 12 "a b" | star 12 | unstar 12 | delete 12 | restore 12
                                label, star, delete (to outputs/trash/) or restore take 12
    log [--n 30]                what the app did (outputs/log.jsonl): takes, cache hits, errors, edits
    rebuild-history             re-create history.jsonl from the per-take .json files

Compare with the base model: add ``--model both`` (or ``base``) to sing, resing, alankar and sargam. ``base`` is the
released SoulX-Singer with Hindi phones mapped to English ones (our zero-shot baseline); ``kalavinka`` is ours.
Cache: a request you made before (same words, tune, voice, model, mode, seed, steps ...) is not sung again; the saved
take is reused. ``--fresh`` forces a new run; ``--seed N`` gives a new take.

Common options (before or after the command): ``--steps N`` (default 32 on a GPU, 16 on CPU; fewer = faster,
rougher), ``--seed 0`` (another seed = another take), ``--fresh``, ``--play`` (play each take), ``--outputs DIR``.
Before the command only: ``--device auto|cuda|mps|cpu`` (default auto), ``--bundle DIR`` (default: the folder this
package sits in).
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


def print_voices(bundle: Path, outputs: Path, gender: str = "any", era: str = "any") -> None:
    """Voice table with profiles (no model needed)."""
    from .voices import filter_profiles, load
    ps = filter_profiles(load(bundle, outputs), gender or "any", era or "any")
    print(f"{'no':>3}  {'gender':7s} {'register':8s} {'era':6s} {'Sa Hz':>6s}  {'singer':24s} source song")
    for p in ps:
        print(f"{p['no']:>3}  {p['gender']:7s} {p['register']:8s} {p['decade']:6s} {p['sa_hz'] or 0:6.0f}  "
              f"{(p['name'] or '-')[:24]:24s} {p['song']} ({p['year']})" + ("  [edited]" if p["edited"] else ""))
    print("\nGender and register are estimated (gender from the song's credited singers, pitch for duets); fix them "
          "with: ./sing voice-edit N --name ... --gender ...")


def resolve_voice(bundle: Path, outputs: Path, spec):
    """--voice value (number or a name / song fragment) -> number."""
    if spec in (None, "") or str(spec).strip().isdigit():
        return int(spec) if spec not in (None, "") else None
    from .voices import load, resolve
    try:
        return resolve(load(bundle, outputs), spec)
    except ValueError as e:
        raise SystemExit(f"--voice: {e}")


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


def sargam_jobs(a):
    """(jobs, settings) for alankar / sargam, checked before the model is loaded."""
    from .sargam import ALANKARS, SargamError, alankar_lines, from_json, parse_sargam
    settings = {}
    if a.cmd == "alankar":
        if a.which != "all" and (not str(a.which).isdigit() or int(a.which) not in ALANKARS):
            raise SargamError(f"there is no alankar {a.which!r}; pick {min(ALANKARS)}-{max(ALANKARS)}, all, or list")
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
    return jobs, settings


def run_sargam(a, singer, hist, steps: int, jobs, settings) -> None:
    """alankar / sargam commands: sing with each model (cached), then print the pitch check side by side."""
    from .seamless import SeamlessSargamSinger
    from .sargam import to_text
    tonic = a.tonic or settings.get("tonic", "auto")
    beat = float(a.beat or settings.get("beat_s", 0.45))
    vowel = a.vowel or settings.get("syllables", "sargam")
    ss = SeamlessSargamSinger(singer)
    voice = a.voice or 1
    for title, lines in jobs:
        print(f"\n== {title}\n" + to_text(lines))
        rows = []
        for model in models_of(a.model):
            req = {"title": title, "sargam": to_text(lines), "model": model, "voice": voice, "tonic": tonic,
                   "beat_s": beat, "syllables": vowel, "seed": a.seed, "steps": steps, "join": a.join,
                   "line_gap": a.line_gap}
            wav, sr, info, path, hit = hist.cached(
                "sargam", req, lambda: ss.sing(lines, model, voice, tonic, beat, vowel, steps, a.seed, join=a.join,
                                               line_gap=a.line_gap),
                f"{model}_{title.split(':')[0]}", a.fresh)
            p = info["pitch"]
            rows.append((model, p, path, hit))
            play(path, a.play)
        print(f"  {'model':10s} {'on the swara (50 c)':>20s} {'in tune (25 c)':>15s} {'mean error':>11s} {'voiced':>7s}")
        for model, p, path, hit in rows:
            print(f"  {model:10s} {p['within_50c'] * 100:19.0f}% {p['within_25c'] * 100:14.0f}% "
                  f"{p['mean_abs_cents']:9.1f} c {p['voiced_share'] * 100:6.0f}%   {'(reused) ' if hit else ''}{path.name}")


def run_compose(a, bundle: Path, hist, singer_factory) -> None:
    """compose command: write a starting composition (--save without a file), or sing one with each model (cached)
    and print how closely each model followed the designed pitch curve."""
    from .compose import EXAMPLE, ComposeError, ComposeSinger, from_sargam, from_tune, normalise
    if a.example:
        comp = EXAMPLE
    elif a.from_tune:
        tu = next((t for t in read_jsonl(bundle / "data" / "tunes.jsonl") if t["no"] == a.from_tune), None)
        if tu is None:
            raise ComposeError(f"there is no tune {a.from_tune} (see: ./sing tunes)")
        comp = from_tune(tu, keep_curve=not a.plain)
    elif a.sargam:
        comp = from_sargam(a.sargam)
    elif a.file:
        comp = normalise(a.file.read_text(encoding="utf-8"))
    else:
        raise ComposeError("give a composition file, or --example / --from-tune N / --sargam \"...\" with --save")
    comp = normalise(comp)
    if a.voice:
        comp["voice"] = a.voice
    if a.save:
        a.save.write_text(json.dumps(comp, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {a.save}; edit it, then: ./sing compose {a.save}")
        if not a.file:
            return
    if a.phrase_gap is not None:
        comp["phrase_gap_s"] = a.phrase_gap
        comp = normalise(comp)
    singer = singer_factory()
    from .singer import default_steps
    steps = int(a.steps or default_steps(singer.device))
    cs = ComposeSinger(singer)
    print(f"== {comp.get('title', 'composition')}")
    for model in models_of(a.model):
        req = {"comp": comp, "model": model, "seed": a.seed, "steps": steps, "join": a.join}
        wav, sr, info, path, hit = hist.cached("compose", req,
                                               lambda: cs.sing(comp, model, None, steps, a.seed, join=a.join),
                                               f"{model}_compose_{comp.get('title', '')}", a.fresh)
        f = info["follow"]
        print(f"  [{model}] {f['within_50c'] * 100:.0f}% of the curve within 50 c, {f['within_25c'] * 100:.0f}% within "
              f"25 c, mean error {f['mean_abs_cents']} c   {'(reused) ' if hit else ''}{path}")
        play(path, a.play)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the command and run it."""
    p = argparse.ArgumentParser(prog="sing", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, default=default_bundle())
    p.add_argument("--device", default="auto")

    def common(q, top: bool):
        """Options accepted before and after the command (after: only if given, so they never reset the value)."""
        d = (lambda v: v) if top else (lambda v: argparse.SUPPRESS)
        q.add_argument("--steps", type=int, default=d(None), help="flow steps (default 32 on a GPU, 16 on CPU)")
        q.add_argument("--seed", type=int, default=d(0))
        q.add_argument("--outputs", type=Path, default=d(None), help="folder for the takes (default: <bundle>/outputs)")
        q.add_argument("--play", action="store_true", default=d(False), help="play each take when it is ready")
        q.add_argument("--fresh", action="store_true", default=d(False),
                       help="sing again even if this exact request is in the history")

    def joins(q):
        """--join / --line-gap."""
        q.add_argument("--join", choices=["seamless", "separate"], default="seamless",
                       help="seamless = one performance, silence only at rests (default); separate = old line-by-line")
        q.add_argument("--line-gap", type=float, default=0.3, help="seconds between lines where the song has no gap")
    common(p, True)
    sub = p.add_subparsers(dest="cmd")
    _add = sub.add_parser

    def add_parser(*args, **kw):
        q = _add(*args, **kw)
        common(q, False)
        return q
    sub.add_parser = add_parser
    s = sub.add_parser("sing", help="sing new lyrics")
    s.add_argument("lyrics")
    s.add_argument("--tune", type=int, default=None)
    s.add_argument("--voice", default=None, help="a voice number, or part of the singer's name / source song")
    joins(s)
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
        g.add_argument("--voice", default=None, help="default: voice 1 (a number or part of the singer's name)")
        joins(g)
        g.add_argument("--tonic", default=None, help="auto (the voice's own Sa), a note like C#3, or Hz")
        g.add_argument("--beat", type=float, default=None, help="seconds per note (default 0.45)")
        g.add_argument("--vowel", choices=["sargam", "aa"], default=None, help="sing the swara names, or 'aa' (akaar)")
    c = sub.add_parser("compose", help="sing your own composition (pitch, timing, ornaments)")
    c.add_argument("file", nargs="?", type=Path, default=None, help="a composition JSON")
    c.add_argument("--example", action="store_true", help="start from the built-in Yaman example")
    c.add_argument("--from-tune", type=int, default=None, help="start from bundled tune N (with its real pitch)")
    c.add_argument("--plain", action="store_true", help="with --from-tune: plain notes, without the real pitch curve")
    c.add_argument("--sargam", default=None, help='start from sargam text, e.g. "Sa Re Ga | ma Pa"')
    c.add_argument("--save", type=Path, default=None, help="write the composition here")
    c.add_argument("--model", choices=MODELS, default="both")
    c.add_argument("--voice", default=None)
    c.add_argument("--join", choices=["seamless", "separate"], default="seamless")
    c.add_argument("--phrase-gap", type=float, default=None, help="seconds between phrases (default: the file's, 0.3)")
    sub.add_parser("compose-prompt", help="the LLM prompt for writing a composition JSON")
    h = sub.add_parser("history", help="list earlier takes")
    h.add_argument("--n", type=int, default=20)
    h.add_argument("--search", default="")
    h.add_argument("--star", action="store_true", help="only starred takes")
    h.add_argument("--kind", default="")
    h.add_argument("--deleted", action="store_true", help="show deleted takes (in outputs/trash/)")
    for name in ("note", "title", "tag"):
        q = sub.add_parser(name, help=f"set a take's {name}")
        q.add_argument("id", type=int)
        q.add_argument("text")
    for name in ("star", "unstar", "delete", "restore"):
        sub.add_parser(name, help=f"{name} a take").add_argument("id", type=int)
    lg = sub.add_parser("log", help="what the app did (outputs/log.jsonl)")
    lg.add_argument("--n", type=int, default=30)
    sub.add_parser("rebuild-history", help="re-create history.jsonl from the per-take .json files")
    ve = sub.add_parser("voice-edit", help="correct or complete a voice's profile")
    ve.add_argument("voice", type=int)
    ve.add_argument("--name", default=None)
    ve.add_argument("--gender", choices=["female", "male", "unsure"], default=None)
    ve.add_argument("--register", choices=["low", "mid", "high"], default=None)
    ve.add_argument("--notes", default=None)
    ve.add_argument("--confirmed", action="store_true", help="the name is confirmed (by ear or by the song)")
    sub.add_parser("prompt", help="the LLM prompt for writing a sargam JSON")
    sub.add_parser("tunes"); sub.add_parser("demo")
    vo = sub.add_parser("voices")
    vo.add_argument("--gender", default="any")
    vo.add_argument("--era", default="any")
    sub.add_parser("fetch", help="download SoulX's base weights (2.7 GB) now")
    u = sub.add_parser("ui", help="open the web page")
    u.add_argument("--share", action="store_true", help="public link (needed on Colab)")
    u.add_argument("--port", type=int, default=7860)
    argv = list(sys.argv[1:] if argv is None else argv)
    cmds = {"sing", "resing", "tunes", "voices", "demo", "ui", "fetch", "alankar", "sargam", "history", "prompt",
            "compose", "compose-prompt", "note", "title", "tag", "star", "unstar", "delete", "restore", "log",
            "rebuild-history", "voice-edit", "-h", "--help"}
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
        print_voices(bundle, outs, a.gender, a.era); return
    if a.cmd == "voice-edit":
        from .voices import save_edit
        fields = {k: v for k, v in (("name", a.name), ("gender", a.gender), ("register", a.register),
                                    ("notes", a.notes)) if v is not None}
        if a.confirmed:
            fields["singer_confirmed"] = True
        save_edit(outs, a.voice, **fields)
        from .history import History
        History(outs).log("voice_edit", voice=a.voice, **fields)
        print_voices(bundle, outs); return
    if hasattr(a, "voice") and a.cmd in ("sing", "alankar", "sargam", "compose"):
        a.voice = resolve_voice(bundle, outs, a.voice)
    if a.cmd == "fetch":
        from .singer import fetch_base_weights
        print(fetch_base_weights(bundle)); return
    from .history import History, describe
    hist = History(outs)
    if a.cmd == "history":
        recs = hist.search(a.search, a.kind, "", a.star, a.deleted, a.n)
        print("\n".join(describe(r) + f"\n      {hist.wav_path(r)}" for r in recs) or "no takes found")
        return
    if a.cmd in ("note", "title", "tag", "star", "unstar", "delete", "restore"):
        if hist.get(a.id) is None:
            raise SystemExit(f"there is no take {a.id} (see: ./sing history)")
        if a.cmd in ("note", "title", "tag"):
            hist.annotate(a.id, {"note": "notes", "title": "title", "tag": "tags"}[a.cmd], a.text)
        elif a.cmd in ("star", "unstar"):
            hist.annotate(a.id, "star", a.cmd == "star")
        else:
            getattr(hist, a.cmd)(a.id)
        print(describe(hist.get(a.id)))
        return
    if a.cmd == "log":
        for e in reversed(hist.events(a.n)):
            extra = {k: v for k, v in e.items() if k not in ("time", "event", "trace")}
            print(f"{e['time']}  {e['event']:14s} {json.dumps(extra, ensure_ascii=False)[:160]}")
        return
    if a.cmd == "rebuild-history":
        print(f"rebuilt history.jsonl from {hist.rebuild()} take files"); return
    if a.cmd == "prompt":
        from .sargam import LLM_PROMPT
        print(LLM_PROMPT); return
    if a.cmd == "compose-prompt":
        from .compose import COMPOSE_PROMPT
        print(COMPOSE_PROMPT); return
    if a.cmd == "compose":
        from .compose import ComposeError
        from .seamless import SeamlessSinger
        from .singer import SingError
        try:
            run_compose(a, bundle, hist, lambda: SeamlessSinger(bundle, device=a.device))
        except (ComposeError, SingError) as e:
            print(f"\nCannot sing that: {e}", file=sys.stderr)
            sys.exit(2)
        return
    if a.cmd == "alankar" and a.which == "list":
        from .sargam import ALANKARS, alankar_lines, to_text
        for no, al in ALANKARS.items():
            print(f"{no}. {al['name']}\n   " + to_text(alankar_lines(no)).replace("\n", "\n   "))
        return

    from .sargam import SargamError
    from .seamless import SeamlessSinger
    from .singer import SingError, default_steps, save_wav
    if a.cmd == "ui":
        from .app import launch
        launch(bundle, device=a.device, share=a.share, port=a.port, outputs=outs, steps=a.steps, seed=a.seed)
        return
    jobs = settings = None
    try:                                   # check the request before loading the model (that takes a while)
        if a.cmd in ("alankar", "sargam"):
            jobs, settings = sargam_jobs(a)
        tunes_file = bundle / "data" / "tunes.jsonl"
        if a.cmd == "sing" and a.tune is not None and tunes_file.exists():
            n_tunes = len(read_jsonl(tunes_file))
            if not 1 <= a.tune <= n_tunes:
                raise SingError(f"there is no tune {a.tune}; pick 1 to {n_tunes} (see: ./sing tunes)")
    except (SingError, SargamError) as e:
        print(f"\nCannot sing that: {e}", file=sys.stderr)
        sys.exit(2)
    singer = SeamlessSinger(bundle, device=a.device)
    steps = int(a.steps or default_steps(singer.device))
    try:
        if a.cmd == "sing":
            mode = "expressive" if a.expressive else "score"
            for model in models_of(a.model):
                req = {"lyrics": a.lyrics, "tune": a.tune, "voice": a.voice, "mode": mode, "model": model,
                       "seed": a.seed, "steps": steps, "join": a.join, "line_gap": a.line_gap}
                wav, sr, info, path, hit = hist.cached(
                    "sing", req, lambda: singer.sing(a.lyrics, a.tune, a.voice, mode, steps, a.seed, model=model,
                                                     join=a.join, line_gap=a.line_gap),
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
                    return w, sr, {"model": out["model_name"], **{k: out[k] for k in ("lyric", "song", "mode", "took_s")}}
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
            run_sargam(a, singer, hist, steps, jobs, settings)
        elif a.cmd == "demo":
            for k, line in enumerate(EXAMPLES):
                for mode in ("score", "expressive"):
                    req = {"lyrics": line, "tune": None, "voice": None, "mode": mode, "model": "kalavinka",
                           "seed": a.seed, "steps": steps, "join": "seamless", "line_gap": 0.3}
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
