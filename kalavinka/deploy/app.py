"""A small web page for the Kalavinka singer (Gradio): type Hindi lyrics, pick a tune and a voice, press Sing.

Tabs:

* **Sing new lyrics**: lyrics box (with examples), tune (auto or a bundled one; optionally only tunes by the chosen
  voice), voice (the tune's own singer, or any voice: filters by gender / era / register, profile card, sample),
  pitch mode (the model's own, or expressive two-stage), Sing -> audio + what was sung where.
* **Compare with the real singer**: pick a tune, re-sing its original words, hear the model and the real clip.
* **Alankars / sargam**: a built-in alankar or your own swaras (text, or an LLM-written JSON), with a note-by-note
  pitch check.
* **Compose (pitch, timing, ornaments)**: write the melody yourself on the F0 curve (``compose_ui``).
* **Voices**: every voice with its profile (estimated gender and register, era, source song, credited singer);
  listen, filter, and correct or add names (``voices_ui``).
* **History**: every take (all tabs and the command line): search, play, label (title, tags, notes, star), delete /
  restore, export, the event log, and **Open in its tab** to change an old take and sing it again (``history_ui``).

Page-wide settings: **Model** (Kalavinka, base SoulX-Singer, or both side by side), steps, seed, and **Join**:
*seamless* (default; lines and phrases are sung as one performance and silence appears only where the score has a
rest, ``seamless``) or *separate* (the old way: each line rendered alone, short gaps between), plus the gap between
lines. Takes are cached in ``outputs/`` (``history``): the same request plays the saved take; change the seed for a
new one. Renders run one at a time (a shared queue), so several browser tabs or users cannot collide on the model.

Launch: ``python -m kalavinka_app ui`` (``--share`` for a public link, needed on Colab), or ``launch(bundle)`` /
``launch(bundle, singer=...)`` to reuse an already loaded singer.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Optional

from .cli import EXAMPLES, read_jsonl

HOW = """**How it works.** Kalavinka is SoulX-Singer fine-tuned on 10.7 hours of Hindi film songs. Your lyrics are
split into syllables, put on the notes of a real song line the model never trained on (the *tune*), and sung in the
*voice* of a short clip. **Model's own pitch** lets the model shape the melody (sounds most natural in our listening
test). **Expressive (two-stage)** first draws a pitch curve with vibrato using our contour model, then sings it.
Lyrics must be in Devanagari; one line per phrase (or separate with |)."""

MODEL_CHOICES = {"Kalavinka (ours)": ["kalavinka"], "Base SoulX-Singer": ["base"],
                 "Both, side by side": ["kalavinka", "base"]}
MODEL_OF = {"kalavinka": "Kalavinka (ours)", "base": "Base SoulX-Singer"}
JOIN_CHOICES = {"Seamless (one performance)": "seamless", "Separate (old: each line alone)": "separate"}
TAB_IDS = {"sing": "t_sing", "resing": "t_resing", "sargam": "t_sargam", "compose": "t_compose"}


def launch(bundle: Path, device: str = "auto", share: bool = False, port: int = 7860, outputs: Optional[Path] = None,
           steps: Optional[int] = None, seed: int = 0, singer=None):
    """Load the singer (unless one is given) and serve the page (blocks)."""
    import gradio as gr
    from .compose_ui import build_tab as build_compose_tab, head_html
    from .history import History
    from .history_ui import build_history_tab
    from .sargam import ALANKARS, LLM_PROMPT, SargamError, alankar_lines, from_json, parse_sargam, to_text
    from .seamless import SeamlessSargamSinger, SeamlessSinger
    from .singer import SingError, default_steps
    from .voices_ui import AUTO, VoiceBook, VoicePicker, build_voices_tab
    bundle = Path(bundle)
    outs = Path(outputs) if outputs else bundle / "outputs"
    if singer is None:
        singer = SeamlessSinger(bundle, device=device)
    elif not isinstance(singer, SeamlessSinger):
        singer.__class__ = SeamlessSinger                    # same object, seamless sing() (no reload)
    hist = History(outs)
    hist.log("app_start", device=singer.device, bundle=str(bundle))
    ss = SeamlessSargamSinger(singer)
    book = VoiceBook(bundle, outs, hist)
    tunes = read_jsonl(bundle / "data" / "tunes.jsonl")

    def tune_label(t):
        p = book.get(t["voice"])
        who = f" · Voice {t['voice']}" + (f" ({p['name']})" if p and p.get("name") else "")
        return f"{t['no']}: {t['song']} ({t['year']}) · {t['syllables']} syll. · {t['seconds']} s{who}"
    tune_names = ["auto (fits the line)"] + [tune_label(t) for t in tunes]
    modes = {"Model's own pitch (recommended)": "score", "Expressive (two-stage, vibrato)": "expressive"}
    mode_name = {v: k for k, v in modes.items()}
    rmodes = {"Model's own pitch": "score", "Expressive (two-stage)": "expressive", "Real singer's pitch": "real-pitch"}
    alankar_names = [f"{n}: {a['name']}" for n, a in ALANKARS.items()]

    def num(choice):
        """Leading number of a dropdown entry or voice value; None for auto / empty."""
        if choice in (None, "", AUTO):
            return None
        head = str(choice).split(":")[0].split("·")[0].strip()
        return int(head) if head.isdigit() else None

    def two(results):
        """Pad a list of (sr, wav) to two players (None = empty)."""
        return (results + [None, None])[:2]

    def do_sing(lyrics, tune, voice, mode, model, steps_, seed_, join, gap):
        """Gradio callback: sing with one or both models (cached) and describe."""
        outs_, notes = [], []
        jn = JOIN_CHOICES.get(join, join)
        for m in MODEL_CHOICES[model]:
            req = {"lyrics": lyrics, "tune": num(tune), "voice": num(voice), "mode": modes[mode], "model": m,
                   "seed": int(seed_), "steps": int(steps_), "join": jn, "line_gap": float(gap)}
            try:
                wav, sr, info, path, hit = hist.cached(
                    "sing", req, lambda m=m: singer.sing(lyrics, num(tune), num(voice), modes[mode], int(steps_),
                                                         int(seed_), model=m, join=jn, line_gap=float(gap)),
                    f"{m}_{modes[mode]}_{lyrics}")
            except SingError as e:
                raise gr.Error(str(e))
            outs_.append((sr, wav))
            lines = "; ".join(f"**{l['line']}** on tune {l['tune']} ({l['tune_song']}), voice {l['voice']}"
                              + (f" (moved {l['shift']:+d} semitones)" if l.get("shift") else "")
                              + f", {int(l['seen_in_training'] * 100)}% seen in training" for l in info["lines"])
            made = "from history" if hit else f"made in {info['took_s']} s"
            joined = f", {info.get('pieces', len(info['lines']))} render(s), {info.get('join', 'separate')}"
            notes.append(f"*{m}*: {lines} ({made}{joined})")
        return *two(outs_), "\n\n".join(notes)

    def do_resing(tune, mode, model, steps_, seed_):
        """Gradio callback: model(s) vs the real singer on a bundled line."""
        n = num(tune) or 1
        m_ = rmodes[mode]
        outs_, about = [], ""
        for m in MODEL_CHOICES[model]:
            req = {"tune": n, "mode": m_, "model": m, "seed": int(seed_), "steps": int(steps_)}

            def make(m=m):
                o = singer.resing(n, m_, int(steps_), int(seed_), model=m)
                w, sr = o["model"]
                return w, sr, {"model": o["model_name"], **{k: o[k] for k in ("lyric", "song", "mode", "took_s")}}
            try:
                wav, sr, info, path, hit = hist.cached("resing", req, make, f"{m}_{m_}_tune{n}")
            except SingError as e:
                raise gr.Error(str(e))
            outs_.append((sr, wav))
            about = f"**{info['song']}**: {info['lyric']}"
        tu = next(t for t in tunes if t["no"] == n)
        return *two(outs_), str(bundle / "data" / tu["item"]["wav"]), about

    def do_sargam(which, text, js, voice, tonic, beat, vowel, model, steps_, seed_, join, gap):
        """Gradio callback: built-in alankar, typed swaras, or JSON; one or both models; pitch table.
        Empty tonic / beat boxes defer to the JSON's own settings, then to the defaults."""
        jn = JOIN_CHOICES.get(join, join)
        try:
            settings = {}
            if js and js.strip():
                lines, settings = from_json(js)
                title = settings.get("title") or "JSON score"
            elif text and text.strip():
                lines, title = parse_sargam(text), "my sargam"
            else:
                no = num(which) or 1
                lines, title = alankar_lines(no), f"alankar {no}: {ALANKARS[no]['name']}"
            vo = num(voice) or 1
            ton = tonic.strip() if tonic and tonic.strip() and tonic.strip() != "auto" else settings.get("tonic", "auto")
            bt = float(beat) if beat not in (None, "", 0) else float(settings.get("beat_s", 0.45))
            vw = settings.get("syllables", vowel)
            outs_, rows = [], []
            for m in MODEL_CHOICES[model]:
                req = {"title": title, "sargam": to_text(lines), "model": m, "voice": vo, "tonic": ton, "beat_s": bt,
                       "syllables": vw, "seed": int(seed_), "steps": int(steps_), "join": jn, "line_gap": float(gap)}
                wav, sr, info, path, hit = hist.cached(
                    "sargam", req, lambda m=m: ss.sing(lines, m, vo, ton, bt, vw, int(steps_), int(seed_),
                                                       join=jn, line_gap=float(gap)),
                    f"{m}_{title.split(':')[0]}")
                p = info["pitch"]
                outs_.append((sr, wav))
                rows.append(f"| {m} | {p['within_50c'] * 100:.0f}% | {p['within_25c'] * 100:.0f}% | "
                            f"{p['mean_abs_cents']} c | {'history' if hit else 'new'} |")
        except (SargamError, SingError, ValueError) as e:
            raise gr.Error(str(e))
        table = ("| model | on the swara (within 50 c) | in tune (within 25 c) | mean error | take |\n|---|---|---|---|---|\n"
                 + "\n".join(rows))
        return *two(outs_), f"**{title}**\n\n```\n{to_text(lines)}\n```\n\n{table}"

    def only_voice_tunes(only, voice):
        """Tune choices, optionally only the chosen voice's own tunes."""
        no = num(voice)
        if not only or no is None:
            return gr.update(choices=tune_names)
        mine = ["auto (fits the line)"] + [tune_label(t) for t in tunes if t["voice"] == no]
        return gr.update(choices=mine, value=mine[1] if len(mine) > 1 else mine[0])

    head = head_html()
    head_on_blocks = "head" in inspect.signature(gr.Blocks.__init__).parameters     # Gradio 4/5; 6 takes it in launch()
    render = {"concurrency_limit": 1, "concurrency_id": "render"}
    with gr.Blocks(title="Kalavinka: Hindi singing", **({"head": head} if head_on_blocks else {})) as page:
        gr.Markdown("# Kalavinka: Hindi film-song singing\n" + HOW)
        with gr.Accordion("Model and settings (all tabs)", open=True):
            with gr.Row():
                model = gr.Radio(list(MODEL_CHOICES), value=list(MODEL_CHOICES)[0], label="Model")
                steps = gr.Slider(8, 64, value=steps or default_steps(singer.device), step=4,
                                  label="Steps (fewer = faster, rougher)")
                seed = gr.Number(value=seed, precision=0, label="Seed (change it for a new take)")
            with gr.Row():
                join = gr.Radio([(k, v) for k, v in JOIN_CHOICES.items()], value="seamless",
                                label="Join lines and phrases")
                gap = gr.Number(value=0.3, minimum=0, maximum=10, label="Gap between lines (s), where the song has none")
        with gr.Tabs() as tabs:
            with gr.Tab("Sing new lyrics", id=TAB_IDS["sing"]):
                lyrics = gr.Textbox(label="Lyrics (Hindi, Devanagari)", value=EXAMPLES[0], lines=3)
                gr.Examples([[e] for e in EXAMPLES], [lyrics])
                vsing = VoicePicker(gr, book, allow_auto=True, label="Voice (who sings)")
                with gr.Row():
                    tune = gr.Dropdown(tune_names, value=tune_names[0], label="Tune", scale=4)
                    only = gr.Checkbox(label="Only tunes sung by this voice", scale=1)
                mode = gr.Radio(list(modes), value=list(modes)[0], label="Pitch")
                go = gr.Button("Sing", variant="primary")
                with gr.Row():
                    a1 = gr.Audio(label="Take 1 (Kalavinka, or the model you chose)", type="numpy")
                    a2 = gr.Audio(label="Take 2 (base SoulX, when comparing)", type="numpy")
                about = gr.Markdown()
                go.click(do_sing, [lyrics, tune, vsing.voice, mode, model, steps, seed, join, gap], [a1, a2, about],
                         **render)
                only.change(only_voice_tunes, [only, vsing.voice], tune)
                vsing.voice.change(only_voice_tunes, [only, vsing.voice], tune)
            with gr.Tab("Compare with the real singer", id=TAB_IDS["resing"]):
                rtune = gr.Dropdown(tune_names[1:], value=tune_names[1], label="Tune")
                rmode = gr.Radio(list(rmodes), value="Model's own pitch", label="Pitch")
                rgo = gr.Button("Re-sing", variant="primary")
                with gr.Row():
                    ra1 = gr.Audio(label="Take 1", type="numpy")
                    ra2 = gr.Audio(label="Take 2 (base, when comparing)", type="numpy")
                    rr = gr.Audio(label="Real singer", type="filepath")
                rabout = gr.Markdown()
                rgo.click(do_resing, [rtune, rmode, model, steps, seed], [ra1, ra2, rr, rabout], **render)
            with gr.Tab("Alankars / sargam", id=TAB_IDS["sargam"]):
                gr.Markdown("Pick a built-in alankar, **or** type swaras (`Sa Re Ga | Re Ga ma`; komal `re(k)`, tivra "
                            "`Ma(t)`, upper `Sa'`, lower `.Ni`, long `Pa:2`, hold `-`, rest `_`), **or** paste a JSON "
                            "score written by an LLM (prompt below). Typed swaras win over the built-in choice, JSON "
                            "wins over both. Leave Sa and the note length empty to use the JSON's own values.")
                which = gr.Dropdown(alankar_names, value=alankar_names[1], label="Built-in alankar")
                stext = gr.Textbox(label="Or your own swaras", lines=2, placeholder="Sa Re Ga | Re Ga ma | Ga ma Pa")
                with gr.Accordion("Or a JSON score (from ChatGPT etc.)", open=False):
                    gr.Markdown("Give this prompt to an LLM, describe the alankar at the end, paste its answer below:")
                    gr.Code(LLM_PROMPT, language=None)
                    sjson = gr.Code(label="JSON score", language="json", value="")
                vsarg = VoicePicker(gr, book, default=1, label="Voice")
                with gr.Row():
                    stonic = gr.Textbox(value="", label="Sa (tonic): empty / auto = the voice's own, or C#3 / 138.6")
                    sbeat = gr.Number(value=None, label="Seconds per note (empty = 0.45 or the JSON's)",
                                      minimum=0.05, maximum=5)
                    svowel = gr.Radio(["sargam", "aa"], value="sargam", label="Sing the swara names, or 'aa'")
                sgo = gr.Button("Sing the alankar", variant="primary")
                with gr.Row():
                    sa1 = gr.Audio(label="Take 1", type="numpy")
                    sa2 = gr.Audio(label="Take 2 (base, when comparing)", type="numpy")
                sabout = gr.Markdown()
                sgo.click(do_sargam, [which, stext, sjson, vsarg.voice, stonic, sbeat, svowel, model, steps, seed,
                                      join, gap], [sa1, sa2, sabout], **render)
            compose = build_compose_tab(gr, singer, hist, tunes, book, model, steps, seed, MODEL_CHOICES, two, join,
                                        VoicePicker)
            build_voices_tab(gr, book, [vsing, vsarg, compose["picker"]])
            H = build_history_tab(gr, hist)

        # ---- History -> "Open in its tab": refill the tab the take came from and switch to it
        def do_open(n):
            n = str(n or "").strip().lstrip("#")
            r = hist.get(int(n), full=True) if n.isdigit() else None
            if r is None:
                raise gr.Error("pick a take first (type its id or click a row)")
            q = r.get("request", {})
            keep = gr.update()
            sing_v = [keep] * 4
            res_v = [keep] * 2
            sarg_v = [keep] * 6
            comp_v = [keep] * len(compose["load_out"])
            kind = r["kind"]
            if kind == "sing":
                t = q.get("tune")
                sing_v = [q.get("lyrics", ""), tune_names[t] if t and t < len(tune_names) else tune_names[0],
                          str(q["voice"]) if q.get("voice") else AUTO, mode_name.get(q.get("mode"), list(modes)[0])]
            elif kind == "resing":
                t = q.get("tune") or 1
                res_v = [tune_names[t] if t < len(tune_names) else tune_names[1],
                         next((k for k, v in rmodes.items() if v == q.get("mode")), "Model's own pitch")]
            elif kind == "sargam":
                sarg_v = [q.get("sargam", ""), "", str(q.get("voice") or 1),
                          "" if q.get("tonic") in (None, "auto") else str(q.get("tonic")), q.get("beat_s"),
                          q.get("syllables", "sargam")]
            elif kind == "compose":
                comp_v = list(compose["load_comp"](q["comp"], f"Opened take #{r['id']} from the history."))
            hist.log("open_in_tab", id=r["id"], kind=kind)
            jn = q.get("join", "separate")
            return (gr.Tabs(selected=TAB_IDS[kind]), MODEL_OF.get(r["model"], list(MODEL_CHOICES)[0]),
                    q.get("steps", keep), q.get("seed", keep), jn, q.get("line_gap", keep),
                    *sing_v, *res_v, *sarg_v, *comp_v, f"Opened #{r['id']} in its tab.")
        H["open_btn"].click(do_open, H["rid"],
                            [tabs, model, steps, seed, join, gap, lyrics, tune, vsing.voice, mode, rtune, rmode,
                             stext, sjson, vsarg.voice, stonic, sbeat, svowel, *compose["load_out"], H["status"]])
        compose["init"](page)
    allowed = [str(bundle / "data"), str(outs)]
    page.queue().launch(share=share, server_port=port, inbrowser=not share, allowed_paths=allowed,
                        **({} if head_on_blocks else {"head": head}))
