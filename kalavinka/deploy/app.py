"""A small web page for the Kalavinka singer (Gradio): type Hindi lyrics, pick a tune and a voice, press Sing.

Two tabs:

* **Sing new lyrics**: lyrics box (with examples), tune (auto or a bundled one), voice (the tune's own or another),
  pitch mode (the model's own, or expressive two-stage), Sing -> audio player + what was sung where.
* **Compare with the real singer**: pick a tune, re-sing its original words, hear the model and the real clip.
* **Alankars / sargam**: a built-in alankar or your own swaras (text, or an LLM-written JSON), with a note-by-note
  pitch check.
* **Compose (pitch, timing, ornaments)**: write the melody yourself on the F0 curve (``compose_ui``): swaras or free
  cents, lengths, lyrics on the notes, ornaments (meend, gamak, kan, murki, andolan, vibrato ...), hand-drawn pitch,
  an instant preview tone, then sing it and see how closely each model followed.
* **History**: every earlier take (all tabs, and the command line), newest first, playable.

Every tab has a **Model** choice: Kalavinka (ours), base SoulX-Singer (zero-shot, Hindi phones mapped to English),
or both side by side. Takes are cached in ``outputs/`` (``kalavinka_app.history``): asking for the same thing again
plays the saved take instead of singing it again; change the seed for a new take.

Launch: ``python -m kalavinka_app ui`` (``--share`` for a public link, needed on Colab), or ``launch(bundle)``.
The model is loaded once when the page starts.
"""
from __future__ import annotations

from pathlib import Path

from .cli import EXAMPLES, read_jsonl

HOW = """**How it works.** Kalavinka is SoulX-Singer fine-tuned on 10.7 hours of Hindi film songs. Your lyrics are
split into syllables, put on the notes of a real song line the model never trained on (the *tune*), and sung in the
*voice* of a short clip. **Model's own pitch** lets the model shape the melody (sounds most natural in our listening
test). **Expressive (two-stage)** first draws a pitch curve with vibrato using our contour model, then sings it.
Lyrics must be in Devanagari; one line per phrase (or separate with |)."""


MODEL_CHOICES = {"Kalavinka (ours)": ["kalavinka"], "Base SoulX-Singer": ["base"],
                 "Both, side by side": ["kalavinka", "base"]}


def launch(bundle: Path, device: str = "auto", share: bool = False, port: int = 7860):
    """Load the singer and serve the page (blocks)."""
    import gradio as gr
    from .history import History, describe
    from .sargam import ALANKARS, LLM_PROMPT, SargamError, SargamSinger, alankar_lines, from_json, parse_sargam, to_text
    from .singer import KalavinkaSinger, SingError, default_steps
    singer = KalavinkaSinger(bundle, device=device)
    hist = History(Path(bundle) / "outputs")
    ss = SargamSinger(singer)
    tunes = read_jsonl(Path(bundle) / "data" / "tunes.jsonl")
    voices = read_jsonl(Path(bundle) / "data" / "voices.jsonl")
    tune_names = ["auto (fits the line)"] + [f"{t['no']}: {t['song']} ({t['year']}), {t['syllables']} syll., "
                                             f"{t['seconds']} s" for t in tunes]
    voice_names = ["the tune's own singer"] + [v["name"] for v in voices]
    modes = {"Model's own pitch (recommended)": "score", "Expressive (two-stage, vibrato)": "expressive"}
    alankar_names = [f"{n}: {a['name']}" for n, a in ALANKARS.items()]

    def num(choice: str):
        """Leading number of a dropdown entry, or None for the first (auto) entry."""
        head = choice.split(":")[0]
        return int(head) if head.isdigit() else (int(head.split()[1]) if head.startswith("Voice ") else None)

    def two(results):
        """Pad a list of (sr, wav) to two players (None = empty)."""
        return (results + [None, None])[:2]

    def do_sing(lyrics, tune, voice, mode, model, steps, seed):
        """Gradio callback: sing with one or both models (cached) and describe."""
        outs, notes = [], []
        for m in MODEL_CHOICES[model]:
            req = {"lyrics": lyrics, "tune": num(tune), "voice": num(voice), "mode": modes[mode], "model": m,
                   "seed": int(seed), "steps": int(steps)}
            try:
                wav, sr, info, path, hit = hist.cached(
                    "sing", req, lambda: singer.sing(lyrics, num(tune), num(voice), modes[mode], int(steps), int(seed),
                                                     model=m), f"{m}_{modes[mode]}_{lyrics}")
            except SingError as e:
                raise gr.Error(str(e))
            outs.append((sr, wav))
            lines = "; ".join(f"**{l['line']}** on tune {l['tune']} ({l['tune_song']}), voice {l['voice']}, "
                              f"{int(l['seen_in_training'] * 100)}% seen in training" for l in info["lines"])
            made = "from history" if hit else f"made in {info['took_s']} s"
            notes.append(f"*{m}*: {lines} ({made})")
        return *two(outs), "\n\n".join(notes)

    def do_resing(tune, mode, model, steps, seed):
        """Gradio callback: model(s) vs the real singer on a bundled line."""
        import soundfile as sf
        n = num(tune) or 1
        m_ = {"Model's own pitch": "score", "Expressive (two-stage)": "expressive", "Real singer's pitch": "real-pitch"}[mode]
        outs, about = [], ""
        for m in MODEL_CHOICES[model]:
            req = {"tune": n, "mode": m_, "model": m, "seed": int(seed), "steps": int(steps)}

            def make(m=m):
                """Re-sing with model m."""
                o = singer.resing(n, m_, int(steps), int(seed), model=m)
                w, sr = o["model"]
                return w, sr, {k: o[k] for k in ("lyric", "song", "mode", "model", "took_s")}
            wav, sr, info, path, hit = hist.cached("resing", req, make, f"{m}_{m_}_tune{n}")
            outs.append((sr, wav))
            about = f"**{info['song']}**: {info['lyric']}"
        tu = next(t for t in tunes if t["no"] == n)
        rw, rsr = sf.read(Path(bundle) / "data" / tu["item"]["wav"], dtype="float32")
        return *two(outs), (rsr, rw), about

    def do_sargam(which, text, js, voice, tonic, beat, vowel, model, steps, seed):
        """Gradio callback: built-in alankar, typed swaras, or JSON; one or both models; pitch table."""
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
            ton = tonic.strip() if tonic and tonic.strip() else settings.get("tonic", "auto")
            bt = float(beat or settings.get("beat_s", 0.45))
            vw = settings.get("syllables", vowel)
            outs, rows = [], []
            for m in MODEL_CHOICES[model]:
                req = {"title": title, "sargam": to_text(lines), "model": m, "voice": vo, "tonic": ton, "beat_s": bt,
                       "syllables": vw, "seed": int(seed), "steps": int(steps)}
                wav, sr, info, path, hit = hist.cached(
                    "sargam", req, lambda: ss.sing(lines, m, vo, ton, bt, vw, int(steps), int(seed)),
                    f"{m}_{title.split(':')[0]}")
                p = info["pitch"]
                outs.append((sr, wav))
                rows.append(f"| {m} | {p['within_50c'] * 100:.0f}% | {p['within_25c'] * 100:.0f}% | "
                            f"{p['mean_abs_cents']} c | {'history' if hit else 'new'} |")
        except (SargamError, SingError) as e:
            raise gr.Error(str(e))
        table = ("| model | on the swara (within 50 c) | in tune (within 25 c) | mean error | take |\n|---|---|---|---|---|\n"
                 + "\n".join(rows))
        return *two(outs), f"**{title}**\n\n```\n{to_text(lines)}\n```\n\n{table}"

    def history_rows():
        """Recent takes for the history table and the picker."""
        recs = hist.recent(50)
        return [describe(r) for r in recs]

    def do_refresh():
        """Re-read the history."""
        rows = history_rows()
        return gr.update(choices=rows, value=rows[0] if rows else None)

    def do_play(choice):
        """Play a take picked from the history."""
        import soundfile as sf
        if not choice:
            return None, ""
        rid = int(choice.split()[0].lstrip("#"))
        r = hist.get(rid)
        w, sr = sf.read(hist.dir / r["wav"], dtype="float32")
        return (sr, w), f"`{hist.dir / r['wav']}`\n\n```\n{__import__('json').dumps(r['request'], ensure_ascii=False, indent=1)}\n```"

    import inspect
    from .compose_ui import build_tab as build_compose_tab, head_html
    head = head_html()
    # Gradio 4/5 take head= on Blocks; Gradio 6 moved it to launch()
    head_on_blocks = "head" in inspect.signature(gr.Blocks.__init__).parameters
    with gr.Blocks(title="Kalavinka: Hindi singing", **({"head": head} if head_on_blocks else {})) as page:
        gr.Markdown("# Kalavinka: Hindi film-song singing\n" + HOW)
        with gr.Accordion("Model and settings (all tabs)", open=True):
            with gr.Row():
                model = gr.Radio(list(MODEL_CHOICES), value=list(MODEL_CHOICES)[0], label="Model")
                steps = gr.Slider(8, 64, value=default_steps(singer.device), step=4, label="Steps (fewer = faster, rougher)")
                seed = gr.Number(value=0, precision=0, label="Seed (change it for a new take)")
        with gr.Tab("Sing new lyrics"):
            lyrics = gr.Textbox(label="Lyrics (Hindi, Devanagari)", value=EXAMPLES[0], lines=3)
            gr.Examples([[e] for e in EXAMPLES], [lyrics])
            with gr.Row():
                tune = gr.Dropdown(tune_names, value=tune_names[0], label="Tune")
                voice = gr.Dropdown(voice_names, value=voice_names[0], label="Voice")
            mode = gr.Radio(list(modes), value=list(modes)[0], label="Pitch")
            go = gr.Button("Sing", variant="primary")
            with gr.Row():
                a1 = gr.Audio(label="Take 1 (Kalavinka, or the model you chose)", type="numpy")
                a2 = gr.Audio(label="Take 2 (base SoulX, when comparing)", type="numpy")
            about = gr.Markdown()
            go.click(do_sing, [lyrics, tune, voice, mode, model, steps, seed], [a1, a2, about])
        with gr.Tab("Compare with the real singer"):
            rtune = gr.Dropdown(tune_names[1:], value=tune_names[1], label="Tune")
            rmode = gr.Radio(["Model's own pitch", "Expressive (two-stage)", "Real singer's pitch"],
                             value="Model's own pitch", label="Pitch")
            rgo = gr.Button("Re-sing", variant="primary")
            with gr.Row():
                ra1 = gr.Audio(label="Take 1", type="numpy")
                ra2 = gr.Audio(label="Take 2 (base, when comparing)", type="numpy")
                rr = gr.Audio(label="Real singer", type="numpy")
            rabout = gr.Markdown()
            rgo.click(do_resing, [rtune, rmode, model, steps, seed], [ra1, ra2, rr, rabout])
        with gr.Tab("Alankars / sargam"):
            gr.Markdown("Pick a built-in alankar, **or** type swaras (`Sa Re Ga | Re Ga ma`; komal `re(k)`, tivra "
                        "`Ma(t)`, upper `Sa'`, lower `.Ni`, long `Pa:2`, hold `-`, rest `_`), **or** paste a JSON score "
                        "written by an LLM (prompt below). Typed swaras win over the built-in choice, JSON wins over both.")
            which = gr.Dropdown(alankar_names, value=alankar_names[1], label="Built-in alankar")
            stext = gr.Textbox(label="Or your own swaras", lines=2, placeholder="Sa Re Ga | Re Ga ma | Ga ma Pa")
            with gr.Accordion("Or a JSON score (from ChatGPT etc.)", open=False):
                gr.Markdown("Give this prompt to an LLM, describe the alankar at the end, paste its answer below:")
                gr.Code(LLM_PROMPT, language=None)
                sjson = gr.Code(label="JSON score", language="json", value="")
            with gr.Row():
                svoice = gr.Dropdown(voice_names[1:], value=voice_names[1], label="Voice")
                stonic = gr.Textbox(value="auto", label="Sa (tonic): auto = the voice's own, or C#3 / 138.6")
                sbeat = gr.Slider(0.2, 1.0, value=0.45, step=0.05, label="Seconds per note")
                svowel = gr.Radio(["sargam", "aa"], value="sargam", label="Sing the swara names, or 'aa'")
            sgo = gr.Button("Sing the alankar", variant="primary")
            with gr.Row():
                sa1 = gr.Audio(label="Take 1", type="numpy")
                sa2 = gr.Audio(label="Take 2 (base, when comparing)", type="numpy")
            sabout = gr.Markdown()
            sgo.click(do_sargam, [which, stext, sjson, svoice, stonic, sbeat, svowel, model, steps, seed],
                      [sa1, sa2, sabout])
        init_compose = build_compose_tab(gr, singer, hist, tunes, voices, model, steps, seed, MODEL_CHOICES, two)
        with gr.Tab("History"):
            gr.Markdown(f"Every take is kept in `{hist.dir}` and logged in `history.jsonl`.")
            rows = history_rows()
            pick = gr.Dropdown(rows, value=rows[0] if rows else None, label="Earlier takes (newest first)")
            with gr.Row():
                hplay = gr.Audio(label="Take", type="numpy")
                hinfo = gr.Markdown()
            hb = gr.Button("Refresh")
            hb.click(do_refresh, None, pick)
            pick.change(do_play, pick, [hplay, hinfo])
        init_compose(page)
    page.queue().launch(share=share, server_port=port, inbrowser=not share,
                        **({} if head_on_blocks else {"head": head}))
