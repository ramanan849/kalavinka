"""A small web page for the Kalavinka singer (Gradio): type Hindi lyrics, pick a tune and a voice, press Sing.

Two tabs:

* **Sing new lyrics**: lyrics box (with examples), tune (auto or a bundled one), voice (the tune's own or another),
  pitch mode (the model's own, or expressive two-stage), Sing -> audio player + what was sung where.
* **Compare with the real singer**: pick a tune, re-sing its original words, hear the model and the real clip.

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


def launch(bundle: Path, device: str = "auto", share: bool = False, port: int = 7860):
    """Load the singer and serve the page (blocks)."""
    import gradio as gr
    from .singer import KalavinkaSinger, SingError, default_steps
    singer = KalavinkaSinger(bundle, device=device)
    tunes = read_jsonl(Path(bundle) / "data" / "tunes.jsonl")
    voices = read_jsonl(Path(bundle) / "data" / "voices.jsonl")
    tune_names = ["auto (fits the line)"] + [f"{t['no']}: {t['song']} ({t['year']}), {t['syllables']} syll., "
                                             f"{t['seconds']} s" for t in tunes]
    voice_names = ["the tune's own singer"] + [v["name"] for v in voices]
    modes = {"Model's own pitch (recommended)": "score", "Expressive (two-stage, vibrato)": "expressive"}

    def num(choice: str):
        """Leading number of a dropdown entry, or None for the first (auto) entry."""
        head = choice.split(":")[0]
        return int(head) if head.isdigit() else (int(head.split()[1]) if head.startswith("Voice ") else None)

    def do_sing(lyrics, tune, voice, mode, steps, seed):
        """Gradio callback: sing and describe."""
        try:
            wav, sr, info = singer.sing(lyrics, num(tune), num(voice), modes[mode], int(steps), int(seed))
        except SingError as e:
            raise gr.Error(str(e))
        lines = "\n".join(f"- **{l['line']}** on tune {l['tune']} ({l['tune_song']}), voice {l['voice']}; "
                          f"{int(l['seen_in_training'] * 100)}% of its 4-syllable phrases occur in the training lyrics"
                          for l in info["lines"])
        return (sr, wav), f"{lines}\n\n{info['seconds']} s of singing, made in {info['took_s']} s on {info['device']}."

    def do_resing(tune, mode, steps, seed):
        """Gradio callback: model vs real singer on a bundled line."""
        n = num(tune) or 1
        m = {"Model's own pitch": "score", "Expressive (two-stage)": "expressive", "Real singer's pitch": "real-pitch"}[mode]
        out = singer.resing(n, m, int(steps), int(seed))
        (w, sr), (rw, rsr) = out["model"], out["real"]
        return (sr, w), (rsr, rw), f"**{out['song']}**: {out['lyric']} (made in {out['took_s']} s)"

    with gr.Blocks(title="Kalavinka: Hindi singing") as page:
        gr.Markdown("# Kalavinka: Hindi film-song singing\n" + HOW)
        with gr.Tab("Sing new lyrics"):
            lyrics = gr.Textbox(label="Lyrics (Hindi, Devanagari)", value=EXAMPLES[0], lines=3)
            gr.Examples([[e] for e in EXAMPLES], [lyrics])
            with gr.Row():
                tune = gr.Dropdown(tune_names, value=tune_names[0], label="Tune")
                voice = gr.Dropdown(voice_names, value=voice_names[0], label="Voice")
            mode = gr.Radio(list(modes), value=list(modes)[0], label="Pitch")
            with gr.Accordion("More settings", open=False):
                steps = gr.Slider(8, 64, value=default_steps(singer.device), step=4, label="Steps (fewer = faster, rougher)")
                seed = gr.Number(value=0, precision=0, label="Seed (change it for another take)")
            go = gr.Button("Sing", variant="primary")
            audio = gr.Audio(label="Kalavinka", type="numpy")
            about = gr.Markdown()
            go.click(do_sing, [lyrics, tune, voice, mode, steps, seed], [audio, about])
        with gr.Tab("Compare with the real singer"):
            rtune = gr.Dropdown(tune_names[1:], value=tune_names[1], label="Tune")
            rmode = gr.Radio(["Model's own pitch", "Expressive (two-stage)", "Real singer's pitch"],
                             value="Model's own pitch", label="Pitch")
            rgo = gr.Button("Re-sing", variant="primary")
            with gr.Row():
                ra = gr.Audio(label="Kalavinka", type="numpy")
                rr = gr.Audio(label="Real singer", type="numpy")
            rabout = gr.Markdown()
            rgo.click(do_resing, [rtune, rmode, steps, seed], [ra, rr, rabout])
    page.queue().launch(share=share, server_port=port, inbrowser=not share)
