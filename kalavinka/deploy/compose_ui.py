"""The web page's **Compose** tab: edit pitch, timing, lyrics and ornaments on the F0 curve, preview, and sing.

Layout:

1. **Start from**: a bundled tune (with the real singer's pitch curve, or as plain notes), Hindi lyrics placed on
   swaras you type, a sargam line or built-in alankar, a JSON composition (for example written by ChatGPT from
   ``COMPOSE_PROMPT``), or the built-in example (a Yaman phrase with kan, meend, vibrato, murki, gamak and andolan).
2. **Settings**: voice, Sa (tonic), tempo, raag (the swaras notes snap to), taal (beat-cycle grid), title.
3. **The editor** (``static/f0_editor.js``): notes as bars with their syllables, the F0 curve, ornaments, pencil and
   eraser, undo, and an instant in-browser preview tone.
4. **Table view**: the same notes as rows, for precise typing (Apply table to use it).
5. **Sing** with the model(s) chosen at the top: the takes, a follow check (how closely each model sang the curve),
   and the sung pitch drawn over the editor in orange.

Python and the editor share the composition through hidden textboxes (``#kv-comp``, ``#kv-view``, ``#kv-sung``) and
a hidden button (``#kv-push``); see ``static/f0_editor.js``. Every take goes through the history cache (kind
``compose``). ``head_html()`` returns the <script>/<style> the page needs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .compose import (COMPOSE_PROMPT, EXAMPLE, RAAGS, TABLE_COLS, TAALS, ComposeError, ComposeSinger, from_lyrics,
                      from_sargam, from_table, from_tune, normalise, to_table)

STATIC = Path(__file__).parent / "static"
SOURCES = ["Example: Yaman phrase with ornaments", "A bundled tune, with the real singer's pitch",
           "A bundled tune, plain notes", "Hindi lyrics on swaras", "Sargam / alankar", "JSON (pasted, e.g. from ChatGPT)",
           "Blank"]


def head_html() -> str:
    """<style> and <script> for the editor, to put in the page's <head>."""
    css = (STATIC / "f0_editor.css").read_text(encoding="utf-8")
    js = (STATIC / "f0_editor.js").read_text(encoding="utf-8")
    return f"<style>{css}</style>\n<script>{js}</script>"


def build_tab(gr, singer, hist, tunes, voices, model, steps, seed, model_choices, two):
    """Add the Compose tab to the current Blocks. ``model``/``steps``/``seed`` are the page-wide controls;
    ``two`` pads a list of takes to the two players. Returns ``init(page)``, which loads the example when the page
    opens."""
    from .sargam import ALANKARS, SargamError
    cs = ComposeSinger(singer)
    tune_names = [f"{t['no']}: {t['song']} ({t['year']}), {t['syllables']} syll., {t['seconds']} s" for t in tunes]
    voice_names = [v["name"] for v in voices]
    alankar_names = ["(type swaras below)"] + [f"{n}: {a['name']}" for n, a in ALANKARS.items()]
    lead = lambda s: int(re.match(r"\D*(\d+)", s).group(1)) if s and re.match(r"\D*(\d+)", s) else None

    def settings_of(comp):
        """Settings controls from a composition."""
        vo = comp.get("voice") or 1
        vname = next((n for n, v in zip(voice_names, voices) if v["no"] == vo), voice_names[0])
        return (vname, str(comp.get("tonic", "auto")), comp.get("tempo_bpm", 80),
                comp.get("raag") if comp.get("raag") in RAAGS else "Chromatic (all 12)",
                comp.get("taal") if comp.get("taal") in TAALS else "none", comp.get("title", ""))

    def refresh(comp):
        """(comp json, view json, table rows, status) for a composition."""
        comp = normalise(comp)
        return (json.dumps(comp, ensure_ascii=False), json.dumps(cs.view(comp), ensure_ascii=False),
                to_table(comp), "")

    def do_load(src, tune, lyrics, swaras, beats, alankar, sargam, js, voice):
        """Make a composition from the chosen starting point."""
        try:
            vo = lead(voice) or 1
            if src.startswith("Example"):
                comp = dict(EXAMPLE)
            elif src.startswith("A bundled tune"):
                t = tunes[(lead(tune) or 1) - 1]
                comp = from_tune(t, keep_curve="real singer" in src)
            elif src.startswith("Hindi lyrics"):
                lines = [ln.strip() for ln in re.split(r"[\n|]+", lyrics or "") if ln.strip()]
                if not lines:
                    raise ComposeError("type some Hindi lyrics first")
                comp = from_lyrics([cs.g2p(ln) for ln in lines], swaras or "", float(beats or 1))
                comp["voice"] = vo
            elif src.startswith("Sargam"):
                comp = from_sargam(sargam or "", lead(alankar) if not (sargam or "").strip() else None)
                comp["voice"] = vo
            elif src.startswith("JSON"):
                comp = normalise(js or "")
                comp.setdefault("voice", vo)
            else:
                comp = {"title": "blank", "voice": vo, "tonic": "auto", "tempo_bpm": 80,
                        "phrases": [{"notes": [{"syl": "", "swara": "Sa", "beats": 1}], "f0_edits": []}]}
            c, v, rows, _ = refresh(comp)
        except (ComposeError, SargamError) as e:
            raise gr.Error(str(e))
        return (c, v, rows, "", *settings_of(json.loads(c)),
                f"Loaded: **{json.loads(c).get('title', '')}**. Edit it below, preview, then Sing.")

    def do_settings(c, voice, tonic, tempo, raag, taal, title):
        """Apply the settings controls to the composition."""
        try:
            comp = normalise(c)
            comp.update({"voice": lead(voice) or 1, "tonic": _num_or_text(tonic or "auto"),
                         "tempo_bpm": float(tempo or 80), "raag": raag, "taal": taal, "title": title or ""})
            return refresh(comp)
        except (ComposeError, SargamError) as e:
            raise gr.Error(str(e))

    def do_push(c):
        """The editor changed the composition: recompute the curve and the table (errors go to the status line)."""
        try:
            comp, view, rows, _ = refresh(c)
            return view, rows, ""
        except (ComposeError, SargamError) as e:
            return gr.update(), gr.update(), f"⚠️ {e}"

    def do_table(rows, c):
        """Use the table's rows as the notes."""
        try:
            data = rows.values.tolist() if hasattr(rows, "values") else rows
            return refresh(from_table(data, c))
        except (ComposeError, SargamError, ValueError) as e:
            raise gr.Error(f"table: {e}")

    def do_sing(c, mdl, stp, sd):
        """Sing the composition with the chosen model(s) (cached), report how closely each followed the curve."""
        try:
            comp = normalise(c)
        except ComposeError as e:
            raise gr.Error(str(e))
        outs, rows, sung = [], [], ""
        for m in model_choices[mdl]:
            req = {"comp": comp, "model": m, "seed": int(sd), "steps": int(stp)}
            try:
                wav, sr, info, path, hit = hist.cached(
                    "compose", req, lambda m=m: cs.sing(comp, m, None, int(stp), int(sd)),
                    f"{m}_compose_{comp.get('title', '')}")
            except (ComposeError, SargamError) as e:
                raise gr.Error(str(e))
            except Exception as e:                    # SingError etc.
                raise gr.Error(str(e))
            outs.append((sr, wav))
            f = info.get("follow", {})
            took = "history" if hit else f"new, {info.get('took_s', 0)} s"
            rows.append(f"| {m} | {f.get('within_50c', 0) * 100:.0f}% | {f.get('within_25c', 0) * 100:.0f}% | "
                        f"{f.get('mean_abs_cents', '')} c | {f.get('frames_voiced_share', 0) * 100:.0f}% | {took} |")
            if not sung and info.get("sung"):
                sung = json.dumps(info["sung"])
        table = ("How closely the model sang your curve (pYIN on the take, frames voiced in both):\n\n"
                 "| model | within 50 c | within 25 c | mean error | voiced | take |\n|---|---|---|---|---|---|\n"
                 + "\n".join(rows) + "\n\nThe orange line in the editor is what the first model sang.")
        return *two(outs), table, sung

    def do_save(c):
        """Write the composition to outputs/compositions/ and offer it for download."""
        comp = normalise(c)
        d = hist.dir / "compositions"
        d.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^\wऀ-ॿ-]+", "_", comp.get("title") or "composition").strip("_")[:40] or "composition"
        p = d / f"{name}.json"
        p.write_text(json.dumps(comp, ensure_ascii=False, indent=1), encoding="utf-8")
        return str(p)

    with gr.Tab("Compose (pitch, timing, ornaments)"):
        gr.Markdown(
            "Write the melody yourself. **Start from** something, then edit it on the pitch curve: drag notes "
            "up/down (pitch, snapping to the raag), drag a note's right edge (length), double-click a note to change "
            "its syllable, click a note to add **ornaments** (meend, gamak, kan, murki, khatka, zamzama, andolan, "
            "vibrato, glides) and set their speed and depth, or use the **pencil** to draw the pitch by hand. "
            "**▶ preview** plays the curve as a plain tone in your browser at once; **Sing** makes the model sing it "
            "(Model choice at the top: Kalavinka, base, or both).")
        with gr.Accordion("1. Start from", open=True):
            with gr.Row():
                src = gr.Dropdown(SOURCES, value=SOURCES[0], label="Start from")
                ctune = gr.Dropdown(tune_names, value=tune_names[2] if len(tune_names) > 2 else tune_names[0],
                                    label="Tune (for the bundled-tune options)")
                calank = gr.Dropdown(alankar_names, value=alankar_names[2], label="Alankar (for Sargam)")
            with gr.Row():
                clyrics = gr.Textbox(label="Hindi lyrics (one phrase per line)", lines=2,
                                     value="ठंडी हवा में गीत नया")
                cswaras = gr.Textbox(label="Swaras, one per syllable (empty = all on Sa)",
                                     value="Ga Pa Dha Pa Ga Re Sa")
                cbeats = gr.Number(value=1, label="Beats per syllable")
                csargam = gr.Textbox(label="Or sargam text (for Sargam)", placeholder="Sa Re Ga ma | Pa:2 Ga Re Sa")
            with gr.Accordion("JSON / ask ChatGPT", open=False):
                gr.Markdown("Give this prompt to an LLM, describe the melody at the end, paste its answer here, "
                            "and pick *JSON* above:")
                gr.Code(COMPOSE_PROMPT, language=None)
                cjson = gr.Code(label="Composition JSON", language="json", value="")
            load = gr.Button("Load", variant="secondary")
        with gr.Row():
            cvoice = gr.Dropdown(voice_names, value=voice_names[3] if len(voice_names) > 3 else voice_names[0],
                                 label="Voice")
            ctonic = gr.Textbox(value="auto", label="Sa (tonic): auto = the voice's own, or C#3 / 138.6")
            ctempo = gr.Number(value=80, label="Tempo (beats per minute)")
            craag = gr.Dropdown(list(RAAGS), value="Chromatic (all 12)", label="Raag / thaat (notes snap to it)")
            ctaal = gr.Dropdown(list(TAALS), value="none", label="Taal (grid)")
            ctitle = gr.Textbox(value="", label="Title")
        gr.HTML('<div id="kv-editor">Loading the editor …</div>')
        cstatus = gr.Markdown()
        comp_box = gr.Textbox(elem_id="kv-comp", elem_classes=["kv-hidden"], label="composition")
        view_box = gr.Textbox(elem_id="kv-view", elem_classes=["kv-hidden"], label="view")
        sung_box = gr.Textbox(elem_id="kv-sung", elem_classes=["kv-hidden"], label="sung")
        push = gr.Button("push", elem_id="kv-push", elem_classes=["kv-hidden"])
        with gr.Accordion("Table view (type exact values, then Apply)", open=False):
            gr.Markdown("Columns: phrase number, syllable (`-` hold, `_` rest, empty = swara name), swara, cents, "
                        "beats **or** sec, ornaments like `meend(to=Pa,start=0.5); vibrato(rate=5,depth=25)`.")
            table = gr.Dataframe(headers=TABLE_COLS, interactive=True, wrap=True)
            tapply = gr.Button("Apply table")
        with gr.Row():
            sing = gr.Button("Sing it", variant="primary")
            save = gr.Button("Save JSON")
            saved = gr.File(label="Composition file", interactive=False)
        with gr.Row():
            ca1 = gr.Audio(label="Take 1", type="numpy")
            ca2 = gr.Audio(label="Take 2 (base, when comparing)", type="numpy")
        cabout = gr.Markdown()

        settings = [cvoice, ctonic, ctempo, craag, ctaal, ctitle]
        load_in = [src, ctune, clyrics, cswaras, cbeats, calank, csargam, cjson, cvoice]
        load_out = [comp_box, view_box, table, sung_box, *settings, cstatus]
        load.click(do_load, load_in, load_out)
        for ctl in settings:
            ctl.input(do_settings, [comp_box, *settings], [comp_box, view_box, table, cstatus])
        push.click(do_push, comp_box, [view_box, table, cstatus])
        tapply.click(do_table, [table, comp_box], [comp_box, view_box, table, cstatus])
        sing.click(do_sing, [comp_box, model, steps, seed], [ca1, ca2, cabout, sung_box])
        save.click(do_save, comp_box, saved)
        comp_box.change(None, comp_box, None, js="(v) => { if (window.KV) window.KV.load(v); return v; }")
        view_box.change(None, view_box, None, js="(v) => { if (window.KV) window.KV.setView(v); return v; }")
        sung_box.change(None, sung_box, None, js="(v) => { if (window.KV) window.KV.setSung(v); return v; }")
    return lambda page: page.load(do_load, load_in, load_out)     # open the page with the example loaded


def _num_or_text(s):
    """'138.6' -> 138.6; anything else stays text (auto, C#3)."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return str(s).strip()
