"""The web page's **Compose** tab: edit pitch, timing, lyrics and ornaments on the F0 curve, preview, and sing.

Layout (numbered steps on the page):

1. **Start from**: pick a source; only that source's own controls are shown (a bundled tune, with or without the
   real singer's pitch curve; Hindi lyrics on swaras you type; a built-in alankar; sargam text; a JSON composition,
   for example written by ChatGPT from ``COMPOSE_PROMPT``; the built-in example; or a blank phrase). **Restore last
   session** brings back the autosaved composition.
2. **Voice and key**: the voice picker (filters, profile card, sample), Sa (tonic), tempo, raag (the swaras notes
   snap to), taal (beat-cycle grid), the gap between phrases, title. Text and number boxes apply when you leave them
   or press Enter (not on every keystroke).
3. **Edit** (``static/f0_editor.js``): notes as bars with their syllables, the F0 curve, ornaments, pencil and
   eraser, undo, an instant in-browser preview tone, a box to type a whole lyric line onto a phrase, and in-place
   syllable editing (Tab moves on). Syllables the model cannot sing are underlined in red. The **table view** shows
   the same notes as rows for precise typing.
4. **Sing** with the model(s) chosen at the top, joined seamlessly (or separately, see Settings): the takes, a
   follow check (how closely each model sang the curve), and the sung pitch drawn over the editor in orange.

Python and the editor share the composition through hidden textboxes (``#kv-comp``, ``#kv-view``, ``#kv-sung``,
``#kv-lyrics``) and hidden buttons (``#kv-push``, ``#kv-lyrics-go``); see ``static/f0_editor.js``. Every edit is
autosaved to ``outputs/compositions/autosave.json``; every take goes through the history cache (kind ``compose``).
``head_html()`` returns the <script>/<style> the page needs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .compose import (COMPOSE_PROMPT, EXAMPLE, RAAGS, TABLE_COLS, TAALS, ComposeError, ComposeSinger, from_lyrics,
                      from_sargam, from_table, from_tune, normalise, to_table)

STATIC = Path(__file__).parent / "static"
SOURCES = ["Example: Yaman phrase with ornaments", "A bundled tune", "Hindi lyrics on swaras", "A built-in alankar",
           "Sargam text", "JSON (pasted, e.g. from ChatGPT)", "Blank"]
HINTS = {
    SOURCES[0]: "A Yaman phrase with kan, meend, vibrato, murki, gamak and andolan: a tour of the ornaments.",
    SOURCES[1]: "A real song line: its syllables and notes, and (ticked) the real singer's pitch curve to edit.",
    SOURCES[2]: "Your lyrics, one syllable per note, on the swaras you type (the last swara repeats).",
    SOURCES[3]: "A sargam exercise; the notes sing the swara names.",
    SOURCES[4]: "Swaras as text: `Sa Re Ga | Re Ga ma`, komal `re(k)`, tivra `Ma(t)`, upper `Sa'`, lower `.Ni`, "
                "long `Pa:2`, rest `_`.",
    SOURCES[5]: "Paste a composition JSON (open the accordion for the prompt to give an LLM).",
    SOURCES[6]: "One note on Sa, to build from scratch.",
}
# which control groups each source shows
SHOWS = {SOURCES[0]: (), SOURCES[1]: ("tune",), SOURCES[2]: ("lyrics",), SOURCES[3]: ("alankar",),
         SOURCES[4]: ("sargam",), SOURCES[5]: ("json",), SOURCES[6]: ()}
GROUPS = ("tune", "lyrics", "alankar", "sargam", "json")


def head_html() -> str:
    """<style> and <script> for the editor, to put in the page's <head>."""
    css = (STATIC / "f0_editor.css").read_text(encoding="utf-8")
    js = (STATIC / "f0_editor.js").read_text(encoding="utf-8")
    return f"<style>{css}</style>\n<script>{js}</script>"


def visibility(src: str):
    """{group: visible} for a source (also used by the tests)."""
    on = SHOWS.get(src, ())
    return {g: g in on for g in GROUPS}


def build_tab(gr, singer, hist, tunes, book, model, steps, seed, model_choices, two, join, picker_cls):
    """Add the Compose tab to the current Blocks. ``model``/``steps``/``seed``/``join`` are the page-wide controls;
    ``book`` is the ``VoiceBook``; ``picker_cls`` the ``VoicePicker`` class. Returns a dict: ``init(page)`` (loads the
    example when the page opens), ``picker``, and ``load_comp`` + ``load_out`` (to open a composition from the
    History tab)."""
    from .sargam import ALANKARS, SargamError
    cs = ComposeSinger(singer)
    tune_names = [f"{t['no']}: {t['song']} ({t['year']}), {t['syllables']} syll., {t['seconds']} s" for t in tunes]
    alankar_names = [f"{n}: {a['name']}" for n, a in ALANKARS.items()]
    lead = lambda s: int(re.match(r"\D*(\d+)", str(s)).group(1)) if s and re.match(r"\D*(\d+)", str(s)) else None
    autosave = hist.dir / "compositions" / "autosave.json"
    errors = (ComposeError, SargamError)

    def save_auto(comp):
        """Keep the latest composition, so a reload or crash never loses edits."""
        try:
            autosave.parent.mkdir(parents=True, exist_ok=True)
            tmp = autosave.with_suffix(".tmp")
            tmp.write_text(json.dumps(comp, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(autosave)
        except OSError:
            pass

    def sa_of(comp):
        """Tonic (MIDI) for the table view."""
        try:
            return cs.sa_midi(comp, comp.get("voice") or 1)
        except Exception:
            return 60.0

    def settings_of(comp):
        """Settings controls from a composition."""
        return (str(comp.get("voice") or 1), str(comp.get("tonic", "auto")), comp.get("tempo_bpm", 80),
                comp.get("raag") if comp.get("raag") in RAAGS else "Chromatic (all 12)",
                comp.get("taal") if comp.get("taal") in TAALS else "none", comp.get("phrase_gap_s", 0.3),
                comp.get("title", ""))

    def refresh(comp):
        """(comp json, view json, table rows, status) for a composition; autosaves it."""
        comp = normalise(comp)
        save_auto(comp)
        return (json.dumps(comp, ensure_ascii=False), json.dumps(cs.view(comp), ensure_ascii=False),
                to_table(comp, sa_of(comp)), "")

    def load_comp(comp, note=""):
        """Values for ``load_out`` from a composition."""
        c, v, rows, _ = refresh(comp)
        cj = json.loads(c)
        return (c, v, rows, "", *settings_of(cj), note or f"Loaded: **{cj.get('title', '')}**. Edit it below, then Sing.")

    def do_load(src, tune, keep_real, lyrics, swaras, beats, alankar, sargam, js, voice):
        """Make a composition from the chosen starting point."""
        try:
            vo = lead(voice) or 1
            if src == SOURCES[0]:
                comp = dict(EXAMPLE)
            elif src == SOURCES[1]:
                t = tunes[(lead(tune) or 1) - 1]
                comp = from_tune(t, keep_curve=bool(keep_real))
            elif src == SOURCES[2]:
                lines = [ln.strip() for ln in re.split(r"[\n|]+", lyrics or "") if ln.strip()]
                if not lines:
                    raise ComposeError("type some Hindi lyrics first")
                comp = from_lyrics([cs.g2p(ln) for ln in lines], swaras or "", float(beats or 1))
                comp["voice"] = vo
            elif src == SOURCES[3]:
                comp = from_sargam("", lead(alankar) or 1)
                comp["voice"] = vo
            elif src == SOURCES[4]:
                if not (sargam or "").strip():
                    raise ComposeError("type some swaras first, e.g. Sa Re Ga | Re Ga ma")
                comp = from_sargam(sargam)
                comp["voice"] = vo
            elif src == SOURCES[5]:
                comp = normalise(js or "")
                comp.setdefault("voice", vo)
            else:
                comp = {"title": "blank", "voice": vo, "tonic": "auto", "tempo_bpm": 80,
                        "phrases": [{"notes": [{"syl": "", "swara": "Sa", "beats": 1}], "f0_edits": []}]}
            out = load_comp(comp)
        except errors as e:
            raise gr.Error(str(e))
        hist.log("compose_load", source=src, title=json.loads(out[0]).get("title", ""))
        return out

    def do_restore():
        """Load the autosaved composition."""
        if not autosave.exists():
            raise gr.Error("nothing autosaved yet")
        try:
            return load_comp(json.loads(autosave.read_text(encoding="utf-8")), "Restored your last session.")
        except (errors + (json.JSONDecodeError,)) as e:
            raise gr.Error(f"the autosave cannot be read: {e}")

    def do_settings(c, voice, tonic, tempo, raag, taal, gap, title):
        """Apply the settings controls to the composition (bad values go to the status line, not an error popup)."""
        try:
            comp = normalise(c)
            comp.update({"voice": lead(voice) or 1, "tonic": _num_or_text(tonic or "auto"),
                         "tempo_bpm": tempo if tempo not in (None, "") else 80, "raag": raag, "taal": taal,
                         "phrase_gap_s": gap if gap not in (None, "") else 0.3, "title": title or ""})
            return refresh(comp)
        except errors as e:
            return gr.update(), gr.update(), gr.update(), f"⚠️ {e}"

    def do_push(c):
        """The editor changed the composition: recompute the curve and the table (errors go to the status line)."""
        try:
            comp, view, rows, _ = refresh(c)
            return view, rows, ""
        except errors as e:
            return gr.update(), gr.update(), f"⚠️ {e}"

    def do_lyrics(req, c):
        """The editor's lyric box: spread a typed line over a phrase."""
        try:
            r = json.loads(req or "{}")
            comp = cs.lyrics(r.get("comp") or c, int(r.get("phrase", 0)), str(r.get("text", "")))
            out = refresh(comp)
            return (*out[:3], f"Lyrics put on phrase {int(r.get('phrase', 0)) + 1}.")
        except (errors + (ValueError,)) as e:
            return gr.update(), gr.update(), gr.update(), f"⚠️ lyrics: {e}"

    def do_table(rows, c):
        """Use the table's rows as the notes."""
        try:
            comp = normalise(c)
            data = rows.values.tolist() if hasattr(rows, "values") else rows
            return refresh(from_table(data, comp, sa_of(comp)))
        except (errors + (ValueError,)) as e:
            raise gr.Error(f"table: {e}")

    def do_sing(c, mdl, stp, sd, jn):
        """Sing the composition with the chosen model(s) (cached), report how closely each followed the curve."""
        try:
            comp = normalise(c)
        except ComposeError as e:
            raise gr.Error(str(e))
        jn = {"Seamless (one performance)": "seamless", "Separate (old: each line alone)": "separate"}.get(jn, jn)
        outs, rows, sung = [], [], ""
        for m in model_choices[mdl]:
            req = {"comp": comp, "model": m, "seed": int(sd), "steps": int(stp), "join": jn}
            try:
                wav, sr, info, path, hit = hist.cached(
                    "compose", req, lambda m=m: cs.sing(comp, m, None, int(stp), int(sd), join=jn),
                    f"{m}_compose_{comp.get('title', '')}")
            except Exception as e:                    # ComposeError, SingError, model errors: shown, and logged
                raise gr.Error(str(e))
            outs.append((sr, wav))
            f = info.get("follow", {})
            took = "history" if hit else f"new, {info.get('took_s', 0)} s"
            rows.append(f"| {m} | {f.get('within_50c', 0) * 100:.0f}% | {f.get('within_25c', 0) * 100:.0f}% | "
                        f"{f.get('mean_abs_cents', '')} c | {f.get('frames_voiced_share', 0) * 100:.0f}% | "
                        f"{info.get('pieces', 1)} ({info.get('join', 'separate')}) | {took} |")
            if not sung and info.get("sung"):
                sung = json.dumps(info["sung"])
        table = ("How closely the model sang your curve (pYIN on the take, frames voiced in both):\n\n"
                 "| model | within 50 c | within 25 c | mean error | voiced | renders | take |\n"
                 "|---|---|---|---|---|---|---|\n" + "\n".join(rows)
                 + "\n\nThe orange line in the editor is what the first model sang.")
        return *two(outs), table, sung

    def do_save(c):
        """Write the composition to outputs/compositions/ and offer it for download."""
        try:
            comp = normalise(c)
        except ComposeError as e:
            raise gr.Error(str(e))
        d = hist.dir / "compositions"
        d.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^\wऀ-ॿ-]+", "_", comp.get("title") or "composition").strip("_")[:40] or "composition"
        p = d / f"{name}.json"
        p.write_text(json.dumps(comp, ensure_ascii=False, indent=1), encoding="utf-8")
        hist.log("compose_save", file=str(p))
        return str(p)

    with gr.Tab("Compose (pitch, timing, ornaments)", id="t_compose"):
        gr.Markdown(
            "Write the melody yourself: **1** pick a starting point, **2** a voice and key, **3** edit it on the pitch "
            "curve (drag notes up/down for pitch and their right edge for length; double-click a note to type its "
            "syllable, or type a whole line in the lyrics box; click a note to add **ornaments**: meend, gamak, kan, "
            "murki, khatka, zamzama, andolan, vibrato, glides; or draw the pitch with the **pencil**), "
            "**▶ preview** in your browser, then **4 Sing**.")
        with gr.Accordion("1. Start from", open=True):
            with gr.Row():
                src = gr.Dropdown(SOURCES, value=SOURCES[0], label="Start from", scale=2)
                hint = gr.Markdown(HINTS[SOURCES[0]])
            with gr.Row(visible=False) as g_tune:
                ctune = gr.Dropdown(tune_names, value=tune_names[2] if len(tune_names) > 2 else tune_names[0],
                                    label="Tune", scale=3)
                ckeep = gr.Checkbox(value=True, label="Keep the real singer's pitch curve (as editable ornaments)")
            with gr.Row(visible=False) as g_lyrics:
                clyrics = gr.Textbox(label="Hindi lyrics (one phrase per line)", lines=2, value="ठंडी हवा में गीत नया")
                cswaras = gr.Textbox(label="Swaras, one per syllable (empty = all on Sa)", value="Ga Pa Dha Pa Ga Re Sa")
                cbeats = gr.Number(value=1, label="Beats per syllable")
            with gr.Row(visible=False) as g_alankar:
                calank = gr.Dropdown(alankar_names, value=alankar_names[1], label="Alankar")
            with gr.Row(visible=False) as g_sargam:
                csargam = gr.Textbox(label="Sargam text", placeholder="Sa Re Ga ma | Pa:2 Ga Re Sa")
            with gr.Group(visible=False) as g_json:
                with gr.Accordion("The prompt to give ChatGPT (or any LLM)", open=False):
                    gr.Code(COMPOSE_PROMPT, language=None)
                cjson = gr.Code(label="Composition JSON", language="json", value="")
            with gr.Row():
                load = gr.Button("Load", variant="primary")
                restore = gr.Button("Restore last session")
        with gr.Accordion("2. Voice and key", open=True):
            picker = picker_cls(gr, book, default=4 if book.get(4) else None, label="Voice")
            with gr.Row():
                ctonic = gr.Textbox(value="auto", label="Sa (tonic): auto = the voice's own, or C#3 / 138.6")
                ctempo = gr.Number(value=80, label="Tempo (beats per minute)", minimum=5, maximum=600)
                craag = gr.Dropdown(list(RAAGS), value="Chromatic (all 12)", label="Raag / thaat (notes snap to it)")
                ctaal = gr.Dropdown(list(TAALS), value="none", label="Taal (grid)")
                cgap = gr.Number(value=0.3, label="Gap between phrases (s)", minimum=0, maximum=10)
                ctitle = gr.Textbox(value="", label="Title")
        gr.Markdown("**3. Edit**")
        gr.HTML('<div id="kv-editor">Loading the editor …</div>')
        cstatus = gr.Markdown()
        comp_box = gr.Textbox(elem_id="kv-comp", elem_classes=["kv-hidden"], label="composition")
        view_box = gr.Textbox(elem_id="kv-view", elem_classes=["kv-hidden"], label="view")
        sung_box = gr.Textbox(elem_id="kv-sung", elem_classes=["kv-hidden"], label="sung")
        lyr_box = gr.Textbox(elem_id="kv-lyrics", elem_classes=["kv-hidden"], label="lyrics request")
        push = gr.Button("push", elem_id="kv-push", elem_classes=["kv-hidden"])
        lyr_go = gr.Button("lyrics", elem_id="kv-lyrics-go", elem_classes=["kv-hidden"])
        with gr.Accordion("Table view (type exact values, then Apply)", open=False):
            gr.Markdown("Columns: `#` (the note's id, leave it; empty for new rows), phrase number, syllable (`-` hold, "
                        "`_` rest, empty = swara name), swara, cents, beats **or** sec, ornaments like "
                        "`meend(to=Pa,start=0.5); vibrato(rate=5,depth=25)`.")
            table = gr.Dataframe(headers=TABLE_COLS, interactive=True, wrap=True)
            tapply = gr.Button("Apply table")
        gr.Markdown("**4. Sing**")
        with gr.Row():
            sing = gr.Button("Sing it", variant="primary")
            save = gr.Button("Save JSON")
            saved = gr.File(label="Composition file", interactive=False)
        with gr.Row():
            ca1 = gr.Audio(label="Take 1", type="numpy")
            ca2 = gr.Audio(label="Take 2 (base, when comparing)", type="numpy")
        cabout = gr.Markdown()

        groups = {"tune": g_tune, "lyrics": g_lyrics, "alankar": g_alankar, "sargam": g_sargam, "json": g_json}

        def on_src(s):
            vis = visibility(s)
            return (HINTS.get(s, ""), *[gr.update(visible=vis[g]) for g in GROUPS])
        src.change(on_src, src, [hint] + [groups[g] for g in GROUPS])

        settings = [picker.voice, ctonic, ctempo, craag, ctaal, cgap, ctitle]
        load_in = [src, ctune, ckeep, clyrics, cswaras, cbeats, calank, csargam, cjson, picker.voice]
        load_out = [comp_box, view_box, table, sung_box, *settings, cstatus]
        load.click(do_load, load_in, load_out)
        restore.click(do_restore, None, load_out)
        set_out = [comp_box, view_box, table, cstatus]
        for ctl in (picker.voice, craag, ctaal):                    # choices: when the user picks one
            ctl.input(do_settings, [comp_box, *settings], set_out, trigger_mode="always_last")
        for ctl in (ctonic, ctempo, cgap, ctitle):                  # typed values: when you leave the box or press Enter
            ctl.blur(do_settings, [comp_box, *settings], set_out, trigger_mode="always_last")
            ctl.submit(do_settings, [comp_box, *settings], set_out, trigger_mode="always_last")
        push.click(do_push, comp_box, [view_box, table, cstatus], trigger_mode="always_last")
        lyr_go.click(do_lyrics, [lyr_box, comp_box], [comp_box, view_box, table, cstatus], trigger_mode="always_last")
        tapply.click(do_table, [table, comp_box], [comp_box, view_box, table, cstatus])
        sing.click(do_sing, [comp_box, model, steps, seed, join], [ca1, ca2, cabout, sung_box],
                   concurrency_limit=1, concurrency_id="render")
        save.click(do_save, comp_box, saved)
        comp_box.change(None, comp_box, None, js="(v) => { if (window.KV) window.KV.load(v); return v; }")
        view_box.change(None, view_box, None, js="(v) => { if (window.KV) window.KV.setView(v); return v; }")
        sung_box.change(None, sung_box, None, js="(v) => { if (window.KV) window.KV.setSung(v); return v; }")
    return {"init": lambda page: page.load(do_load, load_in, load_out), "picker": picker,
            "load_comp": load_comp, "load_out": load_out}


def _num_or_text(s):
    """'138.6' -> 138.6; anything else stays text (auto, C#3)."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return str(s).strip()
