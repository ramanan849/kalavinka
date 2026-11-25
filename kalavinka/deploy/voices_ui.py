"""Web-page pieces for voices (``voices``): a picker used on every tab, and a Voices tab to browse and edit profiles.

* ``VoicePicker``: filters (gender, era, register) in one row, the voice dropdown ("Voice 4 · female · high · 1990s
  · Lata Mangeshkar"), a profile card and a **Listen** player with the voice's own clip. Its ``.voice`` value is the
  voice number as text, or ``"auto"`` (the tune's own singer) where that is allowed.
* ``build_voices_tab``: every voice in a table (filterable), and an **edit** form for the singer's name, gender,
  register and notes. Edits go to ``outputs/voice_profiles.local.json`` (``voices.save_edit``), are logged in the
  history's event log, and refresh every picker at once.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from . import voices as V

AUTO = "auto"
AUTO_LABEL = "the tune's own singer"


class VoiceBook:
    """Profiles of one bundle (re-read after edits), shared by all pickers."""

    def __init__(self, bundle: Path, outputs: Path, hist=None):
        """Remember where the data and the local edits are."""
        self.bundle, self.outputs, self.hist = Path(bundle), Path(outputs), hist
        self.reload()

    def reload(self) -> List[Dict]:
        """Read the profiles again (after an edit)."""
        self.profiles = V.load(self.bundle, self.outputs)
        return self.profiles

    def get(self, no) -> Optional[Dict]:
        """Profile by number (None for auto / unknown)."""
        try:
            n = int(no)
        except (TypeError, ValueError):
            return None
        return next((p for p in self.profiles if p["no"] == n), None)

    def choices(self, gender="any", era="any", register="any", allow_auto=False):
        """(label, value) pairs for a dropdown."""
        ps = V.filter_profiles(self.profiles, gender, era, register)
        return ([(AUTO_LABEL, AUTO)] if allow_auto else []) + [(p["label"], str(p["no"])) for p in ps]

    def sample(self, no) -> Optional[str]:
        """Path of the voice's clip (for the Listen player)."""
        p = self.get(no)
        return str(self.bundle / "data" / p["sample"]) if p else None

    def card(self, no) -> str:
        """Markdown card (or a note for auto)."""
        p = self.get(no)
        return V.card(p) if p else "Each line is sung by the singer of its tune (pick a voice to choose one)."


class VoicePicker:
    """Filters + voice dropdown + profile card + Listen player (create inside a Blocks context)."""

    def __init__(self, gr, book: VoiceBook, default: Optional[int] = None, allow_auto: bool = False,
                 label: str = "Voice", open_card: bool = False):
        """Build the components and wire the filters."""
        self.book, self.allow_auto = book, allow_auto
        value = AUTO if (default is None and allow_auto) else str(default or book.profiles[0]["no"])
        with gr.Group():
            with gr.Row():
                self.gender = gr.Dropdown(["any", "female", "male"], value="any", label="Gender", scale=1, min_width=110)
                self.era = gr.Dropdown(["any", *V.ERAS], value="any", label="Era", scale=1, min_width=110)
                self.register = gr.Dropdown(["any", *V.REGISTERS], value="any", label="Register", scale=1, min_width=110)
                self.voice = gr.Dropdown(book.choices(allow_auto=allow_auto), value=value, label=label, scale=4)
            with gr.Accordion("About this voice", open=open_card):
                with gr.Row():
                    self.card = gr.Markdown(book.card(value))
                    self.sample = gr.Audio(book.sample(value), label="Listen (the voice's own clip)", type="filepath",
                                           interactive=False)
        filt = [self.gender, self.era, self.register]
        for f in filt:
            f.change(self.on_filter, filt + [self.voice], self.voice)
        self.voice.change(self.on_voice, self.voice, [self.card, self.sample])
        self._gr = gr

    def on_filter(self, gender, era, register, current):
        """Narrow the dropdown; keep the current voice if it still matches."""
        ch = self.book.choices(gender, era, register, self.allow_auto)
        values = [v for _, v in ch]
        keep = current if current in values else (values[0] if values else None)
        return self._gr.update(choices=ch, value=keep)

    def on_voice(self, no):
        """Card and sample of the chosen voice."""
        return self.book.card(no), self.book.sample(no)

    def refresh(self, current):
        """New labels after a profile edit."""
        return self._gr.update(choices=self.book.choices(allow_auto=self.allow_auto), value=current)


def table_rows(book: VoiceBook, gender="any", era="any", register="any", text="") -> List[List]:
    """Rows of the voices table."""
    rows = []
    for p in V.filter_profiles(book.profiles, gender, era, register, text):
        rng = f"{p['range_hz'][0]:.0f}-{p['range_hz'][1]:.0f}" if p.get("range_hz") else ""
        rows.append([p["no"], p["name"] or "-", p["gender"], p["register"], p["decade"], f"{p['song']} ({p['year']})",
                     f"{p['sa_hz']:.0f}" if p.get("sa_hz") else "", rng,
                     (p["gender_from"] or "") + (" · edited" if p["edited"] else ""), p["notes"]])
    return rows


COLS = ["no", "singer", "gender", "register", "era", "source song", "Sa (Hz)", "range (Hz)", "from", "notes"]


def build_voices_tab(gr, book: VoiceBook, pickers: List[VoicePicker]):
    """The Voices tab (call inside ``gr.Blocks``): browse, listen, edit; edits refresh every picker."""
    with gr.Tab("Voices"):
        gr.Markdown("All voices you can sing with: real singers' clips from songs the model never trained on. "
                    "Gender and register are **estimated** (gender from the song's credited singers, and from the "
                    "sung pitch in duets); the singer is the credited singer of the source song. Correct anything "
                    "below: your edits are saved in `outputs/voice_profiles.local.json` and logged.")
        with gr.Row():
            g = gr.Dropdown(["any", "female", "male", "unsure"], value="any", label="Gender")
            e = gr.Dropdown(["any", *V.ERAS], value="any", label="Era")
            r = gr.Dropdown(["any", *V.REGISTERS], value="any", label="Register")
            t = gr.Textbox(label="Search (name, song, notes)")
        table = gr.Dataframe(table_rows(book), headers=COLS, interactive=False, wrap=True)
        with gr.Row():
            pick = gr.Dropdown(book.choices(), value=str(book.profiles[0]["no"]), label="Voice to look at / edit",
                               scale=3)
            listen = gr.Audio(book.sample(book.profiles[0]["no"]), label="Listen", type="filepath", interactive=False,
                              scale=2)
        card = gr.Markdown(book.card(book.profiles[0]["no"]))
        with gr.Accordion("Edit this voice", open=False):
            with gr.Row():
                name = gr.Textbox(label="Singer's name")
                gen = gr.Dropdown(list(V.GENDERS), label="Gender")
                reg = gr.Dropdown(list(V.REGISTERS), label="Register")
                conf = gr.Checkbox(label="Name confirmed by ear")
            notes = gr.Textbox(label="Notes (e.g. 'breathy, good for slow songs')", lines=2)
            save = gr.Button("Save changes", variant="primary")
            status = gr.Markdown()
        filt = [g, e, r, t]

        def do_filter(gv, ev, rv, tv):
            return table_rows(book, gv, ev, rv, tv)
        for f in filt:
            f.change(do_filter, filt, table)

        def do_pick(no):
            p = book.get(no)
            if not p:
                return "", None, "", None, None, False, ""
            return (V.card(p), book.sample(no), p["name"], p["gender"], p["register"] or None,
                    p["singer_confirmed"], p["notes"])
        pick.change(do_pick, pick, [card, listen, name, gen, reg, conf, notes])

        def do_save(no, nm, gv, rv, cf, nt, *currents):
            try:
                fields = {"name": (nm or "").strip(), "notes": (nt or "").strip(), "singer_confirmed": bool(cf)}
                if gv:
                    fields["gender"] = gv
                if rv:
                    fields["register"] = rv
                V.save_edit(book.outputs, int(no), **fields)
            except ValueError as err:
                raise gr.Error(str(err))
            if book.hist is not None:
                book.hist.log("voice_edit", voice=int(no), **fields)
            book.reload()
            ups = [pk.refresh(c) for pk, c in zip(pickers, currents)]
            return (f"Saved voice {no}.", V.card(book.get(no)), table_rows(book),
                    gr.update(choices=book.choices(), value=no), *ups)
        save.click(do_save, [pick, name, gen, reg, conf, notes] + [pk.voice for pk in pickers],
                   [status, card, table, pick] + [pk.voice for pk in pickers])
    return pick
