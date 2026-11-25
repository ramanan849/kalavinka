# Kalavinka: Hindi film-song singing

Kalavinka is SoulX-Singer fine-tuned on 10.7 hours of Hindi film songs (our IE624 project). Type a line of Hindi
lyrics, and it sings them to the tune of a real song line, in the voice of a short clip.

## Quick start

| Where | What to do |
|---|---|
| Google Colab | open `Kalavinka_Colab.ipynb` and run the cells top to bottom (see the notebook) |
| Mac | `bash setup_mac.sh` once, then `./sing ui` (see `MAC.md`) |
| Any Linux/Windows PC with Python 3.10+ | `pip install -r requirements.txt`, then `python -m kalavinka_app ui` |

The first run downloads SoulX-Singer's base weights (2.7 GB, from Hugging Face) into
`vendor/SoulX-Singer/pretrained_models/SoulX-Singer/`. After that everything works offline.

## Commands

```
python -m kalavinka_app sing "ठंडी हवा में गीत नया"            # new lyrics, tune picked automatically
python -m kalavinka_app sing "चाँद की रोशनी में तेरा नाम" --tune 7 --voice 2
python -m kalavinka_app sing "बारिश की बूँदों ने कुछ कहा" --expressive   # two-stage pitch (more vibrato)
python -m kalavinka_app sing "पहली पंक्ति | दूसरी पंक्ति"        # several lines, sung one after another
python -m kalavinka_app resing 3                                # a bundled line, next to the real singer
python -m kalavinka_app tunes        # list tunes          python -m kalavinka_app voices   # list voices
python -m kalavinka_app demo         # three example lines in both pitch modes
python -m kalavinka_app ui           # web page            (add --share for a public link)
python -m kalavinka_app sing "..." --model both                 # compare with the base SoulX-Singer
python -m kalavinka_app alankar 2                               # an alankar, both models, pitch checked note by note
python -m kalavinka_app sargam "Sa Re Ga | Re Ga ma"             # your own swaras (or --json score.json)
python -m kalavinka_app prompt       # the prompt that lets ChatGPT write a sargam JSON
python -m kalavinka_app compose mine.json --model both    # your own melody + ornaments (see Compose below)
python -m kalavinka_app history      # earlier takes (each request is cached: asking again reuses the take)
```
Takes are saved in `outputs/` with a `.json` that says what was sung, and logged in `outputs/history.jsonl`.
Options (before the command): `--device auto|cuda|mps|cpu`, `--steps N` (32 on a GPU, 16 on CPU by default),
`--seed N` (another take), `--play`. The word `sing` can be left out: `python -m kalavinka_app "..."`.

## Compose: your own melody, timing and ornaments

The web page's **Compose (pitch, timing, ornaments)** tab lets you write the melody yourself and edit it on the
pitch (F0) curve:

1. **Start from** the Yaman example, a bundled tune (with the real singer's pitch curve, or as plain notes), Hindi
   lyrics on swaras you type (`Ga Pa Dha Pa Ga Re Sa`, one per syllable), a sargam line or alankar, a JSON
   composition (e.g. from ChatGPT: open *JSON / ask ChatGPT* for the prompt), or a blank page. Press **Load**.
2. Set the **voice**, **Sa** (tonic; `auto` = the voice's own), **tempo**, **raag** (notes snap to its swaras) and
   **taal** (the beat-cycle grid).
3. Edit on the curve:
   * drag a note **up/down** to change its pitch (snaps to the raag; hold **Alt** for free cents, e.g. a shruti);
   * drag a note's **right edge** to change its length (quarter beats; Alt = free);
   * **double-click** a note to change its syllable (`-` = hold the previous syllable, a slur; `_` = rest; empty =
     sing the swara name);
   * **click** a note to edit it below the canvas and **add ornaments**, each with its own settings:
     meend (glide to a swara), andolan (slow sway), gamak (fast heavy shake to a neighbour swara), kan (grace note),
     murki (quick turn), khatka, zamzama (fast clusters), vibrato, glide in/out, custom shape. Carnatic names work
     too in JSON: jaru = meend, kampita = gamak, sphurita/nokku = kan;
   * **Draw pitch** (pencil): draw the curve by hand over the sung notes; **Erase drawing** removes it;
   * keys: arrows up/down = next swara, left/right = next note, +/- = length, Delete, Ctrl+Z / Ctrl+Y;
   * **+ phrase**: each phrase is one breath and is sung separately.
4. **▶ preview** plays the curve at once as a plain tone in the browser (no model), so you can check the melody.
5. **Sing it** sings it with the model(s) chosen at the top (Kalavinka, base, or both). The table shows how closely
   each model followed your curve, and the orange line in the editor is what the model actually sang.

Exact values can be typed in the **Table view** (ornaments like `meend(to=Pa,start=0.5); vibrato(rate=5,depth=25)`).
**Save JSON** writes the composition to `outputs/compositions/`.

From the command line:
```
python -m kalavinka_app compose --example --save mine.json          # or --from-tune 3, or --sargam "Sa Re Ga | ma Pa"
python -m kalavinka_app compose mine.json --model both              # sing it; prints how closely each model followed the curve
python -m kalavinka_app compose-prompt                              # prompt for ChatGPT; save its answer as x.json, then compose x.json
```
The JSON format is explained at the top of `kalavinka_app/compose.py`.

## Good to know

* **Lyrics must be Devanagari** (हिंदी). Use a Hindi keyboard or Google Input Tools to convert Hinglish.
* **Tunes**: 17 lines from 10 songs the model never trained on. A tune needs at least as many syllables as your line
  (about one per vowel); leave the tune on *auto* and it picks one that fits. Extra notes of the tune are dropped.
* **Voices**: by default the tune's own singer. Choosing another voice also moves the tune into that voice's range.
* **Pitch modes**: *model's own pitch* (default) sounded most natural in our listening test. *Expressive* is our
  two-stage system: a small contour model draws a vibrato-rich pitch curve and SoulX sings it.
* **Speed** for a 3-second line: about 5 s on a data-centre GPU (Colab's T4 is slower), about 20 s on a 100-core
  server CPU; a laptop CPU takes minutes. On a CPU the default is 16 steps instead of 32.
* **Base model**: `--model base` is the released SoulX-Singer with Hindi phones mapped to English ones (our zero-shot
  baseline); `both` gives one take from each, for comparison.
* "x% of its phrases occur in the training lyrics" tells you how new your line is to the model (0% = all new).

## What is inside

```
kalavinka_app/   singer (KalavinkaSinger), command line, web page, sargam/alankars, history + cache
vendor/          the project code inference needs, and SoulX-Singer's code
weights/         our fine-tune (kalavinka_v4l, 64 MB) and the contour model (contour_cflow, 164 MB)
data/            tunes, voices and their audio clips, the Hindi phone table
```
See `NOTICE.md` for licences and the rules for the audio clips.
