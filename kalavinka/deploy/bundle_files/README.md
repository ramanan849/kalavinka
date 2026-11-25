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
```
Takes are saved in `outputs/` with a `.json` that says what was sung on which tune.
Options (before the command): `--device auto|cuda|mps|cpu`, `--steps N` (32 on a GPU, 16 on CPU by default),
`--seed N` (another take), `--play`. The word `sing` can be left out: `python -m kalavinka_app "..."`.

## Good to know

* **Lyrics must be Devanagari** (हिंदी). Use a Hindi keyboard or Google Input Tools to convert Hinglish.
* **Tunes**: 17 lines from 10 songs the model never trained on. A tune needs at least as many syllables as your line
  (about one per vowel); leave the tune on *auto* and it picks one that fits. Extra notes of the tune are dropped.
* **Voices**: by default the tune's own singer. Choosing another voice also moves the tune into that voice's range.
* **Pitch modes**: *model's own pitch* (default) sounded most natural in our listening test. *Expressive* is our
  two-stage system: a small contour model draws a vibrato-rich pitch curve and SoulX sings it.
* **Speed** for a 3-second line: about 5 s on a data-centre GPU (Colab's T4 is slower), about 20 s on a 100-core
  server CPU; a laptop CPU takes minutes. On a CPU the default is 16 steps instead of 32.
* "x% of its phrases occur in the training lyrics" tells you how new your line is to the model (0% = all new).

## What is inside

```
kalavinka_app/   singer (KalavinkaSinger), command line, web page
vendor/          the project code inference needs, and SoulX-Singer's code
weights/         our fine-tune (kalavinka_v4l, 64 MB) and the contour model (contour_cflow, 164 MB)
data/            tunes, voices and their audio clips, the Hindi phone table
```
See `NOTICE.md` for licences and the rules for the audio clips.
