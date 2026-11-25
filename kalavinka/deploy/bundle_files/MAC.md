# Kalavinka on a Mac

## 1. Set up (once, about 10 minutes)

Open **Terminal**, go to this folder and run the setup script:

```
cd ~/Downloads/kalavinka_mac        # wherever you unzipped it
bash setup_mac.sh
```

It finds Python 3.10 to 3.12, makes a private environment in `.venv/`, installs PyTorch and the other packages,
downloads SoulX-Singer's base weights (2.7 GB, only once) and sings one test line. If it says Python is missing:
`brew install python@3.11` (or install [uv](https://docs.astral.sh/uv/)), then run it again.

You need about **6 GB of free disk** and preferably **16 GB of RAM** (8 GB works, slowly).

## 2. Use it

**The easy way: the web page**

```
./sing ui
```
Your browser opens a page where you type Hindi lyrics, pick a tune and a voice, and press **Sing**. At the top, the
**Model** choice picks Kalavinka (ours), the base SoulX-Singer, or **both side by side**. Tabs:

* **Sing new lyrics**: your lyrics on a bundled tune.
* **Compare with the real singer**: a real song line re-sung, next to the real singer.
* **Alankars / sargam**: a built-in alankar (Sa Re Ga, Re Ga ma, ...), your own swaras, or a JSON score written
  by ChatGPT; you get a table of how many notes land on the right swara.
* **History**: every take you have made (here or in the Terminal), playable.

Stop it with Ctrl+C in the Terminal.

**From the Terminal**

```
./sing "ठंडी हवा में गीत नया"                          # sing a line; saved in outputs/ and played
./sing "चाँद की रोशनी में तेरा नाम" --tune 7 --voice 2   # choose the tune and the voice
./sing "बारिश की बूँदों ने कुछ कहा" --expressive         # two-stage pitch (more vibrato)
./sing "पहली पंक्ति | दूसरी पंक्ति"                    # several lines, one after another
./sing resing 3                                      # tune 3 with its real words, + the real singer
./sing tunes        ./sing voices        ./sing demo        ./sing help
```

**Compare with the base model.** Add `--model both` (or `--model base`) to `sing`, `resing`, `alankar`, `sargam`:

```
./sing "ठंडी हवा में गीत नया" --model both                 # one take from each model
./sing resing 3 --model both                          # both models + the real singer
```

**Alankars and sargam** (these compare both models by default and check the pitch of every note):

```
./sing alankar list                                   # the 7 built-in alankars
./sing alankar 2                                      # Sa Re Ga, Re Ga ma, ... Dha Ni Sa' and back
./sing alankar all --voice 4                          # all of them, in voice 4
./sing sargam "Sa Re Ga | Re Ga ma | Ga ma Pa"         # your own swaras
./sing sargam "Sa re(k) Ga ma | Pa dha(k) Ni Sa'"      # komal: (k), tivra: Ma(t), upper: Sa', lower: .Ni
./sing sargam --json my_alankar.json                  # a score written by ChatGPT (see below)
```
Notation: swaras separated by spaces, phrases by `|` or new lines; `Pa:2` = two beats, `-` = hold one more beat,
`_` = rest. Options: `--voice N`, `--tonic auto|C#3|138.6` (Sa; auto = the voice's own), `--beat 0.45` (seconds
per note), `--vowel aa` (sing "aa" instead of the swara names). The table printed at the end says, per model, how many
notes landed within 50 cents of the written swara ("on the swara"), within 25 cents ("in tune"), and the mean error.

**Let ChatGPT write the score.** `./sing prompt` prints a prompt; paste it into ChatGPT, describe the alankar at the
end ("Sa Ga Re Ma, Re Ma Ga Pa, ... up to Sa' and back"), save its answer as `my_alankar.json`, then
`./sing sargam --json my_alankar.json`. (Or paste the JSON into the web page's Alankars tab.)

**History and cache.** Every take is saved in `outputs/` and logged in `outputs/history.jsonl`. Asking for exactly the
same thing again (same words, tune, voice, model, pitch, seed, steps) plays the saved take instead of singing it
again, so comparisons are instant the second time. `./sing history` lists earlier takes; `--seed 1` gives a new take;
`--fresh` forces a re-run.

| Option | Meaning |
|---|---|
| `--tune N` | which song line to borrow the melody from (`./sing tunes`); default: one that fits your line |
| `--voice N` | whose voice (`./sing voices`); default: the tune's own singer |
| `--model kalavinka\|base\|both` | ours (default), the base SoulX-Singer, or both for comparison |
| `--expressive` | our two-stage system draws the pitch curve (default: the model's own pitch, which sounded more natural) |
| `--seed N` | another take of the same line |
| `--steps N` (before the lyrics) | quality vs speed: 32 on the Mac GPU, 16 on CPU by default |
| `--device cpu` (before the lyrics) | force the CPU (if the Mac GPU misbehaves) |
| `--fresh` (before the lyrics) | sing again even if the same request is in the history |

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
./sing compose --example --save mine.json          # or --from-tune 3, or --sargam "Sa Re Ga | ma Pa"
./sing compose mine.json --model both              # sing it; prints how closely each model followed the curve
./sing compose-prompt                              # prompt for ChatGPT; save its answer as x.json, then compose x.json
```
The JSON format is explained at the top of `kalavinka_app/compose.py`.

## 3. Tips

* **Devanagari only.** On a Mac: System Settings → Keyboard → Input Sources → add *Hindi – Transliteration*;
  then typing `thandi hawa` gives ठंडी हवा.
* A tune needs at least as many syllables as your line (about one per vowel). Very long lines: split with `|`.
* Every take has a `.json` next to it with the tune, the voice and how new the line is.

## 4. If something goes wrong

| Problem | Fix |
|---|---|
| `Not set up yet` | run `bash setup_mac.sh` |
| "this step is not supported on the Mac GPU (MPS) yet; switching to CPU" | nothing to do; it continues on the CPU (slower). We could not test on a Mac GPU before sharing. |
| very slow | add `--steps 12`; close other apps; Intel Macs only have the CPU |
| memory full with `--model both` | both models together need about 5 GB; use one model at a time on an 8 GB Mac |
| download interrupted | run `./sing fetch` again; it resumes |
| "Cannot sing that: ..." | the message says why (script, line too long, unknown tune or voice) |

Licences and the rules for the bundled audio clips: `NOTICE.md`.
