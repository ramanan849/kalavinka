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
Your browser opens a page where you type Hindi lyrics, pick a tune and a voice, and press **Sing**. The second tab
re-sings a real song line so you can compare it with the real singer. Stop it with Ctrl+C in the Terminal.

**From the Terminal**

```
./sing "ठंडी हवा में गीत नया"                          # sing a line; saved in outputs/ and played
./sing "चाँद की रोशनी में तेरा नाम" --tune 7 --voice 2   # choose the tune and the voice
./sing "बारिश की बूँदों ने कुछ कहा" --expressive         # two-stage pitch (more vibrato)
./sing "पहली पंक्ति | दूसरी पंक्ति"                    # several lines, one after another
./sing resing 3                                      # tune 3 with its real words, + the real singer
./sing tunes        ./sing voices        ./sing demo        ./sing help
```

| Option | Meaning |
|---|---|
| `--tune N` | which song line to borrow the melody from (`./sing tunes`); default: one that fits your line |
| `--voice N` | whose voice (`./sing voices`); default: the tune's own singer |
| `--expressive` | our two-stage system draws the pitch curve (default: the model's own pitch, which sounded more natural) |
| `--seed N` | another take of the same line |
| `--steps N` (before the lyrics) | quality vs speed: 32 on the Mac GPU, 16 on CPU by default |
| `--device cpu` (before the lyrics) | force the CPU (if the Mac GPU misbehaves) |

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
| download interrupted | run `./sing fetch` again; it resumes |
| "Cannot sing that: ..." | the message says why (script, line too long, unknown tune or voice) |

Licences and the rules for the bundled audio clips: `NOTICE.md`.
