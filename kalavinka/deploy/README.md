# kalavinka.deploy: shareable inference bundles

Packs the final Kalavinka model into two zips a friend can run without this repo or the dataset:

| Zip | For | Extras |
|---|---|---|
| `kalavinka_colab.zip` | Google Colab | `Kalavinka_Colab.ipynb` (step-by-step notebook) |
| `kalavinka_mac.zip` | a Mac (or any PC) | `setup_mac.sh` (one-time setup), `sing` (launcher), `MAC.md` |

Both contain the same core (see `build_bundle.py`'s docstring for the layout): our weights (V4-L = run `L4_v2`,
and the stage-1 contour model C-flow = run `I2_v1`), copies of the project modules inference needs, SoulX-Singer's
code, and a library of 17 held-out test lines (tunes) with 10 voice clips. SoulX's base `model.pt` (2.7 GB) is
**not** inside; it is downloaded from Hugging Face on first use.

| File | What |
|---|---|
| `singer.py` | `KalavinkaSinger`: load once, then `sing(lyrics, tune, voice, mode)` / `resing(tune, mode)`; device auto (cuda > mps > cpu) with an MPS -> CPU fallback |
| `cli.py` | `python -m kalavinka_app sing/resing/alankar/sargam/compose/compose-prompt/history/prompt/tunes/voices/demo/ui/fetch` inside a bundle; `--model kalavinka\|base\|both`, `--fresh` |
| `app.py` | Gradio page (tabs: new lyrics; compare with the real singer; alankars/sargam; compose; history), model choice kalavinka / base / both |
| `sargam.py` | sargam notation + JSON score (``LLM_PROMPT``), 7 built-in alankars, ``SargamSinger``, note-by-note pitch check (pYIN) |
| `compose.py` | compositions (JSON): swaras/cents, lengths, lyrics, ornaments (meend, gamak, kan, murki, khatka, zamzama, andolan, vibrato, glides, custom), raags/taals, hand-drawn strokes -> 50 fps F0 + SoulX items (melody mode); ``ComposeSinger``, follow check, ``COMPOSE_PROMPT`` |
| `compose_ui.py`, `static/f0_editor.{js,css}` | the web page's Compose tab: canvas F0 editor (drag pitch/length, lyrics on notes, ornament panel, pencil/eraser, undo, in-browser preview tone), table view, sing + sung-pitch overlay |
| `history.py` | ``outputs/history.jsonl`` log and request cache (same request = saved take reused) |
| `build_bundle.py` | chooses the tunes and voices, vendors the code, cleans the checkpoints, writes both zips |
| `bundle_main.py` | becomes `kalavinka_app/__main__.py` in the bundle |
| `bundle_files/` | README, NOTICE, requirements, notebook, Mac scripts copied into the bundles |

Models: `kalavinka` (V4-L) and `base` (released SoulX-Singer, Hindi phones mapped to the nearest English ones = our
zero-shot V0; loaded on first use). Pitch modes: `score` (the model's own pitch, default) and `expressive` (two-stage: C-flow at temperature 0.85 draws
the F0, SoulX sings it in melody mode). `resing` also offers `real-pitch` (the real singer's F0).

## Alankar benchmark

```
cd <bundle> && for v in 1 4 5 7 10; do python -m kalavinka_app alankar all --model both --voice $v; done
cd saregama624 && ../.venv-p1/bin/python -m kalavinka.eval.alankar_eval --history <bundle>/outputs/history.jsonl \
    --out ../runs/eval_v1/alankars          # pitch + Whisper syllable CER, paired bootstrap; alankar_bench.md
```

## Build

```
cd saregama624
../.venv-sx/bin/python -m kalavinka.deploy.build_bundle --out ../bundles      # or: kalavinka/ops/pack_bundles.sh
```

## Tests

```
../.venv-sx/bin/python -m pytest -q kalavinka/tests/test_deploy.py     # zip checks run when ../bundles exists
```

Checked on 25 Nov 2026 in fresh environments: Python 3.12 + newest PyTorch on a GPU (the notebook, run headless),
and Python 3.10 + CPU-only PyTorch (the Mac scripts on Linux). Not tested on a real Mac GPU (MPS).
