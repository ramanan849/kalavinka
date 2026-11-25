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
| `cli.py` | `python -m kalavinka_app sing/resing/tunes/voices/demo/ui/fetch` inside a bundle |
| `app.py` | Gradio page (two tabs: new lyrics; compare with the real singer) |
| `build_bundle.py` | chooses the tunes and voices, vendors the code, cleans the checkpoints, writes both zips |
| `bundle_main.py` | becomes `kalavinka_app/__main__.py` in the bundle |
| `bundle_files/` | README, NOTICE, requirements, notebook, Mac scripts copied into the bundles |

Pitch modes: `score` (the model's own pitch, default) and `expressive` (two-stage: C-flow at temperature 0.85 draws
the F0, SoulX sings it in melody mode). `resing` also offers `real-pitch` (the real singer's F0).

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
