# IE624 Course project (Autumn 2026)
 - Gokularamanan R S (23B1854) 
 - Rushikesh Bhagwan Shinde (23B1529)
***

# Kalavinka: Indian Music Vocals synthesis

Kalavinka is [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) fine-tuned on **10.7 hours of Hindi film
songs** that we collected and labelled ourselves. You give it Hindi lyrics, a tune and a few seconds of a voice, and it
sings. 


**Contents**
1. [Links](#1-links)
2. [Try it: the Colab and Mac apps (no training needed)](#2-try-it-the-colab-and-mac-apps)
3. [Replicate the project, step by step](#3-replicate-the-project-step-by-step)
4. [Where each part of the report lives in the code](#4-where-each-part-of-the-report-lives-in-the-code)
5. [Repository map](#5-repository-map)
6. [Tests](#6-tests)
7. [Credits, license and data](#7-credits-license-and-data)

## 1. Links

| What | Link |
|---|---|
| Code (this repository) | https://github.com/ramanan849/kalavinka |
| Dataset v1: list of the source songs | https://docs.google.com/spreadsheets/d/1LpQ4Byqoeuov-JwNHJD-e4b_Ea7DpTZt4UjNU0904YQ/edit?usp=sharing |
| Gold set (hand-corrected songs) | https://drive.google.com/file/d/1lwxd2_Wfuc7R_cITLlXIqvfaUsXPNv-5/view?usp=sharing |
| App for Google Colab (`kalavinka_colab.zip`) | https://drive.google.com/drive/folders/18IKNkGVkOl9R7MiXcunGDyBLKGAvvXDy?usp=sharing |
| App for Mac (`kalavinka_mac.zip`) | https://drive.google.com/drive/folders/18IKNkGVkOl9R7MiXcunGDyBLKGAvvXDy?usp=sharing |
| Outputs of our models (audio) | https://drive.google.com/drive/folders/1Hozxo1hVehsevUbO7ic8BoGkC9iCuiJg?usp=sharing |
| SoulX-Singer, code (base model) | https://github.com/Soul-AILab/SoulX-Singer |
| SoulX-Singer, weights | https://huggingface.co/Soul-AILab/SoulX-Singer |

## 2. Try it: the Colab and Mac apps

Both apps are self-contained zips (about 230 MB) with our fine-tuned weights, 17 tunes and 24 voices from songs the
model never trained on. You do not need this repository or any training. On first use, each app downloads SoulX-Singer's
base weights (2.7 GB) from Hugging Face, once.

### Option A: Google Colab (nothing to install)

1. Download **`kalavinka_colab.zip`** from the [app folder](https://drive.google.com/drive/folders/18IKNkGVkOl9R7MiXcunGDyBLKGAvvXDy?usp=sharing).
2. Upload the zip, **still zipped**, to the top level of your Google Drive (`My Drive/kalavinka_colab.zip`).
3. On your computer, unzip a copy and find `Kalavinka_Colab.ipynb` inside.
4. Go to [colab.research.google.com](https://colab.research.google.com), choose **File → Upload notebook** and pick
   `Kalavinka_Colab.ipynb`.
5. Choose **Runtime → Change runtime type → T4 GPU**.
6. Run the cells **from top to bottom**. In section 2, keep `source = "drive"`; it mounts your Drive and unzips the app.
   (Or set `source = "upload"` to pick the zip from your computer instead.) The first run takes about 5 minutes.
7. Section 6, **Sing your lyrics**: type Hindi lyrics in Devanagari, then run the cell. Separate lines with `|`.

The later cells compare Kalavinka with the real singer and with base SoulX-Singer, sing alankars, compose a melody,
and open the web page (section 10).

### Option B: Mac (or any Linux / Windows PC)

You need about 6 GB of free disk space and, ideally, 16 GB of RAM (8 GB works, but slowly).

1. Download **`kalavinka_mac.zip`** from the [app folder](https://drive.google.com/drive/folders/18IKNkGVkOl9R7MiXcunGDyBLKGAvvXDy?usp=sharing)
   and unzip it.
2. Open **Terminal** and run the one-time setup (about 10 minutes):
   ```
   cd ~/Downloads/kalavinka_mac      # wherever you unzipped it
   bash setup_mac.sh
   ```
   It finds Python 3.10 to 3.12, installs everything into a private `.venv/` inside the folder, downloads the base
   weights and sings one test line. If it says Python is missing: `brew install python@3.11`, then run it again.
3. Start the web page:
   ```
   ./sing ui
   ```
   Your browser opens. Type Hindi lyrics, pick a tune and a voice, and press **Sing**. Stop it with `Ctrl+C`.
4. Or sing from the Terminal:
   ```
   ./sing "ठंडी हवा में गीत नया"                    # one line, saved in outputs/ and played
   ./sing "ठंडी हवा में गीत नया" --model both        # Kalavinka and base SoulX-Singer, side by side
   ./sing alankar 2                               # a sargam exercise, with a note-by-note pitch check
   ./sing help                                    # every command
   ```

On Linux or Windows, use the same zip: `pip install -r requirements.txt`, then `python -m kalavinka_app ui`.

**What the web page can do**

| Tab | What it does |
|---|---|
| Sing new lyrics | Sings your lyrics on a bundled tune, in a bundled voice. |
| Compare with the real singer | Re-sings a real song line, with the real singer next to it. |
| Alankars / sargam | Built-in alankars or your own swaras, with how many notes land on the right swara. |
| Compose | Draws your own melody, timing and ornaments (meend, gamak, murki, ...) on the pitch curve. |
| Voices | Browses the 24 voices by gender, era and register, with a **Listen** button for each. |
| History | Lists every take: play, label, star, delete and restore, or reopen it in its tab. |

The zip's own `README.md` (and `MAC.md` in the Mac zip) explains every option.

## 3. Replicate the project, step by step

This rebuilds the dataset, trains the models and reproduces the report's numbers. Run the steps in order. Each step
can be re-run safely: the pipeline resumes from where it stopped. Long steps open windows in a tmux session named
`kalavinka`, with logs in `$PROJECT/logs/`.

### Step 0. What you need

- Linux with NVIDIA GPUs. We used single 80 GB H100s per training run; on smaller cards, lower `--max-frames`.
- About 100 GB of free disk space.
- `git`, `tmux`, `ffmpeg` and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

### Step 1. Folder layout, SoulX-Singer and the pretrained models

Everything lives in one project folder, `$PROJECT`. The repository **must** sit directly inside it, because the
scripts find the data relative to it.

```
export PROJECT=~/kalavinka_project && mkdir -p $PROJECT && cd $PROJECT
git clone https://github.com/ramanan849/kalavinka.git
git clone https://github.com/Soul-AILab/SoulX-Singer.git
git -C SoulX-Singer checkout 81aeb3ae772c70093c3de74dc23c92d983801ae4     # the exact version we used, unmodified

HF="uvx --from huggingface_hub hf download"
$HF Soul-AILab/SoulX-Singer            --local-dir SoulX-Singer/pretrained_models/SoulX-Singer
$HF Soul-AILab/SoulX-Singer-Preprocess --local-dir SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess
$HF KimberleyJSN/melbandroformer MelBandRoformer.ckpt --local-dir models/melband-vocals-kj
curl -L -o models/melband-vocals-kj/config_vocals_mel_band_roformer_kj.yaml \
  https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml
```

The ASR, alignment and scoring models (Qwen3-ASR, Whisper-large-v3, MMS_FA, WavLM) download themselves on first use.

When you are done, the folder looks like this. Steps 3 to 11 create the folders marked "later".

```
$PROJECT/
  kalavinka/          this repository
  SoulX-Singer/       base model code + pretrained_models/{SoulX-Singer, SoulX-Singer-Preprocess}
  models/             the full-vocals separator
  dataset/  work/  gold/  export/  runs/  bundles/  logs/      (later)
  .venv-p1/  .venv-sx/                                         (step 2)
```

### Step 2. The two Python environments

```
cd $PROJECT/kalavinka
envs/setup.sh
```

This installs two environments from the uv lock files, with the exact versions of our runs:
- **`.venv-p1`**: Python 3.12, PyTorch 2.8. For annotation and scoring.
- **`.venv-sx`**: Python 3.10, PyTorch 2.2.0, transformers 4.41.2, which are SoulX-Singer's own pins. For training,
  rendering and the apps.

See [`envs/README.md`](envs/README.md). All the commands below run from `$PROJECT/kalavinka`.

### Step 3. Get the songs

We cannot share the audio, because the songs are copyrighted. The [song list](https://docs.google.com/spreadsheets/d/1LpQ4Byqoeuov-JwNHJD-e4b_Ea7DpTZt4UjNU0904YQ/edit?usp=sharing)
gives the year, title and YouTube id of all 1,066 songs.

1. Get the recordings yourself and put them in one folder. Name each file
   `<year> - <title> [<youtube id>].<ext>`, for example `1980 - Aap Jaisa Koi [rY8HQYf7B3I].opus`. Accepted formats:
   opus, m4a, mp3, wav, flac, webm and ogg.
2. Point the pipeline at that folder in `$PROJECT/work/p1.toml`:
   ```toml
   [paths]
   audio_dir = "/full/path/to/the/songs"
   ```

### Step 4. Label the songs (report ch. 4)

The pipeline separates the vocals, tracks the pitch with three trackers, finds the singers and the tonic, gets the
lyrics (two ASR models plus published lyrics), converts Devanagari to phones, aligns the syllables, makes notes and
cuts the songs into clips with quality scores.

```
kalavinka/ops/run_p1_local.sh init          # songs.tsv: one stable id per song
kalavinka/ops/run_p1_local.sh clean 24      # inventory + audio fingerprints (24 CPU workers)
kalavinka/ops/run_p1_local.sh dedupe        # drops re-uploads of the same song
kalavinka/ops/run_p1_local.sh fetch         # published lyrics from LRCLIB (1 request/s)
kalavinka/ops/run_p1_local.sh gpu 18        # separation, pitch, ASR, alignment: one worker per GPU with >= 18 GB free
kalavinka/ops/run_p1_local.sh text 24       # lyrics reconciliation, G2P, notes, clips
kalavinka/ops/run_p1_local.sh retry         # a second lyrics lookup for songs not yet identified
kalavinka/ops/run_p1_local.sh text2 24      # the text stages again, with the bigger lyrics pool
kalavinka/ops/run_p1_local.sh status        # progress and a per-song report
```

Wait for each of the `clean`, `gpu` and `text` steps to finish before you start the next one. Watch them with
`tmux attach -t kalavinka`, or check with `status`.

### Step 5. Gold set and calibration (report sec. 4.2)

```
unzip gold.zip -d $PROJECT                  # the gold set from the links table -> $PROJECT/gold/
$PROJECT/.venv-p1/bin/python -m kalavinka.annot.calibrate --work $PROJECT/work --gold $PROJECT/gold \
    --out $PROJECT/export/calibration --word-acc 0.93
```

This fits the clip-quality thresholds on the hand-corrected songs: accepted clips must be at least 93% word-accurate.
The output is `export/calibration/calibration.json`, with `min_mean_conf = -2.5` and `min_good_src = 1.0`.

### Step 6. Export dataset v1 (report sec. 4.3)

```
$PROJECT/.venv-p1/bin/python -m kalavinka.annot.export_v1 --work $PROJECT/work --out $PROJECT/export/v1 \
    --calibration $PROJECT/export/calibration/calibration.json
$PROJECT/.venv-p1/bin/python -m kalavinka.annot.export_v1 --work $PROJECT/work --out $PROJECT/export/v1_all \
    --calibration $PROJECT/export/calibration/calibration.json --bands accept,review,reject   # only for the noisy-data test
```

Expected result: 6,302 clips from 918 songs, 10.7 h, split by song into train, val and test (6,003, 156 and 143
clips). `export/v1/DATASHEET.md` describes the export.

### Step 7. The Hindi phone table and the training caches

```
$PROJECT/.venv-sx/bin/python -m kalavinka.sx.adapter --phones-out $PROJECT/export/soulx/phone_set_indic.json \
    --check $PROJECT/export/v1/train.jsonl
kalavinka/ops/build_sx_cache.sh $PROJECT/export/v1     $PROJECT/runs/cache_v1
kalavinka/ops/build_sx_cache.sh $PROJECT/export/v1_all $PROJECT/runs/cache_v1_all
```

The first command adds our 62 Hindi phones to SoulX's phone table. The cache script precomputes the mel-spectrograms.

### Step 8. Fine-tune SoulX-Singer (report ch. 6 and 7)

Each run trains a LoRA adapter (r = 32) and the new phone rows on the frozen SoulX-Singer, on one GPU, under an
auto-resume supervisor. The arguments are `<cache> <tag> <min free GB> <runs...>`.

```
kalavinka/ops/launch_sx_runs.sh $PROJECT/runs/cache_v1 v1 16 R1 R2 R3 R4 R5                  # the grid V1 to V5
kalavinka/ops/launch_sx_runs.sh $PROJECT/runs/cache_v1 v2 16 L1 L4 L1b L4b                   # finals V1-L, V4-L, 2 seeds
kalavinka/ops/launch_sx_runs.sh $PROJECT/runs/cache_v1 v2 16 H1 H2 H1b H2b H1c H2c S1 S2 S1b S2b  # held-out phones, 1 hour
kalavinka/ops/launch_sx_runs.sh $PROJECT/runs/cache_v1_all v2 16 N1 N2                      # noisy data
```

Each run writes `runs/<run>_<tag>/` (`best.pt`, `metrics.jsonl`, logs, `live.png`). **Our final model, V4-L, is
`runs/L4_v2`**, which is:

```
$PROJECT/.venv-sx/bin/python -m kalavinka.sx.train --cache $PROJECT/runs/cache_v1 \
    --phones-table $PROJECT/export/soulx/phone_set_indic.json --out $PROJECT/runs/L4_v2 \
    --phones articulatory --bands accept,review --weighting conf \
    --steps 15000 --max-frames 7200 --val-every 500 --patience 8
```

The grid runs take about 40 minutes each and the final runs about 2 hours each, on one H100.

### Step 9. The stage-1 contour models (report ch. 8)

```
kalavinka/ops/launch_ir_runs.sh $PROJECT/export/v1 v1          # I1 = C-reg (regression), I2 = C-flow (flow matching)
$PROJECT/.venv-sx/bin/python -m kalavinka.ir.sweep --run $PROJECT/runs/I2_v1 --export $PROJECT/export/v1
```

The sweep picks C-flow's sampling temperature on the validation split (we got 0.85). It writes the test curves to
`runs/I2_v1/sweep_left/test_f0_tau0.85.json`. I1 writes its curves to `runs/I1_v1/test_f0.json`.

### Step 10. Evaluate (report ch. 6 to 9)

Render the test clips with a model, then score them:

```
SX=$PROJECT/.venv-sx/bin/python; P1=$PROJECT/.venv-p1/bin/python; E=$PROJECT/runs/eval_v1
$SX -m kalavinka.eval.render --export $PROJECT/export/v1 --run $PROJECT/runs/L4_v2 --control score --out $E/L4_score
$SX -m kalavinka.eval.render --export $PROJECT/export/v1 --run zero-shot           --control score --out $E/R0_score
$P1 -m kalavinka.eval.score         --render $E/L4_score --export $PROJECT/export/v1   # CER, pitch error, voicing, SIM
$P1 -m kalavinka.eval.contrast_eval --render $E/L4_score --export $PROJECT/export/v1   # retroflex / aspiration scores
$P1 -m kalavinka.eval.compare --help                                                   # paired song-level bootstrap
```

- **Pitch modes:** `--control melody` follows the real singer's pitch. For the two-stage system, use
  `--control melody-ir --f0-json $PROJECT/runs/I2_v1/sweep_left/test_f0_tau0.85.json` (or I1's `test_f0.json`).
- **Whole queue:** `kalavinka/ops/eval_queue.sh` runs render, contrast and score for many runs in one go.
- **More tests:** the alankar, long-passage and listening tests are in [`kalavinka/README.md`](kalavinka/README.md#evaluation).

### Step 11. Build the app zips

```
kalavinka/ops/pack_bundles.sh               # -> $PROJECT/bundles/kalavinka_colab.zip, kalavinka_mac.zip
```

## 4. Where each part of the report lives in the code

| Report | Code | Name in the code |
|---|---|---|
| Dataset and labelling pipeline (ch. 4) | `saregama/` (stages), `kalavinka/ops/run_p1_local.sh`, `kalavinka/annot/` | P1 |
| Idea (A): articulatory phone embeddings | `kalavinka/sx/phones.py`, `kalavinka/sx/adapter.py` | `--phones articulatory` (vs `nearest`) |
| Idea (B): confidence-weighted training | `kalavinka/sx/train.py` | `--weighting conf` (vs `none`) |
| Idea (C): two-stage contour model | `kalavinka/ir/` | I1 = C-reg, I2 = C-flow |
| Variants V0 and V1 to V5 | `kalavinka/ops/launch_sx_runs.sh` | V0 = zero-shot (R0), V1 to V5 = R1 to R5 |
| Final models V1-L and V4-L | the same | L1 and L4 (seed 1: L1b and L4b); V4-L = `runs/L4_v2` |
| Held-out phones, noisy data, 1 hour of data | the same | HO = H1/H2, NZ = N1/N2, SD = S1/S2 |
| Metrics and statistics | `kalavinka/eval/score.py`, `contrast_eval.py`, `compare.py` | song-level bootstrap with Holm correction |
| Listening check | `kalavinka/eval/listening/` | |
| Beyond the test set (ch. 9) | `kalavinka/eval/sing_new.py`, `alankar_eval.py`, `long_passage.py` | |
| The app | `kalavinka/deploy/` | `KalavinkaSinger`, `kalavinka_app` |

## 5. Repository map

```
kalavinka/       everything specific to this project (see kalavinka/README.md)
  annot/           gold-set calibration, dataset export
  sx/              SoulX adapter: Hindi phone table, ideas (A) and (B), trainer
  ir/              stage-1 contour models, idea (C)
  eval/            rendering, metrics, minimal-pair scores, bootstrap, listening test, demo songs
  deploy/          the Colab / Mac app and the zip builder (see kalavinka/deploy/README.md)
  ops/             run scripts: annotation runner, training launchers, GPU picker, job queue
  tests/           tests for all of the above
saregama/        the annotation package Kalavinka builds on (written for Carnatic music, reused unchanged; saregama/README.md)
annotator_model/ saregama's optional unified annotator model (F0 + phone CTC + onsets); not used for our results
p1_project/      the gold-set annotator (web app), the pipeline benchmark and four example songs
configs/ lexicons/ scripts/ tests/    saregama's default config, lexicon, Carnatic stage runner and tests
envs/            the two uv environments and their lock files
docs/            the final report
```

## 6. Tests

Each environment runs the tests for its own half. One test file in each needs the other environment.

```
$PROJECT/.venv-p1/bin/python -m pytest -q kalavinka/tests tests --ignore=kalavinka/tests/test_sx_ablation.py  # 230 pass
$PROJECT/.venv-sx/bin/python -m pytest -q kalavinka/tests --ignore=kalavinka/tests/test_calibrate.py          # 146 pass
```

## 7. Credits, license and data


**Built on** [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) (Apache-2.0) by Soul AI Lab, and on RMVPE,
FCPE, Mel-Band RoFormer, Qwen3-ASR, Whisper and MMS.

**License.** The code is under the [Apache-2.0 license](LICENSE). The songs are copyrighted, so the audio is not
shared. The example clips in `p1_project/examples/` and the audio inside the apps are excerpts for this course
project only, for research and teaching.

## Use of AI tools
Claude (Anthropic) was used as a coding and debugging assistant. The ideas, the data curation and
annotation, the experimental design and the conclusions are ours, and we checked the results.
