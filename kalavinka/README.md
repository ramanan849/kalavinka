# kalavinka

Hindi film-song singing voice synthesis on SoulX-Singer, built on the `saregama`
annotation package (which stays unmodified; kalavinka wraps it). The [main README](../README.md) gives the
ordered replication steps and the app quick start; this file is the reference for each part.

**Names.** The report calls the ideas (A), (B) and (C); the code and older notes use their working names:
(A) articulatory phone embeddings = `--phones articulatory`; (B) confidence-aware flow matching =
`--weighting conf`; (C) the two-stage system = the contour models in `ir/` (I1 = C-reg, I2 = C-flow).

| Part | What | Status |
|---|---|---|
| `annot/` | P1 finishing: gold-set threshold calibration, v1 export with per-frame confidence, demo-song transcription | done |
| `sx/` | stage 2: SoulX fine-tune -- Indic phone table + `IndicDataProcessor`, articulatory phone embeddings (A), confidence-aware flow matching (B), trainer | done |
| `ir/` | stage 1 of idea (C): expressive-contour model (note scaffold -> expressive F0), temperature sweep | done |
| `eval/` | held-out rendering, CER/F0/SIM, minimal-pair contrasts, paired bootstrap, listening test, demo tracks | done |
| `ops/` | env setup, local (no-Slurm) P1 runner, training launchers, GPU picker, job queue ([`ops/README.md`](ops/README.md)), web fetcher | done |
| `deploy/` | shareable inference bundles: Colab notebook zip and Mac pipeline zip (`KalavinkaSinger`, CLI, web page; base-vs-ours comparison, alankars/sargam, Compose tab (F0 editor with ornaments, lyrics typed onto phrases), seamless joining of lines and phrases, 24 voice profiles with filters, History tab with labels / delete-restore / event log / reopen) | done |

## Layout on disk

`PROJECT` = the repo's parent (here `/workspace/vis-enc/624`). Data never goes in git:

```
PROJECT/
  kalavinka/              this repo (any folder name; it must sit directly in PROJECT)
  SoulX-Singer/           upstream clone + pretrained_models/{SoulX-Singer,SoulX-Singer-Preprocess}
  models/melband-vocals-kj/   full-vocals separator
  dataset/Kalavinka/...   the 1,066 .opus songs
  work/  gold/  export/   P1 state, gold set, exports
  runs/<run>/             training runs (checkpoints, timestamped logs, metrics.jsonl)
  logs/                   setup / P1 worker logs
  .venv-p1  .venv-sx      the two uv venvs
```

## Environments

```
envs/setup.sh                        # from the uv lock files: .venv-p1 (py3.12, torch 2.8) and .venv-sx (py3.10, torch 2.2)
```
See [`../envs/README.md`](../envs/README.md). `.venv-sx` gets the repo and the SoulX clone on its path via a `.pth`
file (`saregama` itself needs py>=3.11, so only its pure-python modules are used there). The older
`kalavinka/ops/setup_envs.sh` builds the same envs with loose pins.

## Annotation run (P1) on a local multi-GPU machine

All steps are resumable; long ones open windows in tmux session `kalavinka`
with logs in `PROJECT/logs/p1/`. `work/p1.toml` points `audio_dir` at the dataset.

```
kalavinka/ops/run_p1_local.sh init          # songs.tsv (stable ids <year>_<slug>__<ytid>)
kalavinka/ops/run_p1_local.sh clean 24      # inventory + fingerprint, 24 CPU workers
kalavinka/ops/run_p1_local.sh dedupe
kalavinka/ops/run_p1_local.sh fetch         # LRCLIB + YouTube oEmbed (1 req/s per host)
kalavinka/ops/run_p1_local.sh gpu 18        # one worker per GPU with >= 18 GiB free
kalavinka/ops/run_p1_local.sh text 24       # refpool + text stages
kalavinka/ops/run_p1_local.sh retry         # second lyrics lookup for unidentified songs
kalavinka/ops/run_p1_local.sh text2 24      # text stages again on the bigger pool
kalavinka/ops/run_p1_local.sh status
```

## Calibration (gold set)

```
.venv-p1/bin/python -m kalavinka.annot.calibrate --work $PROJECT/work --gold $PROJECT/gold --word-acc 0.93
```
(The dataset was built with `--word-acc 0.93`, the target in the report; the default is 0.95.)
Scores every planned clip inside a gold window against the corrected TextGrids
and fits `min_mean_conf`, `min_good_src` for the most accepted time with pooled
word accuracy >= `--word-acc` and onsets within 50 ms >= 0.90. Writes
`export/calibration/{calibration.json, clips.tsv, report.md}`.

## SoulX adapter and fine-tuning

```
.venv-sx/bin/python -m kalavinka.sx.adapter --phones-out $PROJECT/export/soulx/phone_set_indic.json \
    --check $PROJECT/export/v1/train.jsonl          # real DataProcessor check
.venv-sx/bin/python -m kalavinka.sx.train --cache <cache dir> --phones-table <phone_set_indic.json> \
    --out $PROJECT/runs/<run> --phones {nearest|articulatory} --bands accept[,review] --weighting {none|conf}
```
Runs (the arguments of each are in `ops/launch_sx_runs.sh`; report names in brackets):

| Run | Phones | Data / loss | Notes |
|---|---|---|---|
| R0 (V0) | zero-shot, Hindi phones remapped to nearest English tokens | - | eval only (`--run zero-shot`) |
| R1 to R5 (V1 to V5) | nearest / articulatory | accept or accept+review; uniform or confidence-weighted | first grid, <= 5k steps, patience 5 |
| L1, L4 (V1-L, V4-L) | nearest / articulatory | as R1 / R4 | finals: 15k steps, 2x batch (`--max-frames 7200`), patience 8; seed 1 = L1b, L4b |
| H1, H2 (+b, c) (HO-N, HO-A) | nearest / articulatory | accept, without every clip containing Th, Dh, Rh | held-out phone test, 3 seeds |
| N1, N2 (NZ-U, NZ-W) | articulatory | accept+review+reject (cache `v1_all`); uniform / weighted | noisy-data test of (B) |
| S1, S2 (+b) (SD-N, SD-A) | nearest / articulatory | 1 hour of accept data | small-data test of (A) |

Our final model V4-L is `runs/L4_v2` (the one inside the apps).

Smoke checks done: 30-step run and a 300-step overfit on the 25 example clips
(val loss 0.337 -> 0.265).

## Launching the experiments

```
kalavinka/ops/build_sx_cache.sh $PROJECT/export/v1 $PROJECT/runs/cache_v1         # train/val/test .pt (mels via SoulX's encoder)
kalavinka/ops/launch_sx_runs.sh $PROJECT/runs/cache_v1 v1 16 R1 R2 R3 R4          # supervised, one GPU each, + live plots
kalavinka/ops/launch_ir_runs.sh $PROJECT/export/v1 v1                             # I1 (regression) and I2 (flow)
```
**Several machines.** `ops/jobq.py` is a job queue on a shared disk, so idle GPUs on any machine pick up
runs (`QUEUE=1 kalavinka/ops/launch_sx_runs.sh ...`); see [`ops/README.md`](ops/README.md).

Each run lives in `runs/<R>_<tag>/` (`train_<ts>.log`, `supervise_<ts>.log`,
`metrics.jsonl`, `best.pt`, `last.pt`, `live.png`, `DONE`); the val overlay is
`runs/monitoring/val_overlay.png`.

## Evaluation

```
.venv-sx/bin/python -m kalavinka.eval.render --export $PROJECT/export/v1 --run {zero-shot|$PROJECT/runs/R4_v1} \
    --out $PROJECT/runs/eval_v1/<name> --control {score|melody|melody-ir} [--f0-json runs/I2_v1/test_f0.json]
.venv-p1/bin/python -m kalavinka.eval.score --render $PROJECT/runs/eval_v1/<name> --export $PROJECT/export/v1
```
`score` reports loose-phonetic CER (Whisper-large-v3) for the render and the
real clip, RMVPE F0 error against the given and the real contour (cents,
ornament frames, voicing F1) and WavLM speaker similarity to the prompt.

| Module | What |
|---|---|
| `eval/contrast_eval.py` | minimal-pair pronunciation scores (retroflex vs dental, aspirated vs plain) with a character CTC model and forced alignment |
| `eval/compare.py` | paired bootstrap between systems for any per-clip metric (`cer@1`, `place_retro_sum/place_retro_n`, ...); resamples whole **songs** (clips of one song are not independent), two-sided p and Holm over a table's rows |
| `eval/recheck_results.py` | re-runs every comparison table of a RESULTS.md with the song bootstrap + Holm and lists the verdicts that changed (`runs/eval_v1/RESULTS_song_bootstrap.md`) |
| `eval/listening/` | blind naturalness test: `build.py` (page + stimuli + private key), `page.html`, `analyze.py` (two-way bootstrap, screening, comments) |
| `eval/demo_song.py` | one held-out song's clips stitched in order, one track per system plus the real singer |
| `eval/sing_new.py` | new lyrics (checked absent from training) via the dataset G2P, set to a held-out line's tune, sung by a run |
| `eval/demo_song_timed.py` | demo tracks stitched with the song's real gaps (0 s = crossfade) instead of a fixed 0.6 s, so words split across clips are not broken by a pause |
| `eval/demo_song_seamless.py` | a demo song re-rendered as one continuous performance (`deploy.seamless`: one timeline, real gaps, cuts only at rests or inside held notes), per system |
| `eval/long_passage.py` | long-singing test: consecutive test clips joined into 18-28 s passages, rendered clip by clip, in 14 s seamless pieces and in one call; each clip scored against the real singer (does a short-clip model struggle on long lyrics?) |
| `ir/reeval.py` | re-scores a contour run on test with the current evaluation (fair flat-MIDI baselines; sampler time label) -> `test_metrics_v2.json` |
| `ops/shadow_work.py` | a shadow copy of a P1 work dir (links for big files, copies for files stages rewrite) so text stages can be re-run without touching the dataset behind reported results (used for the G2P fix: `work_g2p2` -> `export/v2`) |
| `annot/demo_transcribe.py` | demo-song compositions: `analyse` (lead-vocal separation, F0 ensemble, tonic, swara overview PDF, Whisper timestamps) on any song outside the dataset; `transcribe` a passage into Compose JSON; `compare` a hand-written composition with the real singing note by note and write a fixed copy |
| `eval/alankar_eval.py` | alankar benchmark: base SoulX vs V4-L on sargam exercises (note-by-note pitch, Whisper syllable CER, paired bootstrap) |
| `eval/phone_eval.py` | IPA phone recogniser check (failed its gate; kept as a record) |
| `ops/eval_queue.sh` | per-GPU queue: wait for a run, render, contrast, score |

```
.venv-p1/bin/python -m kalavinka.eval.demo_song --export $PROJECT/export/v1 --song <song id> --out $PROJECT/demo \
    --sys score=$PROJECT/runs/eval_v1/L4_score --sys twostage_i2=$PROJECT/runs/eval_v1/L4_irI2t085
```

## Sharing the model (Colab / Mac)

```
kalavinka/ops/pack_bundles.sh        # -> <PROJECT>/bundles/kalavinka_colab.zip, kalavinka_mac.zip (~230 MB each)
```
See `deploy/README.md`. The zips hold our weights, 17 held-out tunes and 24 voices; SoulX's base `model.pt` is
downloaded from Hugging Face on first use.

## Tests

```
.venv-p1/bin/python -m pytest -q kalavinka/tests tests --ignore=kalavinka/tests/test_sx_ablation.py   # annotation + saregama
.venv-sx/bin/python -m pytest -q kalavinka/tests --ignore=kalavinka/tests/test_calibrate.py           # SoulX parts (py3.10)
```
`test_sx_ablation.py` imports SoulX-Singer (only on `.venv-sx`'s path); `test_calibrate.py` imports `saregama`, which
needs Python >= 3.11. `test_repo_docs.py` checks that the READMEs name only existing paths, modules and links.
