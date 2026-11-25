# kalavinka

Hindi film-song singing voice synthesis on SoulX-Singer, built on the `saregama`
annotation package (which stays unmodified; kalavinka wraps it).

| Part | What | Status |
|---|---|---|
| `annot/` | P1 finishing: gold-set threshold calibration (C3), v1 export with per-frame confidence | calibration done; export in progress |
| `sx/` | stage 2: SoulX fine-tune -- Indic phone table + `IndicDataProcessor`, articulatory phone embeddings (B), confidence-aware flow matching (C), trainer | trainer smoke-tested |
| `ir/` | stage 1: expressive-contour IR model (note scaffold -> expressive F0), temperature sweep | done |
| `eval/` | held-out rendering, CER/F0/SIM, minimal-pair contrasts, paired bootstrap, listening test, demo tracks | done |
| `ops/` | env setup, local (no-Slurm) P1 runner, GPU picker, web fetcher | done |
| `deploy/` | shareable inference bundles: Colab notebook zip and Mac pipeline zip (`KalavinkaSinger`, CLI, web page; base-vs-ours comparison, alankars/sargam, take history + cache) | done |

## Layout on disk

`PROJECT` = the repo's parent (here `/workspace/vis-enc/624`). Data never goes in git:

```
PROJECT/
  saregama624/            this repo
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
kalavinka/ops/setup_envs.sh all      # .venv-p1 (py3.12, torch 2.8, P1) and .venv-sx (py3.10, torch 2.2, SoulX pins)
kalavinka/ops/setup_other_node.sh    # once per extra node's container: links the shared copy of .venv-sx's Python, tmux session
```
`.venv-sx` gets the repo and the SoulX clone on its path via a `.pth` file
(`saregama` itself needs py>=3.11, so only its pure-python modules are used there):
```
echo $PROJECT/saregama624 >  $(.venv-sx/bin/python -c 'import site;print(site.getsitepackages()[0])')/kalavinka_repo.pth
echo $PROJECT/SoulX-Singer >> (same file)
```

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
.venv-p1/bin/python -m kalavinka.annot.calibrate --work $PROJECT/work --gold $PROJECT/gold
```
Scores every planned clip inside a gold window against the corrected TextGrids
and fits `min_mean_conf`, `min_good_src` for the most accepted time with pooled
word accuracy >= 0.95 and onsets within 50 ms >= 0.90. Writes
`export/calibration/{calibration.json, clips.tsv, report.md}`.

## SoulX adapter and fine-tuning

```
.venv-sx/bin/python -m kalavinka.sx.adapter --phones-out $PROJECT/export/soulx/phone_set_indic.json \
    --check $PROJECT/export/v1/train.jsonl          # real DataProcessor check
.venv-sx/bin/python -m kalavinka.sx.train --cache <cache dir> --phones-table <phone_set_indic.json> \
    --out $PROJECT/runs/<run> --phones {nearest|articulatory} --bands accept[,review] --weighting {none|conf}
```
Runs (each <= 5k steps, early stopping at patience 5) -- filled in as they are launched:

| Run | Phones | Data / loss | Command |
|---|---|---|---|
| R0 | zero-shot, Hindi phones remapped to nearest English tokens | - | eval only |
| R1 | nearest-copy | accept, uniform | TBD |
| R2 | articulatory | accept, uniform | TBD |
| R3 | nearest-copy | accept+review, confidence-weighted | TBD |
| R4 | articulatory | accept+review, confidence-weighted | TBD |

Smoke checks done: 30-step run and a 300-step overfit on the 25 example clips
(val loss 0.337 -> 0.265).

## Launching the experiments

```
kalavinka/ops/build_sx_cache.sh $PROJECT/export/v1 $PROJECT/runs/cache_v1         # train/val/test .pt (mels via SoulX's encoder)
kalavinka/ops/launch_sx_runs.sh $PROJECT/runs/cache_v1 v1 16 R1 R2 R3 R4          # supervised, one GPU each, + live plots
kalavinka/ops/launch_ir_runs.sh $PROJECT/export/v1 v1                             # I1 (regression) and I2 (flow)
```
**Two nodes (shared `/workspace`).** `ops/jobq.py` is a job queue on the shared disk: one worker per
node takes the next job when that node has a GPU with room, so idle GPUs on either node get used.
The second node (gokul-bench) is set up once with `ops/setup_other_node.sh`; it runs `sx` jobs (training,
rendering), while `p1` jobs (Whisper / contrast scoring) need the dev container's model caches.

```
# workers (gokul-bench first pick; gokul-dev takes jobs that waited 5 min)
setsid nohup .venv-sx/bin/python kalavinka/ops/jobq.py worker --node gokul-dev --can sx,p1 --max-jobs 4 --delay 300 >> runs/jobq/worker_gokul-dev.log 2>&1 &
ssh -i ~/.ssh/id_ed25519_bench gokula@10.20.227.19 "docker exec -d gokul-bench bash -lc 'cd $PROJECT/saregama624 && exec $PROJECT/.venv-sx/bin/python kalavinka/ops/jobq.py worker --node gokul-bench --can sx >> $PROJECT/runs/jobq/worker_gokul-bench.log 2>&1'"
QUEUE=1 kalavinka/ops/launch_sx_runs.sh $PROJECT/runs/cache_v1 v2 16 L1b L4b             # runs go to the queue
.venv-sx/bin/python kalavinka/ops/jobq.py submit --name X --needs sx --after <path> -- <cmd using $JOB_GPU>
.venv-sx/bin/python kalavinka/ops/jobq.py status
```

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
| `eval/compare.py` | paired bootstrap between systems for any per-clip metric (`cer@1`, `place_retro_sum/place_retro_n`, ...) |
| `eval/listening/` | blind naturalness test: `build.py` (page + stimuli + private key), `page.html`, `analyze.py` (two-way bootstrap, screening, comments) |
| `eval/demo_song.py` | one held-out song's clips stitched in order, one track per system plus the real singer |
| `eval/sing_new.py` | new lyrics (checked absent from training) via the dataset G2P, set to a held-out line's tune, sung by a run |
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
See `deploy/README.md`. The zips hold our weights, 17 held-out tunes and 10 voices; SoulX's base `model.pt` is
downloaded from Hugging Face on first use.

## Tests

```
.venv-p1/bin/python -m pytest -q kalavinka/tests     # annotation parts (SoulX tests skip)
.venv-sx/bin/python -m pytest -q kalavinka/tests/test_sx_adapter_phones.py
```
