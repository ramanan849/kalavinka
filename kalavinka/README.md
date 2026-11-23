# kalavinka

Hindi film-song singing voice synthesis on SoulX-Singer, built on the `saregama`
annotation package (which stays unmodified; kalavinka wraps it).

| Part | What | Status |
|---|---|---|
| `annot/` | P1 finishing: gold-set threshold calibration (C3), v1 export with per-frame confidence | calibration done; export in progress |
| `sx/` | stage 2: SoulX fine-tune -- Indic phone table + `IndicDataProcessor`, articulatory phone embeddings (B), confidence-aware flow matching (C), trainer | trainer smoke-tested |
| `ir/` | stage 1: expressive-contour IR model (note scaffold -> expressive F0) | planned |
| `eval/` | held-out rendering and metrics | planned |
| `ops/` | env setup, local (no-Slurm) P1 runner, GPU picker, web fetcher | done |

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

## Tests

```
.venv-p1/bin/python -m pytest -q kalavinka/tests     # annotation parts (SoulX tests skip)
.venv-sx/bin/python -m pytest -q kalavinka/tests/test_sx_adapter_phones.py
```
