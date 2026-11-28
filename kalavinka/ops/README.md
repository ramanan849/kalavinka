# kalavinka.ops: run scripts

Every script finds the project folder as the repo's parent (`PROJECT`) and uses `$PROJECT/.venv-p1` or
`$PROJECT/.venv-sx` (built by [`envs/setup.sh`](../../envs/README.md)). Long jobs run in windows of the tmux session
`kalavinka`, with timestamped logs. The order in which to use them is in the [main README](../../README.md#3-replicate-the-project-step-by-step).

| Script | What | Used in step |
|---|---|---|
| `run_p1_local.sh <step> [N]` | the annotation pipeline on local GPUs (init, clean, dedupe, fetch, gpu, text, retry, text2, status) | 4 |
| `p1_run.py` | the `saregama` CLI plus a stage-marker status fix; what `run_p1_local.sh` calls | 4 |
| `fetch_local.sh` | the lyrics and metadata fetcher (LRCLIB + YouTube oEmbed, 1 request/s per host) | 4 |
| `build_sx_cache.sh <export> <cache>` | SoulX training caches (train/val/test `.pt`, mels from SoulX's encoder) | 7 |
| `launch_sx_runs.sh <cache> <tag> [GB] runs...` | fine-tuning runs (R1 to R5, L1/L4, H, N, S; arguments in the script), one GPU each | 8 |
| `launch_ir_runs.sh <export> <tag>` | the contour models I1 (C-reg) and I2 (C-flow) | 9 |
| `supervise.sh` | restarts a run from `last.pt` if it crashes; used by both launchers | 8, 9 |
| `eval_queue.sh` | per-GPU queue: wait for a run, render, contrast, score | 10 |
| `pack_bundles.sh` | builds the Colab and Mac zips into `$PROJECT/bundles/` | 11 |
| `gpu_pick.py --min-free-gb N` | lists the GPUs with at least N GB free (cards may be shared) | all |
| `plot_runs.py` | live training plots and the validation overlay | 8 |
| `shadow_work.py` | a shadow copy of a work dir, to re-run text stages without touching the reported dataset | (G2P fix) |
| `setup_envs.sh` | the older, loose-pinned way to build the two envs (kept for reference) | 2 |

## Several machines: the shared-disk job queue

If several machines mount the same project folder, `jobq.py` lets idle GPUs on any of them take runs. Jobs are files
in `$PROJECT/runs/jobq/`, so no SSH between machines is needed. Claiming a job is an atomic rename, so two workers
never take the same job.

```
# one worker per machine; --can says which env's jobs it may run (sx = .venv-sx only; p1 = also the scoring models)
python kalavinka/ops/jobq.py worker --node <machine name> --can sx,p1 --max-jobs 4 [--delay 300]
QUEUE=1 kalavinka/ops/launch_sx_runs.sh $PROJECT/runs/cache_v1 v2 16 L1b L4b      # runs go to the queue
python kalavinka/ops/jobq.py submit --name X --needs sx --after <path> -- <cmd using $JOB_GPU>
python kalavinka/ops/jobq.py status
```

`--delay S` makes a worker take only jobs that have waited S seconds, so the less busy machine gets first pick.
`setup_other_node.sh` is a one-time helper for a second machine whose container cannot see the uv-managed Python
that `.venv-sx` was built with. It links a copy kept on the shared disk.
