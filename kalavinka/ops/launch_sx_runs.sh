#!/usr/bin/env bash
# Launch the stage-2 SoulX fine-tuning runs (R1-R4, optional R5) in tmux session `kalavinka`,
# each under the auto-resume supervisor, on GPUs with enough free memory (gpu_pick.py),
# plus a live-plot window.  Every run writes to its own fresh dir under <PROJECT>/runs/.
#
# Usage: kalavinka/ops/launch_sx_runs.sh <cache dir> <tag> [MIN_FREE_GB] [runs...]
#   e.g. kalavinka/ops/launch_sx_runs.sh $PROJECT/runs/cache_v1 v1 16 R1 R2 R3 R4
set -euo pipefail
CACHE=$1; TAG=$2; MIN=${3:-16}; shift 3 || true
RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=(R1 R2 R3 R4)
REPO=$(cd "$(dirname "$0")/../.." && pwd); PROJECT=$(dirname "$REPO")
PY="$PROJECT/.venv-sx/bin/python"
TABLE="$PROJECT/export/soulx/phone_set_indic.json"
declare -A ARGS=(
  [R1]="--phones nearest --bands accept --weighting none"
  [R2]="--phones articulatory --bands accept --weighting none"
  [R3]="--phones nearest --bands accept,review --weighting conf"
  [R4]="--phones articulatory --bands accept,review --weighting conf"
  [R5]="--phones articulatory --bands accept,review --weighting none"
  # second round (24 Nov): longer headline runs and the targeted tests of (B) and (C)
  [L1]="--phones nearest --bands accept --weighting none --steps 15000 --max-frames 7200 --val-every 500 --patience 8"
  [L4]="--phones articulatory --bands accept,review --weighting conf --steps 15000 --max-frames 7200 --val-every 500 --patience 8"
  [H1]="--phones nearest --bands accept --weighting none --exclude-phones Th,Dh,Rh"
  [H2]="--phones articulatory --bands accept --weighting none --exclude-phones Th,Dh,Rh"
  [N1]="--phones articulatory --bands accept,review,reject --weighting none"     # needs the v1_all cache
  [N2]="--phones articulatory --bands accept,review,reject --weighting conf"     # needs the v1_all cache
  [S1]="--phones nearest --bands accept --weighting none --subset-hours 1"
  [S2]="--phones articulatory --bands accept --weighting none --subset-hours 1"
)
GPUS=($("$PY" "$REPO/kalavinka/ops/gpu_pick.py" --min-free-gb "$MIN" | tr ',' ' '))
[ ${#GPUS[@]} -lt ${#RUNS[@]} ] && { echo "only ${#GPUS[@]} GPUs with >= ${MIN} GiB free for ${#RUNS[@]} runs"; exit 1; }
names=()
for k in "${!RUNS[@]}"; do
  r=${RUNS[$k]}; g=${GPUS[$k]}; name="${r}_${TAG}"; out="$PROJECT/runs/$name"; names+=("$name")
  [ -e "$out" ] && { echo "$out exists -- refusing to reuse a run dir"; exit 1; }
  cmd="cd '$REPO' && CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash kalavinka/ops/supervise.sh '$out' \
'$PY' -m kalavinka.sx.train --cache '$CACHE' --phones-table '$TABLE' --out '$out' --steps 5000 --patience 5 ${ARGS[$r]}; exec bash"   # run args last: they override the defaults
  tmux new-window -d -t kalavinka -n "$name" "$cmd"
  echo "$name on GPU $g: ${ARGS[$r]}"
done
tmux new-window -d -t kalavinka -n "plot_$TAG" "cd '$REPO' && '$PY' kalavinka/ops/plot_runs.py --runs '$PROJECT/runs' ${names[*]} --every 300; exec bash"
