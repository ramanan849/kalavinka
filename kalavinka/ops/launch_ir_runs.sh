#!/usr/bin/env bash
# Launch the stage-1 contour IR runs (I1 regression, I2 flow) on one GPU with enough free memory
# (both are ~15M params and share the card), each under the auto-resume supervisor.
# Usage: kalavinka/ops/launch_ir_runs.sh <export dir> <tag> [MIN_FREE_GB]
set -euo pipefail
EXP=$1; TAG=$2; MIN=${3:-8}
REPO=$(cd "$(dirname "$0")/../.." && pwd); PROJECT=$(dirname "$REPO")
PY="$PROJECT/.venv-sx/bin/python"
G=$("$PY" "$REPO/kalavinka/ops/gpu_pick.py" --min-free-gb "$MIN" --max 1)
[ -z "$G" ] && { echo "no GPU with >= $MIN GiB free"; exit 1; }
for spec in "I1:regress" "I2:flow"; do
  r=${spec%%:*}; mode=${spec##*:}; out="$PROJECT/runs/${r}_${TAG}"
  [ -e "$out" ] && { echo "$out exists"; exit 1; }
  tmux new-window -d -t kalavinka -n "${r}_${TAG}" "cd '$REPO' && CUDA_VISIBLE_DEVICES=$G bash kalavinka/ops/supervise.sh '$out' \
'$PY' -m kalavinka.ir.train --export '$EXP' --out '$out' --mode $mode; exec bash"
  echo "${r}_${TAG} ($mode) on GPU $G"
done
