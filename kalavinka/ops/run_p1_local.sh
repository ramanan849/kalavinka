#!/usr/bin/env bash
# Run the P1 annotation pipeline on this machine (local GPUs, no Slurm).
#
# Each step is one command; long ones are started in their own window of the
# tmux session `kalavinka`, with a timestamped log in <PROJECT>/logs/p1/.
# Steps (in order):
#   init        p1-init: songs.tsv from the dataset folder (ids are stable)
#   clean N     inventory + fingerprint with N CPU workers, then p1-dedupe
#   fetch       p1-fetchlist --kind title, then fetch_local.sh (LRCLIB + oEmbed)
#   gpu MINGB   one GPU worker per card with >= MINGB GiB free (gpu_pick.py)
#   text N      p1-refpool, then N CPU workers over the text stages
#   retry       p1-fetchlist --kind retry + fetch + p1-refpool (second lyrics lookup)
#   text2 N     text stages again with --force (cheap), so identify sees the bigger pool
#   status      p1-status + p1-report
# Stages are resumable (stages.json + claims), so any step can be re-run safely.
#
# Usage: kalavinka/ops/run_p1_local.sh <step> [arg]
set -euo pipefail
STEP=$1; ARG=${2:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PROJECT=${SAREGAMA_PROJECT:-$(dirname "$REPO")}
PY="$PROJECT/.venv-p1/bin/python"
CFG="$PROJECT/work/p1.toml"
WORK="$PROJECT/work"
LOGS="$PROJECT/logs/p1"; mkdir -p "$LOGS"
TS=$(date +%Y%m%d-%H%M%S)
export PATH="$HOME/bin:$PATH" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SG=( "$PY" -m saregama )
cd "$REPO"

win() {   # win <name> <command...>: run in a new tmux window, log to $LOGS/<name>_<ts>.log
    local name=$1; shift
    local log="$LOGS/${name}_$TS.log"
    tmux has-session -t kalavinka 2>/dev/null || tmux new-session -d -s kalavinka -n home
    tmux new-window -d -t kalavinka -n "$name" \
        "cd '$REPO' && export PATH='$HOME/bin':\$PATH PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; $* 2>&1 | tee '$log'; exec bash"
    echo "started tmux kalavinka:$name -> $log"
}

case $STEP in
  init)   "${SG[@]}" p1-init --p1-config "$CFG" ;;
  clean)  N=${ARG:-16}
          for k in $(seq 1 "$N"); do
              win "clean$k" "${SG[*]} p1-run --p1-config '$CFG' --stages clean --budget-min 100000 --include-duplicates"
          done ;;
  dedupe) "${SG[@]}" p1-dedupe --p1-config "$CFG" ;;
  fetch)  "${SG[@]}" p1-fetchlist --p1-config "$CFG" --kind title
          win fetch "bash '$REPO/kalavinka/ops/fetch_local.sh' '$WORK' '$WORK/raw/fetch_title.tsv'" ;;
  gpu)    MIN=${ARG:-20}
          GPUS=$("$PY" "$REPO/kalavinka/ops/gpu_pick.py" --min-free-gb "$MIN")
          [ -z "$GPUS" ] && { echo "no GPU with >= ${MIN} GiB free"; exit 1; }
          for g in ${GPUS//,/ }; do
              win "gpu$g" "CUDA_VISIBLE_DEVICES=$g ${SG[*]} p1-run --p1-config '$CFG' --stages gpu --budget-min 100000"
          done ;;
  text)   "${SG[@]}" p1-refpool --p1-config "$CFG"
          N=${ARG:-24}
          for k in $(seq 1 "$N"); do
              win "text$k" "CUDA_VISIBLE_DEVICES= ${SG[*]} p1-run --p1-config '$CFG' --stages text --budget-min 100000"
          done ;;
  retry)  "${SG[@]}" p1-fetchlist --p1-config "$CFG" --kind retry
          bash "$REPO/kalavinka/ops/fetch_local.sh" "$WORK" "$WORK/raw/fetch_retry.tsv" 2>&1 | tee "$LOGS/retry_$TS.log"
          "${SG[@]}" p1-refpool --p1-config "$CFG" ;;
  text2)  N=${ARG:-24}
          "${SG[@]}" p1-run --p1-config "$CFG" --stages text --force --max-songs 0 --budget-min 1 >/dev/null 2>&1 || true
          for k in $(seq 1 "$N"); do
              win "text2_$k" "CUDA_VISIBLE_DEVICES= ${SG[*]} p1-run --p1-config '$CFG' --stages text --budget-min 100000"
          done ;;
  status) "${SG[@]}" p1-status --p1-config "$CFG"; "${SG[@]}" p1-report --p1-config "$CFG" ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
