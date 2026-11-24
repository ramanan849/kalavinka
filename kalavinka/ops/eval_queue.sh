#!/usr/bin/env bash
# Run a list of evaluation jobs one after another on one GPU, each waiting for what it needs.
#
# Why: the round-2 runs finish at different times and Whisper needs ~14 GB, so every GPU gets its
# own sequential queue (never two scorers on one card), and every job waits for its run's DONE file
# (or for a render it scores) instead of being started by hand.
#
# Usage: kalavinka/ops/eval_queue.sh <gpu> <job> [<job> ...]
#   jobs (EXPORT = <PROJECT>/export/v1, EVAL = <PROJECT>/runs/eval_v1):
#     render:<run>:<control>:<name>[:<split>[:<extra args>]]   wait <run>/DONE, render into EVAL/<name>
#     contrast:<name>[:<split>]                                minimal-pair scores of EVAL/<name>
#     contrast-real:<name>:<out>[:<split>]                      the real clips of EVAL/<name>'s list
#     score:<name>                                             Whisper CER / F0 / SIM of EVAL/<name>
#     wait:<path>                                              wait until <path> exists
#     waitdone:<name>                                          wait until EVAL/<name>.log says "rendered"
#   <run> is a run dir name under runs/ or zero-shot; <extra> uses "+" between arguments.
set -uo pipefail
G=$1; shift
REPO=$(cd "$(dirname "$0")/../.." && pwd); PROJECT=$(dirname "$REPO")
EXPORT=$PROJECT/export/v1; EVAL=$PROJECT/runs/eval_v1; SX=$PROJECT/.venv-sx/bin/python; P1=$PROJECT/.venv-p1/bin/python
cd "$REPO"
export CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date -u +%H:%M:%S)] gpu$G $*"; }
for job in "$@"; do
  IFS=':' read -r kind a b c d e <<<"$job"
  case $kind in
    wait)     log "wait $a"; until [ -e "$a" ]; do sleep 30; done ;;
    waitdone) log "waitdone $a"; until grep -q '^rendered' "$EVAL/$a.log" 2>/dev/null; do sleep 30; done ;;
    render)   run=$a; ctl=$b; name=$c; split=${d:-test}; extra=${e:-}
              [ "$run" != zero-shot ] && { log "wait $run/DONE"; until [ -f "$PROJECT/runs/$run/DONE" ]; do sleep 30; done; run=$PROJECT/runs/$run; }
              [ -f "$EVAL/$name/render.jsonl" ] && grep -q '^rendered' "$EVAL/$name.log" 2>/dev/null && { log "skip render $name (done)"; continue; }
              log "render $name"
              $SX -m kalavinka.eval.render --export "$EXPORT" --split "$split" --run "$run" --out "$EVAL/$name" \
                  --control "$ctl" ${extra//+/ } > "$EVAL/$name.log" 2>&1 || log "render $name FAILED" ;;
    contrast) log "contrast $a"; $P1 -m kalavinka.eval.contrast_eval --render "$EVAL/$a" --export "$EXPORT" --split "${b:-test}" \
                  > "$EVAL/contrast_$a.log" 2>&1 || log "contrast $a FAILED" ;;
    contrast-real) log "contrast-real $a -> $b"; $P1 -m kalavinka.eval.contrast_eval --real --render "$EVAL/$a" --export "$EXPORT" \
                  --split "${c:-test}" --out "$EVAL/$b" > "$EVAL/contrast_$b.log" 2>&1 || log "contrast-real $b FAILED" ;;
    score)    log "score $a"; $P1 -m kalavinka.eval.score --render "$EVAL/$a" --export "$EXPORT" > "$EVAL/score_$a.log" 2>&1 \
                  || log "score $a FAILED" ;;
    *)        log "unknown job $job" ;;
  esac
done
log "queue done"
