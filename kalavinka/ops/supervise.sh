#!/usr/bin/env bash
# Auto-resume supervisor for one training run: re-runs the command until <out>/DONE exists.
# The trainer resumes from <out>/last.pt.  Crash-loop guard: gives up after MAX_RESTARTS failures
# that each lasted under 5 minutes.  All output goes to <out>/supervise_<ts>.log.
# Usage: kalavinka/ops/supervise.sh <out dir> <command...>
OUT=$1; shift
MAX_RESTARTS=${MAX_RESTARTS:-5}
mkdir -p "$OUT"; LOG="$OUT/supervise_$(date +%Y%m%d-%H%M%S).log"
fails=0
while [ ! -f "$OUT/DONE" ]; do
  t0=$(date +%s)
  echo "[$(date)] start: $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ -f "$OUT/DONE" ] && break
  dt=$(( $(date +%s) - t0 ))
  echo "[$(date)] exited rc=$rc after ${dt}s" | tee -a "$LOG"
  if [ "$dt" -lt 300 ]; then fails=$((fails+1)); else fails=0; fi
  if [ "$fails" -ge "$MAX_RESTARTS" ]; then echo "[$(date)] crash loop: giving up" | tee -a "$LOG"; exit 1; fi
  sleep 30
done
echo "[$(date)] DONE" | tee -a "$LOG"
