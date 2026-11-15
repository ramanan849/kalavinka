#!/bin/bash
# Generic P1 job: runs `python -m saregama <args>` from the P1 venv.
# Submit only through ops/submit.sh (it sets partition, QoS, time and resources).
#SBATCH --output=/home/fmea/23b1529/projects/P1/ops/logs/%x_%j.out
#SBATCH --error=/home/fmea/23b1529/projects/P1/ops/logs/%x_%j.err
set -euo pipefail
P1=/home/fmea/23b1529/projects/P1
export PATH="$HOME/bin:$PATH"                       # ffmpeg / ffprobe / fpcalc
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1   # compute nodes are offline
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAREGAMA_PROJECT="$P1" PYTHONUNBUFFERED=1
echo "job $SLURM_JOB_ID on $(hostname) partition=$SLURM_JOB_PARTITION start=$(date -Is)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$P1/saregama624"
"$P1/.venv/bin/python" -m saregama "$@"
echo "end=$(date -Is)"
