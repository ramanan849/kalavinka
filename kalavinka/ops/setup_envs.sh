#!/usr/bin/env bash
# Build the two isolated uv venvs Kalavinka needs (run from anywhere).
#
#   .venv-p1  annotation pipeline (saregama P1): py3.12, torch 2.8 cu128,
#             transformers 4.57, qwen-asr 0.0.6 -- the env the P1 run used.
#   .venv-sx  SoulX-Singer training / inference: py3.10, torch 2.2.0 cu121,
#             transformers 4.41.2 -- upstream pins (DiffLlama subclasses
#             LlamaDecoderLayer, newer transformers changes its signature).
#
# Usage: kalavinka/ops/setup_envs.sh [p1|sx|all]   (PROJECT defaults to the repo's parent)
set -euo pipefail
WHICH=${1:-all}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PROJECT=${SAREGAMA_PROJECT:-$(dirname "$REPO")}
UV=${UV:-$HOME/bin/uv}
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300

if [[ $WHICH == p1 || $WHICH == all ]]; then
  "$UV" venv --python 3.12 "$PROJECT/.venv-p1"
  PY="$PROJECT/.venv-p1/bin/python"
  "$UV" pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu128 \
      "torch==2.8.*" "torchaudio==2.8.*"
  "$UV" pip install --python "$PY" -e "$REPO[test]"
  "$UV" pip install --python "$PY" "transformers==4.57.*" "qwen-asr==0.0.6" speechbrain cmudict \
      torchfcpe torchcrepe librosa einops huggingface_hub ml_collections accelerate omegaconf \
      rotary_embedding_torch beartype loralib "numpy<2.3" soundfile pyloudnorm \
      indic_transliteration rapidfuzz wandb tqdm matplotlib pyyaml
fi

if [[ $WHICH == sx || $WHICH == all ]]; then
  "$UV" venv --python 3.10 "$PROJECT/.venv-sx"
  PY="$PROJECT/.venv-sx/bin/python"
  "$UV" pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu121 \
      "torch==2.2.0" "torchaudio==2.2.0"
  # upstream requirements minus the heavy UI / preprocess-only extras we do not use for training
  grep -vE '^(torch|torchaudio|torchcodec|gradio|nemo_toolkit|funasr|sageattention|wandb)==' \
      "$PROJECT/SoulX-Singer/requirements.txt" > "$PROJECT/logs/sx_requirements.txt"
  "$UV" pip install --python "$PY" -r "$PROJECT/logs/sx_requirements.txt"
  "$UV" pip install --python "$PY" "peft==0.11.*" pytest pandas pyarrow
fi
echo "setup_envs.sh $WHICH: done"
