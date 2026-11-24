#!/usr/bin/env bash
# One-time setup so another node's container (e.g. gokul-bench) can run the project's venvs from the shared disk.
#
# Why: .venv-sx/bin/python is a symlink into /root/.local/share/uv/python/cpython-3.10.20-..., which exists only
# in the container that created the venv.  A copy of that interpreter lives on the shared disk at
# <PROJECT>/.uv-python; this links it into the same path here, so the venv (and every launcher) works unchanged.
# It also creates the tmux session `kalavinka` the launchers add windows to, and checks torch sees the GPUs.
#
# Usage (on the other node, inside its container):  bash kalavinka/ops/setup_other_node.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); PROJECT=$(dirname "$REPO")
SRC=$PROJECT/.uv-python/cpython-3.10.20-linux-x86_64-gnu
DST=/root/.local/share/uv/python
[ -x "$SRC/bin/python3.10" ] || { echo "missing $SRC (copy it from the dev container first)"; exit 1; }
mkdir -p "$DST"
[ -e "$DST/cpython-3.10.20-linux-x86_64-gnu" ] || ln -s "$SRC" "$DST/cpython-3.10.20-linux-x86_64-gnu"
[ -e "$DST/cpython-3.10-linux-x86_64-gnu" ] || ln -s "$DST/cpython-3.10.20-linux-x86_64-gnu" "$DST/cpython-3.10-linux-x86_64-gnu"
tmux has-session -t kalavinka 2>/dev/null || tmux new-session -d -s kalavinka -n home
"$PROJECT/.venv-sx/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
"$PROJECT/.venv-sx/bin/python" "$REPO/kalavinka/ops/gpu_pick.py" --min-free-gb 16 | sed 's/^/GPUs with >= 16 GiB free: /'
echo "ready: e.g. bash kalavinka/ops/launch_sx_runs.sh $PROJECT/runs/cache_v1 v3 16 H1b H2b"
