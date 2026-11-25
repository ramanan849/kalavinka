#!/usr/bin/env bash
# One-time setup of Kalavinka on a Mac (Apple silicon or Intel). Safe to run again.
#   1. finds Python 3.10-3.12 (or uses uv if you have it),
#   2. makes a private environment in ./.venv and installs the packages (PyTorch with Mac-GPU support),
#   3. downloads SoulX-Singer's base weights (2.7 GB, once),
#   4. sings one short test line.
# Afterwards:  ./sing ui      (web page)      or      ./sing "ठंडी हवा में गीत नया"
set -euo pipefail
cd "$(dirname "$0")"
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "1/4  Looking for Python 3.10, 3.11 or 3.12"
if command -v uv >/dev/null 2>&1; then
  [ -x .venv/bin/python ] || uv venv -q -p 3.11 .venv
  PIP="uv pip install -q --python .venv/bin/python"
else
  PY=""
  for c in python3.11 python3.12 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] <= (3,12)))'; then
      PY="$c"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "No Python 3.10-3.12 found. Install one of these, then run this script again:"
    echo "   brew install python@3.11            (Homebrew: https://brew.sh)"
    echo "   or  curl -LsSf https://astral.sh/uv/install.sh | sh     (uv; it brings its own Python)"
    exit 1
  fi
  echo "using $($PY --version)"
  [ -x .venv/bin/python ] || "$PY" -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  PIP=".venv/bin/python -m pip install -q"
fi

say "2/4  Installing packages into ./.venv (a few minutes the first time)"
$PIP torch torchaudio
$PIP -r requirements.txt
.venv/bin/python - <<'PY'
import torch
mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
print(f"PyTorch {torch.__version__}; Mac GPU (MPS): {'yes' if mps else 'no, will use the CPU'}")
PY

say "3/4  Downloading SoulX-Singer base weights (2.7 GB, only once)"
.venv/bin/python -m kalavinka_app fetch

say "4/4  Test: singing one short line"
chmod +x sing
./sing --steps 8 sing "ठंडी हवा में गीत नया" --out outputs/setup_test.wav
command -v afplay >/dev/null 2>&1 && afplay outputs/setup_test.wav || true

say "All set."
echo "  ./sing ui                              open the web page (recommended)"
echo "  ./sing \"चाँद की रोशनी में तेरा नाम\"      sing a line (saved in outputs/ and played)"
echo "  ./sing help                            all commands"
