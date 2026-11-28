#!/usr/bin/env bash
# Build Kalavinka's two Python environments from the uv lock files (exact versions of our reported runs).
#
#   annotate -> $PROJECT/.venv-p1   Python 3.12, PyTorch 2.8 cu128: annotation pipeline, ASR, scoring, tests
#   soulx    -> $PROJECT/.venv-sx   Python 3.10, PyTorch 2.2.0 cu121: SoulX-Singer training, rendering, app zips
#
# The two need different Python and PyTorch versions (SoulX-Singer pins transformers 4.41.2), so each has its own
# uv project in envs/<name>/. The soulx env does not install this repo (saregama needs Python >= 3.11); instead a
# .pth file puts the repo and the SoulX-Singer clone on its path.
#
# Usage:  envs/setup.sh [annotate|soulx|all]        (default: all)
# Switches (environment variables):
#   SAREGAMA_PROJECT   the project folder that holds the repo, SoulX-Singer/, dataset/, runs/ ...
#                      (default: the repo's parent folder)
#   UV                 the uv binary (default: uv on PATH)
set -euo pipefail
WHICH=${1:-all}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PROJECT=${SAREGAMA_PROJECT:-$(dirname "$REPO")}
UV=${UV:-uv}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
command -v "$UV" >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

sync_env() {  # sync_env <uv project> <venv path>: install exactly what the lock file says
  UV_PROJECT_ENVIRONMENT="$2" "$UV" sync --frozen --project "$REPO/envs/$1"
}

if [[ $WHICH == annotate || $WHICH == all ]]; then
  sync_env annotate "$PROJECT/.venv-p1"
fi

if [[ $WHICH == soulx || $WHICH == all ]]; then
  sync_env soulx "$PROJECT/.venv-sx"
  [[ -d "$PROJECT/SoulX-Singer" ]] || echo "note: $PROJECT/SoulX-Singer is missing; clone it (README step 1)" >&2
  SITE=$("$PROJECT/.venv-sx/bin/python" -c 'import site; print(site.getsitepackages()[0])')
  printf '%s\n%s\n' "$REPO" "$PROJECT/SoulX-Singer" > "$SITE/kalavinka_repo.pth"
fi
echo "envs/setup.sh $WHICH: done (project: $PROJECT)"
