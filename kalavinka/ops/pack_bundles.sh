#!/usr/bin/env bash
# Build the shareable zips (Colab notebook + Mac pipeline) into <PROJECT>/bundles/.
# Usage: kalavinka/ops/pack_bundles.sh [--n-tunes 20]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); PROJECT=$(dirname "$REPO")
cd "$REPO"
"$PROJECT/.venv-sx/bin/python" -m kalavinka.deploy.build_bundle --out "$PROJECT/bundles" "$@"
