#!/usr/bin/env bash
# Build the SoulX training caches (train/val/test .pt) from an export dir (kalavinka.sx.data.build_cache).
# Usage: kalavinka/ops/build_sx_cache.sh <export dir> <cache dir>
set -euo pipefail
EXP=$1; OUT=$2
REPO=$(cd "$(dirname "$0")/../.." && pwd); PROJECT=$(dirname "$REPO")
mkdir -p "$OUT"
for s in train val test; do
  "$PROJECT/.venv-sx/bin/python" - "$EXP" "$OUT" "$s" "$PROJECT/export/soulx/phone_set_indic.json" <<'PY' &
import json, sys
from pathlib import Path
from kalavinka.sx.data import build_cache
exp, out, split, table = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
r = build_cache(exp / f"{split}.jsonl", exp, table, out / f"{split}.pt")
print(split, json.dumps(r))
PY
done
wait
