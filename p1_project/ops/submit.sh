#!/bin/bash
# Submit a P1 job following the Prajna rules (see ~/.claude/CLAUDE.md):
# always -p X --qos=X, only the resources used, short --time (backfill starts it sooner).
#
#   ops/submit.sh cpu  <minutes> <name> -- <saregama args...>   # debug partition, no GPU (<= 30 min)
#   ops/submit.sh gpu  <minutes> <name> -- <saregama args...>   # 1 GPU: tries l40, a40, dgx in turn
#   ops/submit.sh test <minutes> <name>                         # pytest in a CPU job
#   P1_DEP=afterany:<jobid> ops/submit.sh gpu ...               # start after another job (chains)
#
# Every submission is appended to ops/jobs.tsv (job id, partition, name, args).
set -uo pipefail
P1=/home/fmea/23b1529/projects/P1
kind=$1; mins=$2; name=$3; shift 3
[ "${1:-}" = "--" ] && shift
ledger="$P1/ops/jobs.tsv"
[ -f "$ledger" ] || printf 'submitted\tjob_id\tpartition\tname\tkind\tminutes\targs\n' > "$ledger"

submit() {   # submit <partition> <extra sbatch args...> -- <script args...>
    local part=$1; shift
    sbatch --parsable -p "$part" --qos="$part" -J "p1-$name" --time="$mins" ${P1_DEP:+--dependency=$P1_DEP} "$@" 2>&1
}

case "$kind" in
  cpu|test)   # debug first; its group limit is often full, then a40 / l40 without a GPU
        for part in debug a40 l40; do
            if [ "$kind" = cpu ]; then
                out=$(submit "$part" --cpus-per-task=4 --mem=16G "$P1/ops/run.sbatch" "$@")
            else
                out=$(submit "$part" --cpus-per-task=4 --mem=16G -o "$P1/ops/logs/%x_%j.out" --wrap \
                    "cd $P1/saregama624 && $P1/.venv/bin/python -m pytest -q ${*:-tests} 2>&1 | tail -25")
            fi
            [[ "$out" =~ ^[0-9]+$ ]] && break
            echo "  $part refused: ${out##*error: }" >&2
        done ;;
  gpu)  for part in ${P1_GPU_PARTS:-a40 l40 dgx}; do   # override: P1_GPU_PARTS="l40 a40"
            out=$(submit "$part" --gres=gpu:1 --cpus-per-task=8 --mem=48G "$P1/ops/run.sbatch" "$@")
            [[ "$out" =~ ^[0-9]+$ ]] && break
            echo "  $part refused: $out" >&2
        done ;;
  *) echo "kind must be cpu | gpu | test" >&2; exit 2 ;;
esac
if [[ "$out" =~ ^[0-9]+$ ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$out" "$part" "$name" "$kind" "$mins" "$*" >> "$ledger"
    echo "$out"
else
    echo "submit failed: $out" >&2; exit 1
fi
