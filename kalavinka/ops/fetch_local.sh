#!/usr/bin/env bash
# Local web fetcher for P1 (this machine has internet; no login-node split needed).
# Same contract as p1_project/ops/fetch.sh -- reads a TSV of
#   <path relative to <work>/raw>\t<url>
# written by `saregama p1-fetchlist`, saves each URL not already present
# (resumable: non-empty files are skipped), 1 request/s per host, generic user agent.
# The list is split by host (lrclib.net / youtube.com) and both halves run in parallel.
#
# Usage: kalavinka/ops/fetch_local.sh <work_dir> <list.tsv>
set -uo pipefail
work=$1; list=$2; raw="$work/raw"

fetch_host() {   # $1 = host substring; reads matching rows of $list
    local host=$1 ok=0 skip=0 fail=0 rel url out code
    while IFS=$'\t' read -r rel url; do
        [ -z "$rel" ] && continue
        [[ $url == *"$host"* ]] || continue
        out="$raw/$rel"
        if [ -s "$out" ]; then skip=$((skip+1)); continue; fi
        mkdir -p "$(dirname "$out")"
        code=$(curl -sS -L --max-time 30 -A "lyrics-research-client/0.1" -o "$out.part" -w "%{http_code}" "$url")
        if [ "$code" = "200" ] || [ "$code" = "404" ]; then
            [ "$code" = "404" ] && echo '{"_status":404}' > "$out.part"
            mv "$out.part" "$out"; ok=$((ok+1))
        else
            rm -f "$out.part"; fail=$((fail+1)); echo "HTTP $code $url" >&2
        fi
        sleep 1
    done < "$list"
    echo "[$host] fetched=$ok skipped=$skip failed=$fail"
}
fetch_host lrclib.net & fetch_host youtube.com & wait
