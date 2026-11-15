#!/bin/bash
# Login-node downloader for P1 (curl only, no python): reads a TSV of
#   <output path relative to P1/work/raw>\t<url>
# written by a `p1-fetchlist` job, and saves each URL that isn't already there.
# Resumable (existing non-empty files are skipped), 1 request/s, generic user agent.
#
#   ops/fetch.sh work/raw/fetch_title.tsv
set -uo pipefail
P1=/home/fmea/23b1529/projects/P1
list=$1
raw="$P1/work/raw"
ok=0; skip=0; fail=0
while IFS=$'\t' read -r rel url; do
    [ -z "$rel" ] && continue
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
echo "fetched=$ok skipped=$skip failed=$fail"
