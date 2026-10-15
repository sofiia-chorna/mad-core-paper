#!/bin/bash

set -uo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
ING=${ING:-$MAD_CORE/mc_archive_survey/tierC_ingest.tsv}
BASE=${BASE:-$MAD_CORE/work/mcloud_tierc}
EX=$BASE/extract; RAW=$BASE/raw
mkdir -p "$EX" "$RAW"
PAT=( '*.db' '*.xyz' '*.extxyz' '*.cif' '*.traj' '*POSCAR*' '*CONTCAR*' '*XDATCAR*' )

for rid in $(cut -f1 "$ING" | sort -u); do
    dst="$EX/$rid"
    [ -f "$dst/.done" ] && { echo "skip $rid (done)"; continue; }
    mkdir -p "$dst" "$RAW/$rid"

    grep -P "^${rid}\t" "$ING" | while IFS=$'\t' read -r r sz url key; do
        [ -f "$RAW/$rid/$key" ] || curl -sL --max-time 1800 "$url" -o "$RAW/$rid/$key"
    done

    for f in "$RAW/$rid"/*; do
        lc=${f,,}
        case "$lc" in
            *.zip)
                unzip -qoC "$f" "${PAT[@]}" -d "$dst" 2>/dev/null ;;
            *.tar.gz|*.tgz|*.tar.bz2|*.tar.xz|*.tar)
                tar -xaf "$f" -C "$dst" --wildcards --no-anchored "${PAT[@]}" 2>/dev/null ;;
            *.gz|*.bz2|*.xz)
                tar -xaf "$f" -C "$dst" --wildcards --no-anchored "${PAT[@]}" 2>/dev/null || {
                    out="$dst/$(basename "${f%.*}")"
                    case "$lc" in
                        *.gz)  gunzip  -c "$f" > "$out" 2>/dev/null ;;
                        *.bz2) bunzip2 -c "$f" > "$out" 2>/dev/null ;;
                        *.xz)  unxz    -c "$f" > "$out" 2>/dev/null ;;
                    esac ; } ;;
            *.7z)
                echo "NO7Z (skipped, no extractor): $rid $(basename "$f")" ;;
        esac
    done

    rm -rf "$RAW/$rid"
    touch "$dst/.done"
    echo "done $rid: $(find "$dst" -type f 2>/dev/null | grep -v '/.done' | wc -l) struct files"
done
echo "DOWNLOAD_EXTRACT_DONE"
