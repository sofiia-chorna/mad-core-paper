#!/bin/bash
# copies every file of archive_files.tsv into results_archive and models_archive
root=$(cd "$(dirname "$0")/.." && pwd)

tail -n +2 $root/provenance/archive_files.tsv | while IFS=$'\t' read -r dst src rest; do
    mkdir -p $root/$(dirname $dst)
    cp -p $src $root/$dst
done
