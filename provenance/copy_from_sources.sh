#!/bin/bash
# copies every file of extracted_files.tsv from its original path into the repository
root=$(cd "$(dirname "$0")/.." && pwd)

tail -n +2 $root/provenance/extracted_files.tsv | while IFS=$'\t' read -r dst src rest; do
    mkdir -p $root/$(dirname $dst)
    cp -p $src $root/$dst
done
