#!/bin/bash
# copies every file listed in extracted_files.csv from where it came from

cd "$(dirname "$0")/.."

tail -n +2 provenance/extracted_files.csv | while IFS=, read -r destination source rest; do
    mkdir -p "$(dirname "$destination")"
    cp -p "$source" "$destination"
done
