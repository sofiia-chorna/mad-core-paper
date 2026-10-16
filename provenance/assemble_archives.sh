#!/bin/bash
# copies every file listed in archive_files.csv into results_archive and models_archive

cd "$(dirname "$0")/.."

tail -n +2 provenance/archive_files.csv | while IFS=, read -r destination source rest; do
    mkdir -p "$(dirname "$destination")"
    cp -p "$source" "$destination"
done
