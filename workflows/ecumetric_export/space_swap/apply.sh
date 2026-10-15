#!/bin/bash
set -euo pipefail

release=${STORE_ROOT}/mad-core/release
backup_dir=$release/_backup_2026-10-08
build_dir=${DEV_ROOT}/phace-atomic/release/build/ecumetric_space_swap
new_checkpoint=${DEV_ROOT}/phace-atomic/model_92_space.ckpt
new_model=$build_dir/staging/ecumetric_model.pt

cd "$build_dir"


copy_and_check() {
    local source=$1
    local target=$2

    cp "$source" "$target"
    chmod 640 "$target"

    local source_hash
    local target_hash
    source_hash=$(sha256sum "$source" | cut -d' ' -f1)
    target_hash=$(sha256sum "$target" | cut -d' ' -f1)

    if [ "$source_hash" != "$target_hash" ]
    then
        echo "copy mismatch for $target"
        exit 1
    fi

    echo "copied $target $target_hash"
}


# stop if a previous run already made the backup
if [ -e "$backup_dir" ]
then
    echo "$backup_dir exists, stopping"
    exit 1
fi

# stage the new files under hidden names and check them before touching the old ones
copy_and_check "$new_checkpoint" "$release/.ecumetric_model.ckpt.new"
copy_and_check "$new_model" "$release/.ecumetric_model.pt.new"

# keep the old files and the current manifest
mkdir "$backup_dir"
cp -p "$release/MANIFEST.json" "$backup_dir/MANIFEST.json"
cp "$build_dir/backup_README.md" "$backup_dir/README.md"
chmod 640 "$backup_dir/README.md"

mv "$release/ecumetric_model.pt" "$backup_dir/ecumetric_model.pt"
mv "$release/ecumetric_model.ckpt" "$backup_dir/ecumetric_model.ckpt"

# put the new files in place
mv "$release/.ecumetric_model.ckpt.new" "$release/ecumetric_model.ckpt"
mv "$release/.ecumetric_model.pt.new" "$release/ecumetric_model.pt"

python3.11 update_manifest.py "$release/MANIFEST.json"

echo "apply done $(date -Is)"
