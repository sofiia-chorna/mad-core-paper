#!/usr/bin/env bash
set -euo pipefail

cd ${DEV_ROOT}/phace-atomic
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
scripts=release/build/matcloud/mad-cosmo-v1
run=matcloud-upload/mad-cosmo-v1
mode=${1:?choose pilot or first}

case "$mode" in
    pilot)
        report=$run/cell0-preflight.json
        staging=$run/pilot-staging
        output=$run/pilot/files
        label=madcosmo_cells_0-1
        ;;
    first)
        report=$run/preflight.json
        staging=$run/first-staging
        output=$run/first/files
        label=madcosmo_cells_0-2pow18
        ;;
    *) exit 2 ;;
esac

/usr/bin/python3.11 "$scripts/export.py" --preflight "$report" --staging "$staging" --output "$output"
/usr/bin/python3.11 "$scripts/validate.py" --staging "$staging" --output "$output" --manifest "${output%/files}/${label}_build.json" --report "${output%/files}/validation.json"
