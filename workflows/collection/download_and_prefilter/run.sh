#!/bin/bash
# Run pyykko_filter_parquet on every shard of a dataset, one at a time,
# with --workers 1. Useful when the parallel driver crashes (OOM, segfault)
# and you want a) to see which shard fails, b) to recover output as far as
# possible without losing already-done shards.
#
# Each shard writes its own output + rejected parquet into
# <out_root>/<dataset>/co_passed/ and co_rejected/. The driver is
# idempotent: shards with an existing output file are skipped on rerun.
#
# Usage:
#   ./filter_shards_serial.sh <input_dataset_dir> <out_root> [SCALE]
#
# Example (your dataset):
#   ./filter_shards_serial.sh \
#       colabfit_data/colabfit_Alexandria_geometry_optimization_paths_PBE_2D \
#       filtered 0.5

set -uo pipefail   # not -e: we want to continue past individual failures

IN_DS="${1:?input dataset dir required (e.g. colabfit_data/<name>)}"
OUT_ROOT="${2:?output root required (e.g. filtered)}"
SCALE="${3:-0.5}"

FILTER="${FILTER:-pyykko_filter_parquet.py}"
INPUT_SUBDIR="${INPUT_SUBDIR:-co}"

# Strip trailing slashes for clean basename
IN_DS="${IN_DS%/}"
OUT_ROOT="${OUT_ROOT%/}"

if [[ ! -d "$IN_DS/$INPUT_SUBDIR" ]]; then
    echo "ERROR: $IN_DS/$INPUT_SUBDIR/ does not exist"
    echo "  (set INPUT_SUBDIR=co_passed to filter a previously-filtered dataset)"
    exit 1
fi

ds_name=$(basename "$IN_DS")
out_ds="$OUT_ROOT/$ds_name"
out_pass="$out_ds/co_passed"
out_rej="$out_ds/co_rejected"
log_dir="$out_ds/.logs"
mkdir -p "$out_pass" "$out_rej" "$log_dir"

# Discover shards
mapfile -t shards < <(find "$IN_DS/$INPUT_SUBDIR" -maxdepth 1 -name '*.parquet' \
                          -type f | sort)
n_total=${#shards[@]}
if (( n_total == 0 )); then
    echo "No parquet files found in $IN_DS/$INPUT_SUBDIR/"
    exit 1
fi

echo "Dataset:    $ds_name"
echo "Shards:     $n_total in $IN_DS/$INPUT_SUBDIR/"
echo "Output:     $out_pass/"
echo "Rejected:   $out_rej/"
echo "Logs:       $log_dir/"
echo "Scale:      $SCALE"
echo

n_ok=0 n_skip=0 n_fail=0
t_start=$SECONDS

for shard in "${shards[@]}"; do
    base=$(basename "$shard" .parquet)
    out_p="$out_pass/${base}_passed.parquet"
    out_r="$out_rej/${base}_rejected.parquet"
    log="$log_dir/${base}.log"

    if [[ -f "$out_p" ]]; then
        printf "[SKIP] %-30s (output exists)\n" "$base"
        n_skip=$((n_skip + 1))
        continue
    fi

    t0=$SECONDS
    printf "[RUN ] %-30s " "$base"
    if python "$FILTER" "$shard" \
            -o "$out_p" --rejected "$out_r" \
            --scale "$SCALE" --workers 8 \
            > "$log" 2>&1; then
        elapsed=$((SECONDS - t0))
        summary=$(grep '^Done:' "$log" | tail -1)
        printf "OK   (%4ds)  %s\n" "$elapsed" "$summary"
        n_ok=$((n_ok + 1))
    else
        elapsed=$((SECONDS - t0))
        # Worker may have crashed mid-write; partial output is unsafe
        rm -f "$out_p" "$out_r"
        printf "FAIL (%4ds)  -> %s\n" "$elapsed" "$log"
        n_fail=$((n_fail + 1))
    fi
done

t_total=$((SECONDS - t_start))
echo
echo "============================================================"
echo "Summary for $ds_name:"
echo "  ran:     $n_ok"
echo "  skipped: $n_skip  (already had output)"
echo "  failed:  $n_fail  (see $log_dir/ for which)"
echo "  wall:    ${t_total}s"

if (( n_fail > 0 )); then
    echo
    echo "Failed shards (check the logs):"
    for shard in "${shards[@]}"; do
        base=$(basename "$shard" .parquet)
        log="$log_dir/${base}.log"
        out_p="$out_pass/${base}_passed.parquet"
        # A failed shard has a log but no output
        if [[ -f "$log" && ! -f "$out_p" ]]; then
            echo "  $base"
            tail -3 "$log" | sed 's/^/    /'
        fi
    done
    exit 2
fi
