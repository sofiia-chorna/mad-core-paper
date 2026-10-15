#!/bin/bash
# filter_lmdb_dir.sh
#
# Parallel filter for a directory of FAIRChem .aselmdb files. One Python
# worker process per file; N files run concurrently. Each LMDB is
# independent, so this scales linearly with worker count up to I/O limits.
#
# Usage:
#   ./filter_lmdb_dir.sh <input_dir> <output_dir> <rejected_dir> [N_WORKERS] [SCALE]
#
# Example:
#   ./filter_lmdb_dir.sh train train_passed train_rejected 16 0.5
#
# Idempotent: files whose output already exists are skipped on rerun.
# Per-file logs go to <output_dir>/.logs/<stem>.log so failures are
# diagnosable. Failed runs delete their partial output so they re-run
# cleanly on the next invocation.

set -uo pipefail   # NOT -e: we want to continue past individual file failures

IN_DIR="${1:?input dir required}"
OUT_DIR="${2:?output dir required}"
REJ_DIR="${3:?rejected dir required}"
N_WORKERS="${4:-16}"
SCALE="${5:-0.5}"

# Path to the Python filter. Override with FILTER=... in the environment.
FILTER="${FILTER:-filter_lmdb.py}"

mkdir -p "$OUT_DIR" "$REJ_DIR" "$OUT_DIR/.logs"

# -----------------------------------------------------------------------------
# Per-file worker. Exported for xargs's child shells.
# -----------------------------------------------------------------------------
filter_one() {
    local in="$1"
    local stem
    stem=$(basename "$in" .aselmdb)
    local out="${OUT_DIR}/${stem}.aselmdb"
    local rej="${REJ_DIR}/${stem}.aselmdb"
    local log="${OUT_DIR}/.logs/${stem}.log"

    if [[ -f "$out" ]]; then
        printf "[SKIP] %-30s (output exists)\n" "$stem"
        return 0
    fi

    local t0=$SECONDS
    printf "[START] %s\n" "$stem"
    if python "$FILTER" "$in" \
            -o "$out" --rejected "$rej" --scale "$SCALE" \
            > "$log" 2>&1; then
        local elapsed=$((SECONDS - t0))
        # Pull the filter's final "Done:" summary for the master log
        local summary
        summary=$(grep '^Done:' "$log" | tail -1)
        printf "[DONE]  %-30s (%ds) %s\n" "$stem" "$elapsed" "$summary"
    else
        # Partial output is unsafe; delete so the next run retries cleanly.
        rm -f "$out" "$rej"
        printf "[FAIL]  %-30s -- see %s\n" "$stem" "$log"
        return 1
    fi
}
export -f filter_one
export OUT_DIR REJ_DIR SCALE FILTER

# -----------------------------------------------------------------------------
# Discover inputs and dispatch
# -----------------------------------------------------------------------------
mapfile -t inputs < <(find "$IN_DIR" -maxdepth 1 -name '*.aselmdb' -type f | sort)
n_total=${#inputs[@]}

if (( n_total == 0 )); then
    echo "No .aselmdb files found in $IN_DIR"
    exit 1
fi

echo "Found $n_total LMDBs in $IN_DIR"
echo "Workers: $N_WORKERS"
echo "Scale:   $SCALE"
echo "Output:  $OUT_DIR"
echo "Logs:    $OUT_DIR/.logs/"
echo

printf "%s\n" "${inputs[@]}" \
    | xargs -n 1 -P "$N_WORKERS" -I{} bash -c 'filter_one "$@"' _ {}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo
echo "=== Summary ==="
n_done=$(find "$OUT_DIR" -maxdepth 1 -name '*.aselmdb' -type f | wc -l)
n_fail=$((n_total - n_done))

echo "input:    $n_total files"
echo "passed:   $n_done files in $OUT_DIR"
if (( n_fail > 0 )); then
    echo "missing:  $n_fail files (failed or not yet processed)"
    echo "  re-run this script to retry them"
fi

# Aggregate counts across all per-file logs
echo
echo "=== Aggregate counts ==="
python3 - "$OUT_DIR/.logs" <<'PY'
import re, sys, pathlib
log_dir = pathlib.Path(sys.argv[1])
total_p = total_r = total_s = total_t = 0
n_logs = 0
pat = re.compile(r'Done: passed=(\d+) rejected=(\d+) skipped=(\d+) total=(\d+)')
for log in log_dir.glob("*.log"):
    txt = log.read_text(errors="replace")
    for m in pat.finditer(txt):
        p, r, s, t = (int(x) for x in m.groups())
        total_p += p; total_r += r; total_s += s; total_t += t
        n_logs += 1
        break
print(f"  files contributing:  {n_logs:,}")
print(f"  total structures in: {total_t:,}")
print(f"  passed:              {total_p:,}")
print(f"  rejected:            {total_r:,}")
print(f"  skipped:             {total_s:,}")
PY
