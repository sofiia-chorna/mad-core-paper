#!/bin/bash
set -euo pipefail

# reorganises aa002 mad-core/lda_pbe into four top level folders
# final            the dataset
# dft_outputs      every raw FHI-aims output tar (results, EPFL deliveries, Clariden batches)
# build            inputs and scripts the dataset was built from
# campaign_history everything else from the campaign
# only renames, nothing is copied or deleted, every move is logged in MOVES_20261005.tsv

R=${STORE_ROOT}/mad-core/lda_pbe
VR=${STORE_ROOT}/mad-core/voronoi_remaining
M="$R/MOVES_20261005.tsv"

for d in dft_outputs build campaign_history; do
    test ! -e "$R/$d"
done
test -d "$R/product/final"

mkdir -p "$R/dft_outputs/epfl_deliveries" "$R/build" "$R/campaign_history"
printf 'old\tnew\n' > "$M"

mv_log() {
    mv "$1" "$2"
    printf '%s\t%s\n' "$1" "$2" >> "$M"
}

mv_log "$R/product/final" "$R/final"

for e in computed_20260923 computed_20261005 INVENTORY.md madcore_recheck README.md references rerun_hamiltonians scripts sets share snapshot_history; do
    mv_log "$R/product/$e" "$R/build/$e"
done
rmdir "$R/product"

mv_log "$R/campaign/results" "$R/dft_outputs/results"

for e in 20260922_round2 20260923_round3 20260923_round4_delta 20260923_round5_delta 20260923_supp_lda2 20260927_round6_delta 20260928_round7_delta 20260929_round8_delta 20260929_round9_delta 20260929_round11_delta retrieve_20260923; do
    mv_log "$VR/$e" "$R/dft_outputs/epfl_deliveries/$e"
done
rmdir "$VR"

mv_log "$R/campaign/from_lyra/20260922" "$R/dft_outputs/epfl_deliveries/20260922_round1"
rmdir "$R/campaign/from_lyra"

for e in from_kuma clariden_handoff scratch_rescue_20260925 rerun_hamiltonians_20260914 benchmark_20260925; do
    mv_log "$R/campaign/$e" "$R/dft_outputs/$e"
done

for e in $(ls "$R/campaign"); do
    mv_log "$R/campaign/$e" "$R/campaign_history/$e"
done
rmdir "$R/campaign"

echo "=== lda_pbe"
ls -la "$R"
echo "=== dft_outputs"
ls "$R/dft_outputs"
echo "=== epfl_deliveries"
ls "$R/dft_outputs/epfl_deliveries"
echo "=== moves logged"
wc -l "$M"

cd "$R/final"
sha256sum -c SHA256SUMS --quiet
echo "final verified"
