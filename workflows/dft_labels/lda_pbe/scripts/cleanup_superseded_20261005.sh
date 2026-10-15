#!/bin/bash
set -euo pipefail

# removes the snapshots superseded by labelled_20261005 and the staging dirs
# every voronoi frame of every removed snapshot is byte identical in labelled_20261005
# madcore random and random_test files are identical across all of them
# checksum lists and build logs of each removed snapshot are kept in product/snapshot_history

PRODUCT=${STORE_ROOT}/mad-core/lda_pbe/product
SCRATCH=${DEV_ROOT}/phace-atomic/mad-core-valid/dft/lda_pbe

cd "$PRODUCT/labelled_20261005"
sha256sum -c SHA256SUMS --quiet
echo "labelled_20261005 verified"

test -d "$PRODUCT/snapshot_history/labelled_20260929b"
echo "snapshot_history present"

chmod -R u+w "$PRODUCT/labelled_20260923c"

rm -rf "$PRODUCT/labelled"
rm -rf "$PRODUCT/labelled_20260922"
rm -rf "$PRODUCT/labelled_20260923"
rm -rf "$PRODUCT/labelled_20260923b"
rm -rf "$PRODUCT/labelled_20260923c"
rm -rf "$PRODUCT/labelled_20260923d"
rm -rf "$PRODUCT/labelled_20260929"
rm -rf "$PRODUCT/labelled_20260929b"

rm -rf "$PRODUCT/computed"
rm -rf "$PRODUCT/computed_20260929"
rm -rf "$PRODUCT/computed_20260929b"

ln -sfn labelled_20261005 "$PRODUCT/labelled_final"

rm -rf "$SCRATCH/import_20260923b"
rm -rf "$SCRATCH/import_20260923c"
rm -rf "$SCRATCH/import_20260923d"
rm -rf "$SCRATCH/import_20260929b"
rm -rf "$SCRATCH/import_20261005"
rm -rf "$SCRATCH/completion_20260929"
rm -rf "$SCRATCH/labelled"
rm -rf "$SCRATCH/labelled_20260929b"
rm -rf "$SCRATCH/computed"

echo "remaining in $PRODUCT"
ls -la "$PRODUCT"

echo "dangling links under computed_20261005 (must be 0)"
find "$PRODUCT/computed_20261005" -xtype l | wc -l

cd "$PRODUCT/labelled_20261005"
sha256sum -c SHA256SUMS --quiet
echo "labelled_20261005 still verified"
