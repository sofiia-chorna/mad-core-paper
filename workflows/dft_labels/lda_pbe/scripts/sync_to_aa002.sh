#!/bin/bash
set -uo pipefail

CAMPAIGN_SRC=${DEV_ROOT}/lda_pbe
PRODUCT_SRC=${DEV_ROOT}/phace-atomic/mad-core-valid/dft/lda_pbe
DEST=${STORE_ROOT}/mad-core/lda_pbe

DRY="--dry-run"
[ "${1:-}" = "--go" ] && DRY=""

RSYNC="rsync -a --ignore-existing --info=stats2 $DRY"

echo "=== 1. product/share ==="
$RSYNC "$PRODUCT_SRC/share/" "$DEST/product/share/"

echo "=== 2. campaign top-level directories never synced ==="
# benchmark_20260925 already lives, curated and checksummed, at campaign/benchmark_20260925
for d in logs docs archive merged to_kuma to_lyra; do
  echo "-- $d"
  $RSYNC --exclude='benchmark_20260925/' "$CAMPAIGN_SRC/$d/" "$DEST/campaign/$d/"
done

echo "=== 3. campaign reports and common, additive only ==="
$RSYNC "$CAMPAIGN_SRC/reports/" "$DEST/campaign/reports/"
$RSYNC "$CAMPAIGN_SRC/common/" "$DEST/campaign/common/"

echo "=== 3b. refresh files that changed since they were first copied ==="
rsync -a --update --info=stats2 $DRY "$CAMPAIGN_SRC/common/" "$DEST/campaign/common/"
rsync -a --update --info=stats2 $DRY "$CAMPAIGN_SRC/reports/" "$DEST/campaign/reports/"

echo "=== 4. campaign top-level files ==="
$RSYNC "$CAMPAIGN_SRC/CLAUDE.md" "$CAMPAIGN_SRC/requirements.txt" "$DEST/campaign/"

echo "=== 5. per-block frame censuses (manifest.txt), the authoritative completeness record ==="
cd "$CAMPAIGN_SRC" && $RSYNC --include='*/' --include='manifest.txt' --exclude='*' \
  runs/ "$DEST/campaign/runs/"

echo "=== 6. converged runs that were never banked into results/ (text outputs only) ==="
cd "$CAMPAIGN_SRC" && $RSYNC --include='*/' \
  --include='aims.out' --include='aims.err' --include='geometry.in' --include='control.in' \
  --exclude='*' runs/ "$DEST/campaign/runs_unbanked/"

echo "=== 6b. the campaign documentation and inventory ==="
$RSYNC --update "$PRODUCT_SRC/../DFT_CAMPAIGN.md" "$DEST/../DFT_CAMPAIGN.md"
$RSYNC --update "$PRODUCT_SRC/../DFT_CAMPAIGN.md" "$DEST/campaign/DFT_CAMPAIGN.md"
$RSYNC --update "$PRODUCT_SRC/INVENTORY.md" "$DEST/product/INVENTORY.md"

echo "=== 7. the 2026-09-14 reruns: scripts, reports and outputs ==="
for d in madcore_recheck rerun_hamiltonians; do
  echo "-- $d"
  $RSYNC "$PRODUCT_SRC/$d/" "$DEST/product/$d/"
done

echo "=== 8. regenerated Hamiltonians for the 22 RERUN_FROM_SCRATCH frames ==="
while read -r arm kind block frame; do
  name=$(printf 'frame_%05d' "$frame")
  src="$CAMPAIGN_SRC/runs/$arm/$kind/$block/$name"
  dst="$DEST/campaign/rerun_hamiltonians_20260914/$arm/$kind/$block/$name"
  [ -d "$src" ] || { echo "  skip, missing: $src"; continue; }
  # rsync only creates the final path component, so make the parents first
  [ -n "$DRY" ] || mkdir -p "$dst"
  $RSYNC "$src/" "$dst/" < /dev/null || echo "  FAILED: $src"
done < "$PRODUCT_SRC/rerun_hamiltonians/lists/all_frames.txt"

echo
[ -n "$DRY" ] && echo "DRY RUN - rerun with --go to copy"
