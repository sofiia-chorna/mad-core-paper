#!/usr/bin/env python
import glob
import hashlib
import json
import os
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

STORE = "${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92"
SIDECARS = "${DEV_ROOT}/phace-atomic/featurize_zbl/sidecars"
MODEL = "${STORE_ROOT}/mad-core/models/phace_atomic_ep92.pt"
MANIFESTS = [
    "${DEV_ROOT}/phace-atomic/featurize_zbl/manifests/feat_manifest.txt",
    "${DEV_ROOT}/phace-atomic/featurize_zbl/manifests/feat_manifest_gdb13.txt",
]
# outputs that are expected to be ABSENT because they were split into __pNN parts
SPLIT_WHOLES = {f"{STORE}/mcloudarchive__jk-9v/mcloudarchive__jk-9v/part_0000.parquet"}


def sidecar_for(rel):
    p = os.path.join(SIDECARS, rel[:-len(".parquet")] + ".json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def main():
    expected = set()
    for m in MANIFESTS:
        if os.path.exists(m):
            for ln in open(m):
                if ln.strip():
                    expected.add(ln.split("\t")[1])

    actual = set(glob.glob(f"{STORE}/**/*.parquet", recursive=True))
    actual = {a for a in actual if "/_pilot/" not in a}  # ignore any stray pilot files
    actual -= SPLIT_WHOLES  # the 3.6M whole is replaced by its __pNN parts; never count it

    missing = (expected - actual) - SPLIT_WHOLES
    extra = actual - expected  # split parts (__pNN) + anything unexpected
    print(f"expected outputs : {len(expected):,}")
    print(f"actual outputs   : {len(actual):,}")
    print(f"missing          : {len(missing):,}  (run NOT complete if >0)")
    print(f"split-whole skips: {len(SPLIT_WHOLES & expected)}  extra (split parts etc): {len(extra):,}")
    if missing:
        for p in list(sorted(missing))[:10]:
            print("   MISSING:", p.replace(STORE + "/", ""))

    rows = []
    tot_struct = tot_dropped = 0
    for f in sorted(actual):
        rel = f[len(STORE) + 1:]
        parts = rel.split("/")
        dataset = parts[0]
        variant = parts[-2] if len(parts) >= 2 else ""
        n_out = pq.ParquetFile(f).metadata.num_rows
        sc = sidecar_for(rel) or {}
        tot_struct += n_out
        tot_dropped += int(sc.get("n_dropped_nsites", 0) or 0)
        rows.append({
            "dataset": dataset, "variant": variant, "relpath": rel,
            "n_out_rows": n_out,
            "n_src_rows": int(sc.get("n_src_rows", 0) or 0),
            "n_dropped_nsites": int(sc.get("n_dropped_nsites", 0) or 0),
            "struct_per_s": float(sc.get("struct_per_s", 0) or 0),
            "host": sc.get("host", ""), "completed_at": sc.get("completed_at", ""),
        })

    idx = pa.Table.from_pylist(rows)
    pq.write_table(idx, f"{STORE}/index.parquet", compression="snappy")

    md5 = hashlib.md5(open(MODEL, "rb").read()).hexdigest() if os.path.exists(MODEL) else "(model not copied)"
    readme = f"""# phace-atomic dataset store (phace_atomic_ep92)

PhACE features (checkpoint **phace_atomic_ep92.pt**, md5 `{md5}`) for every structure in
`${STORE_ROOT}/zbl_filtering/output` (all variants) **plus the full gdb13**
(`mad-core/work/gdb13/structures_only`).

- Built: {datetime.now(timezone.utc).isoformat()}
- Output files: {len(actual):,}   Structures: {tot_struct:,}   Dropped (nsites>1000): {tot_dropped:,}
- Layout mirrors the source tree; one parquet per source file (the 3.6M mcloudarchive__jk-9v file
  is split into `part_0000__pNN.parquet`).

## Columns
- Provenance: `configuration_id` (+ `structure_hash`, `dataset_id`, `names` where present)
- Geometry: `atomic_numbers`, `positions`, `cell`, `pbc`, `nsites`
- ZBL: `zbl_max_force`, `zbl_energy_per_atom`, `zbl_max_atomic_energy`
- PhACE (ep92): `phace_energy`, `phace_features_bb_mean` (mean+std, 512), `phace_non_conservative_forces`

## Policy
- No DFT labels, no `passed`. No ZBL filter at storage (apply zbl/nsites thresholds at FPS time).
- Only filter: `nsites > 1000` skipped (featurizer OOM). gdb13 excluded from the zbl source tree and
  featurized separately from its full source.
- `index.parquet`: one row per output file (dataset, variant, relpath, rows, dropped, struct/s).
"""
    open(f"{STORE}/README.md", "w").write(readme)
    print(f"\nwrote {STORE}/index.parquet ({len(rows):,} rows) and README.md")
    print(f"TOTAL structures in store: {tot_struct:,}")


if __name__ == "__main__":
    main()
