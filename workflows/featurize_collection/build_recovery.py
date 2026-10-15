import glob
import os
import sys

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SRC_ROOT = "${STORE_ROOT}/zbl_filtering/output"
STORE = "${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92"
RECOVERY_INPUT_ROOT = "${STORE_ROOT}/zbl_filtering/output_recovered"
RECOVERY_MANIFEST = "${DEV_ROOT}/phace-atomic/featurize_zbl/manifests/feat_manifest_recovery.txt"

HEAVY = ("ODAC", "OMol25", "OC20", "OC22", "OC25", "solvated_protein",
         "Chig-AIMD", "OrbNet_Denali", "OMC25", "Open_Molecular_Crystals")

families = sys.argv[1:]


def store_shards(family):
    return [s for s in glob.glob(f"{STORE}/{family}/**/*.parquet", recursive=True)
            if not s.endswith("index.parquet") and "/_pilot/" not in s]


def concat_column(shards, column):
    def load(path):
        return pq.read_table(path, columns=[column]).column(0).combine_chunks().cast(pa.large_string())
    with ThreadPoolExecutor(32) as pool:
        arrays = [a for a in pool.map(load, shards) if len(a)]
    return pa.concat_arrays(arrays) if arrays else pa.array([], pa.large_string())


manifest_lines = []

for family in families:
    input_shards = sorted(glob.glob(f"{SRC_ROOT}/{family}/**/*.parquet", recursive=True))

    stored_cids = concat_column(store_shards(family), "configuration_id")

    per_shard_cids = []
    per_shard_nsites = []
    rows_per_shard = []
    for shard in input_shards:
        table = pq.read_table(shard, columns=["configuration_id", "nsites"])
        per_shard_cids.append(table.column("configuration_id").combine_chunks().cast(pa.large_string()))
        per_shard_nsites.append(table.column("nsites").combine_chunks())
        rows_per_shard.append(table.num_rows)

    input_cids = pa.concat_arrays(per_shard_cids)
    input_nsites = pa.concat_arrays(per_shard_nsites)

    present = pc.is_in(input_cids, value_set=stored_cids)
    recoverable = pc.less_equal(input_nsites, 1000)
    missing_mask = pc.and_(pc.invert(present), recoverable).to_numpy(zero_copy_only=False)
    missing_positions = np.sort(np.flatnonzero(missing_mask))

    print(f"{family}: store={len(stored_cids):,}  input={len(input_cids):,}  "
          f"missing_recoverable={len(missing_positions):,}", flush=True)

    if len(missing_positions) == 0:
        continue

    offsets = np.cumsum([0] + rows_per_shard)
    recovered_parts = []
    for shard_index, shard in enumerate(input_shards):
        low, high = offsets[shard_index], offsets[shard_index + 1]
        start, stop = np.searchsorted(missing_positions, low), np.searchsorted(missing_positions, high)
        if start == stop:
            continue
        local_rows = (missing_positions[start:stop] - low).astype(np.int64)
        recovered_parts.append(pq.read_table(shard).take(pa.array(local_rows)))

    recovered = pa.concat_tables(recovered_parts)

    source_out = f"{RECOVERY_INPUT_ROOT}/{family}/recovery.parquet"
    os.makedirs(os.path.dirname(source_out), exist_ok=True)
    pq.write_table(recovered, source_out, compression="snappy")

    store_out = f"{STORE}/{family}/_recovered/recovery.parquet"
    batch = 4 if any(h in family for h in HEAVY) else 32
    manifest_lines.append(f"{source_out}\t{store_out}\t{batch}")
    print(f"  wrote {recovered.num_rows:,} rows -> {source_out}", flush=True)

os.makedirs(os.path.dirname(RECOVERY_MANIFEST), exist_ok=True)
with open(RECOVERY_MANIFEST, "w") as fh:
    fh.write("\n".join(manifest_lines) + ("\n" if manifest_lines else ""))
print(f"\nwrote recovery manifest: {len(manifest_lines)} units -> {RECOVERY_MANIFEST}", flush=True)
