import glob
import os

from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq


SRC = "${STORE_ROOT}/zbl_filtering/output"
STORE = "${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92"


def recoverable_rows(shard):
    nsites = pq.read_table(shard, columns=["nsites"]).column("nsites").to_numpy()
    return int((nsites <= 1000).sum())


def row_count(shard):
    return pq.ParquetFile(shard).metadata.num_rows


families = sorted(f for f in os.listdir(SRC)
                  if os.path.isdir(os.path.join(SRC, f)) and f != "gdb13")

print(f"comparing {len(families)} families (gdb13 excluded)\n")
print(f"{'family':44s} {'in_sh':>6s} {'out_sh':>7s} {'input_rec':>13s} {'store':>13s}  flag")

flagged = 0
with ThreadPoolExecutor(96) as pool:
    for family in families:
        input_shards = glob.glob(f"{SRC}/{family}/**/*.parquet", recursive=True)
        store_shards = [s for s in glob.glob(f"{STORE}/{family}/**/*.parquet", recursive=True)
                        if not s.endswith("index.parquet") and "/_pilot/" not in s]

        n_in, n_out = len(input_shards), len(store_shards)
        input_rec = sum(pool.map(recoverable_rows, input_shards))
        store_rows = sum(pool.map(row_count, store_shards)) if store_shards else 0

        repartitioned = n_out != n_in
        row_mismatch = store_rows != input_rec

        if repartitioned or row_mismatch:
            flagged += 1
            tags = []
            if repartitioned:
                tags.append("REPARTITIONED(cid-diff needed)" if n_out > n_in else "MISSING-SHARDS")
            if row_mismatch:
                tags.append(f"rows {'short' if store_rows < input_rec else 'over'} {abs(store_rows - input_rec):,}")
            print(f"  {family[:42]:42s} {n_in:>6,} {n_out:>7,} {input_rec:>13,} {store_rows:>13,}  {' | '.join(tags)}")

print(f"\n{flagged} families differ from a clean 1:1 match; all others are 1:1 with store==recoverable-input (complete).")
