import glob
import sys

from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SRC = "${STORE_ROOT}/zbl_filtering/output"
STORE = "${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92"

families = sys.argv[1:]


def store_shards(family):
    return [s for s in glob.glob(f"{STORE}/{family}/**/*.parquet", recursive=True)
            if not s.endswith("index.parquet") and "/_pilot/" not in s]


def load_cids(shards):
    def load(path):
        return pq.read_table(path, columns=["configuration_id"]).column(0).combine_chunks().cast(pa.large_string())
    with ThreadPoolExecutor(32) as pool:
        arrays = [a for a in pool.map(load, shards) if len(a)]
    return pa.concat_arrays(arrays) if arrays else pa.array([], pa.large_string())


def load_input_recoverable_cids(shards):
    def load(path):
        table = pq.read_table(path, columns=["configuration_id", "nsites"])
        cids = table.column("configuration_id").combine_chunks().cast(pa.large_string())
        keep = pc.less_equal(table.column("nsites").combine_chunks(), 1000)
        return cids.filter(keep)
    with ThreadPoolExecutor(32) as pool:
        arrays = [a for a in pool.map(load, shards) if len(a)]
    return pa.concat_arrays(arrays) if arrays else pa.array([], pa.large_string())


print(f"{'family':40s} {'store_rows':>13s} {'unique':>13s} {'dups':>11s} {'input_rec':>13s} {'missing':>11s} {'extra':>10s}")

for family in families:
    input_shards = sorted(glob.glob(f"{SRC}/{family}/**/*.parquet", recursive=True))

    store_cids = load_cids(store_shards(family))
    unique_store = pc.unique(store_cids)
    input_recoverable = load_input_recoverable_cids(input_shards)
    input_all = load_cids(input_shards)

    store_rows = len(store_cids)
    unique = len(unique_store)
    dups = store_rows - unique
    n_input_rec = len(input_recoverable)
    missing = n_input_rec - int(pc.sum(pc.is_in(input_recoverable, value_set=unique_store).cast(pa.int64())).as_py())
    extra = unique - int(pc.sum(pc.is_in(unique_store, value_set=input_all).cast(pa.int64())).as_py())

    print(f"  {family[:38]:38s} {store_rows:>13,} {unique:>13,} {dups:>11,} {n_input_rec:>13,} {missing:>11,} {extra:>10,}", flush=True)
