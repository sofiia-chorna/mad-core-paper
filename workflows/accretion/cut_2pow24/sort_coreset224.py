import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

scratch_out = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/out"
store_out = "${STORE_ROOT}/mad-core/accrete/phace_atomic_ep92/coreset224/out"
num_rows = 2 ** 24
slice_rows = 1_000_000


def with_large_lists(schema):
    fields = []
    for field in schema:
        if pa.types.is_list(field.type):
            fields.append(pa.field(field.name, pa.large_list(field.type.value_type)))
        else:
            fields.append(field)
    return pa.schema(fields)


for name in ["coreset_224_light.parquet", "coreset_224_full.parquet"]:
    current = pq.read_table(f"{store_out}/{name}", columns=["fps_order"]).column("fps_order").to_numpy()
    if np.array_equal(current, np.arange(num_rows)):
        print(f"{name}: already sorted, skip", flush=True)
        continue

    print(f"reading {name}", flush=True)
    table = pq.read_table(f"{store_out}/{name}")
    assert table.num_rows == num_rows
    original_schema = table.schema
    table = table.cast(with_large_lists(original_schema))
    order = pc.sort_indices(table.column("fps_order"))
    print("  cast to large_list + sort indices computed, writing in slices", flush=True)

    writer = pq.ParquetWriter(f"{scratch_out}/{name}.sorted", original_schema, compression="zstd")
    for lo in range(0, num_rows, slice_rows):
        chunk = table.take(order.slice(lo, slice_rows))
        writer.write_table(chunk.cast(original_schema))
        if lo % 4_000_000 == 0:
            print(f"  written {lo:,}", flush=True)
    writer.close()
    del table

    check = pq.read_table(f"{scratch_out}/{name}.sorted", columns=["fps_order"]).column("fps_order").to_numpy()
    assert np.array_equal(check, np.arange(num_rows)), "sorted output is not 0..2^24-1"
    schema_check = pq.ParquetFile(f"{scratch_out}/{name}.sorted").schema_arrow
    assert schema_check.equals(original_schema), "schema changed"
    print("  re-read + schema checks OK; replacing scratch + store copies", flush=True)
    os.replace(f"{scratch_out}/{name}.sorted", f"{scratch_out}/{name}")
    shutil.copy2(f"{scratch_out}/{name}", f"{store_out}/{name}.new")
    os.replace(f"{store_out}/{name}.new", f"{store_out}/{name}")
    print(f"  {name} DONE", flush=True)

print("ALL SORTED", flush=True)
