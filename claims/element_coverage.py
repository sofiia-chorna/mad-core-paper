# the pick at which the MAD-CORE ordering and the MAD-NEXT ordering first contain every one of
# the 102 elements of the collection (text: "the first 480 picks already contain all 102
# elements" for MAD-NEXT, "all 102 elements of the collection are covered within the first
# 2,158 selections" for MAD-CORE). reads the elements column of the two orderings from the
# aa002 store; written for this release, run on 2026-10-16
import argparse
import json
import os

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
N_ELEMENTS = 102


# the ordering is the fps_order column; the first prefix that holds all elements
def first_prefix_with_all_elements(parquet):
    table = pq.read_table(parquet, columns=["fps_order", "elements"])
    order = np.argsort(table.column("fps_order").to_numpy())
    elements = table.column("elements").to_pylist()
    seen = set()
    for position, row in enumerate(order):
        seen.update(int(z) for z in elements[row])
        if len(seen) >= N_ELEMENTS:
            return position + 1, len(elements)
    raise SystemExit(f"{parquet}: only {len(seen)} elements in the whole ordering")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True, help="the aa002 mad-core store, holding mad_union/ and accrete/")
    parser.add_argument("--out", default=os.path.join(HERE, "out", "element_coverage.json"))
    args = parser.parse_args()

    madcore = f"{args.store_root}/accrete/phace_atomic_ep92/coreset224/out/coreset_224_light.parquet"
    madnext = f"{args.store_root}/mad_union/mad_union_full.parquet"
    result = {}
    for name, path in (("mad-core", madcore), ("mad-next", madnext)):
        prefix, rows = first_prefix_with_all_elements(path)
        result[name] = {"source": os.path.basename(path), "rows": rows, "first_prefix_with_all_102_elements": prefix}
        print(f"{name}: all {N_ELEMENTS} elements within the first {prefix} picks of {rows:,}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
