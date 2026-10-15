import json
import os

import numpy as np

RELEASE = "${STORE_ROOT}/mad-core/release"
VORONOI = "${DEV_ROOT}/phace-atomic/voronoi"
OUT = f"{VORONOI}/out/cell_fps"

ROWS_PER_CHUNK = 4_000_000
N_BUCKETS = 128
MEGA_MIN_POPULATION = 2 ** 21


def main():
    populations = np.load(f"{VORONOI}/out/analysis/populations.npz")
    offsets = populations["level_offsets"]
    level24 = np.asarray(populations["populations"][offsets[24]:offsets[25]], dtype=np.int64)
    mega_cells = np.where(level24 >= MEGA_MIN_POPULATION)[0]

    sources = []
    group_names = sorted(os.listdir(f"{RELEASE}/collection"))
    for name in group_names:
        path = f"{RELEASE}/collection/{name}"
        ns = int(np.load(f"{path}/ns.npy")[0])
        sources.append({"name": name, "path": path, "ns": ns, "tier": "collection"})
    ns_mad = int(np.load(f"{RELEASE}/mad_coreset/ns.npy")[0])
    sources.append({"name": "mad_coreset", "path": f"{RELEASE}/mad_coreset",
                    "ns": ns_mad, "tier": "mad_coreset"})

    chunks = []
    for source_index, source in enumerate(sources):
        for start in range(0, source["ns"], ROWS_PER_CHUNK):
            stop = min(start + ROWS_PER_CHUNK, source["ns"])
            chunks.append([source_index, start, stop])

    total_rows = sum(s["ns"] for s in sources)
    plan = {
        "release": RELEASE,
        "out": OUT,
        "n_buckets": N_BUCKETS,
        "rows_per_chunk": ROWS_PER_CHUNK,
        "mega_min_population": MEGA_MIN_POPULATION,
        "mega_cells": mega_cells.tolist(),
        "mega_populations": level24[mega_cells].tolist(),
        "total_rows": total_rows,
        "sources": sources,
        "chunks": chunks,
    }

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/plan.json", "w") as fh:
        json.dump(plan, fh)

    print(f"sources {len(sources)}  chunks {len(chunks)}  total rows {total_rows:,}")
    print(f"mega cells {len(mega_cells)}  mega points {int(level24[mega_cells].sum()):,}")


if __name__ == "__main__":
    main()
