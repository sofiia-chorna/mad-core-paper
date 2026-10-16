#!/usr/bin/env python
import concurrent.futures as cf
import glob
import hashlib
import json
import os
from datetime import datetime, timezone

import pyarrow.parquet as pq

SRC_ROOT = "${STORE_ROOT}/zbl_filtering/output"
STORE = "${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92"
MODEL = "${STORE_ROOT}/mad-core/models/phace_atomic_ep92.pt"
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "feat_manifest.txt")

EXCLUDE_TOP = {"gdb13"}  # treated separately later
# big-molecule datasets to small batch (avoid OOM) matched as substring of the top-level dir
HEAVY = ("ODAC", "OMol25", "OC20", "OC22", "OC25", "solvated_protein",
         "Chig-AIMD", "OrbNet_Denali", "OMC25", "Open_Molecular_Crystals")
KEEP_COLS = ("configuration_id,structure_hash,dataset_id,names,atomic_numbers,positions,"
             "cell,pbc,nsites,zbl_max_force,zbl_energy_per_atom,zbl_max_atomic_energy")


def main():
    files = [
        f for f in glob.glob(f"{SRC_ROOT}/**/*.parquet", recursive=True)
        if f[len(SRC_ROOT) + 1:].split("/")[0] not in EXCLUDE_TOP
    ]
    print(f"globbed {len(files):,} in-scope parquet files (gdb13 excluded)")

    def nrows(f):
        try:
            return pq.ParquetFile(f).metadata.num_rows
        except Exception:
            return -1

    with cf.ThreadPoolExecutor(max_workers=48) as ex:
        rows = list(ex.map(nrows, files))

    entries, bad = [], 0
    for f, n in zip(files, rows):
        if n < 0:
            bad += 1
            continue
        rel = f[len(SRC_ROOT) + 1:]
        top = rel.split("/")[0]
        batch = 4 if any(h in top for h in HEAVY) else 32
        out = os.path.join(STORE, rel)
        entries.append((n, f, out, batch))
    entries.sort(key=lambda e: -e[0])  # big first

    with open(MANIFEST, "w") as fh:
        for n, src, out, batch in entries:
            fh.write(f"{src}\t{out}\t{batch}\n")

    total = sum(e[0] for e in entries)
    md5 = hashlib.md5(open(MODEL, "rb").read()).hexdigest()
    os.makedirs(STORE, exist_ok=True)
    manifest_json = {
        "source_root": SRC_ROOT,
        "store_root": STORE,
        "model": MODEL,
        "model_md5": md5,
        "keep_cols": KEEP_COLS,
        "added_cols": ["phace_energy", "phace_features_bb_mean (mean+std, 2xD)",
                       "phace_non_conservative_forces"],
        "filters": {"max_nsites": 1000, "zbl": "none (selection-time concern)"},
        "meanstd": True, "nl_backend": "metatomic", "dtype": "float32",
        "excluded": sorted(EXCLUDE_TOP),
        "n_units": len(entries), "n_bad": bad, "total_src_rows": total,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(STORE, "MANIFEST.json"), "w") as fh:
        json.dump(manifest_json, fh, indent=2)

    print(f"units={len(entries):,}  bad={bad}  total_src_rows={total:,}")
    print(f"biggest unit={entries[0][0]:,} rows  ->  {entries[0][1]}")
    print(f"wrote {MANIFEST}")
    print(f"wrote {STORE}/MANIFEST.json")


if __name__ == "__main__":
    main()
