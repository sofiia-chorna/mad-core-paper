import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

RELEASE = "${STORE_ROOT}/mad-core/release"
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = ("${DEV_ROOT}/phace-atomic/voronoi/out/"
            "manifest_all.tsv")
OUT = f"{HERE}/dataset_metadata.tsv"

COLUMNS = ["dataset_id", "dataset_group", "colabfit_ds_id", "rows", "source_kind",
           "store_subfolder", "doi", "upstream_url", "license"]


def name_to_ds_id():
    """Invert the rename map: readable name -> the ColabFit DS-id it replaced."""
    with open(f"{HERE}/dataset_id_to_name.json") as fh:
        forward = json.load(fh)
    # the map predates the colabfit_ prefix fix, so register both spellings
    inverse = defaultdict(list)
    for ds_id, name in forward.items():
        inverse[name].append(ds_id)
        if not name.startswith("colabfit_"):
            inverse[f"colabfit_{name}"].append(ds_id)
    return inverse


def store_subfolders():
    """Every store subfolder a group's shards live in, from the frozen manifest."""
    folders = defaultdict(set)
    with open(MANIFEST) as fh:
        next(fh)
        for line in fh:
            path = line.split("\t")[0]
            parts = path.split("/")
            if "phace_atomic_ep92" in parts:
                index = parts.index("phace_atomic_ep92")
                group = parts[index + 1]
                if len(parts) > index + 2:
                    folders[group].add(parts[index + 2])
    return folders


def classify(dataset_id):
    if dataset_id.startswith("colabfit_"):
        return "colabfit"
    if dataset_id == "nomad":
        return "nomad record"
    if dataset_id.startswith("mcloudarchive__"):
        return "materials cloud archive record"
    if dataset_id.startswith(("mcloud__", "oqmd__")):
        return "optimade"
    return "standalone dataset"


def main():
    inverse = name_to_ds_id()
    folders = store_subfolders()

    rows = []
    for group in sorted(os.listdir(f"{RELEASE}/collection")):
        path = f"{RELEASE}/collection/{group}"
        dataset_id = np.load(f"{path}/dataset_id.npy", mmap_mode="r")
        values, counts = np.unique(np.asarray(dataset_id), return_counts=True)
        for value, count in zip(values, counts):
            name = value.decode()
            ds_ids = inverse.get(name, [])
            rows.append({
                "dataset_id": name,
                "dataset_group": group,
                "colabfit_ds_id": ";".join(sorted(ds_ids)),
                "rows": int(count),
                "source_kind": classify(name),
                "store_subfolder": ";".join(sorted(folders.get(group, []))[:4]),
                "doi": "", "upstream_url": "", "license": "",
            })

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    resolved = sum(1 for r in rows if r["colabfit_ds_id"])
    print(f"wrote {OUT}")
    print(f"  dataset entries : {len(rows):,}")
    print(f"  with colabfit DS-id recovered : {resolved:,}")
    print(f"  total rows accounted : {sum(r['rows'] for r in rows):,}")


if __name__ == "__main__":
    main()
