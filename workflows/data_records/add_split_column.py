import collections
import hashlib
import json
import os

import numpy as np

TIER = "${STORE_ROOT}/mad-core/release/mad_coreset"
SPLIT_MAP = "split_map_mad15.json"
AMENDMENT = ("2026-08-07: added split.npy carrying the official MAD-1.5 "
             "train/val/test assignment; empty on rows with no official split")


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


with open(SPLIT_MAP) as handle:
    split_of_id = json.load(handle)

cid = np.load(f"{TIER}/configuration_id.npy")
group = np.load(f"{TIER}/dataset_group.npy")
ns = len(cid)

split = np.zeros(ns, dtype="S5")

for row in range(ns):
    value = split_of_id.get(cid[row].decode())

    if value is not None:
        split[row] = value.encode()

labelled = split != b""
mad15_rows = group == b"mad15"

counts = collections.Counter(split[labelled].tolist())

print("rows:", ns)
print("labelled:", int(labelled.sum()), counts)
print("labelled set == mad15 set:", bool(np.array_equal(labelled, mad15_rows)))
print("unlabelled by group:",
      collections.Counter(group[~labelled].tolist()))

assert np.array_equal(labelled, mad15_rows), "labelled rows are not exactly the mad15 rows"
assert counts == {b"train": 180184, b"val": 18305, b"test": 18314}, counts

target = f"{TIER}/split.npy"
staging = f"{target}.tmp"

with open(staging, "wb") as handle:
    np.save(handle, split)

os.replace(staging, target)

written = np.load(target)
assert np.array_equal(written, split), "readback differs"

print("wrote", target, written.dtype, written.shape)

with open(f"{TIER}/MANIFEST.json") as handle:
    manifest = json.load(handle)

manifest["files"]["split.npy"] = {
    "bytes": os.path.getsize(target),
    "sha256": sha256(target),
    "shape": [ns],
    "dtype": "|S5",
}
manifest["files"] = dict(sorted(manifest["files"].items()))
manifest["amendments"] = manifest.get("amendments", []) + [AMENDMENT]

staging = f"{TIER}/MANIFEST.json.tmp"

with open(staging, "w") as handle:
    json.dump(manifest, handle, indent=1)

os.replace(staging, f"{TIER}/MANIFEST.json")

print("manifest updated:", manifest["files"]["split.npy"])
