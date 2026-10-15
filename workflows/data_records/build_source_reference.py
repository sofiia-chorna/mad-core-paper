import argparse
import hashlib
import json
import os

import numpy as np
import pyarrow.parquet as pq

RELEASE = "${STORE_ROOT}/mad-core/release"
MANIFEST = ("${DEV_ROOT}/phace-atomic/voronoi/out/"
            "manifest_all.tsv")
MAD_GROUPS = ("mad-1.5", "mad-next", "mad_union")


def group_of(shard_path):
    parts = shard_path.split("/")
    for index, part in enumerate(parts):
        if part == "phace_atomic_ep92":
            return parts[index + 1]
    return None


def shards_by_group():
    """Frozen shard order per group — this IS the release row order."""
    groups = {}
    with open(MANIFEST) as fh:
        next(fh)
        for line in fh:
            path, rows, _ = line.rstrip("\n").split("\t")
            name = group_of(path)
            if name and name not in MAD_GROUPS:
                groups.setdefault(name, []).append((path, int(rows)))
    return groups


def references_of_shard(path):
    """One reference per row: every `names` entry joined by '|'.

    Non-ColabFit ingests give '<dataset>__<source_file>__frame<N>' plus any native
    id; ColabFit ingests give the upstream configuration labels. Joining keeps
    every element, including OMol25's 'subset:...' tag.

    A few ColabFit datasets carry no names at all (Cu_FHI-aims_NPJCM_2021 has
    empty lists for every row); those fall back to the configuration_id, which is
    itself the upstream ColabFit coordinate, so no row is left without a
    reference.
    """
    table = pq.read_table(path, columns=["names", "configuration_id"])
    names = table.column("names").combine_chunks()
    ids = table.column("configuration_id").combine_chunks()
    out = []
    for value, identifier in zip(names, ids):
        joined = "|".join(value.as_py() or [])
        out.append((joined or identifier.as_py() or "").encode())
    return out


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def build(group, shards, apply):
    path = f"{RELEASE}/collection/{group}"
    ns = int(np.load(f"{path}/ns.npy")[0])
    if sum(rows for _, rows in shards) != ns:
        return {"group": group, "status": "SKIPPED row mismatch",
                "manifest_rows": sum(r for _, r in shards), "release_ns": ns}

    if not apply:
        total = count = longest = 0
        for shard, _ in shards:
            for value in references_of_shard(shard):
                total += len(value)
                longest = max(longest, len(value))
                count += 1
        if count != ns:
            return {"group": group, "status": "SKIPPED count mismatch",
                    "counted": count, "release_ns": ns}
        return {"group": group, "status": "dry-run", "ns": ns,
                "bytes": total + 8 * (ns + 1), "longest": longest}

    bytes_path = f"{path}/source_reference_bytes.bin"
    offsets_path = f"{path}/source_reference_offsets.npy"
    offsets = np.lib.format.open_memmap(f"{offsets_path}.new", mode="w+",
                                        dtype=np.int64, shape=(ns + 1,))
    position, row = 0, 0
    offsets[0] = 0
    with open(f"{bytes_path}.new", "wb") as out:
        for shard, _ in shards:
            values = references_of_shard(shard)
            out.write(b"".join(values))
            for value in values:
                position += len(value)
                row += 1
                offsets[row] = position
    if row != ns:
        os.remove(f"{bytes_path}.new")
        os.remove(f"{offsets_path}.new")
        return {"group": group, "status": "SKIPPED count mismatch",
                "counted": row, "release_ns": ns}

    offsets.flush()
    del offsets
    os.replace(f"{bytes_path}.new", bytes_path)
    os.replace(f"{offsets_path}.new", offsets_path)

    manifest_path = f"{path}/MANIFEST.json"
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    manifest["files"]["source_reference_bytes.bin"] = {
        "bytes": os.path.getsize(bytes_path), "sha256": sha256_of(bytes_path),
        "shape": [position], "dtype": "|u1"}
    manifest["files"]["source_reference_offsets.npy"] = {
        "bytes": os.path.getsize(offsets_path), "sha256": sha256_of(offsets_path),
        "shape": [ns + 1], "dtype": "<i8"}
    with open(f"{manifest_path}.new", "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(f"{manifest_path}.new", manifest_path)

    return {"group": group, "status": "applied", "ns": ns,
            "bytes": position + 8 * (ns + 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--only", default=None,
                        help="comma-separated group names to rebuild")
    args = parser.parse_args()

    groups = shards_by_group()
    selected = sorted(groups)
    if args.only:
        wanted = set(args.only.split(","))
        selected = [g for g in selected if g in wanted]
    names = selected[args.task_id::args.num_tasks]

    results = []
    for index, group in enumerate(names):
        try:
            results.append(build(group, groups[group], args.apply))
        except Exception as error:
            results.append({"group": group, "status": f"EXCEPTION {error!r}"})
        print(f"task {args.task_id}: {index + 1}/{len(names)} {group} "
              f"{results[-1].get('status')}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/source_{args.task_id:04d}.json", "w") as fh:
        json.dump(results, fh, indent=1)

    bad = [r for r in results if r.get("status") not in ("dry-run", "applied")]
    print(f"task {args.task_id}: {len(results)} groups, {len(bad)} problems", flush=True)


if __name__ == "__main__":
    main()
