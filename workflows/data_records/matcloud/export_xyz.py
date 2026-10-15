import argparse
import datetime
import gzip
import hashlib
import json
import multiprocessing
import os
import shutil

import h5py
import numpy as np

SOURCE_RELEASE = "${STORE_ROOT}/mad-core/release"
SOURCE_MADCORE = SOURCE_RELEASE + "/mad-core"
SOURCE_EXTXYZ = SOURCE_MADCORE + "/extxyz"
CATALOGUE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "dataset_metadata.csv")

EXPORT_VERSION = "xyz-2pow24-v1"
DOI = "10.24435/materialscloud:rs-tt"
FEATURE_DIM = 512
CHUNK_ROWS = 256
TOTAL_ROWS = 16777216
FIRST_INTERVAL_ROWS = 262144
GZIP_LEVEL = 6

HEADER_COLUMNS = ["epsilon_sq", "k_set", "voronoi_population",
                  "zbl_max_force", "zbl_energy_per_atom", "zbl_max_atomic_energy"]

# the 67,975 structures with non-finite features were placed in cell 0 and are not in MAD-COSMO
VORONOI_POPULATION = {0: 5272}

def intervals():
    edges = [0, FIRST_INTERVAL_ROWS]
    while edges[-1] < TOTAL_ROWS:
        edges.append(edges[-1] * 2)
    return list(zip(edges, edges[1:]))


def label(row):
    return "0" if row == 0 else f"2pow{row.bit_length() - 1}"


def interval_name(lo, hi):
    return f"{label(lo)}-{label(hi)}"


# index.json sits next to the files folder, it is kept with the record but not uploaded
def index_path(files_dir):
    return os.path.join(os.path.dirname(os.path.abspath(files_dir)), "index.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 26), b""):
            digest.update(block)
    return digest.hexdigest()


EPSILON_OVERRIDE = os.environ.get("MADCORE_EPSILON_SQ")


def release_array(name, lo, hi):
    if name == "epsilon_sq.npy" and EPSILON_OVERRIDE:
        return np.asarray(np.load(EPSILON_OVERRIDE, mmap_mode="r")[lo:hi])
    return np.asarray(np.load(f"{SOURCE_MADCORE}/{name}", mmap_mode="r")[lo:hi])


def source_references(lo, hi):
    offsets = np.asarray(np.load(SOURCE_MADCORE + "/source_reference_offsets.npy", mmap_mode="r")[lo:hi + 1])
    with open(SOURCE_MADCORE + "/source_reference_bytes.bin", "rb") as fh:
        fh.seek(int(offsets[0]))
        blob = fh.read(int(offsets[-1] - offsets[0]))
    start = offsets - offsets[0]
    return [blob[start[i]:start[i + 1]] for i in range(len(start) - 1)]


def header_suffixes(lo, hi):
    columns = {key: release_array(key + ".npy", lo, hi) for key in HEADER_COLUMNS}

    # the release counts what MAD-COSMO left out, the record counts only what it holds
    populations = np.array(columns["voronoi_population"])
    for row, value in VORONOI_POPULATION.items():
        if lo <= row < hi:
            populations[row - lo] = value
    columns["voronoi_population"] = populations

    references = source_references(lo, hi)
    suffixes = []
    for row in range(hi - lo):
        parts = [f"{key}={columns[key][row]}" for key in columns]
        suffixes.append(" " + " ".join(parts) + " source_reference=" + references[row].decode())
    assert not any(" " in reference.decode() for reference in references[:1000]), "source_reference must not contain spaces"
    return [suffix.encode() for suffix in suffixes]


def append_headers(data, suffixes, first):
    lines = data.split(b"\n")
    out = []
    index, frame = 0, 0
    while index + 1 < len(lines) and lines[index]:
        count = int(lines[index])
        out.append(lines[index])
        out.append(lines[index + 1] + suffixes[first + frame])
        out.extend(lines[index + 2:index + 2 + count])
        index += 2 + count
        frame += 1
    return b"\n".join(out) + b"\n", frame


def chunks_for(manifest, lo, hi):
    inside = sorted((v["rows"], k) for k, v in manifest["files"].items() if v["rows"][0] >= lo and v["rows"][1] <= hi)
    covered = [rows for rows, _ in inside]
    assert covered[0][0] == lo and covered[-1][1] == hi, (lo, hi, covered[0], covered[-1])
    assert all(a[1] == b[0] for a, b in zip(covered, covered[1:])), "gap between release chunks"
    return inside


def write_xyz(out_dir, manifest, lo, hi, suffixes):
    path = os.path.join(out_dir, f"madcore_{interval_name(lo, hi)}.extxyz.gz")
    frames = 0
    with gzip.GzipFile(path, "wb", compresslevel=GZIP_LEVEL, mtime=0) as dst:
        for rows, chunk in chunks_for(manifest, lo, hi):
            with open(os.path.join(SOURCE_EXTXYZ, chunk), "rb") as src:
                text, written = append_headers(src.read(), suffixes, rows[0] - lo)
            assert written == rows[1] - rows[0], (chunk, written)
            dst.write(text)
            frames += written
    assert frames == hi - lo, (lo, hi, frames)
    return path


def write_features(out_dir, lo, hi):
    path = os.path.join(out_dir, f"madcore_features_{interval_name(lo, hi)}.h5")
    scaler = np.load(SOURCE_RELEASE + "/ecumetric_scaler.npz")
    rows = hi - lo
    with open(SOURCE_MADCORE + "/ecumetric_features.bin", "rb") as src, h5py.File(path, "w") as fh:
        src.seek(lo * FEATURE_DIM * 4)
        dataset = fh.create_dataset("features", shape=(rows, FEATURE_DIM), dtype=np.float32,
                                    chunks=(CHUNK_ROWS, FEATURE_DIM), compression="gzip",
                                    compression_opts=4, shuffle=True)
        block_rows = 1 << 16
        for start in range(0, rows, block_rows):
            take = min(block_rows, rows - start)
            buffer = np.frombuffer(src.read(take * FEATURE_DIM * 4), dtype=np.float32).reshape(take, FEATURE_DIM)
            dataset[start:start + take] = buffer
        fh.create_dataset("mean", data=scaler["mean"].astype(np.float32))
        fh.create_dataset("scale", data=np.float32(scaler["scale"]))
        fh.attrs["rows"] = [lo, hi]
        fh.attrs["row_order"] = "row i = i-th FPS pick, matches frame i of the matching extxyz file"
    return path


def write_image_flags(out_dir):
    path = os.path.join(out_dir, "image_flags.npz")
    rows = np.load(SOURCE_MADCORE + "/image_flag_rows.npy")
    na = np.load(SOURCE_MADCORE + "/na.npy", mmap_mode="r")
    counts = np.array([na[row + 1] - na[row] for row in rows], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    flags = np.fromfile(SOURCE_MADCORE + "/image_flags.bin", dtype=np.int64).reshape(-1, 3)
    cells = np.fromfile(SOURCE_MADCORE + "/image_flag_cells.bin", dtype=np.float64).reshape(-1, 3, 3)
    assert len(flags) == offsets[-1] and len(cells) == len(rows)
    np.savez(path, rows=rows, offsets=offsets, flags=flags, cells=cells)
    return path


def build_interval(job):
    out_dir, manifest, lo, hi, xyz_only = job
    suffixes = header_suffixes(lo, hi)
    outputs = [(write_xyz(out_dir, manifest, lo, hi, suffixes), "xyz")]
    if not xyz_only:
        outputs.append((write_features(out_dir, lo, hi), "features"))
    entries = {}
    for path, family in outputs:
        entries[os.path.basename(path)] = {"family": family, "rows": [lo, hi],
                                           "bytes": os.path.getsize(path), "sha256": sha256_file(path)}
    sizes = ", ".join(f"{v['bytes'] / 1e9:.2f} GB" for v in entries.values())
    print(f"{interval_name(lo, hi)} done: {sizes}", flush=True)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--processes", type=int, default=7)
    parser.add_argument("--xyz-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest = json.load(open(SOURCE_EXTXYZ + "/MANIFEST.json"))
    ranges = intervals()
    assert ranges[0][0] == 0 and ranges[-1][1] == TOTAL_ROWS
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))

    jobs = [(args.out, manifest, lo, hi, args.xyz_only) for lo, hi in ranges]
    with multiprocessing.Pool(min(args.processes, len(jobs))) as pool:
        results = pool.map(build_interval, jobs)

    files = {}
    for entries in results:
        files.update(entries)

    # the other files of the record are unchanged, their entries are merged into index.json later
    if args.xyz_only:
        with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "xyz_entries.json"), "w") as fh:
            json.dump(files, fh, indent=1, sort_keys=True)
        print(f"xyz export complete: {len(files)} files", flush=True)
        return

    shutil.copyfile(CATALOGUE, os.path.join(args.out, "dataset_metadata.csv"))
    for name in ("dataset_metadata.csv", os.path.basename(write_image_flags(args.out))):
        path = os.path.join(args.out, name)
        files[name] = {"family": "global", "bytes": os.path.getsize(path), "sha256": sha256_file(path)}

    index = {
        "record": {"name": "MAD-CORE", "export_version": EXPORT_VERSION, "doi": DOI,
                   "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
                   "rows": [0, TOTAL_ROWS]},
        "files": files,
    }
    with open(index_path(args.out), "w") as fh:
        json.dump(index, fh, indent=1, sort_keys=True)

    total = sum(entry["bytes"] for entry in files.values())
    print(f"export complete: {len(files) + 1} files, {total / 1e9:.2f} GB, rows [0, {TOTAL_ROWS})", flush=True)


if __name__ == "__main__":
    main()
