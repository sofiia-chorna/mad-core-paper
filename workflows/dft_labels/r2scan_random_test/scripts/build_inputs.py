import json
import shutil
from collections import Counter

import numpy as np
from ase.data import atomic_numbers, chemical_symbols

from paths import BUNDLE, LOGS, MANIFESTS, PBE_XYZ, SPECIES_DIR, make_logger, parse_frames, read_tsv, sha256, soft_hash, write_tsv

TEMPLATE_OF_BUCKET = {"isolated_fff": "control_r2scan_isolated_fff.in", "periodic_ttt": "control_r2scan_periodic_ttt.in",
                      "surfaces_ttf": "control_r2scan_surfaces_ttf.in"}
FRAME_COLUMNS = ["sample_rank", "frame_dir", "configuration_id", "dataset_group", "natoms", "n_species", "elements", "pbc", "dft_bucket",
                 "control_template", "geometry_hash", "r2scan_status", "geometry_sha256", "control_sha256"]


def species_block(symbol):
    path = SPECIES_DIR / "{:02d}_{}_default".format(atomic_numbers[symbol], symbol)
    return path.name, path.read_text()


def write_geometry(path, numbers, positions, cell, pbc):
    lines = []
    if pbc.any():
        assert pbc.all(), "only fully periodic or fully isolated frames are expected"
        for v in cell:
            lines.append("lattice_vector {:.12f} {:.12f} {:.12f}".format(*v))
    for z, p in zip(numbers, positions):
        lines.append("atom {:.12f} {:.12f} {:.12f} {}".format(p[0], p[1], p[2], chemical_symbols[int(z)]))
    path.write_text("\n".join(lines) + "\n")


def read_geometry(path):
    cell, numbers, positions = [], [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "lattice_vector":
            cell.append([float(v) for v in parts[1:4]])
        elif parts[0] == "atom":
            positions.append([float(v) for v in parts[1:4]])
            numbers.append(atomic_numbers[parts[4]])
    cell = np.array(cell) if cell else np.zeros((3, 3))
    return np.array(numbers), np.array(positions), cell, np.array([bool(len(cell.nonzero()[0]))] * 3)


def copy_frame_bytes(src, offset, natoms, dst):
    src.seek(offset)
    for _ in range(natoms + 2):
        dst.write(src.readline())


def main():
    say = make_logger(LOGS / "build_inputs.log")
    status = read_tsv(MANIFESTS / "test_1000_status.tsv")
    by_idx = {int(r["pbe_frame_idx"]): r for r in status}
    templates = {k: (BUNDLE / "control_templates" / v).read_text() for k, v in TEMPLATE_OF_BUCKET.items()}
    inputs = BUNDLE / "inputs"
    if inputs.exists():
        shutil.rmtree(inputs)
    inputs.mkdir()

    frames_out = []
    species_used = Counter()
    labels = open(BUNDLE / "pbe_labels" / "test_1000_pbe.xyz", "wb")
    raw = open(PBE_XYZ, "rb")
    n_label_frames = 0
    for idx, (offset, info, numbers, positions, cell, pbc) in enumerate(parse_frames(PBE_XYZ)):
        if idx not in by_idx:
            continue
        r = by_idx[idx]
        assert info["configuration_id"] == r["configuration_id"]
        assert soft_hash(numbers, positions, cell, pbc) == r["geometry_hash"]
        assert (r["dft_bucket"] == "isolated_fff") == (not pbc.any()), r["configuration_id"]
        copy_frame_bytes(raw, offset, len(numbers), labels)
        n_label_frames += 1

        row = {k: r[k] for k in ("sample_rank", "configuration_id", "dataset_group", "natoms", "n_species", "elements", "pbc", "dft_bucket",
                                 "geometry_hash", "r2scan_status")}
        row["frame_dir"] = "inputs/frame_{:04d}".format(int(r["sample_rank"]))
        row["control_template"] = TEMPLATE_OF_BUCKET[r["dft_bucket"]]
        if r["r2scan_status"] != "required":
            frames_out.append(row)
            continue

        frame_dir = BUNDLE / row["frame_dir"]
        frame_dir.mkdir()
        write_geometry(frame_dir / "geometry.in", numbers, positions, cell, pbc)
        n2, p2, c2, pbc2 = read_geometry(frame_dir / "geometry.in")
        assert np.array_equal(n2, numbers) and np.abs(p2 - positions).max() < 1e-9 and np.abs(c2 - cell).max() < 1e-9 and np.array_equal(pbc2, pbc)

        blocks = []
        for symbol in sorted({chemical_symbols[int(z)] for z in numbers}, key=lambda s: atomic_numbers[s]):
            name, text = species_block(symbol)
            species_used[name] += 1
            blocks.append(text)
        (frame_dir / "control.in").write_text(templates[r["dft_bucket"]] + "\n" + "\n".join(blocks))
        (frame_dir / "frame.json").write_text(json.dumps({"sample_rank": int(r["sample_rank"]), "configuration_id": r["configuration_id"],
                                                          "natoms": int(r["natoms"]), "pbc": r["pbc"], "dft_bucket": r["dft_bucket"],
                                                          "geometry_hash": r["geometry_hash"]}, indent=1))
        row["geometry_sha256"] = sha256(frame_dir / "geometry.in")
        row["control_sha256"] = sha256(frame_dir / "control.in")
        frames_out.append(row)
        if len(frames_out) % 200 == 0:
            say(f"  {len(frames_out)} frames")
    labels.close()
    raw.close()

    frames_out.sort(key=lambda r: int(r["sample_rank"]))
    write_tsv(BUNDLE / "manifests" / "frames.tsv", frames_out, FRAME_COLUMNS)
    required = [r for r in frames_out if r["r2scan_status"] == "required"]
    say(f"frames listed: {len(frames_out)}, input directories written: {len(required)}, PBE label frames copied: {n_label_frames}")
    say(f"species files used: {len(species_used)}; buckets: {dict(Counter(r['dft_bucket'] for r in required))}")
    with open(LOGS / "build_inputs.json", "w") as fh:
        json.dump({"frames": len(frames_out), "required": len(required), "species_used": dict(species_used),
                   "buckets": dict(Counter(r["dft_bucket"] for r in required))}, fh, indent=1)


if __name__ == "__main__":
    main()
