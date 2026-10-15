import argparse
import gzip
import json
import os
from collections import Counter

import chemiscope
import h5py
import numpy as np
import torch
import torch.nn as nn
from chemiscope.structures import structures_to_json

from export_xyz import DOI, FEATURE_DIM, SOURCE_RELEASE, index_path, interval_name, intervals, label, sha256_file

TOP_GROUPS = 20
GZIP_LEVEL = 6
PROJECT_CHUNK = 1 << 16
LOD_ROWS = 1 << 18
PROJECTOR_FILE = SOURCE_RELEASE + "/ecumetric_sketchmap.pt"
PROJECTOR_SPEC = SOURCE_RELEASE + "/ecumetric_sketchmap.json"
META_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "chemiscope_meta.md")

HEADER_KEYS = ("configuration_id", "dataset_group", "ecumetric_energy", "fps_order",
               "k_set", "voronoi_population")

PROPERTY_NOTES = {
    "k_set": "smallest k for which the structure is within the first 2^k rows, that is "
             "max(1, ceil(log2(fps_order + 1))); rows 0 and 1 are both level 1",
    "fps_order": "global rank of the structure in the coreset, also its row in every record file",
    "log10 voronoi_population": "log10 of how many MAD-COSMO structures have this structure as "
                                "their nearest neighbour among all 2^24 coreset members",
    "dataset_group": f"source dataset group, the {TOP_GROUPS} largest kept and the rest as other",
    "ecumetric_energy per atom": "model-predicted energy divided by the number of atoms, not a DFT value",
    "configuration_id": "unique structure id, the same id as in the extended xyz files",
    "chemical_formula": "chemical formula in Hill order, carbon and hydrogen first then alphabetical",
}


def read_meta(rows):
    sections, current = {}, None

    for line in open(META_FILE):
        if line.startswith("# "):
            current = line[2:].strip()
            sections[current] = []
        elif line.strip():
            sections[current].append(line.strip())

    missing = {"title", "description"} - set(sections)
    assert not missing, f"{META_FILE} is missing the {missing} section"

    fields = {"structures": f"{rows:,}", "level": rows.bit_length() - 1}

    return (" ".join(sections["title"]).format(**fields),
            " ".join(sections["description"]).format(**fields))


def covering_intervals(rows):
    chosen = []
    for lo, hi in intervals():
        if lo >= rows:
            break
        chosen.append((lo, hi, min(hi, rows) - lo))
    return chosen


def chemical_formula(names):
    counts = Counter(names)

    ordered = [symbol for symbol in ("C", "H") if symbol in counts]
    ordered += sorted(symbol for symbol in counts if symbol not in ("C", "H"))

    return "".join(symbol + (str(counts[symbol]) if counts[symbol] > 1 else "")
                   for symbol in ordered)


def parse_header(line):
    info, index, length = {}, 0, len(line)

    while index < length:
        while index < length and line[index] == " ":
            index += 1

        equals = line.find("=", index)
        if equals < 0:
            break

        key = line[index:equals]
        if equals + 1 < length and line[equals + 1] == '"':
            end = line.find('"', equals + 2)
            info[key] = line[equals + 2:end]
            index = end + 1
        else:
            end = line.find(" ", equals + 1)
            end = length if end < 0 else end
            info[key] = line[equals + 1:end]
            index = end

    return info


def read_structures(path, limit):
    structures, headers = [], []

    with gzip.open(path, "rt") as stream:
        lines = stream.read().split("\n")

    index = 0
    while len(structures) < limit:
        count = int(lines[index])
        info = parse_header(lines[index + 1])

        names, x, y, z = [], [], [], []
        for atom in lines[index + 2:index + 2 + count]:
            species, px, py, pz = atom.split(None, 4)[:4]
            names.append(species)
            x.append(float(px))
            y.append(float(py))
            z.append(float(pz))

        structure = {"size": count, "names": names, "x": x, "y": y, "z": z}
        if "Lattice" in info:
            structure["cell"] = [float(value) for value in info["Lattice"].split()]

        structures.append(structure)
        headers.append({key: info[key] for key in HEADER_KEYS})
        index += 2 + count

    return structures, headers


def check_parser(path, frames):
    from ase.io import read

    reference = read(path, index=f":{frames}", format="extxyz")
    expected = structures_to_json(reference)
    structures, headers = read_structures(path, frames)

    assert structures == expected, "the fast reader disagrees with ase plus chemiscope"
    for atoms, structure, header in zip(reference, structures, headers):
        for key in HEADER_KEYS:
            assert str(atoms.info[key]) == header[key], (key, atoms.info[key], header[key])
        formula = chemical_formula(structure["names"])
        assert formula == atoms.get_chemical_formula(), (formula, atoms.get_chemical_formula())

    return frames


def load_projector():
    checkpoint = torch.load(PROJECTOR_FILE, map_location="cpu", weights_only=False)
    activation = {"relu": nn.ReLU, "silu": nn.SiLU}[checkpoint["activation"]]

    layers, width = [], FEATURE_DIM
    for hidden_width in checkpoint["hidden"]:
        layers += [nn.Linear(width, hidden_width), activation()]
        width = hidden_width
    layers.append(nn.Linear(width, checkpoint["d"]))

    model = nn.Sequential(*layers)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, checkpoint


def sketchmap_coordinates(files_dir, ranges):
    model, checkpoint = load_projector()
    feature_mean = torch.as_tensor(checkpoint["feature_mean"])
    feature_scale = torch.as_tensor(checkpoint["feature_scale"])
    output_mean = torch.as_tensor(np.asarray(checkpoint["projection_mean"], np.float32))
    output_scale = torch.as_tensor(np.asarray(checkpoint["projection_scale"], np.float32))

    blocks = []
    with torch.no_grad():
        for lo, hi, take in ranges:
            path = os.path.join(files_dir, f"madcore_features_{interval_name(lo, hi)}.h5")

            with h5py.File(path, "r") as fh:
                scaler_mean = torch.as_tensor(fh["mean"][:])
                scaler_scale = float(fh["scale"][()])

                for start in range(0, take, PROJECT_CHUNK):
                    rows = min(PROJECT_CHUNK, take - start)
                    block = torch.as_tensor(fh["features"][start:start + rows])
                    block = (block - scaler_mean) / scaler_scale
                    block = (block - feature_mean) / feature_scale
                    blocks.append((model(block) * output_scale + output_mean).numpy())

    return np.concatenate(blocks)


def build_properties(structures, headers, coordinates):
    groups = [info["dataset_group"] for info in headers]
    frequent = {name for name, _ in Counter(groups).most_common(TOP_GROUPS)}

    natoms = np.array([structure["size"] for structure in structures], dtype=np.int64)
    energy = np.array([float(info["ecumetric_energy"]) for info in headers], dtype=np.float64)
    population = np.array([int(info["voronoi_population"]) for info in headers], dtype=np.int64)

    values = {
        "k_set": [int(info["k_set"]) for info in headers],
        "fps_order": [int(info["fps_order"]) for info in headers],
        "log10 voronoi_population": np.log10(np.maximum(population, 1)).tolist(),
        "dataset_group": [name if name in frequent else "other" for name in groups],
        "ecumetric_energy per atom": (energy / natoms).tolist(),
        "configuration_id": [info["configuration_id"] for info in headers],
        "chemical_formula": [chemical_formula(structure["names"]) for structure in structures],
    }

    properties = {
        f"smap {axis + 1}": {
            "target": "structure",
            "values": coordinates[:, axis].tolist(),
            "description": f"sketch-map dimension {axis + 1} of 8, from the d = 8 sketch-map "
                           "projector fitted on the level-20 coreset, applied to the features "
                           "in this record",
        }
        for axis in range(coordinates.shape[1])
    }

    for name, column in values.items():
        properties[name] = {"target": "structure", "values": column,
                            "description": PROPERTY_NOTES[name]}
    properties["ecumetric_energy per atom"]["units"] = "ev/atom"

    return properties


def build_settings(rows):
    settings = {
        "target": "structure",
        "map": {
            "x": {"property": "smap 1", "scale": "linear"},
            "y": {"property": "smap 2", "scale": "linear"},
            "z": {"property": "", "scale": "linear"},
            "symbol": "dataset_group",
            "color": {"property": "dataset_group"},
            "size": {"property": "log10 voronoi_population", "factor": 25, "mode": "linear"},
            "markerOutline": False,
            "joinPoints": False,
        },
        "pinned": [0],
        "structure": [{
            "bonds": True, "atoms": True, "spaceFilling": False, "atomLabels": False,
            "shape": "", "unitCell": True, "rotation": False, "cartoon": False,
            "supercell": {"0": 1, "1": 1, "2": 1}, "axes": "off", "keepOrientation": False,
            "playbackDelay": 700,
            "environments": {"activated": False, "bgColor": "grey", "bgStyle": "ball-stick",
                             "center": False, "cutoff": 0},
            "color": {"property": "element", "min": 0, "max": 0, "transform": "linear", "palette": "bwr"},
        }],
    }

    if rows > LOD_ROWS:
        settings["map"]["useLOD"] = True

    return settings


def write_one(out_dir, rows, structures, headers, coordinates, spec):
    title, description = read_meta(rows)
    properties = build_properties(structures[:rows], headers[:rows], coordinates[:rows])

    data = chemiscope.create_input(structures=structures[:rows], properties=properties,
                                   settings=build_settings(rows),
                                   metadata={"name": title, "description": description,
                                             "references": [f"https://doi.org/{DOI}"]})
    payload = json.dumps(data).encode()

    path = os.path.join(out_dir, f"madcore_0-{label(rows)}.chemiscope.json.gz")
    with gzip.GzipFile(path, "wb", compresslevel=GZIP_LEVEL, mtime=0) as fh:
        fh.write(payload)

    name = os.path.basename(path)
    entry = {"family": "chemiscope", "rows": [0, rows],
             "bytes": os.path.getsize(path), "sha256": sha256_file(path)}

    index_file = index_path(out_dir)
    if os.path.exists(index_file):
        index = json.load(open(index_file))
        index["files"][name] = entry
        with open(index_file, "w") as fh:
            json.dump(index, fh, indent=1, sort_keys=True)

    print(f"{name}: {entry['bytes'] / 1e6:.1f} MB gzipped, {len(payload) / 1e6:.1f} MB of json, "
          f"{rows:,} structures, {int(sum(s['size'] for s in structures[:rows])):,} atoms, "
          f"{len(properties)} properties, projector {spec['tag']}", flush=True)

    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", required=True)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    out_dir = args.out or args.files
    sizes = sorted(args.rows)
    ranges = covering_intervals(sizes[-1])
    assert sum(take for _, _, take in ranges) == sizes[-1], ranges

    structures, headers = [], []
    for lo, hi, take in ranges:
        name = f"madcore_{interval_name(lo, hi)}.extxyz.gz"
        path = os.path.join(args.files, name)
        print(f"checking the header parser against ase on {check_parser(path, 200)} frames of {name}",
              flush=True)

        block_structures, block_headers = read_structures(path, take)
        structures.extend(block_structures)
        headers.extend(block_headers)
        print(f"read {name}: {len(structures):,} frames so far", flush=True)

    orders = np.array([int(info["fps_order"]) for info in headers], dtype=np.int64)
    assert np.array_equal(orders, np.arange(sizes[-1])), "fps_order is not the row index"

    coordinates = sketchmap_coordinates(args.files, ranges)
    assert len(coordinates) == sizes[-1], (len(coordinates), sizes[-1])
    print(f"projected {len(coordinates):,} rows to {coordinates.shape[1]} sketch-map dimensions",
          flush=True)

    spec = json.load(open(PROJECTOR_SPEC))
    for rows in sizes:
        write_one(out_dir, rows, structures, headers, coordinates, spec)


if __name__ == "__main__":
    main()
