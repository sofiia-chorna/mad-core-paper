import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols

TEST = Path(__file__).resolve().parent.parent
VALID = TEST.parent
sys.path.insert(0, str(VALID / "delta_pbe_10k_vs_mad16" / "scripts"))

from delta_common import read_tsv, soft_hash, write_tsv

R2SCAN_XYZ = TEST / "results" / "test_1000_r2scan.xyz"
PBE_XYZ = TEST / "bundle" / "pbe_labels" / "test_1000_pbe.xyz"
STATUS_TSV = TEST / "manifests" / "test_1000_status.tsv"
XYZ_DIR = TEST / "manifests" / "xyz"
SET_TAG = "test_r2scan_1000"

DELTA_MANIFESTS = VALID / "delta_pbe_10k_vs_mad16" / "manifests"
DISJOINT_FROM = ("madcore_10k", "mad16_10k", "val", "test_mad16", "holdout_madcore")

PROPERTIES = "species:S:1:pos:R:3:forces:R:3:forces_pbe:R:3:delta_forces:R:3"
POS_TOL = 1e-5
VACUUM_MIN = 20.0
FORCE_CUT = 100.0


def read_frames(path):
    frames = {}
    with open(path, "rb") as fh:
        while True:
            line = fh.readline()
            if not line:
                return frames

            natoms = int(line)
            header = fh.readline().decode()
            numbers = np.empty(natoms, dtype=np.int64)
            positions = np.empty((natoms, 3))
            forces = np.empty((natoms, 3))
            for i in range(natoms):
                parts = fh.readline().split()
                numbers[i] = chemical_symbols.index(parts[0].decode())
                positions[i] = [float(v) for v in parts[1:4]]
                forces[i] = [float(v) for v in parts[4:7]]

            lattice = re.search(r'Lattice="([^"]*)"', header)
            pbc = re.search(r'pbc="([^"]*)"', header)
            info = {m.group(1): m.group(2).strip('"') for m in re.finditer(r'(\w+)=("[^"]*"|\S+)', header)}
            frames[info["configuration_id"]] = {
                "natoms": natoms, "numbers": numbers, "positions": positions, "forces": forces,
                "lattice": lattice.group(1) if lattice else None, "pbc": pbc.group(1) if pbc else "F F F",
                "energy": float(info["energy"]), "atomization_energy": float(info["atomization_energy"])}


def main():
    XYZ_DIR.mkdir(parents=True, exist_ok=True)

    r2scan = read_frames(R2SCAN_XYZ)
    pbe = read_frames(PBE_XYZ)
    status = read_tsv(STATUS_TSV)
    group_of = {r["configuration_id"]: r["dataset_group"] for r in status}
    rank_of = {r["configuration_id"]: int(r["sample_rank"]) for r in status}
    print(f"r2scan converged frames {len(r2scan)}; pbe frames {len(pbe)}")

    out_xyz = XYZ_DIR / f"{SET_TAG}.xyz"
    rows = []
    pbc_taken_from_r2scan = 0

    with open(out_xyz, "w") as fh:
        for cid in sorted(r2scan, key=lambda c: rank_of[c]):
            r2f = r2scan[cid]
            pf = pbe[cid]

            assert np.array_equal(r2f["numbers"], pf["numbers"]), f"{cid}: species differ"
            assert np.abs(r2f["positions"] - pf["positions"]).max() < POS_TOL, f"{cid}: positions differ"

            if r2f["pbc"] != pf["pbc"]:
                lengths = np.linalg.norm(np.array(r2f["lattice"].split(), float).reshape(3, 3), axis=1)
                differing = [i for i, (a, b) in enumerate(zip(r2f["pbc"].split(), pf["pbc"].split())) if a != b]
                assert all(lengths[i] >= VACUUM_MIN for i in differing), f"{cid}: pbc differ along a non-vacuum axis"
                pbc_taken_from_r2scan += 1

            natoms = r2f["natoms"]
            delta_e = r2f["atomization_energy"] - pf["atomization_energy"]
            delta_f = r2f["forces"] - pf["forces"]
            cell = np.array(r2f["lattice"].split(), float).reshape(3, 3) if r2f["lattice"] else np.zeros((3, 3))
            pbc_flags = np.array([c == "T" for c in r2f["pbc"].split()])
            structure_hash = soft_hash(r2f["numbers"], r2f["positions"], cell, pbc_flags)

            fields = [f'configuration_id="{cid}"', f'hash="{structure_hash}"', f'subset="{group_of.get(cid, "")}"',
                      f'frame_id="{rank_of[cid]}"', 'r2scan_source="lyra_r2scan_1000"', 'pbe_source="madcore-pbe"', 'fps_order="-1"',
                      f"energy_r2scan={r2f['energy']!r}", f"energy_pbe={pf['energy']!r}",
                      f"atomization_energy={r2f['atomization_energy']!r}", f"atomization_energy_pbe={pf['atomization_energy']!r}",
                      f"delta_energy={delta_e!r}"]
            if r2f["lattice"] is not None:
                fields.append(f'Lattice="{r2f["lattice"]}"')
            fields += [f"Properties={PROPERTIES}", f'pbc="{r2f["pbc"]}"']

            fh.write(f"{natoms}\n" + " ".join(fields) + "\n")
            for i in range(natoms):
                fh.write(f"{chemical_symbols[r2f['numbers'][i]]:2s} "
                         + " ".join(f"{v:16.8f}" for v in r2f["positions"][i])
                         + " ".join(f"{v:16.8f}" for v in r2f["forces"][i])
                         + " ".join(f"{v:16.8f}" for v in pf["forces"][i])
                         + " ".join(f"{v:16.8f}" for v in delta_f[i]) + "\n")

            rows.append({"configuration_id": cid, "hash": structure_hash, "subset": group_of.get(cid, ""),
                         "natoms": natoms, "pbc": r2f["pbc"].replace(" ", ""),
                         "n_species": len(set(r2f["numbers"].tolist())),
                         "atomization_e_r2scan": r2f["atomization_energy"], "atomization_e_pbe": pf["atomization_energy"],
                         "delta_e": delta_e,
                         "fmax_r2scan": round(float(np.linalg.norm(r2f["forces"], axis=1).max()), 4),
                         "fmax_pbe": round(float(np.linalg.norm(pf["forces"], axis=1).max()), 4)})

    write_tsv(TEST / "manifests" / f"{SET_TAG}.tsv", rows, list(rows[0]))

    over_cut = [r["configuration_id"] for r in rows if max(r["fmax_r2scan"], r["fmax_pbe"]) > FORCE_CUT]
    hashes = {r["hash"] for r in rows}
    overlaps = {}
    for name in DISJOINT_FROM:
        other = {r["hash"] for r in read_tsv(DELTA_MANIFESTS / f"{name}.tsv")}
        overlaps[name] = len(hashes & other)
        assert not (hashes & other), f"{SET_TAG} overlaps {name}"

    natoms = np.array([r["natoms"] for r in rows], float)
    summary = {"set": SET_TAG, "n_structures": len(rows), "total_atoms": int(natoms.sum()),
               "natoms_mean": round(float(natoms.mean()), 2), "natoms_median": float(np.median(natoms)),
               "natoms_max": int(natoms.max()), "unique_hashes": len(hashes),
               "elements": len({z for r in r2scan.values() for z in r["numbers"].tolist()}),
               "pbc": dict(Counter(r["pbc"] for r in rows)),
               "dataset_groups": dict(Counter(r["subset"] for r in rows)),
               "pbc_taken_from_r2scan_where_pbe_differs": pbc_taken_from_r2scan,
               "structures_over_force_cut": over_cut,
               "overlap_with_delta_sets": overlaps,
               "fmax_r2scan_max": round(max(r["fmax_r2scan"] for r in rows), 4),
               "fmax_pbe_max": round(max(r["fmax_pbe"] for r in rows), 4)}

    with open(TEST / "manifests" / f"{SET_TAG}.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    print(json.dumps(summary, indent=1))
    print(f"wrote {out_xyz}")


if __name__ == "__main__":
    main()
