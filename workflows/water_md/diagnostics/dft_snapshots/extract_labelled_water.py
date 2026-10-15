import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VALID = HERE.parents[3]
FINAL = VALID / "dft" / "lda_pbe" / "final"
SHARED = VALID / "experiments" / "delta_lda_pbe_18k" / "manifests" / "xyz"
OUT = HERE / "existing_labels"

ARMS = ("random_25k", "random_test_6k")
FUNCTIONALS = ("lda", "pbe")
MIN_ATOMS = 24
AMU_G = 1.66053906660e-24


def frames(path):
    with open(path, "rb") as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            n = int(line)
            header = fh.readline().decode()
            rows = [fh.readline().decode().split() for _ in range(n)]
            yield n, header, rows


def cid_of(header):
    return re.search(r'configuration_id="?([^" ]+)"?', header).group(1)


def main():
    shared_ids = set()
    for name in ("train", "val", "test_madcore", "test_random"):
        for _, header, _ in frames(SHARED / f"{name}.xyz"):
            shared_ids.add((name, cid_of(header)))

    record = {}
    for functional in FUNCTIONALS:
        kept = []
        for arm in ARMS:
            for n, header, rows in frames(FINAL / arm / f"{functional}.xyz"):
                species = {r[0] for r in rows}
                if not species <= {"H", "O"} or n < MIN_ATOMS:
                    continue
                cid = cid_of(header)
                n_h = sum(1 for r in rows if r[0] == "H")
                lattice = re.search(r'Lattice="([^"]+)"', header)
                cell = np.array(lattice.group(1).split(), dtype=float).reshape(3, 3) if lattice else None
                mass_g = (n_h * 1.00794 + (n - n_h) * 15.9994) * AMU_G
                density = mass_g / (abs(np.linalg.det(cell)) * 1e-24) if cell is not None else None
                header_out = header.strip()
                header_out = re.sub(r'\barm="[^"]*"', f'arm="{arm}"', header_out)
                kept.append((n, header_out, rows))
                info = record.setdefault(cid, {"arm": arm, "natoms": n, "n_h": n_h, "n_o": n - n_h, "periodic": cell is not None,
                                               "density_g_cm3": density, "cell_A": [float(cell[i, i]) for i in range(3)] if cell is not None else None,
                                               "in_shared_sets": sorted(s for s, c in shared_ids if c == cid)})
                info[f"{functional}_free_energy_eV"] = float(re.search(r"free_energy=(\S+)", header).group(1))
        with open(OUT / f"water_{functional}.xyz", "w") as fh:
            for n, header, rows in kept:
                fh.write(f"{n}\n{header}\n")
                fh.write("\n".join(" ".join(r) for r in rows) + "\n")
        print(functional, len(kept), "frames written")

    (OUT / "frames.json").write_text(json.dumps(record, indent=1) + "\n")
    for cid, info in sorted(record.items(), key=lambda kv: -kv[1]["natoms"]):
        print(cid, info["arm"], info["natoms"], "atoms", "periodic" if info["periodic"] else "isolated",
              f"density {info['density_g_cm3']:.3f} g/cm3" if info["density_g_cm3"] else "", "shared sets:", info["in_shared_sets"])


if __name__ == "__main__":
    main()
