import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from ase.data import chemical_symbols

HERE = Path(__file__).resolve().parent

FAMILIES = {
    "hydrogen": {"H"},
    "s-block metals": set("Li Na K Rb Cs Fr Be Mg Ca Sr Ba Ra".split()),
    "p-block elements": set("He B C N O F Ne Al Si P S Cl Ar Ga Ge As Se Br Kr In Sn Sb Te I Xe "
                            "Tl Pb Bi Po At Rn".split()),
    "transition metals": set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
                             "Hf Ta W Re Os Ir Pt Au Hg".split()),
    "lanthanides/actinides": set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
                                 "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split()),
}
FAM_ORDER = ["hydrogen", "s-block metals", "p-block elements", "transition metals",
             "lanthanides/actinides"]
CAT_ORDER = ["3D bulk", "molecules", "clusters", "surfaces/2D", "molecular crystals",
             "solvated", "atom"]


def write(name, header, rows):
    with open(HERE / "data" / name, "w") as fh:
        fh.write("\t".join(header) + "\n")

        for row in rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    print(f"wrote data/{name}")


def main():
    parser = argparse.ArgumentParser(description="rebuild data/ from the large source files")
    parser.add_argument("--store-root", required=True,
                        help="root of the aa002 mad-core store, the directory holding release/ and accrete/")
    parser.add_argument("--labels", choices=["run", "release"], default="release")
    args = parser.parse_args()
    run = f"{args.store_root}/accrete/phace_atomic_ep92/coreset224/out"

    # the light parquet has the same non feature columns as the full one
    info = f"{run}/coreset_224_light.parquet"
    categories = f"{run}/picks_category_map_224.json"
    release_groups = f"{args.store_root}/release/mad-core/dataset_group.npy"

    t = pq.read_table(info, columns=["fps_order", "epsilon_sq", "family",
                                     "configuration_id", "elements"])
    order = np.argsort(t.column("fps_order").to_numpy())
    n = len(order)
    fam = np.asarray(t.column("family").to_pylist())[order]
    eps = np.sqrt(np.clip(t.column("epsilon_sq").to_numpy()[order], 0, None))
    cids = t.column("configuration_id").to_pylist()
    cids = [cids[i] for i in order]
    els = t.column("elements").to_pylist()
    els = [els[i] for i in order]
    del t

    if args.labels == "release":
        fam = np.load(release_groups).astype(str)
        assert fam.shape == (n,), fam.shape
    ks = [1 << e for e in range(1, 25)]
    assert ks[-1] == n, n

    groups, codes = np.unique(fam, return_inverse=True)
    group_rows = []

    for g, name in enumerate(groups):
        cum = np.cumsum(codes == g)
        group_rows += [(k, name, int(cum[k - 1])) for k in ks]
    write("group_counts.tsv", ["k_set_size", "group", "structures"], group_rows)
    del codes

    print("categories...", flush=True)
    cat_map = json.load(open(categories))
    cat = np.asarray([cat_map[c] for c in cids])
    del cat_map, cids
    present = [c for c in CAT_ORDER if (cat == c).any()]
    ccum = {c: np.cumsum(cat == c) for c in present}
    write("category_counts.tsv", ["k_set_size"] + present,
          [[k] + [int(ccum[c][k - 1]) for c in present] for k in ks])

    print("element families...", flush=True)
    z_of = {s: z for z, s in enumerate(chemical_symbols)}
    fam_of_z = {z_of[s]: FAM_ORDER.index(f) for f, syms in FAMILIES.items() for s in syms

                if s in z_of}

    # each distinct element of a structure counts once for its family
    per = np.zeros((n, len(FAM_ORDER)), dtype=np.int32)

    for i, e in enumerate(els):
        for z in set(int(z) for z in e):
            f = fam_of_z.get(z)

            if f is not None:
                per[i, f] += 1
    elfam_cum = np.cumsum(per, axis=0)
    write("element_family_counts.tsv", ["k_set_size"] + FAM_ORDER,
          [[k] + [int(v) for v in elfam_cum[k - 1]] for k in ks])

    write("coverage_radius.tsv", ["k_set_size", "epsilon"],
          [(k, repr(float(eps[k - 1]))) for k in ks])


if __name__ == "__main__":
    main()
