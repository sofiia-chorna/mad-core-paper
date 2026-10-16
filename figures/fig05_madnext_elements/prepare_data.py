from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from ase.data import chemical_symbols

HERE = Path(__file__).resolve().parent

import argparse

parser = argparse.ArgumentParser(description="rebuild data/ from the large source files")
parser.add_argument("--store-root", required=True,
                    help="root of the aa002 mad-core store, the directory holding release/, mad_union/ and accrete/")
STORE = parser.parse_args().store_root
INFO = f"{STORE}/mad_union/mad_union_full.parquet"
MAX_Z = 118

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


def main():
    t = pq.read_table(INFO, columns=["fps_order", "elements"])
    order = np.argsort(t.column("fps_order").to_numpy())
    els = t.column("elements").to_pylist()
    els = [els[i] for i in order]
    n = len(els)
    ks = [1 << e for e in range(1, int(np.floor(np.log2(n))) + 1)] + [n]
    z_of = {s: z for z, s in enumerate(chemical_symbols)}
    fz = {z_of[s]: fam for fam, syms in FAMILIES.items() for s in syms if s in z_of}
    fam_idx = {f: i for i, f in enumerate(FAM_ORDER)}

    # each DISTINCT element of a structure contributes one count to its family
    per = np.zeros((n, len(FAM_ORDER)), dtype=np.int32)
    for i, e in enumerate(els):
        for z in e:
            f = fz.get(int(z))
            if f is not None:
                per[i, fam_idx[f]] += 1
    cum = np.cumsum(per, axis=0)
    with open(HERE / "data" / "element_family_counts.tsv", "w") as fh:
        fh.write("k_set_size\t" + "\t".join(FAM_ORDER) + "\n")
        for k in ks:
            fh.write(f"{k}\t" + "\t".join(str(int(v)) for v in cum[k - 1]) + "\n")

    counts = {}
    for e in els:
        for z in set(int(x) for x in e):
            counts[z] = counts.get(z, 0) + 1
    with open(HERE / "data" / "element_structure_counts.tsv", "w") as fh:
        fh.write("element\tstructures\n")
        for z in sorted(counts):
            if 0 < z <= MAX_Z:
                fh.write(f"{chemical_symbols[z]}\t{counts[z]}\n")
    print(f"wrote data/element_family_counts.tsv and data/element_structure_counts.tsv, {n:,} structures")

if __name__ == "__main__":
    main()
