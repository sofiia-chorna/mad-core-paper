from __future__ import annotations
import argparse

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import vesin

MAX_ATOMS = 1000   # project rule: drop structures with more than 1000 atoms

KE2 = 14.399645
A_PREFAC = 0.46850
_C = np.array([0.18175, 0.50986, 0.28022, 0.02817])
_D = np.array([3.19980, 0.94229, 0.40290, 0.20162])

MERGED_COLS = ["configuration_id", "dataset_id", "source_id", "names",
               "atomic_numbers", "positions", "cell", "pbc", "nsites"]
PHACE_COLS = ["phace_energy", "phace_features_bb_mean", "phace_non_conservative_forces"]


def zbl_quantities(nl, numbers, pos, cell, pbc, cutoff):
    n = len(numbers)
    box = np.asarray(cell, dtype=np.float64).copy()
    pbc = np.asarray(pbc, dtype=bool)

    if not pbc.all():
        for ax in range(3):
            if not pbc[ax] and np.linalg.norm(box[ax]) < 2 * cutoff:
                span = float(pos[:, ax].max() - pos[:, ax].min()) if n else 0.0
                v = np.zeros(3); v[ax] = span + 4 * cutoff
                box[ax] = v

    i, j, r, D = nl.compute(points=np.asarray(pos, dtype=np.float64), box=box,
                            periodic=pbc, quantities="ijdD")
    if i.size == 0:
        return 0.0, 0.0, 0.0

    i = i.astype(np.intp); j = j.astype(np.intp)
    Zi = numbers[i].astype(np.float64); Zj = numbers[j].astype(np.float64)
    a = A_PREFAC / (Zi ** 0.23 + Zj ** 0.23)
    x = r / a
    e = np.exp(-_D[:, None] * x[None, :])
    phi = (_C[:, None] * e).sum(0)
    dphi = -(_C[:, None] * _D[:, None] * e).sum(0)
    A = KE2 * Zi * Zj

    dVdr = A * (-phi / r ** 2 + dphi / (a * r))
    f = (dVdr / r)[:, None] * D
    F = np.empty((n, 3))
    for c in range(3):
        F[:, c] = (np.bincount(i, weights=f[:, c], minlength=n)
                   - np.bincount(j, weights=f[:, c], minlength=n))
    max_force = float(np.sqrt((F * F).sum(1)).max())

    pair_e = A / r * phi
    half = 0.5 * pair_e
    e_atom = np.bincount(i, half, n) + np.bincount(j, half, n)
    return max_force, float(e_atom.sum() / n), float(e_atom.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structures", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zbl-sidecar", default=None)
    ap.add_argument("--cutoff", type=float, default=5.0)
    args = ap.parse_args()

    st = pq.read_table(args.structures)
    st = st.filter(pc.less_equal(st.column("nsites"), MAX_ATOMS))   # drop > 1000-atom structures
    ft = pq.read_table(args.features, columns=["configuration_id"] + PHACE_COLS)

    ft_df = ft.to_pandas().drop_duplicates("configuration_id")
    st_df = st.select([c for c in MERGED_COLS if c in st.column_names]).to_pandas().drop_duplicates("configuration_id")
    merged = st_df.merge(ft_df, on="configuration_id", how="inner").reset_index(drop=True)

    def as2d(v):
        return np.array([np.asarray(x, dtype=np.float64) for x in v])

    nl = vesin.NeighborList(cutoff=args.cutoff, full_list=False)
    mf = np.empty(len(merged)); epa = np.empty(len(merged)); mae = np.empty(len(merged))

    for k in range(len(merged)):
        numbers = np.asarray([int(z) for z in merged["atomic_numbers"].iloc[k]], dtype=np.int64)
        pos = as2d(merged["positions"].iloc[k])
        cell = as2d(merged["cell"].iloc[k]).reshape(3, 3)
        pbc = np.asarray([bool(b) for b in merged["pbc"].iloc[k]], dtype=bool)
        mf[k], epa[k], mae[k] = zbl_quantities(nl, numbers, pos, cell, pbc, args.cutoff)

    merged["zbl_max_force"] = mf
    merged["zbl_energy_per_atom"] = epa
    merged["zbl_max_atomic_energy"] = mae

    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), args.out)

    if args.zbl_sidecar:
        side = pa.table({"configuration_id": merged["configuration_id"].to_numpy(),
                         "zbl_energy_per_atom": epa, "zbl_max_atomic_energy": mae})
        pq.write_table(side, args.zbl_sidecar)

    mx = lambda a: (float(a.max()) if len(a) else 0.0)

    print(f"[merge] {len(merged):,} rows -> {args.out}  "
          f"max_force[max]={mx(mf):.1f} epa[max]={mx(epa):.1f} atomic[max]={mx(mae):.1f}", flush=True)


if __name__ == "__main__":
    main()
