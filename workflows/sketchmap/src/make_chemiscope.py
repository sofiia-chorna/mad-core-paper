import argparse
from collections import Counter
from pathlib import Path

import chemiscope as cs
import numpy as np
import pyarrow.parquet as pq
from ase import Atoms

run_dir = "${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap"
SMAP_DEFAULT = f"{run_dir}/bundle_2p20_d8/sketchmap_2p20_d8.parquet"
CORESET_DEFAULT = ("${STORE_ROOT}/mad-core/accrete/phace_atomic_ep92"
                   "/coreset224/out/coreset_224_full.parquet")
D = 8

MAD_PARTS = ("mad-1.5", "mad-next", "mad_union", "mad-union")
MAD_LABEL = "MAD"
CATALYST_PREFIXES = ("OC20", "OC22", "OC25")
ANI_PREFIXES = ("ANI-1", "ANI-2")
RENAME = {"Open_Molecular_Crystals_2025_OMC25": "OMC25",
          "Materials_Project": "Materials Project",
          "mcloudarchive__jk-9v": "Li-ion conductors"}


def label_of(name):
    if name in MAD_PARTS:
        return MAD_LABEL
    if name.upper().startswith(CATALYST_PREFIXES):
        return "Open Catalyst"
    if name.upper().startswith(ANI_PREFIXES):
        return "ANI"
    if name.startswith("JARVIS"):
        return "JARVIS"
    if name.startswith(("mcloudarchive", "mcloud__")):
        return "Materials Cloud"
    return RENAME.get(name, name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketchmap", default=SMAP_DEFAULT)
    parser.add_argument("--coreset", default=CORESET_DEFAULT,
                        help="coreset parquet with structures; rows fps_order<2^20 are the smap points")
    parser.add_argument("--max-k", type=int, default=20,
                        help="include the first 2**max_k centres (nested FPS prefix)")
    parser.add_argument("--top-families", type=int, default=20,
                        help="keep the N most common families as-is, collapse the rest to 'other'; "
                             "chemiscope drops string properties with more than 21 distinct values")
    parser.add_argument("--output", default=None)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    K = 2 ** args.max_k
    smap_path = Path(args.sketchmap)
    out_path = (Path(args.output) if args.output
                else smap_path.parent / f"chemiscope_2p20_d8_2p{args.max_k}.json.gz")
    name = args.name or "mad-core coreset 2^20 - sketch-map d=8"

    print(f"K = 2^{args.max_k} = {K:,}")
    print(f"output: {out_path}")

    smap = pq.read_table(args.sketchmap)
    coreset_index = smap.column("coreset_index").to_numpy()
    if not np.array_equal(coreset_index, np.arange(len(coreset_index))):
        raise SystemExit("sketchmap parquet is not sorted by coreset_index 0..N-1")
    N = len(coreset_index)
    if K > N:
        print(f"  warn: 2^{args.max_k} > {N:,} rows; clamping")
        K = N

    cids = smap.column("configuration_id").to_pylist()[:K]
    family = [label_of(f) for f in smap.column("family").to_pylist()[:K]]
    top = set(f for f, _ in Counter(family).most_common(args.top_families))
    family = [f if f in top else "other" for f in family]
    print(f"family merged and collapsed to top {args.top_families} + other: {sorted(top)}")
    pops = smap.column("voronoi_population").to_numpy()[:K]
    log_pops = smap.column("log10_population").to_numpy()[:K]

    print(f"\nstreaming structures from {args.coreset} ...")
    cols = ["fps_order", "configuration_id", "atomic_numbers",
            "positions", "cell", "pbc"]
    numbers = [None] * K
    positions = [None] * K
    cells = [None] * K
    pbcs = [None] * K
    n_found = 0
    pf = pq.ParquetFile(args.coreset)
    for b, batch in enumerate(pf.iter_batches(batch_size=32768, columns=cols)):
        fo = batch["fps_order"].to_numpy()
        sel = np.nonzero(fo < K)[0]
        for i in sel:
            s = int(fo[i])
            if batch["configuration_id"][i].as_py() != cids[s]:
                raise SystemExit(f"configuration_id mismatch at fps_order={s}")
            numbers[s] = np.asarray(batch["atomic_numbers"][i].as_py(), dtype=np.int32)
            positions[s] = np.asarray(batch["positions"][i].as_py(),
                                      dtype=np.float64).reshape(-1, 3)
            cells[s] = np.asarray(batch["cell"][i].as_py(), dtype=np.float64).reshape(3, 3)
            pbcs[s] = np.asarray(batch["pbc"][i].as_py(), dtype=bool)
        n_found += len(sel)
        if (b + 1) % 32 == 0 or n_found == K:
            print(f"  batch {b + 1:>4}  found {n_found:>9,}/{K:,}", flush=True)
        if n_found == K:
            break
    if n_found < K:
        raise SystemExit(f"only {n_found:,}/{K:,} centres found in the coreset parquet")

    print(f"\nbuilding {K:,} ASE Atoms ...")
    frames = [Atoms(numbers=numbers[i], positions=positions[i],
                    cell=cells[i], pbc=pbcs[i]) for i in range(K)]
    formulas = [a.get_chemical_formula(mode="hill") for a in frames]

    k_set = [1 if i == 0 else int(np.floor(np.log2(i)) + 1) for i in range(K)]

    properties = {
        "coreset_index":      {"target": "structure", "values": list(range(K))},
        "k_set":              {"target": "structure", "values": k_set},
        "family":             {"target": "structure", "values": family},
        "configuration_id":   {"target": "structure", "values": cids},
        "chemical_formula":   {"target": "structure", "values": formulas},
        "voronoi_population": {"target": "structure", "values": pops.tolist()},
        "log10_population":   {"target": "structure", "values": log_pops.tolist()},
    }
    for i in range(D):
        properties[f"smap_{i + 1}"] = {
            "target": "structure",
            "values": smap.column(f"x{i}").to_numpy()[:K].tolist()}

    print(f"\nwriting {out_path} ...")
    cs.write_input(
        str(out_path),
        structures=frames,
        properties=properties,
        metadata={"name": name,
                  "description": ("sketch-map: d=8, sigma=1.24, A=4, B=2, a=2, b=2, "
                                  "wgamma=0.25, PCA init, 400 iterations")},
        settings={"map": {
            "x": {"property": "smap_1"},
            "y": {"property": "smap_2"},
            "color": {"property": "family"},
            "symbol": "family",
            "size": {"property": "log10_population"},
        }},
    )
    print(f"saved => {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
