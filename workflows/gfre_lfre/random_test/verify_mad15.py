import argparse
import importlib

import numpy as np
from ase import io as ase_io

from common import STORE, Writer

UNIVERSAL = f"{STORE}/universal_features"
XYZ = f"{UNIVERSAL}/mad-1.5/mad-1.5-r2scan-all.xyz"
COMBINED = f"{STORE}/mad-core/models/training/atomic_combined"
N_MAD15 = 216803

MODULE = {
    "pet-mad": "featurize_metatomic",
    "mace-omat": "featurize_mace",
    "uma-omat": "featurize_uma",
    "dpa-omat": "featurize_dpa",
    "dpa-omol": "featurize_dpa",
}

# the normalized bins the union figure was built from, with the scale of build_stats.json
COMBINED_BIN = {
    "pet-mad": ("pet.bin", 512, 13.143182633340421),
    "mace-omat": ("mace.bin", 144, 3.3791315950729905),
    "uma-omat": ("uma.bin", 128, 0.7853647808071987),
    "dpa-omat": ("dpa_omat.bin", 240, 3.3262480000814945),
    "dpa-omol": ("dpa_omol.bin", 240, 22.518889804836927),
}

# raw per atom arrays left in the store, only the PET one is from the run the paper used
RAW_ARRAY = {
    "pet-mad": f"{UNIVERSAL}/mad-1.5/pet/mad-s-v1.5.0_ll.npy",
    "mace-omat": f"{UNIVERSAL}/mad-1.5/mace/mace_full.npy",
    "dpa-omat": f"{UNIVERSAL}/mad-1.5/dpa/dpa-Omat24_ll.npy",
}


def compare(label, new, reference, frame_of_row, centered=False):
    finite = np.isfinite(new).all(axis=1) & np.isfinite(reference).all(axis=1)
    new = new[finite].astype(np.float64)
    reference = reference[finite].astype(np.float64)

    if centered:
        new = new - new.mean(axis=0)
        reference = reference - reference.mean(axis=0)

    difference = np.abs(new - reference)
    scale = np.abs(reference).mean()
    correlation = np.corrcoef(new.ravel(), reference.ravel())[0, 1]
    print(f"  {label} rows {len(new)}, max diff {difference.max():.3e}, median diff {np.median(difference):.3e}, "
          f"mean abs reference {scale:.3e}, correlation {correlation:.6f}", flush=True)

    frames = frame_of_row[finite]
    per_frame = {int(frame): float(difference[frames == frame].max()) for frame in np.unique(frames)}
    worst = sorted(per_frame.items(), key=lambda item: -item[1])[:5]
    n_bad = sum(value > 0.05 * scale for value in per_frame.values())
    print(f"    frames with max diff above 5 percent of the mean, {n_bad}/{len(per_frame)}, worst {worst}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODULE))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--start", type=int, default=150000)
    args = parser.parse_args()
    start, n = args.start, args.n

    frames = list(ase_io.iread(XYZ, index=f"{start}:{start + n}"))
    na_all = np.load(f"{COMBINED}/na.npy")
    na = na_all[start:start + n + 1] - na_all[start]
    assert all(len(atoms) == na[i + 1] - na[i] for i, atoms in enumerate(frames)), "natoms differ from na.npy"
    first_row = int(na_all[start])
    n_rows = int(na[-1])
    print(f"{n} MAD-1.5 frames from {start}, {n_rows} atoms", flush=True)

    writer = Writer(frames)
    module = importlib.import_module(MODULE[args.model])
    module.featurize(frames, writer, args.model)
    new = writer.features
    print(f"  new features {new.shape}, excluded {dict(writer.excluded)}", flush=True)

    frame_of_row = np.repeat(np.arange(start, start + n), np.diff(na))

    bin_name, dim, scale = COMBINED_BIN[args.model]
    canonical = np.memmap(f"{COMBINED}/{bin_name}", dtype="float32", mode="r", shape=(int(na_all[-1]), dim))
    compare("canonical bin, centred and scaled,", new / scale, np.asarray(canonical[first_row:first_row + n_rows]),
            frame_of_row, centered=True)

    if args.model in RAW_ARRAY:
        raw = np.load(RAW_ARRAY[args.model], mmap_mode="r")
        compare("raw store array,", new, np.asarray(raw[first_row:first_row + n_rows]), frame_of_row)


if __name__ == "__main__":
    main()
