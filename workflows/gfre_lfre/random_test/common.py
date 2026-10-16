import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
from ase import io as ase_io

ROOT = "${DEV_ROOT}"
STORE = "${STORE_ROOT}"
SETS = f"{ROOT}/phace-atomic/mad-core-valid/dft/lda_pbe/sets/random_test_6k"
SPLIT_FILES = ["fff", "ttt", "surfaces"]
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = f"{HERE}/results_random_test/features"
MODELS_DIR = f"{ROOT}/teacher_models"

CHECKPOINTS = {
    "phace-atomic": f"{STORE}/mad-core/release/ecumetric_model.pt",
    "pet-mad": f"{MODELS_DIR}/pet-mad-s-v1.5.0.pt",
    "mace-omat": f"{MODELS_DIR}/mace-omat-0-medium.model",
    "uma-omat": f"{MODELS_DIR}/uma-s-1p2.pt",
    "dpa-omat": f"{MODELS_DIR}/DPA-3.2-5M-OMat24.pth",
    "dpa-omol": f"{MODELS_DIR}/DPA3-Omol-Large.pt",
}


def parse_args(models):
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=models)
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


# the canonical frame order is the concatenation of the three split files
def load_frames(limit=None):
    frames = []
    meta = defaultdict(list)

    for split_file in SPLIT_FILES:
        for row, atoms in enumerate(ase_io.iread(f"{SETS}/{split_file}.extxyz", index=":")):
            frames.append(atoms)
            meta["configuration_id"].append(atoms.info["configuration_id"])
            meta["dataset_group"].append(atoms.info["dataset_group"])
            meta["split_file"].append(split_file)
            meta["split_row"].append(row)
            meta["natoms"].append(len(atoms))

    if limit:
        rng = np.random.default_rng(0)
        picks = np.sort(rng.choice(len(frames), limit, replace=False))
        frames = [frames[i] for i in picks]
        meta = {key: [values[i] for i in picks] for key, values in meta.items()}

    return frames, {key: np.array(values) for key, values in meta.items()}


class Writer:
    # per atom features in canonical frame order, NaN rows for excluded structures

    def __init__(self, frames):
        self.natoms = np.array([len(atoms) for atoms in frames], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.natoms)])
        self.features = None
        self.excluded = defaultdict(list)

    def put(self, frame, rows):
        if self.features is None:
            self.features = np.full((int(self.offsets[-1]), rows.shape[1]), np.nan, np.float32)

        start, end = int(self.offsets[frame]), int(self.offsets[frame + 1])
        assert rows.shape[0] == end - start, f"frame {frame} has {rows.shape[0]} rows for {end - start} atoms"
        self.features[start:end] = rows

    def exclude(self, frame, reason):
        self.excluded[reason].append(int(frame))

    def valid_frames(self):
        return np.array([
            np.isfinite(self.features[self.offsets[i]:self.offsets[i + 1]]).all()
            for i in range(len(self.natoms))
        ])

    def save(self, model, meta):
        os.makedirs(FEATURE_DIR, exist_ok=True)
        assert self.features is not None, "no features were written"

        valid = self.valid_frames()
        listed = {frame for frames in self.excluded.values() for frame in frames}
        non_finite = [int(i) for i in np.where(~valid)[0] if i not in listed]
        if non_finite:
            self.excluded["non_finite_output"] = non_finite

        np.save(f"{FEATURE_DIR}/{model}_ll.npy", self.features)
        summary = {
            "model": model,
            "n_frames": int(len(self.natoms)),
            "n_atoms": int(self.offsets[-1]),
            "dim": int(self.features.shape[1]),
            "n_valid_frames": int(valid.sum()),
            "excluded": {reason: sorted(frames) for reason, frames in self.excluded.items()},
            "excluded_configuration_ids": {
                reason: [str(meta["configuration_id"][i]) for i in sorted(frames)]
                for reason, frames in self.excluded.items()
            },
        }
        with open(f"{FEATURE_DIR}/{model}_exclusions.json", "w") as handle:
            json.dump(summary, handle, indent=1)

        counts = ", ".join(f"{reason} {len(frames)}" for reason, frames in self.excluded.items())
        print(f"{model} valid {valid.sum()}/{len(valid)} frames, dim {self.features.shape[1]}, "
              f"excluded {counts or 'none'}", flush=True)


def run_batched(frames, writer, embed, batch, groups=None):
    if groups is None:
        groups = [list(range(len(frames)))]

    started = time.time()
    done = 0

    for indices in groups:
        for i in range(0, len(indices), batch):
            chunk = indices[i:i + batch]

            try:
                block = embed([frames[j] for j in chunk])
                position = 0
                for j in chunk:
                    writer.put(j, block[position:position + len(frames[j])])
                    position += len(frames[j])

            except Exception as batch_error:
                print(f"  batch at frame {chunk[0]} failed with {type(batch_error).__name__}, "
                      f"retrying one by one", flush=True)
                for j in chunk:
                    try:
                        writer.put(j, embed([frames[j]]))
                    except Exception as frame_error:
                        writer.exclude(j, f"runtime_error {type(frame_error).__name__}")
                        print(f"  frame {j} dropped, {type(frame_error).__name__} "
                              f"{str(frame_error)[:120]}", flush=True)

            done += len(chunk)
            if (done // batch) % 20 == 0:
                print(f"  {done}/{len(frames)} frames, {time.time() - started:.0f} s", flush=True)


def run(featurize, models):
    args = parse_args(models)
    frames, meta = load_frames(args.limit)
    print(f"{len(frames)} frames, {sum(len(atoms) for atoms in frames)} atoms", flush=True)

    if not args.limit:
        os.makedirs(FEATURE_DIR, exist_ok=True)
        np.savez(f"{FEATURE_DIR}/frames_meta.npz", **meta)

    writer = Writer(frames)
    featurize(frames, writer, args.model)

    name = args.model if not args.limit else f"{args.model}_smoke{args.limit}"
    writer.save(name, meta)
