from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from ase import io as ase_io
import metatomic.torch as mta
import vesin.metatomic as vesin_metatomic
from metatrain.utils.io import load_model

ROOT = "${DEV_ROOT}"
ATOMIC_COMBINED = f"{ROOT}/phace-atomic/data/atomic_combined"
MODEL_PATH = f"{ROOT}/phace-atomic/pre/phace_atomic_ep92.pt"
BLOCKS = [
    ("mad_next", 216803, 638443, f"{ROOT}/phace-atomic/data/mad_next/mad_next.xyz"),
    ("mad_sol", 638443, 677454, f"{ROOT}/phace-atomic/data/mad_sol/mad_sol.xyz"),
]
OUT_DIR = f"{ROOT}/phace-atomic/data/union_test"
OUT_FEATURES = f"{OUT_DIR}/phace_atomic_ep92_nextsol_ll.npy"
OUT_META = f"{OUT_DIR}/phace_atomic_ep92_nextsol_meta.npz"
BATCH = 16


def load_selected_frames():
    atom_offsets = np.load(f"{ATOMIC_COMBINED}/na.npy")
    test_ids = np.loadtxt(f"{ATOMIC_COMBINED}/test_indices_in_all.txt", dtype=np.int64)

    frames, global_ids = [], []
    for name, block_start, block_end, xyz in BLOCKS:
        selected = np.sort(test_ids[(test_ids >= block_start) & (test_ids < block_end)])
        wanted = set((selected - block_start).tolist())
        last_local = int(selected[-1] - block_start)

        found = 0
        for local_id, atoms in enumerate(ase_io.iread(xyz, index=":")):
            if local_id in wanted:
                gid = block_start + local_id
                expected = int(atom_offsets[gid + 1] - atom_offsets[gid])
                assert len(atoms) == expected, \
                    f"{name}[{local_id}]: natoms {len(atoms)} != na diff {expected}"
                frames.append(atoms)
                global_ids.append(gid)
                found += 1
            if found == len(selected) and local_id >= last_local:
                break
        assert found == len(selected), f"{name}: found {found}/{len(selected)} test frames"
        print(f"{name}: {found} test frames", flush=True)

    order = np.argsort(np.array(global_ids))
    return [frames[i] for i in order], np.array(global_ids, dtype=np.int64)[order]


def main():
    frames, global_ids = load_selected_frames()
    natoms = np.array([len(a) for a in frames], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(natoms)])
    print(f"selected {len(frames):,} structures, {int(natoms.sum()):,} atoms", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_PATH).to(device)
    length_unit = model.capabilities().length_unit
    nl_calculators = vesin_metatomic.neighbor_lists_for_model(length_unit, model)
    options = mta.ModelEvaluationOptions(
        length_unit=length_unit,
        outputs={"features": mta.ModelOutput(per_atom=True)},
        selected_atoms=None)
    print(f"model loaded  device={device}", flush=True)

    def embed(atoms_list):
        systems = mta.systems_to_torch(atoms_list)
        if not isinstance(systems, list):
            systems = [systems]
        for calculator in nl_calculators:
            calculator.add_neighbor_list(systems)
        systems = [s.to(dtype=torch.float32, device=device) for s in systems]
        with torch.no_grad():
            out = model(systems, options, check_consistency=False)
        return out["features"].block(0).values.detach().cpu().numpy().astype(np.float32)

    probe = embed([frames[0]])
    dim = probe.shape[1]
    features = np.full((int(offsets[-1]), dim), np.nan, dtype=np.float32)
    print(f"phace feature dim = {dim}", flush=True)

    n_failed = 0
    t_start = time.time()
    for i in range(0, len(frames), BATCH):
        chunk = frames[i:i + BATCH]
        try:
            block = embed(chunk)
            row = int(offsets[i])
            features[row:row + block.shape[0]] = block
        except Exception as batch_err:
            print(f"  batch at frame {i} failed ({type(batch_err).__name__}); per-structure retry", flush=True)
            for c, atoms in enumerate(chunk):
                try:
                    single = embed([atoms])
                    row = int(offsets[i + c])
                    features[row:row + single.shape[0]] = single
                except Exception as frame_err:
                    n_failed += 1
                    print(f"  gid {int(global_ids[i + c])} dropped "
                          f"({type(frame_err).__name__}: {frame_err}) -> NaN", flush=True)
        if (i // BATCH) % 100 == 0:
            print(f"  {i + len(chunk)}/{len(frames)} frames  ({time.time() - t_start:.0f}s)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(OUT_FEATURES, features)
    np.savez(OUT_META, gids=global_ids, natoms=natoms)
    print(f"saved {OUT_FEATURES}  shape={features.shape}  failed={n_failed}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
