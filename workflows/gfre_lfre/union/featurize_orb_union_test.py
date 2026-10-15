from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from ase import io as ase_io
from orb_models.forcefield import pretrained

ROOT = "${DEV_ROOT}"
ATOMIC_COMBINED = f"{ROOT}/phace-atomic/data/atomic_combined"
BLOCKS = [
    ("mad_next", 216803, 638443, f"{ROOT}/phace-atomic/data/mad_next/mad_next.xyz"),
    ("mad_sol", 638443, 677454, f"{ROOT}/phace-atomic/data/mad_sol/mad_sol.xyz"),
]
OUT_DIR = f"{ROOT}/phace-atomic/data/union_test"
OUT_FEATURES = f"{OUT_DIR}/orb-v3-omat_nextsol_ll.npy"
OUT_META = f"{OUT_DIR}/orb-v3-omat_nextsol_meta.npz"
BATCH = 128


def uniform_pbc(atoms):
    if atoms.pbc.any() and not atoms.pbc.all():
        atoms.pbc = [True, True, True]
    return atoms


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
                frames.append(uniform_pbc(atoms))
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
    orbff, adapter = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
    orbff.eval()

    def embed(atoms_list):
        batch = adapter.from_ase_atoms_list(atoms_list, device=device)
        with torch.no_grad():
            return orbff.model(batch)["node_features"].float().cpu().numpy()

    probe = embed([frames[0]])
    dim = probe.shape[1]
    features = np.full((int(offsets[-1]), dim), np.nan, dtype=np.float32)
    print(f"orb node feature dim = {dim}  device={device}", flush=True)

    def place(node_features, frame_start, counts):
        pos = 0
        for c, nat in enumerate(counts):
            row = int(offsets[frame_start + c])
            features[row:row + nat] = node_features[pos:pos + nat]
            pos += nat

    n_failed = 0
    t_start = time.time()
    for i in range(0, len(frames), BATCH):
        chunk = frames[i:i + BATCH]
        counts = [len(a) for a in chunk]
        try:
            place(embed(chunk), i, counts)
        except Exception as batch_err:
            print(f"  batch at frame {i} failed ({type(batch_err).__name__}); per-structure retry", flush=True)
            for c, atoms in enumerate(chunk):
                try:
                    place(embed([atoms]), i + c, [len(atoms)])
                except Exception as frame_err:
                    n_failed += 1
                    print(f"  gid {int(global_ids[i + c])} dropped "
                          f"({type(frame_err).__name__}: {frame_err}) -> NaN", flush=True)
        if (i // BATCH) % 20 == 0:
            print(f"  {i + len(chunk)}/{len(frames)} frames  ({time.time() - t_start:.0f}s)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(OUT_FEATURES, features)
    np.savez(OUT_META, gids=global_ids, natoms=natoms)
    print(f"saved {OUT_FEATURES}  shape={features.shape}  failed={n_failed}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
