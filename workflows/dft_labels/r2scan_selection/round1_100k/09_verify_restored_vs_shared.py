import re

import numpy as np

project_dir = "${DEV_ROOT}/phace-atomic"
shared_path = f"{project_dir}/cscs-madcore_first100k_non-mad_max300 (3).xyz"
restored_path = f"{project_dir}/botr/round1_100k/out/cscs-madcore_first100k_non-mad_max300.xyz"
id_map_path = f"{project_dir}/botr/round1_100k/out/round1_id_map.tsv"

rank_pattern = re.compile(r"fps_rank=(\d+)")
cid_pattern = re.compile(r"configuration_id=(\S+)")
lattice_pattern = re.compile(r'Lattice="([^"]*)"')
pbc_pattern = re.compile(r'pbc="([^"]*)"')


def parse_cell(comment):
    match = lattice_pattern.search(comment)
    if match is None:
        return np.zeros((3, 3))
    return np.array([float(v) for v in match.group(1).split()]).reshape(3, 3)


def parse_pbc(comment):
    match = pbc_pattern.search(comment)
    if match is None:
        return [False, False, False]
    return [flag == "T" for flag in match.group(1).split()]


def read_frames(path):
    frames = []
    with open(path) as handle:
        line = handle.readline()
        while line:
            n_atoms = int(line)
            comment = handle.readline()
            species, positions = [], np.empty((n_atoms, 3))
            for i in range(n_atoms):
                fields = handle.readline().split()
                species.append(fields[0])
                positions[i] = [float(v) for v in fields[1:4]]

            frames.append({
                "rank": int(rank_pattern.search(comment).group(1)),
                "cid": cid_pattern.search(comment).group(1),
                "species": species,
                "positions": positions,
                "cell": parse_cell(comment),
                "pbc": parse_pbc(comment),
            })
            line = handle.readline()
    return frames


shared_frames = read_frames(shared_path)
restored_frames = read_frames(restored_path)
print(f"shared file   : {len(shared_frames):,} frames  ({shared_path.split('/')[-1]})")
print(f"restored file : {len(restored_frames):,} frames")
assert len(shared_frames) == len(restored_frames)

shared_frames.sort(key=lambda frame: frame["rank"])
restored_frames.sort(key=lambda frame: frame["rank"])

max_position_delta = 0.0
max_cell_delta = 0.0
wrapped_only_ranks = []
mismatched_ranks = []

for shared, restored in zip(shared_frames, restored_frames):
    assert shared["rank"] == restored["rank"]
    assert shared["species"] == restored["species"], f"species differ at rank {shared['rank']}"
    assert shared["pbc"] == restored["pbc"], f"pbc differs at rank {shared['rank']}"

    max_cell_delta = max(max_cell_delta, np.abs(shared["cell"] - restored["cell"]).max())
    delta = shared["positions"] - restored["positions"]
    direct = np.abs(delta).max()

    if direct <= 1e-4:
        max_position_delta = max(max_position_delta, direct)
        continue

    if any(restored["pbc"]) and np.abs(np.linalg.det(restored["cell"])) > 1e-8:
        fractional = delta @ np.linalg.inv(restored["cell"])
        residual = np.abs((fractional - np.round(fractional)) @ restored["cell"]).max()
        if residual <= 1e-3:
            wrapped_only_ranks.append(shared["rank"])
            continue

    mismatched_ranks.append((shared["rank"], float(direct)))

print()
print(f"ranks identical, in order          : True")
print(f"species identical (every frame)    : True")
print(f"pbc identical (every frame)        : True")
print(f"max |cell| difference              : {max_cell_delta:.2e} A")
print(f"max |position| difference          : {max_position_delta:.2e} A  (frames equal atom-for-atom)")
print(f"frames differing by a lattice shift: {len(wrapped_only_ranks)} -> ranks {wrapped_only_ranks}")
print(f"genuinely mismatched frames        : {len(mismatched_ranks)} {mismatched_ranks[:5]}")

with open(id_map_path, "w") as handle:
    handle.write("fps_rank\tconfiguration_id_pre_unification\tconfiguration_id_release\n")
    for shared, restored in zip(shared_frames, restored_frames):
        handle.write(f"{shared['rank']}\t{shared['cid']}\t{restored['cid']}\n")
print(f"\nwrote {id_map_path}")
