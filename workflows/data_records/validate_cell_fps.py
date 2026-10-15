import argparse
import json
from pathlib import Path

import numpy as np

RELEASE = Path("${STORE_ROOT}/mad-core/release")
CELL_FPS = RELEASE / "mad-core" / "cell_fps"
N_CELLS = 2**24
N_MEMBERS = 1_694_502_355


def load_member_arrays():
    offsets = np.load(CELL_FPS / "center_offsets.npy")
    rank = np.load(CELL_FPS / "fps_order_in_cell.npy", mmap_mode="r")
    eps = np.load(CELL_FPS / "epsilon_sq.npy", mmap_mode="r")
    id_off = np.load(CELL_FPS / "configuration_id_offsets.npy", mmap_mode="r")
    id_bytes = np.memmap(CELL_FPS / "configuration_id_bytes.bin", dtype=np.uint8, mode="r")
    rel_dir = np.load(CELL_FPS / "release_directory.npy", mmap_mode="r")
    rel_row = np.load(CELL_FPS / "release_row_in_directory.npy", mmap_mode="r")
    return offsets, rank, eps, id_off, id_bytes, rel_dir, rel_row


def member_id(id_off, id_bytes, i):
    return id_bytes[id_off[i]:id_off[i + 1]].tobytes()


def check_structure(offsets, rank, eps, id_off, id_bytes, rel_dir, rel_row):
    assert offsets.shape == (N_CELLS + 1,) and offsets.dtype == np.int64
    assert offsets[0] == 0 and offsets[-1] == N_MEMBERS
    assert np.all(np.diff(offsets) >= 0), "center_offsets not monotone"

    assert rank.shape == (N_MEMBERS,) and rank.dtype == np.int32
    assert eps.shape == (N_MEMBERS,) and eps.dtype == np.float32
    assert id_off.shape == (N_MEMBERS + 1,) and id_off[0] == 0
    assert id_off[-1] == len(id_bytes)
    assert rel_dir.shape == (N_MEMBERS,) and rel_row.shape == (N_MEMBERS,)

    names = np.load(CELL_FPS / "release_directory_names.npy")
    assert len(names) == 455
    missing = [n for n in names.tolist() if not (RELEASE / n.decode()).is_dir()]
    assert not missing, f"directory names not found in release: {missing}"

    print(f"structure OK: {N_CELLS} cells, {N_MEMBERS} members, 455 directories resolve")
    return names


def check_centers_match_tier():
    centers = np.load(CELL_FPS / "center_configuration_id.npy", mmap_mode="r")
    tier = np.load(RELEASE / "mad-core" / "configuration_id.npy", mmap_mode="r")
    assert centers.shape == tier.shape

    chunk = 1 << 20
    for start in range(0, N_CELLS, chunk):
        stop = min(start + chunk, N_CELLS)
        assert np.array_equal(centers[start:stop], tier[start:stop]), \
            f"center id mismatch in rows {start}:{stop}"

    print("center_configuration_id == mad-core/configuration_id row-for-row")


def check_population_closure(offsets):
    population = np.load(RELEASE / "mad-core" / "voronoi_population.npy")
    counts = np.diff(offsets)
    assert np.array_equal(counts, population), "member counts != voronoi_population"
    print(f"member counts == voronoi_population for all cells (sum {counts.sum()})")


def sample_cells(offsets, n_random, seed):
    rng = np.random.default_rng(seed)
    cells = set(rng.integers(0, N_CELLS, n_random).tolist())

    counts = np.diff(offsets)
    cells.add(int(np.argmax(counts)))

    flagged = np.load(RELEASE / "mad-core" / "image_flag_rows.npy")
    cells.update(int(r) for r in flagged)

    return sorted(cells)


def check_unranked_total(rank):
    total = 0
    chunk = 1 << 27
    for start in range(0, N_MEMBERS, chunk):
        total += int((np.asarray(rank[start:start + chunk]) == -1).sum())
    assert total == 486_653, f"rank==-1 rows: {total} != documented 486,653"
    print(f"rank==-1 rows: {total} (matches documented breakdown)")


def check_within_cells(offsets, rank, eps, id_off, id_bytes, cells):
    centers = np.load(CELL_FPS / "center_configuration_id.npy", mmap_mode="r")
    seed_mismatches = 0
    boundary_steps = 0
    tie_groups_checked = 0

    for cell in cells:
        start, stop = int(offsets[cell]), int(offsets[cell + 1])
        n = stop - start
        r = np.asarray(rank[start:stop])
        e = np.asarray(eps[start:stop])

        unranked = r == -1
        k = n - int(unranked.sum())
        assert k >= 1 and not unranked[:k].any(), \
            f"cell {cell}: unranked rows not all at the end (ranked count {k})"

        ranked = r[:k]
        assert ranked[0] == 0, f"cell {cell}: first ranked row has rank {ranked[0]}"
        assert np.all(np.diff(ranked) >= 0), f"cell {cell}: ranks not non-decreasing"
        distinct_ranks, first_index = np.unique(ranked, return_index=True)
        assert np.array_equal(distinct_ranks, np.arange(len(distinct_ranks), dtype=ranked.dtype)), \
            f"cell {cell}: distinct ranks not 0..{len(distinct_ranks) - 1}"

        assert np.isnan(e[k:]).all() and not np.isnan(e[:k]).any(), \
            f"cell {cell}: epsilon NaN pattern does not match rank==-1 pattern"

        if len(distinct_ranks) >= 3:
            body = e[first_index][1:]
            rises = np.diff(body) > 1e-3 * np.abs(body[:-1]) + 2e-6
            assert not rises.any(), \
                f"cell {cell}: epsilon_sq rises beyond resume-boundary tolerance at {np.flatnonzero(rises)[:5]}"
            boundary_steps += int((np.diff(body) > 0).sum())

        tie_starts = np.flatnonzero(np.diff(ranked) == 0)
        for t in tie_starts[:5]:
            first = member_id(id_off, id_bytes, start + int(t))
            second = member_id(id_off, id_bytes, start + int(t) + 1)
            assert first.rpartition(b":")[2] == second.rpartition(b":")[2], \
                f"cell {cell}: tied rank {ranked[t]} on different structures " \
                f"{first!r} vs {second!r}"
            tie_groups_checked += 1

        if member_id(id_off, id_bytes, start) != centers[cell].tobytes().rstrip(b"\x00"):
            seed_mismatches += 1

    print(f"within-cell OK for {len(cells)} cells: distinct-id ranks contiguous, "
          f"unranked tail + NaN pattern, epsilon non-increasing "
          f"({boundary_steps} sub-tolerance resume steps), {tie_groups_checked} tie groups id-verified; "
          f"seed==center mismatches: {seed_mismatches}")


def check_pointers(offsets, id_off, id_bytes, rel_dir, rel_row, names, n_samples, seed):
    rng = np.random.default_rng(seed + 1)
    members = np.unique(rng.integers(0, N_MEMBERS, n_samples))

    by_dir = {}
    for i in members.tolist():
        d = int(rel_dir[i])
        by_dir.setdefault(d, []).append(i)

    cell_of = dict(zip(
        members.tolist(),
        (np.searchsorted(offsets, members, side="right") - 1).tolist(),
    ))

    checked = 0
    for d, idxs in by_dir.items():
        dir_path = RELEASE / names[d].decode()
        cid = np.load(dir_path / "configuration_id.npy", mmap_mode="r")
        ns = len(cid)
        va = np.memmap(dir_path / "voronoi_a.bin", dtype=np.int32, mode="r")
        levels = len(va) // ns
        va = va.reshape(ns, levels)

        for i in idxs:
            row = int(rel_row[i])
            expected = member_id(id_off, id_bytes, i)
            actual = cid[row].tobytes().rstrip(b"\x00")
            assert actual == expected, \
                f"member {i}: pointer {names[d].decode()}[{row}] id {actual!r} != blob {expected!r}"

            cell = cell_of[i]
            assigned = int(va[row, levels - 1])
            assert assigned == cell, \
                f"member {i}: voronoi_a level-24 {assigned} != cell {cell}"
            checked += 1

    print(f"pointers OK for {checked} sampled members across {len(by_dir)} directories "
          f"(id match + voronoi_a level-24 agreement)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=2000)
    parser.add_argument("--members", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    offsets, rank, eps, id_off, id_bytes, rel_dir, rel_row = load_member_arrays()

    names = check_structure(offsets, rank, eps, id_off, id_bytes, rel_dir, rel_row)
    check_centers_match_tier()
    check_population_closure(offsets)
    check_unranked_total(rank)

    cells = sample_cells(offsets, args.cells, args.seed)
    check_within_cells(offsets, rank, eps, id_off, id_bytes, cells)

    check_pointers(offsets, id_off, id_bytes, rel_dir, rel_row, names, args.members, args.seed)

    print("ALL CELL_FPS CHECKS PASSED")


if __name__ == "__main__":
    main()

