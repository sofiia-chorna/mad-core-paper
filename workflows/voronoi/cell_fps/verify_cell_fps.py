import glob
import json

import numpy as np
import pyarrow.parquet as pq

OUT = "${DEV_ROOT}/phace-atomic/voronoi/out/cell_fps"
N_CELLS = 2 ** 24


def main():
    with open(f"{OUT}/plan.json") as fh:
        plan = json.load(fh)

    shuffle_tasks = sorted(glob.glob(f"{OUT}/shuffle/logs/task_*.json"))
    rows_seen = rows_clean = rows_mega = 0
    for path in shuffle_tasks:
        with open(path) as fh:
            task = json.load(fh)
        rows_seen += task["rows_seen"]
        rows_clean += task["rows_clean"]
        rows_mega += task["rows_mega"]
    print(f"shuffle tasks {len(shuffle_tasks)}: seen {rows_seen:,} "
          f"clean {rows_clean:,} mega {rows_mega:,}")
    print(f"plan total rows {plan['total_rows']:,} "
          f"(seen == plan: {rows_seen == plan['total_rows']})")

    expected = np.zeros(N_CELLS, dtype=np.int64)
    for path in sorted(glob.glob(f"{OUT}/shuffle/logs/cell_counts_*.npy")):
        expected += np.load(path).astype(np.int64)
    print(f"expected pool rows {expected.sum():,} over "
          f"{int((expected > 0).sum()):,} nonempty cells")

    got = np.zeros(N_CELLS, dtype=np.int64)
    ranked_total = 0
    problem_summaries = []
    for path in sorted(glob.glob(f"{OUT}/fps/b*.parquet")) \
            + sorted(glob.glob(f"{OUT}/fps/cell_*.parquet")):
        cells = pq.read_table(path, columns=["cell"]).column("cell").to_numpy()
        got += np.bincount(cells, minlength=N_CELLS)
        ranked_total += len(cells)
    for path in sorted(glob.glob(f"{OUT}/fps/*.json")):
        with open(path) as fh:
            summary = json.load(fh)
        bad = {key: value for key, value in summary["problems"].items()
               if isinstance(value, (int, bool)) and value
               and key not in ("centers_present", "seed_is_center",
                               "selected_is_permutation", "all_rows_this_cell")}
        for key in ("seed_is_center", "selected_is_permutation", "all_rows_this_cell"):
            if key in summary["problems"] and not summary["problems"][key]:
                bad[key] = False
        if bad:
            problem_summaries.append((path, bad))

    mismatched = np.nonzero(expected != got)[0]
    covered = np.nonzero(got > 0)[0]
    print(f"ranked rows {ranked_total:,}  covered cells {len(covered):,}")
    print(f"cells with count mismatch (expected vs ranked): {len(mismatched):,}")
    if len(mismatched):
        for cell in mismatched[:10]:
            print(f"  cell {cell}: expected {expected[cell]} got {got[cell]}")
    if problem_summaries:
        print("summaries with problems:")
        for path, bad in problem_summaries:
            print(f"  {path}: {bad}")
    else:
        print("no problems flagged in any summary")

    complete = (rows_seen == plan["total_rows"] and len(mismatched) == 0
                and ranked_total == expected.sum() and not problem_summaries)
    print(f"ALL GATES {'PASS' if complete else 'NOT YET PASSING'}")


if __name__ == "__main__":
    main()
