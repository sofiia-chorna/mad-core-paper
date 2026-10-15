import json
import os

import numpy as np

HERE = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert"
POP = f"{HERE}/out/population"
OUT = f"{HERE}/out/stageA_split"
N_TASKS = 5


def main():
    os.makedirs(OUT, exist_ok=True)
    families = json.load(open(f"{HERE}/out/pool_families.json"))
    done = {name[:-4] for name in os.listdir(f"{HERE}/out/stageA")}

    big_todo = [name for name in families["big_families"] if f"big__{name}" not in done]
    small_todo = [name for name in families["small_families"] if f"small__{name}" not in done]

    task_names = [[] for _ in range(N_TASKS)]
    task_rows = np.zeros(N_TASKS, dtype=np.int64)

    if "anchor__mad-next-coreset" not in done:
        task_names[N_TASKS - 1].append("mad-next-coreset")

    for index, name in enumerate(big_todo):
        task_names[index % N_TASKS].append(name)
        task_rows[index % N_TASKS] += np.load(f"{POP}/collection__{name}.npy", mmap_mode="r").shape[0]

    small_sizes = sorted(((np.load(f"{POP}/collection__{name}.npy", mmap_mode="r").shape[0], name) for name in small_todo), reverse=True)
    for size, name in small_sizes:
        target = int(np.argmin(task_rows))
        task_names[target].append(name)
        task_rows[target] += size

    for task_id, names in enumerate(task_names):
        with open(f"{OUT}/task_{task_id}.txt", "w") as f:
            f.write(",".join(names))
        print(f"task {task_id}: {len(names)} folders, {task_rows[task_id]:,} pool rows")


if __name__ == "__main__":
    main()
