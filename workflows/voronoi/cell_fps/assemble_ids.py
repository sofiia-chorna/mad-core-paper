import argparse
import glob
import os

import numpy as np


def part_spans(cell, offsets):
    boundaries = np.concatenate([[0], np.where(np.diff(cell) != 0)[0] + 1, [len(cell)]])
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        c = int(cell[lo])
        start = int(offsets[c])
        if int(offsets[c + 1]) - start != hi - lo:
            raise RuntimeError(f"cell {c}: span {int(offsets[c+1]) - start} "
                               f"but part holds {hi - lo}")
        yield start, lo, hi - lo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", choices=["lengths", "bytes"], required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    args = parser.parse_args()

    offsets = np.load(f"{args.out}/center_offsets.npy", mmap_mode="r")
    n = int(offsets[-1])
    parts = sorted(glob.glob(f"{args.out}/ids/part_*.npz"))

    if args.phase == "lengths":
        lengths = np.lib.format.open_memmap(f"{args.out}/_id_lengths.npy", mode="w+",
                                            dtype=np.uint16, shape=(n,))
        for index, path in enumerate(parts):
            data = np.load(path)
            cell, part_lengths = data["cell"], data["lengths"].astype(np.uint16)
            for start, local, count in part_spans(cell, offsets):
                lengths[start:start + count] = part_lengths[local:local + count]
            if index % 20 == 0:
                print(f"lengths {index + 1}/{len(parts)}", flush=True)
        lengths.flush()

        total = np.zeros(n + 1, dtype=np.int64)
        step = 50_000_000
        running = 0
        for s in range(0, n, step):
            block = np.asarray(lengths[s:min(s + step, n)], dtype=np.int64)
            total[s + 1:s + 1 + len(block)] = np.cumsum(block) + running
            running = int(total[s + len(block)])
        np.save(f"{args.out}/configuration_id_offsets.npy", total)
        print(f"wrote configuration_id_offsets.npy  total bytes {running:,}", flush=True)

        with open(f"{args.out}/configuration_id_bytes.bin", "wb") as fh:
            fh.truncate(running)
        print(f"allocated configuration_id_bytes.bin ({running / 1e9:.1f} GB)",
              flush=True)
        return

    byte_offsets = np.load(f"{args.out}/configuration_id_offsets.npy", mmap_mode="r")
    mine = parts[args.task_id::args.num_tasks]
    # runs out of walltime on a big one must not redo the parts it already wrote
    done_dir = f"{args.out}/logs/parts"
    os.makedirs(done_dir, exist_ok=True)
    with open(f"{args.out}/configuration_id_bytes.bin", "r+b") as fh:
        for index, path in enumerate(mine):
            stamp = f"{done_dir}/{os.path.basename(path)}.done"
            if os.path.exists(stamp):
                print(f"task {args.task_id}: {index + 1}/{len(mine)} already done",
                      flush=True)
                continue
            data = np.load(path)
            cell, part_lengths = data["cell"], data["lengths"].astype(np.int64)
            payload = memoryview(data["payload"].data)
            local_offsets = np.concatenate([[0], np.cumsum(part_lengths)])

            spans = list(part_spans(cell, offsets))
            # 13.6 GB offsets memmap page-fault individually and dominate runtime
            row_starts = np.fromiter((s for s, _, _ in spans), dtype=np.int64,
                                     count=len(spans))
            file_starts = np.asarray(byte_offsets[row_starts])

            # spans are already in increasing global position, so writes move
            for (start, local, count), where in zip(spans, file_starts):
                lo = int(local_offsets[local])
                hi = int(local_offsets[local + count])
                fh.seek(int(where))
                fh.write(payload[lo:hi])
            fh.flush()
            with open(stamp, "w") as marker:
                marker.write("done\n")
            print(f"task {args.task_id}: bytes {index + 1}/{len(mine)} "
                  f"({len(spans):,} cells)", flush=True)

    os.makedirs(f"{args.out}/logs", exist_ok=True)
    with open(f"{args.out}/logs/ids_{args.task_id:04d}.done", "w") as fh:
        fh.write("done\n")
    print(f"task {args.task_id}: wrote {len(mine)} parts", flush=True)


if __name__ == "__main__":
    main()
