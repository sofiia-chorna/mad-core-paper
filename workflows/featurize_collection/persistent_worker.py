#!/usr/bin/env python
import argparse
import glob
import os
import shutil
import signal
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_inference as ri  # noqa: E402

STALE_LOCK_SECONDS = 50000

_current_output = {"path": None}


def _release_on_term(signum, frame):
    out = _current_output["path"]
    if out:
        for tmp in glob.glob(out + ".tmp.*"):
            try:
                os.remove(tmp)
            except OSError:
                pass
        try:
            os.rmdir(out + ".lock")
        except OSError:
            pass
    os._exit(0)


signal.signal(signal.SIGTERM, _release_on_term)
signal.signal(signal.SIGINT, _release_on_term)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--store-root", required=True)
    p.add_argument("--sidecar-root", required=True)
    p.add_argument("--keep-cols", default=None)
    p.add_argument("--keep-cols-missing-ok", action="store_true")
    p.add_argument("--meanstd", action="store_true")
    p.add_argument("--max-nsites", type=int, default=None)
    p.add_argument("--nl-backend", choices=["metatomic", "ase"], default="metatomic")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--chunk-rows", type=int, default=50_000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-atomic", default=None)  # unused here; kept for run_inference compat
    p.add_argument("--offset", type=int, default=-1,
                   help="first manifest line to try (-1 = auto, spread by host/gpu/pid)")
    return p.parse_args()


def claim(output):
    if os.path.exists(output):
        return False

    os.makedirs(os.path.dirname(output), exist_ok=True)
    lock = output + ".lock"

    try:
        os.mkdir(lock)
        return True
    except FileExistsError:
        try:
            age = time.time() - os.path.getmtime(lock)
        except OSError:
            return False
        if os.path.exists(output) or age < STALE_LOCK_SECONDS:
            return False
        try:
            os.rmdir(lock)
            os.mkdir(lock)
            return True
        except OSError:
            return False


def relocate_sidecar(args, output):
    relative = output[len(args.store_root):].lstrip("/")
    dest = os.path.join(args.sidecar_root, relative[:-len(".parquet")] + ".json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    try:
        shutil.move(output[:-len(".parquet")] + ".json", dest)  # mv: handles store->scratch cross-FS
    except OSError:
        pass


def main():
    args = parse_args()
    ctx = ri.build_context(args)

    lines = [ln.rstrip("\n") for ln in open(args.manifest) if ln.strip()]
    total = len(lines)
    if total == 0:
        return 0
    if args.offset >= 0:
        start = args.offset % total
    else:
        seed = f"{socket.gethostname()}-{os.environ.get('CUDA_VISIBLE_DEVICES', '')}-{os.getpid()}"
        start = abs(hash(seed)) % total
    order = list(range(start, total)) + list(range(0, start))

    ri.log(f"persistent worker: {total} files, start={start}, host={socket.gethostname()}")
    n_done = 0
    for i in order:
        source, output, batch = lines[i].split("\t")
        if os.path.exists(output) or not claim(output):
            continue
        _current_output["path"] = output
        try:
            if os.path.exists(output):
                continue
            ri.featurize_file(args, ctx, source, output, int(batch))
            relocate_sidecar(args, output)
            n_done += 1
        except Exception as exc:
            ri.log(f"FAILED {source}: {exc!r}")
        finally:
            try:
                os.rmdir(output + ".lock")
            except OSError:
                pass
            _current_output["path"] = None
    ri.log(f"persistent worker finished: featurized {n_done} files this pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
