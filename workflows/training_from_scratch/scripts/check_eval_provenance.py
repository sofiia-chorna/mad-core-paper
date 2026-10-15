import hashlib
import json
import re
import sys
from pathlib import Path

import torch
import metatomic.torch  # noqa: F401
from metatomic.torch import load_atomistic_model

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/scripts")
from common import read_tsv  # noqa: E402

# provenance of every evaluated row of this round: checkpoint path and hash, which state the exported model carries
# (best or final, verified tensor by tensor against the checkpoint), prediction cache path, test file hash; every path
# must resolve inside this round's directory (tests are the shared frozen files of pbe_20k); exits nonzero otherwise
S = Path(__file__).resolve().parent.parent
RUNS = S / "runs"
REPORTS = S / "reports"
TESTS_DIR = Path("${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/manifests/xyz")
EXPECTED_TEST_SHA = {"test_random": "9b5ecf69abbdec32ce197c2b271e8d5851db65568d0c01006f556de00ace24af", "test_madcore": "4ca941bf622d364d992f37054d61307453be9d152ea152c4c777c548a075cb60"}
GUARD = len(sys.argv) > 1 and sys.argv[1] == "guard"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(path, root):
    return str(Path(path).resolve()).startswith(str(root.resolve()) + "/")


def exported_tensors(path):
    model = load_atomistic_model(str(path))
    return {k.split(".", 1)[1] if k.startswith("module.") else k: v for k, v in model.state_dict().items()}


def state_match(exported, state):
    shared = [k for k in state if k in exported and isinstance(state[k], torch.Tensor) and state[k].dtype.is_floating_point]
    equal = sum(torch.equal(exported[k].to(state[k].dtype), state[k]) for k in shared)
    return len(shared), equal


def main():
    rows = []
    problems = []
    tables = {"selected": REPORTS / "selected_checkpoints" / "metrics.tsv", "final": REPORTS / "final_checkpoints" / "metrics_final.tsv"}
    ckpt_cache = {}
    for kind, table in tables.items():
        if not table.exists():
            continue
        for r in read_tsv(table):
            run, test = r["model"], r["test"]
            run_dir = RUNS / run
            ckpt = run_dir / "model.ckpt"
            exported = run_dir / ("model_best.pt" if kind == "selected" else "model_final.pt")
            pred = REPORTS / "selected_checkpoints" / "preds" / f"{run}@{test}.xyz" if kind == "selected" else REPORTS / "final_checkpoints" / f"{run}@{test}.xyz"
            cache = REPORTS / "selected_checkpoints" / "cache" / f"{run}@{test}.npz" if kind == "selected" else REPORTS / "final_checkpoints" / f"{run}@{test}.npz"
            eval_log = run_dir / f"eval@{test}.log" if kind == "selected" else REPORTS / "final_checkpoints" / f"{run}@{test}.log"
            package = re.findall(r"Package directory: (\S+)", eval_log.read_text(errors="replace")) if eval_log.exists() else []
            if not package or not all(x.startswith("${HOME}/venv-metatrain-fix/") for x in package):
                problems.append(f"{kind} {run} {test}: evaluation log {eval_log} does not show the corrected package ({package})")
            for p in (run_dir, ckpt, exported, pred, cache):
                if not p.exists():
                    problems.append(f"{kind} {run} {test}: missing {p}")
                if not inside(p, S):
                    problems.append(f"{kind} {run} {test}: {p} resolves outside {S}")
            if run not in ckpt_cache:
                c = torch.load(ckpt, map_location="cpu", weights_only=False)
                ckpt_cache[run] = {"sha": sha256(ckpt), "epoch": c.get("epoch"), "best_epoch": c.get("best_epoch"), "best": c["best_model_state_dict"], "final": c["model_state_dict"],
                                   "best_pt": exported_tensors(run_dir / "model_best.pt") if (run_dir / "model_best.pt").exists() else None,
                                   "final_pt": exported_tensors(run_dir / "model_final.pt") if (run_dir / "model_final.pt").exists() else None}
            info = ckpt_cache[run]
            tensors = info["best_pt"] if kind == "selected" else info["final_pt"]
            n, eq_intended = state_match(tensors, info["best"] if kind == "selected" else info["final"])
            _, eq_other = state_match(tensors, info["final"] if kind == "selected" else info["best"])
            state_ok = n > 0 and eq_intended == n and (eq_other < n or info["best_epoch"] == info["epoch"])
            if not state_ok:
                problems.append(f"{kind} {run}: exported model matches {eq_intended}/{n} tensors of the intended state, {eq_other}/{n} of the other")
            test_sha = sha256(TESTS_DIR / f"{test}.xyz")
            if test_sha != EXPECTED_TEST_SHA[test]:
                problems.append(f"{test}: unexpected test file hash {test_sha}")
            rows.append({"checkpoint": kind, "run": run, "test": test, "checkpoint_path": str(ckpt.resolve()), "checkpoint_sha256": info["sha"], "checkpoint_epoch": info["epoch"], "best_epoch": info["best_epoch"],
                         "state": "best_model_state_dict (selected epoch)" if kind == "selected" else "model_state_dict (final epoch)", "state_tensors_matching": f"{eq_intended}/{n}", "other_state_matching": f"{eq_other}/{n}",
                         "evaluation_package": ";".join(sorted(set(package))), "exported_model": str(exported.resolve()), "exported_sha256": sha256(exported), "predictions": str(pred.resolve()), "cache": str(cache.resolve()), "test_xyz": str(TESTS_DIR / f"{test}.xyz"), "test_sha256": test_sha})
    cols = list(rows[0]) if rows else []
    with open(REPORTS / "eval_provenance.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    json.dump({"rows": len(rows), "problems": problems}, open(REPORTS / "eval_provenance_check.json", "w"), indent=1)
    for r in rows:
        print(f"{r['checkpoint']:8s} {r['run']:30s} {r['test']:12s} ckpt {r['checkpoint_sha256'][:12]} epoch {r['checkpoint_epoch']} best {r['best_epoch']} state {r['state_tensors_matching']} (other {r['other_state_matching']}) export {r['exported_sha256'][:12]} pred {r['predictions'].replace(str(S), '.')}")
    print("problems:", problems or "none")
    if problems:
        print("PROVENANCE CHECK FAILED")
        sys.exit(1)
    print("PROVENANCE CHECK PASSED")


if __name__ == "__main__":
    main()
