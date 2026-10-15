import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import metatrain

D = Path(__file__).resolve().parent.parent
XYZ = D / "manifests" / "xyz"
PREDS = D / "reports" / "preds"
TESTS = {"smoke": ("smoke_test_madcore", "smoke_test_random"), "main": ("test_madcore", "test_random")}
KEYS = {"delta_lda": ("delta_energy_lda", "delta_forces_lda"), "delta_pbe": ("delta_energy_pbe", "delta_forces_pbe")}

OPTIONS = """systems:
  read_from: {xyz}
  length_unit: angstrom
targets:
  "mtt::delta_energy":
    quantity: energy
    key: {energy_key}
    unit: eV
    forces:
      key: {forces_key}
    stress: false
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# the two correction runs of a stage, each with the target keys of its functional
def models(stage):
    runs = sorted(p.parent for p in (D / "runs" / stage).glob("*/options.yaml"))
    assert len(runs) == 2, runs
    return {r.name: (r, r / "model_best.pt", KEYS[r.name.split("--")[0]]) for r in runs}


def evaluate(name, run_dir, exported, keys, stage, work):
    # the exported best model, the best state of the checkpoint as metatrain export writes it
    if not exported.exists():
        subprocess.run([sys.executable, "-m", "metatrain", "export", "model.ckpt", "-o", exported.name], cwd=run_dir, check=True)
    for test in TESTS[stage]:
        out = PREDS / f"{stage}-{name}@{test}.xyz"
        if out.exists():
            print("have", out.name, flush=True)
            continue

        xyz = XYZ / f"{test}.xyz"
        options = work / f"eval-options-{name}-{test}.yaml"
        options.write_text(OPTIONS.format(xyz=xyz, energy_key=keys[0], forces_key=keys[1]))

        result = subprocess.run([sys.executable, "-m", "metatrain", "eval", str(exported), options.name, "-o", out.name, "-b", "16"],
                                cwd=work, capture_output=True, text=True)
        (work / f"eval-{name}@{test}.log").write_text(result.stdout + result.stderr)
        produced = work / out.name
        if result.returncode != 0 or not produced.exists():
            sys.exit(f"evaluation failed for {name} on {test}, see {work}/eval-{name}@{test}.log")
        produced.rename(out)

        provenance = {"model": name, "model_best_pt": str(exported), "model_best_pt_sha256": sha256(exported), "test_xyz": str(xyz), "test_xyz_sha256": sha256(xyz),
                      "predictions_sha256": sha256(out), "metatrain_package": str(Path(metatrain.__file__).parent)}
        (PREDS / f"{stage}-{name}@{test}.provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
        print("wrote", out.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    PREDS.mkdir(parents=True, exist_ok=True)
    assert "venv-metatrain-fix" in metatrain.__file__, metatrain.__file__

    work = D / "runs" / args.stage / "eval_work"
    work.mkdir(exist_ok=True)
    for name, (run_dir, exported, keys) in models(args.stage).items():
        if args.model and name != args.model:
            continue
        evaluate(name, run_dir, exported, keys, args.stage, work)
    print("EVALUATE DONE")


if __name__ == "__main__":
    main()
