import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import metatrain

D = Path(__file__).resolve().parent.parent
E = D.parent
XYZ = E / "delta_lda_pbe_18k" / "manifests" / "xyz"
PREDS = D / "reports" / "preds"
TESTS = {"smoke": ("smoke_test_madcore", "smoke_test_random"), "main": ("test_madcore", "test_random")}
OUTPUT = "mtt::r2scan_energy"

OPTIONS = """systems:
  read_from: {xyz}
  length_unit: angstrom
targets:
  "mtt::r2scan_energy":
    quantity: energy
    key: atomization_energy
    unit: eV
    forces:
      key: forces
    stress: false
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_dir(stage):
    runs = sorted(p.parent for p in (D / "runs" / stage).glob("*/options.yaml"))
    assert len(runs) == 1, runs
    return runs[0]


# standalone r2scan predictions of the validation best model, no DFT energy or force is added
def evaluate(stage):
    run = run_dir(stage)
    exported = run / "model_best.pt"
    work = D / "runs" / stage / "eval_work"
    work.mkdir(exist_ok=True)
    if not exported.exists():
        subprocess.run([sys.executable, "-m", "metatrain", "export", "model.ckpt", "-o", exported.name], cwd=run, check=True)

    for test in TESTS[stage]:
        out = PREDS / f"{stage}-{run.name}@{test}.xyz"
        if out.exists():
            print("have", out.name, flush=True)
            continue
        xyz = XYZ / f"{test}.xyz"
        options = work / f"eval-options-{test}.yaml"
        options.write_text(OPTIONS.format(xyz=xyz))
        result = subprocess.run([sys.executable, "-m", "metatrain", "eval", str(exported), options.name, "-o", out.name, "-b", "16"], cwd=work, capture_output=True, text=True)
        (work / f"eval-{test}.log").write_text(result.stdout + result.stderr)
        produced = work / out.name
        if result.returncode != 0 or not produced.exists():
            sys.exit(f"evaluation failed on {test}, see {work}/eval-{test}.log")
        produced.rename(out)
        provenance = {"model": run.name, "model_best_pt": str(exported), "model_best_pt_sha256": sha256(exported), "test_xyz": str(xyz), "test_xyz_sha256": sha256(xyz),
                      "predictions_sha256": sha256(out), "metatrain_package": str(Path(metatrain.__file__).parent)}
        (PREDS / f"{stage}-{run.name}@{test}.provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
        print("wrote", out.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    PREDS.mkdir(parents=True, exist_ok=True)
    assert "venv-metatrain-fix" in metatrain.__file__, metatrain.__file__
    evaluate(args.stage)
    print("EVALUATE DONE")


if __name__ == "__main__":
    main()
