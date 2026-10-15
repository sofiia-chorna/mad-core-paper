import hashlib
import json
import subprocess
import sys
from pathlib import Path

import metatomic.torch  # noqa: F401
import metatrain
import torch

D = Path(__file__).resolve().parent.parent
RUN = D / "runs" / "omat-s--madcore_20k_aligned--s42"
TESTS = Path("${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/manifests/xyz")
PREDS = D / "reports" / "preds"
OPTIONS = """systems:
  read_from: {xyz}
  length_unit: angstrom
targets:
  energy:
    key: atomization_energy
    unit: eV
    forces:
      key: forces
    stress: false
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


# the selected checkpoint exported once and evaluated on both full frozen tests
def main():
    assert "venv-metatrain-fix" in metatrain.__file__, metatrain.__file__
    PREDS.mkdir(parents=True, exist_ok=True)
    exported = RUN / "model_best.pt"
    if not exported.exists():
        subprocess.run([sys.executable, "-m", "metatrain", "export", "model.ckpt", "-o", exported.name], cwd=RUN, check=True)
    ckpt = torch.load(RUN / "model.ckpt", map_location="cpu", weights_only=False)

    for test in ("test_random", "test_madcore"):
        out = PREDS / f"{RUN.name}@{test}.xyz"
        if not out.exists():
            options = RUN / f"eval-options-{test}.yaml"
            options.write_text(OPTIONS.format(xyz=TESTS / f"{test}.xyz"))
            result = subprocess.run([sys.executable, "-m", "metatrain", "eval", exported.name, options.name, "-o", out.name, "-b", "16"],
                                    cwd=RUN, capture_output=True, text=True)
            (RUN / f"eval@{test}.log").write_text(result.stdout + result.stderr)
            if result.returncode != 0 or not (RUN / out.name).exists():
                sys.exit(f"evaluation failed on {test}, see {RUN}/eval@{test}.log")
            (RUN / out.name).rename(out)
        provenance = {"checkpoint_sha256": sha256(RUN / "model.ckpt"), "epochs": ckpt["epoch"] + 1, "best_epoch_index": ckpt["best_epoch"],
                      "best_metric_mae_prod": ckpt["best_metric"], "model_best_pt_sha256": sha256(exported), "test_xyz": str(TESTS / f"{test}.xyz"),
                      "test_xyz_sha256": sha256(TESTS / f"{test}.xyz"), "predictions_sha256": sha256(out), "metatrain_package": str(Path(metatrain.__file__).parent)}
        (PREDS / f"{RUN.name}@{test}.provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
        print("evaluated", test, flush=True)
    print("EVALUATE DONE")


if __name__ == "__main__":
    main()
