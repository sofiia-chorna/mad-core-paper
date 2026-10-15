import shutil
import sys
from pathlib import Path

import metatomic.torch as mta
import torch
from metatrain.utils.io import model_from_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from water import FLOAT64, LDA_PT, LIPS, PBE_PT, sha256  # noqa: E402


# float64 copies of the validation best weights of both corrections, exported from the checkpoint next to each model_best.pt
def export(run_pt, target):
    checkpoint = torch.load(run_pt.parent / "model.ckpt", map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint, context="export")
    exported = model.to(torch.float64).export()
    exported.save(str(target))

    loaded = mta.load_atomistic_model(str(target))
    assert loaded.capabilities().dtype == "float64", loaded.capabilities().dtype
    print("wrote", target, "best epoch", checkpoint["best_epoch"], "dtype", loaded.capabilities().dtype, "sha256", sha256(target)[:16])


def main():
    FLOAT64.mkdir(parents=True, exist_ok=True)
    export(LDA_PT, FLOAT64 / "delta_lda.float64.pt")
    export(PBE_PT, FLOAT64 / "delta_pbe.float64.pt")

    # the baseline float64 copy of the LiPS experiment, unchanged
    for name in ("baseline_inner.float64.pt", "baseline_inner.float32.pt"):
        shutil.copyfile(LIPS / "verification" / "float64" / name, FLOAT64 / name)
        print("copied", name, "sha256", sha256(FLOAT64 / name)[:16])


if __name__ == "__main__":
    main()
