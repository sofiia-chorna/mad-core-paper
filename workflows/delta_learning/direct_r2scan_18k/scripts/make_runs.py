import sys
from pathlib import Path

D = Path(__file__).resolve().parent.parent
E = D.parent
TEMPLATE = E / "delta_pbe_10k_vs_mad16" / "config" / "options_direct.yaml"
XYZ = E / "delta_lda_pbe_18k" / "manifests" / "xyz"
CHECKPOINT = E / "models" / "pet-omat-s-v1.0.0.ckpt"
RUN = "direct_r2scan--omat-s--madcore_shared18000--s42"
STAGES = {
    "smoke": {"train": XYZ / "smoke_train.xyz", "val": XYZ / "smoke_val.xyz", "epochs": 2, "ckpt_interval": 1},
    "main": {"train": XYZ / "train.xyz", "val": XYZ / "val.xyz", "epochs": 150, "ckpt_interval": 25},
}


def options(stage):
    text = TEMPLATE.read_text()

    # the target is named apart from the pretrained energy so that metatrain prunes the pretrained head and starts a fresh one,
    # the same head replacement the delta runs get through their mtt::delta_energy target; labels stay r2scan atomization energies and forces
    assert text.count("    energy:\n      key: atomization_energy\n") == 2
    text = text.replace("    energy:\n      key: atomization_energy\n", '    "mtt::r2scan_energy":\n      quantity: energy\n      key: atomization_energy\n')
    text = text.replace("direct-r2scan, full, delta-experiment", "direct-r2scan, full, direct-shared-split, fresh-head, corrected-attention-mask")

    settings = STAGES[stage]
    return text.format(
        seed=42, ckpt=CHECKPOINT, batch=64, epochs=settings["epochs"], ckpt_interval=settings["ckpt_interval"],
        train=settings["train"], val=settings["val"], name=f"direct_r2scan_18k-{stage}-{RUN}", group=f"direct_r2scan_18k-{stage}",
        arm="madcore_shared18000", model="omat-s",
    )


def main():
    stage = sys.argv[1]
    run_dir = D / "runs" / stage / RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "options.yaml").write_text(options(stage))
    print("wrote", run_dir / "options.yaml")


if __name__ == "__main__":
    main()
