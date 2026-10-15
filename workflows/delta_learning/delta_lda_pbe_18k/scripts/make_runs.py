import sys
from pathlib import Path

D = Path(__file__).resolve().parent.parent
E = D.parent
TEMPLATE = E / "delta_pbe_10k_vs_mad16" / "config" / "options_delta.yaml"
XYZ = D / "manifests" / "xyz"
CHECKPOINT = E / "models" / "pet-omat-s-v1.0.0.ckpt"

# run name, target keys of the energy and the forces, wandb tag
RUNS = {
    "delta_lda--omat-s--madcore_shared18000--s42": ("delta_energy_lda", "delta_forces_lda", "delta-r2scan-lda"),
    "delta_pbe--omat-s--madcore_shared18000--s42": ("delta_energy_pbe", "delta_forces_pbe", "delta-r2scan-pbe"),
}
STAGES = {
    "smoke": {"train": XYZ / "smoke_train.xyz", "val": XYZ / "smoke_val.xyz", "epochs": 2, "ckpt_interval": 1},
    "main": {"train": XYZ / "train.xyz", "val": XYZ / "val.xyz", "epochs": 150, "ckpt_interval": 25},
}


def options(run, stage):
    energy_key, forces_key, tag = RUNS[run]
    text = TEMPLATE.read_text()

    # only the target keys and the tags differ from the established delta template
    assert text.count("key: delta_energy\n") == 2 and text.count("key: delta_forces\n") == 2
    text = text.replace("key: delta_energy\n", f"key: {energy_key}\n").replace("key: delta_forces\n", f"key: {forces_key}\n")
    text = text.replace("delta-r2scan-pbe, full, delta-experiment", f"{tag}, full, delta-shared-split, corrected-attention-mask")

    settings = STAGES[stage]
    return text.format(
        seed=42, ckpt=CHECKPOINT, batch=64, epochs=settings["epochs"], ckpt_interval=settings["ckpt_interval"],
        train=settings["train"], val=settings["val"], name=f"delta_lda_pbe_18k-{stage}-{run}", group=f"delta_lda_pbe_18k-{stage}",
        arm="madcore_shared18000", model="omat-s",
    )


def main():
    stage = sys.argv[1]
    for run in RUNS:
        run_dir = D / "runs" / stage / run
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "options.yaml").write_text(options(run, stage))
        print("wrote", run_dir / "options.yaml")


if __name__ == "__main__":
    main()
