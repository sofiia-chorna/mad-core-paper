import json
from pathlib import Path

S = Path(__file__).resolve().parent.parent
RUNS = S / "runs"
MANIFESTS = S / "manifests"
VAL_XYZ = Path("${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v2/manifests/xyz/val_combined_1250.xyz")
LAUNCHER = "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/scripts/launch_runs_fix.sh"
UPDATES = 187200
MAX_SNAPSHOTS = 480
NODES = {"a": ["scratchv3-s--madcore_2k--s42", "scratchv3-s--madcore_5k--s42", "scratchv3-s--madcore_10k--s42", "scratchv3-s--madcore_20k--s42"],
         "b": ["scratchv3-s--random_2k--s42", "scratchv3-s--random_5k--s42", "scratchv3-s--random_10k--s42", "scratchv3-s--random_20k--s42"]}

OPTIONS = """seed: 42
device: cuda
base_precision: 32

architecture:
  name: pet
  atomic_types: {atomic_types}
  model:
    cutoff: 8.0
    cutoff_width: 0.5
    d_pet: 256
    d_head: 256
    d_node: 1024
    d_feedforward: 512
    num_heads: 8
    num_attention_layers: 1
    num_gnn_layers: 3
    normalization: "RMSNorm"
    activation: "SwiGLU"
    transformer_type: "PreLN"
    featurizer_type: "feedforward"
    zbl: false
    cutoff_function: "Bump"
    num_neighbors_adaptive: 16
  training:
    batch_size: 64
    num_epochs: {num_epochs}
    learning_rate: 3e-4
    warmup_fraction: 0.01
    log_mae: true
    log_interval: 1
    checkpoint_interval: {checkpoint_interval}
    best_model_metric: mae_prod

training_set:
  systems:
    read_from: {train_xyz}
    length_unit: angstrom
  targets:
    energy:
      key: atomization_energy
      unit: eV
      forces:
        key: forces
      stress: false

validation_set:
  systems:
    read_from: {val_xyz}
    length_unit: angstrom
  targets:
    energy:
      key: atomization_energy
      unit: eV
      forces:
        key: forces
      stress: false

test_set: 0.0

wandb:
  project: mad-core-valid
  name: scratch_scaling_v3-{name}
  group: scratch_scaling_v3
  tags: [{train}, scratch, mse, pet-s, pbe, corrected-attention-mask, val-random1000-madcore250, isolated-atom-references]
"""

SUBMIT = """#!/bin/bash
set -uo pipefail
G={runs}
unfinished=0
for run in {run_list}; do
  grep -q "Training finished!" $G/$run/train.log 2>/dev/null || unfinished=1
done
if [ "$unfinished" = "1" ]; then
  sbatch --dependency=afterany:$SLURM_JOB_ID $G/submit_node_{node}.sh >> $G/chain_node_{node}.log 2>&1 || true
fi
{launcher} wait $G {run_list}
"""

SMOKE = """#!/bin/bash
set -uo pipefail
export UENV_WARN_MIGRATE=1
cd {exp}
{launcher} skip {runs}/smoke scratchv3-s--smoke_mixed--s42
uenv run pytorch/v2.9.1:v2 --view=default -- ${HOME}/venv-metatrain-fix/bin/python scripts/check_scratch_init.py {runs}/smoke/scratchv3-s--smoke_mixed--s42
uenv run pytorch/v2.9.1:v2 --view=default -- ${HOME}/venv-metatrain-fix/bin/python scripts/check_selection.py {runs}/smoke/scratchv3-s--smoke_mixed--s42
uenv run pytorch/v2.9.1:v2 --view=default -- ${HOME}/venv-metatrain-fix/bin/python scripts/check_references.py {runs}/smoke/scratchv3-s--smoke_mixed--s42
"""


def main():
    sets = json.load(open(MANIFESTS / "SETS.json"))
    atomic_types = "[" + ", ".join(str(z) for z in sets["atomic_types"]) + "]"
    provenance = json.load(open(RUNS / "PROVENANCE.json")) if (RUNS / "PROVENANCE.json").exists() else {"atomic_types": sets["atomic_types"], "runs": {}}
    assert provenance["atomic_types"] == sets["atomic_types"]
    for node, run_list in NODES.items():
        for name in run_list:
            train = name.split("--")[1]
            run_dir = RUNS / name
            steps = sets["sets"][train]["steps_per_epoch_batch_64_drop_last"]
            epochs = round(UPDATES / steps)
            interval = max(5, -(-epochs // MAX_SNAPSHOTS))
            if name in provenance["runs"]:
                assert provenance["runs"][name]["epochs"] == epochs, name
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "options.yaml").write_text(OPTIONS.format(atomic_types=atomic_types, num_epochs=epochs, checkpoint_interval=interval,
                                                                 train_xyz=MANIFESTS / "xyz" / f"{train}_aug.xyz", val_xyz=VAL_XYZ, name=name, train=train))
            provenance["runs"][name] = {"train": train, "epochs": epochs, "steps_per_epoch": steps, "optimizer_updates": steps * epochs,
                                        "checkpoint_interval": interval, "node": node}
        if not (RUNS / f"submit_node_{node}.sh").exists():
            (RUNS / f"submit_node_{node}.sh").write_text(SUBMIT.format(node=node, runs=RUNS, run_list=" ".join(run_list), launcher=LAUNCHER))
    smoke = RUNS / "smoke" / "scratchv3-s--smoke_mixed--s42"
    if not (smoke / "options.yaml").exists():
        smoke.mkdir(parents=True, exist_ok=True)
        (smoke / "options.yaml").write_text(OPTIONS.format(atomic_types=atomic_types, num_epochs=2, checkpoint_interval=1, train_xyz=MANIFESTS / "xyz" / "smoke_mixed.xyz", val_xyz=VAL_XYZ,
                                                           name="smoke-scratchv3-s--smoke_mixed--s42", train="smoke_mixed"))
        (RUNS / "smoke" / "submit_smoke.sh").write_text(SMOKE.format(runs=RUNS, exp=S, launcher=LAUNCHER))
    json.dump(provenance, open(RUNS / "PROVENANCE.json", "w"), indent=1)
    for name, v in provenance["runs"].items():
        print(f"{name:36s} {v['epochs']:5d} epochs x {v['steps_per_epoch']:3d} steps = {v['optimizer_updates']} updates, snapshot every {v.get('checkpoint_interval', 5)} epochs, node {v['node']}")


if __name__ == "__main__":
    main()
