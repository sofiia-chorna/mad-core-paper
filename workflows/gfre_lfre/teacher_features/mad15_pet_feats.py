import os
import sys

import ase.io
import metatomic.torch as mta
import numpy as np
import torch
import vesin.metatomic as vesin_metatomic
from metatrain.utils.io import load_model as load_metatrain_model

DATASET_PATH = "data/mad-1.5/mad-1.5-r2scan-all.xyz"
OUTPUT_DIR = "data/mad-1.5/pet"

MODEL_PATHS = {
    "mad": "${OTHER_WORKSPACE}/models/pet-mad-v1.0.2.pt",
    "mad-xs-v1.5.0": "${KUMA_WORK_ROOT}/data/models/pet-mad-xs-v1.5.0.pt",
    "mad-s-v1.5.0": "${KUMA_WORK_ROOT}/data/models/pet-mad-s-v1.5.0.pt",
}

if len(sys.argv) < 2:
    print("Usage: python3 mad15_pet_feats.py <MODEL_NAME> [LL|BB]")
    print(f"  Available models: {list(MODEL_PATHS.keys())}")
    sys.exit(1)

MODEL_KEY = sys.argv[1].lower()
VARIANT = sys.argv[2].upper() if len(sys.argv) > 2 else "LL"

if MODEL_KEY not in MODEL_PATHS:
    print(f"Error: unknown model '{MODEL_KEY}'. Available: {list(MODEL_PATHS.keys())}")
    sys.exit(1)

if VARIANT not in ["LL", "BB"]:
    print(f"Error: unknown variant '{VARIANT}'. Available: ['LL', 'BB']")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = MODEL_PATHS[MODEL_KEY]
device = "cuda" if torch.cuda.is_available() else "cpu"

requested_key = (
    "mtt::aux::energy_last_layer_features" if VARIANT == "LL" else "features"
)

suffix = "_ll" if VARIANT == "LL" else "_bb"
output_path = f"{OUTPUT_DIR}/{MODEL_KEY}{suffix}.npy"

print(f"Model: {MODEL_KEY}")
print(f"Variant: {VARIANT}")
print(f"Dataset: {DATASET_PATH}")
print(f"Model path: {MODEL_PATH}")
print(f"Output: {output_path}")
print(f"Requested key: {requested_key}")
print(f"Using device: {device}")

print(f"Loading model '{MODEL_KEY}' from {MODEL_PATH}")
model = load_metatrain_model(MODEL_PATH).to(device)

structures = ase.io.read(DATASET_PATH, ":")
print(f"Loaded {len(structures)} frames from {DATASET_PATH}")

length_unit = model.capabilities().length_unit
batch_size = 64
dtype = torch.float32
check_consistency = False

print("Converting structures to torch...")
systems = mta.systems_to_torch(structures)
vesin_metatomic.compute_requested_neighbors(systems, length_unit, model)
systems = [s.to(dtype=dtype, device=device) for s in systems]

options = mta.ModelEvaluationOptions(
    length_unit=length_unit,
    outputs={requested_key: mta.ModelOutput(per_atom=True)},
    selected_atoms=None,
)

print("Running inference...")
outputs = []
for i_sys in range(0, len(systems), batch_size):
    if i_sys % 100 == 0:
        print(f"Processing batch {i_sys}/{len(systems)}")
    batch = systems[i_sys : i_sys + batch_size]
    with torch.jit.optimized_execution(False):
        with torch.no_grad():
            batch_out = model(batch, options, check_consistency=check_consistency)
    outputs.append(batch_out[requested_key].block().values.detach().cpu().numpy())

outputs_np = np.concatenate(outputs)
print(f"Per-atom features shape: {outputs_np.shape}")
np.save(output_path, outputs_np)
print(f"Saved features to {output_path}")
