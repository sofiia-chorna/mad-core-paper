import os
import sys
import time
import tempfile

import ase.io
import numpy as np
import torch
from mace import data, tools
from mace.data import KeySpecification
from mace.modules import LLPRModel
from mace.tools import torch_geometric
from mace.tools.scripts_utils import get_dataset_from_xyz

start_total = time.time()

DATASET_PATH = "data/mad-1.5/mad-1.5-r2scan-all.xyz"


def filter_structures_for_mace(structures, supported_atomic_numbers):
    filtered = []
    kept_ids = []
    excluded_ids = []
    unsupported_elements_found = set()

    for i, s in enumerate(structures):
        atomic_set = set(s.get_atomic_numbers().tolist())
        unsupported = atomic_set - supported_atomic_numbers
        if unsupported:
            excluded_ids.append(i)
            unsupported_elements_found.update(unsupported)
            continue
        filtered.append(s)
        kept_ids.append(i)

    print(
        f"model supports {len(supported_atomic_numbers)} elements: Z in {sorted(supported_atomic_numbers)[:10]}...{sorted(supported_atomic_numbers)[-5:]}"
    )
    print(
        f"filtered out {len(excluded_ids)} structures with unsupported atomic numbers"
    )
    if unsupported_elements_found:
        print(
            f"  unsupported elements found in data: {sorted(unsupported_elements_found)}"
        )
    return (
        filtered,
        np.array(kept_ids, dtype=np.int64),
        np.array(excluded_ids, dtype=np.int64),
    )


OUTPUT_DIR = "data/mad-1.5/mace"

MODEL_DICT = {
    "mace-mp-0b3": {
        "model_path": "${OTHER_WORKSPACE}/models/mace-mp-0b3-medium.model",
    },
    "mace-omat-0-medium": {
        "model_path": "${KUMA_WORK_ROOT}/data/models/mace-omat-0-medium.model",
    },
}

if len(sys.argv) < 2:
    print("Usage: python3 mad15_mace_feats.py <MODEL_KEY> [LL|BB]")
    print(f"  Available models: {list(MODEL_DICT.keys())}")
    sys.exit(1)

KEY = sys.argv[1].lower()
VARIANT = sys.argv[2].upper() if len(sys.argv) > 2 else "LL"

if KEY not in MODEL_DICT:
    print(f"Error: unknown model '{KEY}'. Available: {list(MODEL_DICT.keys())}")
    sys.exit(1)

if VARIANT not in ["LL", "BB"]:
    print(f"Error: unknown variant '{VARIANT}'. Available: ['LL', 'BB']")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.set_default_dtype(torch.float32)
device = tools.init_device("cuda")

MODEL_PATH = MODEL_DICT[KEY]["model_path"]
suffix = "_ll" if VARIANT == "LL" else "_bb"
OUTPUT_PATH = f"{OUTPUT_DIR}/{KEY}{suffix}"

print(f"Model: {KEY}")
print(f"Variant: {VARIANT}")
print(f"Dataset: {DATASET_PATH}")
print(f"Model path: {MODEL_PATH}")
print(f"Output: {OUTPUT_PATH}.npy")

t0 = time.time()
model = torch.load(MODEL_PATH, weights_only=False)
model = model.float()
mace_llpr = LLPRModel(model).to(device)
print(f"loaded model and moved to {device} in {time.time() - t0:.2f} s")

supported_atomic_numbers = set(model.atomic_numbers.tolist())

print("loading structures...")
all_structures = ase.io.read(DATASET_PATH, ":")
print(f"loaded {len(all_structures)} structures")
filtered_structures, kept_ids, filtered_ids = filter_structures_for_mace(
    all_structures, supported_atomic_numbers
)
print(f"using {len(filtered_structures)} structures after filtering")

ids_path = f"{OUTPUT_DIR}/{KEY}_ids.npz"
np.savez(ids_path, kept=kept_ids)
print(f"saved indices to {ids_path}")

temp_xyz = tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False)
temp_path = temp_xyz.name
temp_xyz.close()
ase.io.write(temp_path, filtered_structures)
print(f"wrote filtered structures to temp file: {temp_path}")

atomic_numbers = model.atomic_numbers.tolist()
atomic_energies = model.atomic_energies_fn.atomic_energies.squeeze().tolist()
stats = {
    "atomic_numbers": atomic_numbers,
    "atomic_energies": {
        int(z): float(e) for z, e in zip(atomic_numbers, atomic_energies)
    },
    "r_max": model.r_max.item(),
}

key_specification_config = KeySpecification(
    info_keys={"energy": "energy", "stress": "stress", "head": "config_type"},
    arrays_keys={"forces": "forces"},
)

config_type_weights = {"Default": 1.0}
z_table = tools.get_atomic_number_table_from_zs(stats["atomic_numbers"])

t0 = time.time()
collections, atomic_energies_dict = get_dataset_from_xyz(
    work_dir="",
    train_path=temp_path,
    valid_path="${OTHER_WORKSPACE}/data/raw/dummy_valid.xyz",
    test_path="${OTHER_WORKSPACE}/data/raw/dummy_test.xyz",
    valid_fraction=0,
    config_type_weights=config_type_weights,
    key_specification=key_specification_config,
)
print(f"Loaded dataset in {time.time() - t0:.2f} s")

os.unlink(temp_path)

train_loader = torch_geometric.dataloader.DataLoader(
    dataset=[
        data.AtomicData.from_config(config, z_table=z_table, cutoff=stats["r_max"])
        for config in collections.train
    ],
    batch_size=64,
    shuffle=False,
    drop_last=False,
)


def get_mace_descriptors(model, node_feats, invariants_only=True):
    # node_feats from LLPRModel is all interaction layers concatenated
    irreps_out = model.products[0].linear.irreps_out
    l_max = irreps_out.lmax

    num_invariant_features = 0
    for mul, ir in irreps_out:
        if ir.l == 0:
            num_invariant_features += mul

    num_interactions = len(model.interactions)

    per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
    per_layer_features[-1] = num_invariant_features

    if invariants_only:
        descriptors_list = []
        for i in range(num_interactions):
            start = i * irreps_out.dim
            end = start + num_invariant_features
            descriptors_list.append(node_feats[:, start:end])
        return torch.cat(descriptors_list, dim=-1)
    else:
        to_keep = sum(per_layer_features)
        return node_feats[:, :to_keep]


def aggregate_features(mace_llpr, ll_feats):
    # apply the readout MLP (or identity if linear) to each layer's features, then concat
    ll_feats_list = torch.split(ll_feats, mace_llpr.hidden_sizes_before_readout, dim=-1)
    ll_feats_list = [
        (ll_feats if is_linear else readout.non_linearity(readout.linear_1(ll_feats)))[
            :, :size
        ]
        for ll_feats, readout, size, is_linear in zip(
            ll_feats_list,
            mace_llpr.orig_model.readouts.children(),
            mace_llpr.hidden_sizes,
            mace_llpr.readouts_are_linear,
        )
    ]
    return torch.cat(ll_feats_list, dim=-1)


t0 = time.time()
all_feats = []
mace_llpr.eval()

print("Running inference...")
with torch.no_grad():
    for i, batch in enumerate(train_loader):
        if i % 100 == 0:
            print(f"Processing batch {i}")

        batch = batch.to(device)
        outputs = mace_llpr(batch, compute_force=False, compute_stress=False)

        if VARIANT == "BB":
            # backbone invariant features from each interaction layer
            feats = get_mace_descriptors(
                model=model, node_feats=outputs["node_feats"], invariants_only=True
            )
        else:
            # last layer features after the readout MLP
            feats = aggregate_features(mace_llpr, outputs["node_feats"])

        all_feats.append(feats.detach().cpu())

all_feats = torch.cat(all_feats, dim=0)
all_feats_np = all_feats.numpy()

print(f"Per-atom features shape: {all_feats_np.shape}")
print(f"Number of atoms: {len(all_feats_np)}")

np.save(OUTPUT_PATH, all_feats_np)
print(f"Saved {VARIANT} features to {OUTPUT_PATH}.npy")

print(f"Feature extraction completed in {time.time() - t0:.2f} s")
print(f"Total runtime: {time.time() - start_total:.2f} s")
