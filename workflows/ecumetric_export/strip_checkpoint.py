import argparse
import hashlib

import metatomic.torch as mta
import torch
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.metadata import merge_metadata
from omegaconf import OmegaConf


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--reference-model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

torch.set_num_interop_threads(1)
torch.set_num_threads(1)


def sha256_of(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(64 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def assert_same_tensors(reference, candidate, label):
    reference_tensors = {key: value for key, value in reference.items() if torch.is_tensor(value)}
    candidate_tensors = {key: value for key, value in candidate.items() if torch.is_tensor(value)}

    if reference_tensors.keys() != candidate_tensors.keys():
        raise SystemExit(f"{label} has different tensor names")

    for key, value in reference_tensors.items():
        if not torch.equal(value, candidate_tensors[key]):
            raise SystemExit(f"{label} differs in {key}")

    print(f"{label} identical, {len(reference_tensors)} tensors")


full_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
best_weights = full_checkpoint["best_model_state_dict"]

metadata = merge_metadata(
    full_checkpoint["metadata"],
    mta.ModelMetadata(**OmegaConf.to_container(OmegaConf.load(args.metadata))),
)

# only the keys metatrain reads to export or finetune, the best weights in both state dicts
stripped_checkpoint = {
    "architecture_name": full_checkpoint["architecture_name"],
    "model_ckpt_version": full_checkpoint["model_ckpt_version"],
    "metadata": metadata,
    "model_data": full_checkpoint["model_data"],
    "epoch": None,
    "best_epoch": None,
    "model_state_dict": dict(best_weights),
    "best_model_state_dict": dict(best_weights),
}

print("removed keys", sorted(set(full_checkpoint) - set(stripped_checkpoint)))

torch.save(stripped_checkpoint, args.output)


# the file read back from disk has the same best weights
reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
assert_same_tensors(best_weights, reloaded["best_model_state_dict"], "stripped best weights")
assert_same_tensors(best_weights, reloaded["model_state_dict"], "stripped model weights")

# metatrain builds the same model from both files
full_model = model_from_checkpoint(torch.load(args.checkpoint, map_location="cpu", weights_only=False), context="export")
stripped_model = model_from_checkpoint(reloaded, context="export")
assert_same_tensors(full_model.state_dict(), stripped_model.state_dict(), "model from stripped vs full checkpoint")

# the export of the stripped checkpoint matches the shipped model
exported = stripped_model.export()
reference_model = mta.load_atomistic_model(args.reference_model)
assert_same_tensors(reference_model.state_dict(), exported.state_dict(), "export vs reference model")

if str(exported.metadata()) != str(reference_model.metadata()):
    raise SystemExit("export metadata differs from the reference model")

print("export metadata identical to the reference model")
print(exported.metadata())

print(f"{args.checkpoint} {sha256_of(args.checkpoint)}")
print(f"{args.output} {sha256_of(args.output)}")
