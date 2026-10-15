import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SMAP = Path(__file__).resolve().parent.parent
PAPER = SMAP.parent
VALID = PAPER.parent
PROD = VALID / "prod_experiment"
sys.path.insert(0, str(PROD / "scripts"))

from common import MANIFESTS as PROD_MANIFESTS, RELEASE, make_logger, read_tsv, write_tsv  # noqa: E402,F401

REPORTS = SMAP / "reports"
FIGURES = REPORTS / "figures"
LOGS = SMAP / "logs"

COLLECTION = RELEASE / "collection"
CORESET = RELEASE / "mad-core"
MAD_ANCHOR = RELEASE / "mad-next-coreset"
SCALER_FILE = RELEASE / "ecumetric_scaler.npz"
PROJECTOR_FILE = RELEASE / "ecumetric_sketchmap.pt"
POPULATION_FILE = CORESET / "voronoi_population.npy"

EVAL = PAPER / "pbe_evaluation"
AUG = PAPER / "pbe_augmentation"
MIX = PAPER / "pbe_mixtures"

N_CORESET = 2 ** 24
FEATURE_DIM = 512

MAIN_SETS = {
    "random_test": (EVAL / "manifests" / "random_test_common.tsv", "test"),
    "random_holdout": (EVAL / "manifests" / "random_holdout_common.tsv", "holdout_all"),
    "madcore_holdout": (EVAL / "manifests" / "madcore_holdout_common.tsv", "holdout_all"),
}

AUG_SETS = {
    "random_test": (AUG / "manifests" / "frozen_random_test.tsv", "test"),
    "random_remainder": (AUG / "manifests" / "frozen_random_remainder_v18664.tsv", "holdout_all"),
    "madcore_remainder": (AUG / "manifests" / "frozen_madcore_remainder_v18664.tsv", "holdout_all"),
}

MAIN_MODELS = {
    "random_10k": PROD / "reports" / "cache" / "pbe-stage1_primary-omat-s--random_10k--s42__best",
    "madcore_10k": PROD / "reports" / "cache" / "pbe-stage1_primary-omat-s--madcore_10k--s42__best",
    "voronoi_10k": PROD / "reports" / "cache" / "pbe-stage1_primary-omat-s--voronoi_10k--s42__best",
}

AUG_MODELS = {
    "random_addition": AUG / "reports" / "cache" / "pbe-augmentation-omat-s--random_addition--s42__best",
    "madcore_top": AUG / "reports" / "cache" / "pbe-augmentation-omat-s--madcore_top--s42__best",
    "voronoi_addition": AUG / "reports" / "cache" / "pbe-augmentation-omat-s--voronoi_addition--s42__best",
}

TRUTH = {"test": MIX / "reports" / "cache" / "_truth.npz",
         "holdout_all": MIX / "reports" / "cache" / "_truth@holdout_all.npz"}


def frame_order(source):
    return np.load(TRUTH[source], allow_pickle=True)["configuration_id"].astype(str)


def load_errors(stem, source):
    suffix = "" if source == "test" else f"@{source}"
    data = np.load(Path(f"{stem}{suffix}.npz"))

    return np.abs(data["de"]), np.abs(data["df_mean"])


def load_projector(device="cpu"):
    checkpoint = torch.load(PROJECTOR_FILE, map_location=device, weights_only=False)
    activation = {"relu": nn.ReLU, "silu": nn.SiLU}[checkpoint.get("activation", "relu")]

    layers, width = [], FEATURE_DIM
    for hidden_width in checkpoint["hidden"]:
        layers += [nn.Linear(width, hidden_width), activation()]
        width = hidden_width
    layers.append(nn.Linear(width, checkpoint["d"]))

    model = nn.Sequential(*layers).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scaler = np.load(SCALER_FILE)
    constants = {
        "feature_offset": torch.as_tensor(scaler["mean"], device=device),
        "feature_divisor": float(scaler["scale"]),
        "projector_mean": torch.as_tensor(checkpoint["feature_mean"], device=device),
        "projector_scale": torch.as_tensor(checkpoint["feature_scale"], device=device),
        "output_mean": torch.as_tensor(np.asarray(checkpoint["projection_mean"], np.float32), device=device),
        "output_scale": torch.as_tensor(np.asarray(checkpoint["projection_scale"], np.float32), device=device),
        "d": checkpoint["d"],
    }

    return model, constants


def project(features, model, constants, device="cpu", chunk=262_144):
    coordinates = np.empty((len(features), constants["d"]), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(features), chunk):
            block = torch.as_tensor(np.ascontiguousarray(features[start:start + chunk]), device=device)
            block = (block - constants["feature_offset"]) / constants["feature_divisor"]
            block = (block - constants["projector_mean"]) / constants["projector_scale"]
            block = model(block) * constants["output_scale"] + constants["output_mean"]
            coordinates[start:start + chunk] = block.float().cpu().numpy()

    return coordinates


def feature_directory(group):
    directory = COLLECTION / group

    return directory if (directory / "configuration_id.npy").exists() else MAD_ANCHOR


def read_blocks(path, chunk):
    header = np.load(path, mmap_mode="r")
    offset, dtype, total = header.offset, header.dtype, header.shape[0]
    del header

    with open(path, "rb") as handle:
        handle.seek(offset)
        start = 0
        while start < total:
            take = min(chunk, total - start)
            block = np.frombuffer(handle.read(take * dtype.itemsize), dtype=dtype)
            yield start, block
            start += take


def resolve_rows(directory, wanted, chunk=4_194_304):
    path = directory / "configuration_id.npy"
    width = np.load(path, mmap_mode="r").dtype.itemsize

    targets = np.array([w.encode() for w in wanted], dtype=f"S{width}")
    offset = wanted[0].index(":") + 1
    assert all(w.index(":") + 1 == offset for w in wanted), directory

    target_keys = np.ascontiguousarray(
        targets.view(np.uint8).reshape(len(targets), width)[:, offset:offset + 8]).view(np.uint64).ravel()
    order = np.argsort(target_keys)
    sorted_keys = target_keys[order]

    lookup = {t: i for i, t in enumerate(targets)}
    found = np.full(len(targets), -1, dtype=np.int64)

    for start, block in read_blocks(path, chunk):
        keys = np.ascontiguousarray(
            block.view(np.uint8).reshape(len(block), width)[:, offset:offset + 8]).view(np.uint64).ravel()

        position = np.searchsorted(sorted_keys, keys)
        np.clip(position, 0, len(sorted_keys) - 1, out=position)
        candidates = np.flatnonzero(sorted_keys[position] == keys)

        for local in candidates:
            index = lookup.get(bytes(block[local]))
            if index is not None and found[index] < 0:
                found[index] = start + local

    return found


def directory_features(directory, rows):
    features = np.memmap(directory / "ecumetric_features.bin",
                         dtype=np.float32, mode="r").reshape(-1, FEATURE_DIM)

    return np.asarray(features[rows])
