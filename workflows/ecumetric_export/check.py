import argparse

import numpy as np
import torch
from ase import Atoms

import metatomic.torch as mta
import vesin.metatomic as vesin_metatomic


release = "${STORE_ROOT}/mad-core/release/"
largest_cell_center_row = 16_531_443
batch_size = 32

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--explorer", required=True)
args = parser.parse_args()


# the first 64 fps picks are the most extreme structures, the rest is a fixed random sample
def select_rows():
    na = np.load(release + "mad-core/na.npy", mmap_mode="r")
    n_structures = len(na) - 1

    generator = np.random.default_rng(20261008)
    random_rows = generator.choice(
        np.arange(64, n_structures),
        size=192,
        replace=False,
    )

    return np.concatenate([np.arange(64), np.sort(random_rows), [largest_cell_center_row]])


def read_frames(rows):
    na = np.load(release + "mad-core/na.npy", mmap_mode="r")
    numbers = np.memmap(release + "mad-core/a.bin", dtype=np.int32, mode="r")
    positions = np.memmap(release + "mad-core/x.bin", dtype=np.float32, mode="r").reshape(-1, 3)
    cells = np.memmap(release + "mad-core/c.bin", dtype=np.float32, mode="r").reshape(-1, 3, 3)
    pbcs = np.load(release + "mad-core/pbc.npy", mmap_mode="r")

    frames = []
    for row in rows:
        start, stop = int(na[row]), int(na[row + 1])

        atoms = Atoms(
            numbers=np.asarray(numbers[start:stop]),
            positions=np.asarray(positions[start:stop], dtype=np.float64),
            cell=np.asarray(cells[row], dtype=np.float64),
            pbc=np.asarray(pbcs[row]),
        )
        atoms.wrap(pbc=atoms.cell.any(axis=1))
        frames.append(atoms)

    return frames


# projector built from the release files, independent of the explorer model
def offline_projection(descriptor, scaler, checkpoint):
    activation = {"relu": torch.nn.ReLU, "silu": torch.nn.SiLU}[checkpoint["activation"]]

    layers = []
    width = descriptor.shape[1]
    for hidden_width in checkpoint["hidden"]:
        layers.append(torch.nn.Linear(width, hidden_width))
        layers.append(activation())
        width = hidden_width
    layers.append(torch.nn.Linear(width, checkpoint["d"]))

    projector = torch.nn.Sequential(*layers)
    projector.load_state_dict(checkpoint["state_dict"])
    projector.eval()

    scaled = (descriptor - scaler["mean"]) / scaler["scale"]
    normalized = (scaled - checkpoint["feature_mean"]) / checkpoint["feature_scale"]

    with torch.no_grad():
        output = projector(torch.as_tensor(normalized, dtype=torch.float32)).numpy()

    return output * checkpoint["projection_scale"] + checkpoint["projection_mean"], scaled


def evaluate(model, frames, outputs, device):
    model = model.to(device)
    calculators = vesin_metatomic.neighbor_lists_for_model("angstrom", model)

    options = mta.ModelEvaluationOptions(
        length_unit="angstrom",
        outputs=outputs,
        selected_atoms=None,
    )

    collected = {name: ([], []) for name in outputs}

    for start in range(0, len(frames), batch_size):
        systems = mta.systems_to_torch(frames[start : start + batch_size], dtype=torch.float64)

        for calculator in calculators:
            calculator.add_neighbor_list(systems)

        systems = [system.to(dtype=torch.float32, device=device) for system in systems]

        with torch.no_grad():
            output = model(systems, options, check_consistency=False)

        for name in outputs:
            block = output[name].block()
            collected[name][0].append(block.samples.column("system").cpu().numpy() + start)
            collected[name][1].append(block.values.cpu().numpy().astype(np.float64))

    return {
        name: (np.concatenate(system_index), np.concatenate(values))
        for name, (system_index, values) in collected.items()
    }


def by_system(system_index, values):
    return values[np.argsort(system_index, kind="stable")]


# same pooling as the release features, mean and population std over atoms
def pool(system_index, values, n_systems):
    descriptors = np.zeros((n_systems, 2 * values.shape[1]))

    for system in range(n_systems):
        atom_values = values[system_index == system]
        descriptors[system] = np.concatenate([atom_values.mean(axis=0), atom_values.std(axis=0)])

    return descriptors


def report(label, reference, candidate):
    gap = np.abs(reference - candidate)
    print(f"  {label:52s} max abs {gap.max():.3e}   median abs {np.median(gap):.3e}")


rows = select_rows()
frames = read_frames(rows)

features_file = np.memmap(release + "mad-core/ecumetric_features.bin", dtype=np.float32, mode="r").reshape(-1, 512)
energy_file = np.memmap(release + "mad-core/ecumetric_energy.bin", dtype=np.float32, mode="r")

released_features = np.asarray(features_file[rows], dtype=np.float64)
released_energy = np.asarray(energy_file[rows], dtype=np.float64)

scaler = np.load(release + "ecumetric_scaler.npz")
projector_checkpoint = torch.load(release + "ecumetric_sketchmap.pt", map_location="cpu", weights_only=False)
released_projection, released_scaled = offline_projection(released_features, scaler, projector_checkpoint)


model = mta.load_atomistic_model(args.model)
explorer = mta.load_atomistic_model(args.explorer)

print(f"== metadata, metatomic {mta.__version__}")
print(model.metadata())
print(explorer.metadata())

model_outputs = {
    "energy": mta.ModelOutput(sample_kind="system"),
    "feature": mta.ModelOutput(sample_kind="atom"),
}
explorer_outputs = {
    "feature": mta.ModelOutput(sample_kind="system"),
    "feature/ecumetric": mta.ModelOutput(sample_kind="system"),
}

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")

results = {}
for device in devices:
    model_result = evaluate(model, frames, model_outputs, device)
    explorer_result = evaluate(explorer, frames, explorer_outputs, device)

    results[device] = {
        "energy": by_system(*model_result["energy"])[:, 0],
        "pooled features": pool(*model_result["feature"], len(frames)),
        "explorer feature": by_system(*explorer_result["feature"]),
        "explorer feature/ecumetric": by_system(*explorer_result["feature/ecumetric"]),
    }

cpu = results["cpu"]

print(f"\n== against the release files on cpu, {len(frames)} structures")
report("model energy vs ecumetric_energy.bin", released_energy, cpu["energy"])
report("model pooled features vs ecumetric_features.bin", released_features, cpu["pooled features"])
report("explorer feature/ecumetric vs scaled ecumetric_features.bin", released_scaled, cpu["explorer feature/ecumetric"])
report("explorer feature vs offline projector", released_projection, cpu["explorer feature"])

# distance in the scaled ecumetric metric, to compare with the level 24 separation of 0.218
metric_gap = np.linalg.norm((cpu["pooled features"] - released_features) / scaler["scale"], axis=1)
print(f"  model ecumetric distance to the released rows max {metric_gap.max():.3e}")

if "cuda" in results:
    print("\n== cuda vs cpu")
    for name in cpu:
        report(name, cpu[name], results["cuda"][name])
else:
    print("\nno cuda device, cuda vs cpu skipped")

print("\n== largest Voronoi cell center")
print(f"  explorer  {np.round(cpu['explorer feature'][-1], 4)}")
print(f"  released  {np.round(released_projection[-1], 4)}")
