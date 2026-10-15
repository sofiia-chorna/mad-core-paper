import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols
from metatensor.torch import Labels, TensorBlock
from metatomic.torch import ModelOutput, systems_to_torch

# the 102 isolated atoms as metatomic systems with an explicitly empty neighbour list in the model's requested format,
# so no neighbour list library is needed; predictions are complete energies (scaler and composition included)
ELEMENTS = [chemical_symbols[z] for z in range(1, 103)]


def empty_neighbor_block(dtype):
    return TensorBlock(values=torch.zeros((0, 3, 1), dtype=dtype), samples=Labels(["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"], torch.zeros((0, 5), dtype=torch.int32)),
                       components=[Labels(["xyz"], torch.tensor([[0], [1], [2]], dtype=torch.int32))], properties=Labels(["distance"], torch.tensor([[0]], dtype=torch.int32)))


def isolated_systems(model, dtype=torch.float32):
    systems = systems_to_torch([Atoms(el, positions=[[0.0, 0.0, 0.0]], cell=np.zeros((3, 3)), pbc=False) for el in ELEMENTS], dtype=torch.float64)
    out = []
    for s in systems:
        s = s.to(dtype=dtype)
        for options in model.requested_neighbor_lists():
            s.add_neighbor_list(options, empty_neighbor_block(dtype))
        out.append(s)
    return out


def predict_isolated(model, systems):
    model.eval()
    with torch.no_grad():
        values = model(systems, {"energy": ModelOutput(quantity="energy", unit="eV", per_atom=False)})["energy"].block(0).values.detach().double().numpy()[:, 0]
    return dict(zip(ELEMENTS, values))


def summary(pred):
    a = np.abs(np.array([pred[e] for e in ELEMENTS]))
    return {"median_abs_eV": float(np.median(a)), "max_abs_eV": float(a.max()), "max_element": ELEMENTS[int(a.argmax())], "H_eV": float(pred["H"]), "Ne_eV": float(pred["Ne"]), "n_above_1eV": int((a > 1.0).sum())}
