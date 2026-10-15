from __future__ import annotations

from typing import List

import ase
import numpy as np
import torch
import vesin
from metatensor.torch import Labels, TensorBlock
from metatomic.torch import NeighborListOptions, System, register_autograd_neighbors


def system_to_ase(system: System) -> ase.Atoms:
    positions = system.positions.detach().cpu().numpy()
    numbers = system.types.detach().cpu().numpy()
    cell = system.cell.detach().cpu().numpy()
    pbc = list(cell.any(axis=1))
    return ase.Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)


def _compute_single_neighbor_list(atoms: ase.Atoms, options: NeighborListOptions) -> TensorBlock:
    assert options.full_list, "nl_ase only supports full_list=True neighbor lists"
    nl_i, nl_j, nl_S, nl_D = vesin.ase_neighbor_list("ijSD", atoms, cutoff=options.cutoff)

    samples = np.concatenate([nl_i[:, None], nl_j[:, None], nl_S], axis=-1, dtype=np.int32)
    return TensorBlock(
        values=torch.from_numpy(nl_D).reshape(-1, 3, 1),
        samples=Labels(
            names=["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
            values=torch.from_numpy(samples),
            assume_unique=True,
        ),
        components=[Labels.range("xyz", 3)],
        properties=Labels.range("distance", 1),
    )


def get_system_with_neighbor_lists(system: System, neighbor_lists: List[NeighborListOptions]) -> System:
    atoms = system_to_ase(system)
    for options in neighbor_lists:
        if options not in system.known_neighbor_lists():
            neighbors = _compute_single_neighbor_list(atoms, options).to(
                device=system.device, dtype=system.dtype
            )
            register_autograd_neighbors(system, neighbors)
            system.add_neighbor_list(options, neighbors)
    return system
