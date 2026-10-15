from pathlib import Path

import metatomic.torch as mta
import numpy as np
import torch
from ase.io import read
from vesin.metatomic import compute_requested_neighbors

D = Path(__file__).resolve().parent.parent
VALID = D.parent.parent
DELTA_RUN = VALID / "experiments" / "delta_lda" / "runs" / "main_clariden" / "delta_lda--omat-s--madcore_10k--s42"
DELTA_PBE_RUN = VALID / "experiments" / "delta_lda" / "runs" / "main_clariden" / "delta_pbe--omat-s--madcore_10k--s42"
BASELINE_PT = D / "models" / "pet-mad-m-v1.6.0.fixed.pt"
UPSTREAM_PT = D / "models" / "pet-mad-m-v1.6.0.upstream.pt"
DELTA_PT = DELTA_RUN / "model_best.pt"
DELTA_PBE_PT = DELTA_PBE_RUN / "model_best.pt"
BASELINE_OUTPUT = "energy"
DELTA_OUTPUT = "mtt::delta_energy"
STARTS = VALID / "md" / "md_augmentation" / "inputs"

# folder of each phase under runs, the short pilot test runs share one folder
PHASE_DIRS = {
    "pilot_nve": "pilots/nve",
    "pilot_nve_half_dt": "pilots/nve_half_dt",
    "pilot_nvt": "pilots/nvt",
    "production": "production",
}
LAMMPS_TYPES = {1: 3, 2: 15, 3: 16}
LIPS_MASSES = {3: 6.94, 15: 30.973761998, 16: 32.06}
CORRECTIONS = {
    "lda": {"run": DELTA_RUN, "pt": DELTA_PT, "functional": "lda", "preds": "main_clariden-delta_lda--omat-s--madcore_10k--s42"},
    "pbe": {"run": DELTA_PBE_RUN, "pt": DELTA_PBE_PT, "functional": "pbe", "preds": "main_clariden-delta_pbe--omat-s--madcore_10k--s42"},
}


# one model, one energy output, conservative forces as the negative energy gradient, one structure per call
class Potential:
    def __init__(self, path, output, device):
        self.model = mta.load_atomistic_model(str(path)).to(device)
        self.output = output
        self.device = device
        self.dtype = getattr(torch, self.model.capabilities().dtype)
        self.options = mta.ModelEvaluationOptions(
            length_unit="angstrom",
            outputs={output: mta.ModelOutput(quantity="energy", unit="eV", per_atom=False)},
        )

    def __call__(self, numbers, positions, cell, pbc):
        pbc = np.asarray(pbc, dtype=bool)
        cell = np.array(cell, dtype=float)
        cell[~pbc] = 0.0

        positions = torch.tensor(np.asarray(positions, dtype=float), dtype=self.dtype, device=self.device, requires_grad=True)
        system = mta.System(
            types=torch.tensor(np.asarray(numbers), dtype=torch.int32, device=self.device),
            positions=positions,
            cell=torch.tensor(cell, dtype=self.dtype, device=self.device),
            pbc=torch.tensor(pbc, device=self.device),
        )
        compute_requested_neighbors(system, system_length_unit="angstrom", model=self.model)

        outputs = self.model([system], self.options, check_consistency=False)
        energy = outputs[self.output].block().values.sum()
        (gradient,) = torch.autograd.grad(energy, positions)

        return float(energy.detach().double().cpu()), -gradient.detach().double().cpu().numpy()


def atoms_call(potential, atoms):
    return potential(atoms.numbers, atoms.positions, atoms.cell.array, atoms.pbc)


# a LiPS start in the LAMMPS frame of the previous production runs, velocities in ASE units
def lips_start(k):
    atoms = read(STARTS / f"lips_start{k}.data", format="lammps-data", atom_style="atomic", units="metal", Z_of_type=LAMMPS_TYPES)
    atoms.pbc = True
    atoms.set_masses([LIPS_MASSES[z] for z in atoms.numbers])
    return atoms
