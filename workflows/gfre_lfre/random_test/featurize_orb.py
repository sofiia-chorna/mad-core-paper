import torch
from orb_models.forcefield import pretrained

from common import run, run_batched

BATCH = 128


def featurize(frames, writer, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    orbff, adapter = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
    orbff.eval()

    def embed(atoms_list):
        for atoms in atoms_list:
            if atoms.pbc.any() and not atoms.pbc.all():
                atoms.pbc = [True, True, True]

        graph = adapter.from_ase_atoms_list(atoms_list, device=device)

        with torch.no_grad():
            return orbff.model(graph)["node_features"].float().cpu().numpy()

    # a batch must not mix periodic and non periodic systems
    periodic = [i for i, atoms in enumerate(frames) if atoms.pbc.any()]
    non_periodic = [i for i, atoms in enumerate(frames) if not atoms.pbc.any()]

    run_batched(frames, writer, embed, BATCH, groups=[periodic, non_periodic])


if __name__ == "__main__":
    run(featurize, ["orb-v3-omat"])
