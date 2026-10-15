import numpy as np
import torch
from mace import data, tools
from mace.data import config_from_atoms
from mace.modules import LLPRModel
from mace.tools import torch_geometric

from common import CHECKPOINTS, run, run_batched

BATCH = 64


# the readout features of every layer, as aggregate_features in mad15_mace_feats.py
def aggregate(llpr, node_feats):
    parts = torch.split(node_feats, llpr.hidden_sizes_before_readout, dim=-1)
    out = []

    for part, readout, size, is_linear in zip(
        parts,
        llpr.orig_model.readouts.children(),
        llpr.hidden_sizes,
        llpr.readouts_are_linear,
    ):
        if is_linear:
            out.append(part[:, :size])
        else:
            out.append(readout.non_linearity(readout.linear_1(part))[:, :size])

    return torch.cat(out, dim=-1)


def featurize(frames, writer, model_name):
    torch.set_default_dtype(torch.float32)
    device = tools.init_device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(CHECKPOINTS[model_name], weights_only=False, map_location="cpu").float()
    llpr = LLPRModel(model).to(device).eval()

    supported = set(model.atomic_numbers.tolist())
    z_table = tools.get_atomic_number_table_from_zs(sorted(supported))
    r_max = float(model.r_max)
    print(f"{model_name} supports {len(supported)} elements, r_max {r_max}, "
          f"hidden sizes {llpr.hidden_sizes}", flush=True)

    kept = []
    for i, atoms in enumerate(frames):
        if set(atoms.get_atomic_numbers().tolist()) - supported:
            writer.exclude(i, "unsupported_element")
        else:
            kept.append(i)

    def embed(atoms_list):
        dataset = [
            data.AtomicData.from_config(config_from_atoms(atoms), z_table=z_table, cutoff=r_max)
            for atoms in atoms_list
        ]
        loader = torch_geometric.dataloader.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
        batch = next(iter(loader)).to(device)

        with torch.no_grad():
            output = llpr(batch, compute_force=False, compute_stress=False)
            return aggregate(llpr, output["node_feats"]).cpu().numpy().astype(np.float32)

    run_batched(frames, writer, embed, BATCH, groups=[kept])


if __name__ == "__main__":
    run(featurize, ["mace-omat"])
