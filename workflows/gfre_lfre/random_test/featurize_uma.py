import torch
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit import load_predict_unit

from common import CHECKPOINTS, run, run_batched

TASK = "omat"
BATCH = 128


def featurize(frames, writer, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = load_predict_unit(CHECKPOINTS[model_name], device=device)

    # the same two hooks as the expose_feat fairchem fork used for the MAD-NEXT features
    captured = {}
    energy_block = predictor.model.module.output_heads.energyandforcehead.head.energy_block
    energy_block.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("bb_feat", inputs[0].detach())
    )
    energy_block[-1].register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("ll_feat", inputs[0].detach())
    )

    try:
        max_num_elements = predictor.model.module.backbone.max_num_elements
    except AttributeError:
        max_num_elements = 100
    print(f"{model_name} task {TASK}, max_num_elements {max_num_elements}", flush=True)

    data_list = {}
    kept = []
    for i, atoms in enumerate(frames):
        numbers = atoms.get_atomic_numbers()

        if (numbers >= max_num_elements).any() or (numbers < 1).any():
            writer.exclude(i, "unsupported_element")
            continue

        if not (atoms.pbc.all() or (~atoms.pbc).all()):
            writer.exclude(i, "mixed_pbc")
            continue

        clean = atoms.copy()
        clean.calc = None
        try:
            data_list[i] = AtomicData.from_ase(clean, r_edges=False, task_name=TASK)
            kept.append(i)
        except Exception as error:
            writer.exclude(i, f"from_ase_error {type(error).__name__}")

    index_of = {id(frames[i]): i for i in kept}

    def embed(atoms_list):
        indices = [index_of[id(atoms)] for atoms in atoms_list]
        batch = atomicdata_list_to_batch([data_list[i] for i in indices])
        batch.dataset = [TASK] * len(indices)

        captured.clear()
        with torch.no_grad():
            predictor.predict(batch)

        return captured["ll_feat"].to(torch.float32).cpu().numpy()

    # the backbone asserts that a batch is all periodic or all non periodic
    periodic = [i for i in kept if frames[i].pbc.all()]
    non_periodic = [i for i in kept if not frames[i].pbc.all()]

    run_batched(frames, writer, embed, BATCH, groups=[periodic, non_periodic])


if __name__ == "__main__":
    run(featurize, ["uma-omat"])
