from collections import defaultdict

import numpy as np
from deepmd.infer import DeepPot

from common import CHECKPOINTS, run, run_batched

SUPPORTED_ELEMENTS = {
    "dpa-omat": None,
    "dpa-omol": 83,
}
BATCH = {
    "dpa-omat": 64,
    "dpa-omol": 128,
}


def featurize(frames, writer, model_name):
    dp = DeepPot(CHECKPOINTS[model_name])
    type_map = dp.get_type_map()
    n_supported = SUPPORTED_ELEMENTS[model_name]
    supported = set(type_map[:n_supported] if n_supported else type_map)

    # DPA3-Omol-Large takes charge and spin multiplicity per frame, its model card
    charge_spin = [0.0, 1.0] if dp.has_chg_spin_ebd() else None
    print(f"{model_name} type map {len(type_map)}, supported {len(supported)}, "
          f"charge_spin {charge_spin}", flush=True)

    # a batch must share the number of atoms and the periodicity
    groups = defaultdict(list)
    for i, atoms in enumerate(frames):
        if not set(atoms.get_chemical_symbols()).issubset(supported):
            writer.exclude(i, "unsupported_element")
            continue

        groups[(len(atoms), bool(atoms.pbc.all()))].append(i)

    def embed(atoms_list):
        n_frames = len(atoms_list)
        periodic = bool(atoms_list[0].pbc.all())

        coords = np.stack([atoms.get_positions() for atoms in atoms_list]).reshape(n_frames, -1)
        cells = None
        if periodic:
            cells = np.stack([atoms.get_cell().array for atoms in atoms_list]).reshape(n_frames, -1)
        types = np.array(
            [[type_map.index(symbol) for symbol in atoms.get_chemical_symbols()] for atoms in atoms_list],
            dtype=np.int32,
        )

        extra = {}
        if charge_spin:
            extra["charge_spin"] = np.tile(np.array(charge_spin), (n_frames, 1))

        fit = np.asarray(dp.eval_fitting_last_layer(coords, cells, types, mixed_type=True, **extra))

        return fit.astype(np.float32).reshape(-1, fit.shape[-1])

    run_batched(frames, writer, embed, BATCH[model_name], groups=list(groups.values()))


if __name__ == "__main__":
    run(featurize, list(SUPPORTED_ELEMENTS))
