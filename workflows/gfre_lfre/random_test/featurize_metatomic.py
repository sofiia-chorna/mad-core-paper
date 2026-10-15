import numpy as np
import torch
import metatomic.torch as mta
import vesin.metatomic as vesin_metatomic
from metatrain.utils.io import load_model

from common import CHECKPOINTS, run, run_batched

OUTPUT_KEY = {
    "phace-atomic": "features",
    "pet-mad": "mtt::aux::energy_last_layer_features",
}
BATCH = {
    "phace-atomic": 16,
    "pet-mad": 64,
}


def featurize(frames, writer, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CHECKPOINTS[model_name]).to(device)
    length_unit = model.capabilities().length_unit
    output_key = OUTPUT_KEY[model_name]
    options = mta.ModelEvaluationOptions(
        length_unit=length_unit,
        outputs={output_key: mta.ModelOutput(per_atom=True)},
        selected_atoms=None,
    )
    print(f"{model_name} output {output_key} on {device}", flush=True)

    def embed(atoms_list):
        systems = mta.systems_to_torch(atoms_list)
        if not isinstance(systems, list):
            systems = [systems]

        vesin_metatomic.compute_requested_neighbors(systems, length_unit, model)
        systems = [system.to(dtype=torch.float32, device=device) for system in systems]

        with torch.no_grad():
            output = model(systems, options, check_consistency=False)

        return output[output_key].block().values.detach().cpu().numpy().astype(np.float32)

    run_batched(frames, writer, embed, BATCH[model_name])


if __name__ == "__main__":
    run(featurize, list(OUTPUT_KEY))
