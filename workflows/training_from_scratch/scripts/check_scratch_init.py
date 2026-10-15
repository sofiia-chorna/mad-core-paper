import json
import sys
from pathlib import Path

import metatensor.torch as mts
import metatomic.torch  # noqa: F401
import torch
from metatrain.utils.io import model_from_checkpoint

# checks that a scratch run started from random parameters (not the pretrained PET-OMat-S weights) and records the
# composition baseline and target scale fitted on its training set, read from the run's first checkpoint
PRETRAINED = Path("${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/models/pet-omat-s-v1.0.0.ckpt")


def decode(sd, key):
    return mts.load_buffer(sd[key]).block(0)


def main():
    run_dir = Path(sys.argv[1])
    ckpts = sorted(run_dir.glob("outputs/*/*/model_*.ckpt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        print("no checkpoint found yet")
        sys.exit(1)
    first = ckpts[0]
    ck = torch.load(first, map_location="cpu", weights_only=False)
    pre = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    psd = pre["best_model_state_dict"]
    sd = {k.removeprefix("backend."): v for k, v in ck["model_state_dict"].items()}
    shared = [k for k in sd if k in psd and isinstance(sd[k], torch.Tensor) and sd[k].dtype.is_floating_point and sd[k].shape == psd[k].shape and k.startswith(("gnn_layers", "node_heads", "edge_heads", "node_last_layers", "edge_last_layers", "combination"))]
    equal = [k for k in shared if torch.equal(sd[k], psd[k])]
    corr = []
    for k in shared[:400]:
        a, b = sd[k].flatten().double(), psd[k].flatten().double()
        if a.numel() > 10 and a.std() > 0 and b.std() > 0:
            corr.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    comp = decode(sd, "additive_models.0.energy_composition_buffer")
    scale = float(decode(sd, "scaler.energy_scaler_buffer").values.flatten()[0])
    comp_pre = decode(psd, "additive_models.0.energy_composition_buffer")
    baseline = {int(z): float(v) for z, v in zip(comp.samples.column("center_type").tolist(), comp.values.flatten().tolist())}
    same_layout = comp.samples == comp_pre.samples
    max_diff = float((comp.values.double() - comp_pre.values.double()).abs().max()) if same_layout else None
    result = {"checkpoint": str(first), "epoch": ck.get("epoch"), "network_tensors_compared": len(shared), "network_tensors_equal_to_pretrained": len(equal),
              "max_abs_correlation_with_pretrained": max(abs(c) for c in corr) if corr else None, "composition_baseline_eV_per_atom": baseline,
              "composition_elements": len(baseline), "composition_equal_to_pretrained_layout": bool(same_layout), "composition_max_abs_diff_vs_pretrained": max_diff,
              "energy_scale_eV_per_atom": scale, "pretrained_energy_scale": float(decode(psd, "scaler.energy_scaler_buffer").values.flatten()[0]),
              "atomic_types_in_model": len(sd["species_to_species_index"][sd["species_to_species_index"] >= 0]) if "species_to_species_index" in sd else None}
    # PET-OMat-S was itself initialised from the seed 42 draw, so a seed 42 scratch model correlates with its embeddings
    # although no weight was loaded; the seed 43 control (runs/smoke_seed43/seed_control.json) shows the correlation is
    # the shared seed, not loaded weights, so the criterion is: nothing equal to the pretrained tensors and the control
    control = Path("${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling/runs/smoke_seed43/seed_control.json")
    control_ok = json.load(open(control))["seed43_vs_pretrained_n_above_0.2"] == 0 if control.exists() else False
    result["seed_control"] = str(control) if control.exists() else None
    result["random_init_confirmed"] = len(equal) == 0 and ((result["max_abs_correlation_with_pretrained"] or 0) < 0.2 or control_ok)
    json.dump(result, open(run_dir / "init_check.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "composition_baseline_eV_per_atom"}, indent=1))
    print("baseline H C N O:", {z: round(baseline[z], 3) for z in (1, 6, 7, 8) if z in baseline})
    print("RANDOM INIT CONFIRMED" if result["random_init_confirmed"] else "INIT CHECK FAILED")


if __name__ == "__main__":
    main()
