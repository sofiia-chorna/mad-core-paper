import json
import re
import sys
from pathlib import Path

import torch
import metatomic.torch  # noqa: F401
from metatrain.utils.data import read_systems
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from isolated_atoms import ELEMENTS, isolated_systems, predict_isolated, summary

# smoke check of the isolated atom references: zero neighbours in the model's own neighbour list calculation, finite
# training loss on the mixed batches, finite predictions for all 102 references from the first snapshot
S = Path(__file__).resolve().parent.parent
REFS = S / "manifests" / "xyz" / "isolated_atom_references.xyz"
EPOCH_LINE = re.compile(r"Epoch: +\d+ \|.*training loss: ([0-9.e+-]+|nan|inf)")


def main():
    run_dir = Path(sys.argv[1])
    ckpt_path = sorted(run_dir.glob("outputs/*/*/model_*.ckpt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
    model = model_from_checkpoint(torch.load(ckpt_path, map_location="cpu", weights_only=False), context="restart").to(dtype=torch.float32).eval()
    systems = [s.to(dtype=torch.float32) for s in read_systems(str(REFS))]
    assert len(systems) == 102
    neighbours = []
    for s in systems:
        s = get_system_with_neighbor_lists(s, model.requested_neighbor_lists())
        neighbours.append(sum(len(s.get_neighbor_list(o).samples) for o in model.requested_neighbor_lists()))
    pred_vesin = predict_isolated(model, [get_system_with_neighbor_lists(s, model.requested_neighbor_lists()) for s in systems])
    pred_manual = predict_isolated(model, isolated_systems(model))
    losses = [float(m.group(1)) for m in EPOCH_LINE.finditer((run_dir / "train.log").read_text(errors="replace"))]
    result = {"checkpoint": str(ckpt_path), "references": 102, "max_neighbours_in_pet_neighbor_list": max(neighbours), "all_zero_neighbours": bool(max(neighbours) == 0),
              "training_losses_logged": losses, "losses_finite": bool(all(l == l and abs(l) != float("inf") for l in losses)),
              "predictions_finite": bool(all(v == v for v in pred_vesin.values())), "manual_empty_list_equals_vesin_list": bool(max(abs(pred_vesin[e] - pred_manual[e]) for e in ELEMENTS) < 1e-6),
              "isolated_atom_summary_at_snapshot": summary(pred_vesin)}
    result["passed"] = bool(result["all_zero_neighbours"] and result["losses_finite"] and result["predictions_finite"] and result["manual_empty_list_equals_vesin_list"])
    json.dump(result, open(run_dir / "references_check.json", "w"), indent=1)
    print(json.dumps(result, indent=1))
    print("REFERENCES CHECK PASSED" if result["passed"] else "REFERENCES CHECK FAILED")


if __name__ == "__main__":
    main()
