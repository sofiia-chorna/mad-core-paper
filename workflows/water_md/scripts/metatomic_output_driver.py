import hashlib
import json

import metatomic.torch as mta
import numpy as np
from ipi.pes.metatomic import MetatomicDriver
from ipi.utils.units import unit_to_user

__DRIVER_NAME__ = "metatomic_output"
__DRIVER_CLASS__ = "MetatomicOutputDriver"


# the i-PI metatomic driver with a selectable energy output, conservative forces only, and a log of the geometries it receives
class MetatomicOutputDriver(MetatomicDriver):
    def __init__(self, template, model, output="energy", geometry_log="", log_calls=0, *args, **kwargs):
        self.output = str(output)
        self.geometry_log = str(geometry_log)
        self.log_calls = int(log_calls)
        self.calls = 0

        kwargs["non_conservative"] = False
        kwargs["energy_ensemble"] = False
        kwargs["uncertainty_threshold"] = 0.0
        super().__init__(template, model, *args, **kwargs)

    def _register_model_outputs(self):
        if self.output not in self.model.capabilities().outputs:
            raise ValueError(f"model has no output {self.output}")

        self.evaluation_options = mta.ModelEvaluationOptions(
            length_unit="Angstrom",
            outputs={self.output: mta.ModelOutput(quantity="energy", unit="eV", per_atom=False)},
        )

    def _process_standard_outputs(self, outputs, systems, strains, device):
        renamed = {f"energy{self.energy_suffix}": outputs[self.output]}
        return super()._process_standard_outputs(renamed, systems, strains, device)

    def compute_structure(self, cell, pos):
        result = super().compute_structure(cell, pos)

        if self.geometry_log and self.calls < self.log_calls:
            cell = np.ascontiguousarray(cell, dtype=np.float64)
            pos = np.ascontiguousarray(pos, dtype=np.float64)
            record = {
                "call": self.calls,
                "output": self.output,
                "positions_sha256": hashlib.sha256(pos.tobytes()).hexdigest(),
                "cell_sha256": hashlib.sha256(cell.tobytes()).hexdigest(),
                "first_atom_angstrom": unit_to_user("length", "angstrom", pos.reshape(-1, 3)[0]).tolist(),
                "cell_angstrom": unit_to_user("length", "angstrom", cell).tolist(),
                "energy_eV": float(unit_to_user("energy", "electronvolt", result[0])),
            }
            with open(self.geometry_log, "a") as fh:
                fh.write(json.dumps(record) + "\n")

        self.calls += 1
        return result
