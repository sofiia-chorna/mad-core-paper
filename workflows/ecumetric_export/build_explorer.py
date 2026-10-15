import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

import metatensor.torch as mts
import metatomic.torch as mta


class EcumetricExplorer(torch.nn.Module):
    def __init__(self, ecumetric, scaler, projector_checkpoint):
        super().__init__()

        self.ecumetric = ecumetric

        constants = {
            "scaler_mean": scaler["mean"],
            "scaler_scale": scaler["scale"],
            "feature_mean": projector_checkpoint["feature_mean"],
            "feature_scale": projector_checkpoint["feature_scale"],
            "projection_mean": projector_checkpoint["projection_mean"],
            "projection_scale": projector_checkpoint["projection_scale"],
        }

        for name, value in constants.items():
            self.register_buffer(name, torch.as_tensor(np.asarray(value), dtype=torch.float32))

        activation = {
            "relu": torch.nn.ReLU,
            "silu": torch.nn.SiLU,
        }[projector_checkpoint["activation"]]

        layers = []
        width = self.scaler_mean.shape[0]

        for hidden_width in projector_checkpoint["hidden"]:
            layers.append(torch.nn.Linear(width, hidden_width))
            layers.append(activation())
            width = hidden_width

        layers.append(torch.nn.Linear(width, projector_checkpoint["d"]))

        self.projector = torch.nn.Sequential(*layers)
        self.projector.load_state_dict(projector_checkpoint["state_dict"])

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[mts.Labels],
    ) -> Dict[str, mts.TensorMap]:
        atom_outputs = {"feature": mta.ModelOutput(sample_kind="atom")}
        atom_features = self.ecumetric(systems, atom_outputs, selected_atoms)
        atom_block = atom_features["feature"].block()

        values = atom_block.values
        system_index = atom_block.samples.column("system")
        n_systems = len(systems)

        # mean and population std of the per-atom features of each system
        counts = torch.zeros(n_systems, dtype=values.dtype, device=values.device)
        counts = counts.index_add(0, system_index, torch.ones_like(values[:, 0])).reshape(-1, 1)

        totals = torch.zeros(n_systems, values.shape[1], dtype=values.dtype, device=values.device)
        mean = totals.index_add(0, system_index, values) / counts

        centered = values - mean[system_index]
        variance = totals.index_add(0, system_index, centered * centered) / counts

        descriptor = torch.cat([mean, torch.sqrt(variance)], dim=1)
        scaled = (descriptor - self.scaler_mean) / self.scaler_scale

        samples = mts.Labels(
            "system",
            torch.arange(n_systems, device=values.device).reshape(-1, 1),
        )
        keys = mts.Labels("_", torch.tensor([[0]], device=values.device))

        results: Dict[str, mts.TensorMap] = {}

        if "feature" in outputs:
            normalized = (scaled - self.feature_mean) / self.feature_scale
            projection = self.projector(normalized) * self.projection_scale + self.projection_mean

            block = mts.TensorBlock(
                values=projection,
                samples=samples,
                components=[],
                properties=mts.Labels(
                    "sketchmap",
                    torch.arange(projection.shape[1], device=values.device).reshape(-1, 1),
                ),
            )
            results["feature"] = mts.TensorMap(keys, [block])

        if "feature/ecumetric" in outputs:
            block = mts.TensorBlock(
                values=scaled,
                samples=samples,
                components=[],
                properties=mts.Labels(
                    "ecumetric",
                    torch.arange(scaled.shape[1], device=values.device).reshape(-1, 1),
                ),
            )
            results["feature/ecumetric"] = mts.TensorMap(keys, [block])

        return results


parser = argparse.ArgumentParser()
parser.add_argument("--ecumetric-model", required=True)
parser.add_argument("--scaler", required=True)
parser.add_argument("--projector", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()


ecumetric_model = mta.load_atomistic_model(args.ecumetric_model)
ecumetric_capabilities = ecumetric_model.capabilities()

scaler = np.load(args.scaler)
projector_checkpoint = torch.load(args.projector, map_location="cpu", weights_only=False)

explorer = EcumetricExplorer(ecumetric_model.module, scaler, projector_checkpoint).eval()

# chemiscope picks the first available device but does not move the model there, so cpu goes first
supported_devices = sorted(
    ecumetric_capabilities.supported_devices,
    key=lambda device: device != "cpu",
)

capabilities = mta.ModelCapabilities(
    outputs={
        "feature": mta.ModelOutput(
            sample_kind="system",
            description="8-d sketch-map coordinates of the MAD-CORE ecumetric descriptor",
        ),
        "feature/ecumetric": mta.ModelOutput(
            sample_kind="system",
            description="scaled 512-d ecumetric descriptor",
        ),
    },
    atomic_types=ecumetric_capabilities.atomic_types,
    interaction_range=ecumetric_capabilities.interaction_range,
    length_unit=ecumetric_capabilities.length_unit,
    supported_devices=supported_devices,
    dtype=ecumetric_capabilities.dtype,
)

metadata = mta.ModelMetadata(**OmegaConf.to_container(OmegaConf.load(args.metadata)))

model = mta.AtomisticModel(explorer, metadata, capabilities)
model.save(args.output)

reloaded = mta.load_atomistic_model(args.output)
print(reloaded.metadata())

for options in reloaded.requested_neighbor_lists():
    print(f"neighbor list cutoff {options.cutoff} full {options.full_list} strict {options.strict}")

print(f"supported devices {reloaded.capabilities().supported_devices}")

for name, output in reloaded.capabilities().outputs.items():
    print(f"output {name} sample_kind {output.sample_kind}")

print(f"saved {args.output}")
