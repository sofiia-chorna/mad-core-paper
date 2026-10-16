import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# the projector of the release and the level 24 populations of the record
PROJECTOR = str(ROOT / "data" / "models" / "ecumetric_sketchmap_projector_p20d8_rec_silu.pt")
POPULATIONS = str(ROOT / "results_archive" / "data" / "voronoi_population_record_level24.npy")
N_CORESET = 2 ** 24

MAD_PARTS = ("mad-1.5", "mad-next", "mad_union")
MAD_LABEL = "MAD-NEXT"
CATALYST_PREFIXES = ("OC20", "OC22", "OC25")
ANI_PREFIXES = ("ANI-1", "ANI-2")
RENAME = {"Open_Molecular_Crystals_2025_OMC25": "OMC25",
          "Materials_Project": "Materials Project",
          "mcloudarchive__jk-9v": "Li-ion conductors",
          "Carolina_Materials": "Carolina",
          "MatPES-2025.1": "MatPES",
          "Finding_new_crystal_compounds_using_chemical_similarity": "Chem. similarity"}

PALETTE5 = ["#C94E3F", "#00A79F", "#F0C93B", "#4B4A86", "#403C3A"]
OTHER_COLOUR = "#7E7368"
COLOURS = PALETTE5 + [OTHER_COLOUR]
Z_BOOST = {"OMat24": 1.0, "JARVIS": 1.0, "Alexandria": -0.5}

# members listed in the color order of PALETTE5 (first member is rosy red, ...)
GROUPS = [
    ("MAD-NEXT", [MAD_LABEL]),
    ("Bulk materials", ["OMat24", "Alexandria", "JARVIS"]),
    ("Repositories", ["nomad", "Materials Project", "Materials Cloud"]),
    ("Molecular", ["OMol25", "ANI", "gdb13"]),
    ("Catalysis and interfaces", ["Open Catalyst", "ODAC"]),
]

LABEL = {"HEA25_high_entropy_transition-metal_alloys": "HEA25",
         "HEA25S_high_entropy_alloys": "HEA25S",
         "UNEP_v1_2023": "UNEP-v1",
         "MatPES-2025.1": "MatPES",
         "Carolina_Materials": "Carolina",
         "Finding_new_crystal_compounds_using_chemical_similarity": "Chem. similarity",
         "Open_Molecular_Crystals_2025_OMC25": "OMC25",
         "Si_Al_Ti_Seko_PRB_2019": "Si-Al-Ti",
         "AlNiTi_CMS_2019": "AlNiTi",
         "WBe_PRB2019": "W-Be",
         "nomad": "NOMAD",
         "ANI": "ANI-1/2x"}

parser = argparse.ArgumentParser(description="rebuild data/smap_points.npz from the release files")
parser.add_argument("--store-root", required=True,
                    help="root of the aa002 mad-core store, the directory holding release/")
parser.add_argument("--projector", default=PROJECTOR, help="sketch-map projector checkpoint")
parser.add_argument("--populations", default=POPULATIONS,
                    help="level-24 cell populations, the released .npy or a populations.npz "
                         "with level_offsets")
parser.add_argument("--labels", default=None,
                    help="dataset group of each center, the released .npy or a parquet "
                         "with a family column")
parser.add_argument("--output", default=str(HERE / "data" / "smap_points.npz"))
parser.add_argument("--n-show", type=int, default=2 ** 20)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--always-top", type=int, default=512)
parser.add_argument("--background", type=int, default=2 ** 20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--smin", type=float, default=2.0)
parser.add_argument("--smax", type=float, default=60.0)
parser.add_argument("--size-power", type=float, default=2.5)
parser.add_argument("--force-top", default="",
                    help="draw this dataset above all others in its panel")
parser.add_argument("--threads", type=int, default=16)
args = parser.parse_args()
RELEASE = f"{args.store_root}/release"
if args.labels is None:
    args.labels = f"{RELEASE}/mad-core/dataset_group.npy"


def project():
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.projector, map_location="cpu", weights_only=False)
    activation = {"relu": nn.ReLU, "silu": nn.SiLU}[checkpoint["activation"]]
    layers, width = [], 512
    for hidden_width in checkpoint["hidden"]:
        layers += [nn.Linear(width, hidden_width), activation()]
        width = hidden_width
    layers.append(nn.Linear(width, checkpoint["d"]))
    model = nn.Sequential(*layers)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # the released features are raw, the projector expects them in the scaled ecumetric metric
    scaler = np.load(ROOT / "data" / "ecumetric_scaler.npz")
    scaler_mean = torch.as_tensor(scaler["mean"])
    scaler_scale = float(scaler["scale"])
    feature_mean = torch.as_tensor(checkpoint["feature_mean"])
    feature_scale = torch.as_tensor(checkpoint["feature_scale"])
    features = np.memmap(f"{RELEASE}/mad-core/ecumetric_features.bin", dtype=np.float32,
                         mode="r", shape=(N_CORESET, 512))
    projection = np.empty((N_CORESET, checkpoint["d"]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N_CORESET, 262_144):
            end = min(start + 262_144, N_CORESET)
            block = torch.as_tensor(np.ascontiguousarray(features[start:end]))
            block = ((block - scaler_mean) / scaler_scale - feature_mean) / feature_scale
            projection[start:end] = model(block).numpy()
    projection = (projection * np.asarray(checkpoint["projection_scale"], dtype=np.float32)
                  + np.asarray(checkpoint["projection_mean"], dtype=np.float32))
    return projection[:, :2].astype(np.float32)


def label_of(name):
    if name in MAD_PARTS or name == "mad-union":
        return MAD_LABEL
    if name.upper().startswith(CATALYST_PREFIXES):
        return "Open Catalyst"
    if name.upper().startswith(ANI_PREFIXES):
        return "ANI"
    if name.startswith("JARVIS"):
        return "JARVIS"
    if name.startswith(("mcloudarchive", "mcloud__")):
        return "Materials Cloud"
    return RENAME.get(name, name)

coordinates = project()

if args.populations.endswith(".npz"):
    blob = np.load(args.populations)
    offsets = blob["level_offsets"]
    population = blob["populations"][offsets[24]:offsets[25]].astype(np.float64)
else:
    population = np.load(args.populations).astype(np.float64)
assert population.shape == (N_CORESET,), population.shape
print(f"level-24 populations: sum {population.sum():,.0f}, max {population.max():,.0f}")

if args.labels.endswith(".parquet"):
    import pyarrow.parquet as pq
    source = np.asarray(pq.read_table(args.labels, columns=["family"]).column("family")
                        .to_pylist(), dtype=object)
else:
    source = np.load(args.labels).astype(str).astype(object)
assert source.shape == (N_CORESET,), source.shape
family = np.asarray([label_of(name) for name in source], dtype=object)

# population-weighted sample without replacement, Efraimidis-Spirakis keys
generator = np.random.default_rng(args.seed)
weight = population ** args.gamma
keys = -np.log(generator.random(N_CORESET)) / weight
shown = np.argsort(keys)[:args.n_show]
top = np.argsort(population)[::-1][:args.always_top]
shown = np.union1d(shown, top)
print(f"showing {shown.size:,} of {N_CORESET:,} cells "
      f"(gamma {args.gamma}, forced top {args.always_top}); shown cells hold "
      f"{population[shown].sum() / population.sum():.1%} of the collection")

background = np.sort(generator.choice(N_CORESET, args.background, replace=False))

full_counts = collections.Counter(family.tolist())
size_fraction = (np.log10(population) / np.log10(population.max())) ** args.size_power
marker_size = args.smin + (args.smax - args.smin) * size_fraction

named = {name for _, members in GROUPS for name in members}
panels = GROUPS + [("Other datasets", ["__other__"])]

data = {"background_xy": coordinates[background],
        "low": coordinates.min(0), "high": coordinates.max(0),
        "population_max": population.max(),
        "smin": args.smin, "smax": args.smax, "size_power": args.size_power}
legends = []
shown_family = family[shown]
for index, (title, members) in enumerate(panels):
    colour_of = {name: (len(PALETTE5) if name == "__other__" else i % len(PALETTE5))
                 for i, name in enumerate(members)}
    if members == ["__other__"]:
        in_panel = shown[~np.isin(shown_family, list(named))]
        point_colour = np.full(in_panel.size, len(PALETTE5), dtype=np.int8)
        panel_family = np.full(in_panel.size, "", dtype=object)
    else:
        mask = np.isin(shown_family, members)
        in_panel = shown[mask]
        panel_family = shown_family[mask]
        point_colour = np.asarray([colour_of[f] for f in panel_family], dtype=np.int8)

    # z-order, log10 population plus a per-dataset boost and jitter
    log_pop = np.log10(np.maximum(population[in_panel], 1.0))
    boost = np.asarray([Z_BOOST.get(f, 0.0) for f in panel_family])
    jitter = (generator.random(in_panel.size) - 0.5) * 0.5
    key = log_pop + boost + jitter
    if args.force_top:
        forced = panel_family == args.force_top
        order = np.lexsort((key, forced))
    else:
        order = np.argsort(key, kind="stable")
    drawn = in_panel[order]
    data[f"panel{index}_xy"] = coordinates[drawn]
    data[f"panel{index}_size"] = marker_size[drawn]
    data[f"panel{index}_colour"] = point_colour[order]
    legends.append({
        "title": title,
        "entries": [[f"{len(full_counts) - len(named)} datasets" if name == "__other__"
                     else LABEL.get(name, name), COLOURS[colour_of[name]]]
                    for name in members]})

data["colours"] = np.asarray(COLOURS)
data["panels"] = np.asarray(json.dumps(legends))
np.savez_compressed(args.output, **data)
print("wrote", args.output)
for legend in legends:
    print(f"  {legend['title']}: {', '.join(label for label, _ in legend['entries'])}")
