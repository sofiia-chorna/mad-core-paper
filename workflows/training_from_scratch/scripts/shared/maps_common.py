import sys
from pathlib import Path

S = Path(__file__).resolve().parent.parent
PAPER = S.parent / "paper_demo"
OLD = PAPER / "sketchmap_error" / "scripts"
OLD5 = PAPER / "sketchmap_error_5model"
sys.path.insert(0, str(OLD))
from common_smap import FEATURE_DIM, POPULATION_FILE, directory_features, feature_directory, load_projector, make_logger, project, read_tsv, resolve_rows, write_tsv  # noqa: E402,F401
import bin_errors as B  # noqa: E402,F401
import plot_maps as P  # noqa: E402,F401

PBE = S.parent / "voronoi_weighted" / "pbe_20k"
MANIFESTS = PBE / "manifests"
REPORTS = S / "reports"
MAPS = REPORTS / "maps"
FIGURES = REPORTS / "figures"
LOGS = S / "logs"
LIBRARY_XY = OLD5.parent / "sketchmap_error" / "reports" / "cache" / "library_xy.npy"
SETS = {"test_random": "test_random", "test_madcore": "test_madcore"}
SIZES = ("5k", "10k", "20k")
ARMS = ("random", "madcore")
MODELS = [f"scratch-s--{arm}_{size}--s42" for arm in ARMS for size in SIZES]
ARM_LABEL = {"random": "random training", "madcore": "MAD-CORE training", "voronoi": "Voronoi training",
             "random_atoms": "random training\n(atom matched)", "voronoi_atoms": "Voronoi training\n(atom matched)"}
# figure sets: rows of panels (one arm per row, one size per column); "scaling" is the delivered six run figure
FIGURE_SETS = {"scaling": ("random", "madcore"), "voronoi": ("random", "voronoi", "madcore"), "atoms": ("random_atoms", "voronoi_atoms", "madcore")}
SET_LABEL = {"test_random": "random test (5,847)", "test_madcore": "MAD-CORE holdout (2,000)"}


def model_of(arm, size):
    return f"scratch-s--{arm}_{size}--s42"
