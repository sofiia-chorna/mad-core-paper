import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VALID = ROOT.parent
REPO = VALID.parent
sys.path.insert(0, str(VALID / "delta_pbe_10k_vs_mad16" / "scripts"))

from delta_common import (INDEX as DELTA_INDEX, MANIFESTS as DELTA_MANIFESTS, MAD16_R2SCAN, BOTR_XYZ, E0_R2SCAN_JSON,  # noqa: E402,F401
                          MONOMERS_PBE, PROD_MANIFESTS, parse_frames, read_tsv, write_tsv, soft_hash)

MANIFESTS = ROOT / "manifests"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
BUNDLE = ROOT / "bundle"

PARENT = PROD_MANIFESTS / "test.tsv"
LABELLED_TEST = VALID / "lda_pbe" / "labelled" / "random_test_6k"
PBE_XYZ = LABELLED_TEST / "pbe.xyz"
PBE_MANIFEST = LABELLED_TEST / "manifest.tsv"

PIPELINE = Path("${DEV_ROOT}/lda_pbe")
SPECIES_DIR = PIPELINE / "common" / "defaults_2020" / "tight"
PBE_CONTROL_DIR = PIPELINE / "common" / "control"
MAD_RECORD = VALID / "sources" / "mad-1.6" / "record_2026.167_v4"
MAD_CONTROL = MAD_RECORD / "control.in"

SEED = 20260908
SAMPLE_SIZE = 1000
TRAIN_VAL_SETS = ("madcore_10k", "mad16_10k", "mad16_atoms_66336", "val")
REPORT_ONLY_SETS = ("test_mad16", "holdout_madcore")

TEST_COLUMNS = ["sample_rank", "configuration_id", "frozen_configuration_id", "dataset_group", "dataset_id", "natoms", "n_species",
                "elements", "pbc", "vacuum_axes", "dft_bucket", "geometry_hash", "invariant_fp", "pbe_frame_idx", "e_pbe", "free_energy_pbe",
                "atomization_e_pbe", "fmax_pbe", "eligible_rank"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_logger(path):
    log = open(path, "w")

    def say(msg=""):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    return say
