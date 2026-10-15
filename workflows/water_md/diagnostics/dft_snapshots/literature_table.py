import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE.parents[1]
ANALYSIS = D / "analysis"
TABLES = HERE / "tables"

CELL_A = 9.855479942744763
N_O = 32
DISTASIO_CUTOFF_A = 3.28

OURS = {"baseline": "PET-MAD-1.6-M", "lda_corrected": "approx. LDA (18k)", "pbe_corrected": "approx. PBE (18k)"}

# Todorova et al. 2006, Table 4 (pm converted to A) and Tables 5 and 6; DiStasio et al. 2014, Table I
LITERATURE = {
    "LDA, Todorova 2006, 350 K": {"r1": 2.59, "g1": 5.07, "fwhm": 0.21, "rmin": 3.12, "gmin": 0.23, "n": None, "r2": None, "g2": None, "note": "max deviation over the sampling window 0.02 A and 0.07 on the first peak"},
    "PBE, Todorova 2006, 350 K": {"r1": 2.70, "g1": 2.99, "fwhm": 0.37, "rmin": 3.29, "gmin": 0.47, "n": None, "r2": None, "g2": None, "note": "max deviation 0.05 A and 0.08"},
    "PBE, DiStasio 2014, 300 K": {"r1": 2.69, "g1": 3.28, "fwhm": None, "rmin": 3.28, "gmin": 0.37, "n": 4.03, "r2": 4.44, "g2": 1.44, "note": "n_OO integrated to the first minimum"},
    "experiment, Skinner 2013, 295 K (as quoted by DiStasio)": {"r1": 2.80, "g1": 2.57, "fwhm": None, "rmin": 3.45, "gmin": 0.84, "n": 4.3, "r2": 4.50, "g2": 1.12, "note": ""},
    "experiment, Soper and Benmore 2008, 298 K (as quoted by DiStasio)": {"r1": 2.76, "g1": 2.62, "fwhm": None, "rmin": 3.42, "gmin": 0.82, "n": 4.67, "r2": 4.43, "g2": 1.14, "note": ""},
}

PROTOCOL = [
    ("quantity", "this work", "Todorova 2006 (LDA and PBE rows)", "DiStasio 2014 (PBE row)"),
    ("potential", "PET-MAD-1.6-M minus a learned r2SCAN minus LDA or PBE correction, machine-learned approximation", "Kohn-Sham DFT, LDA in the Pade parametrisation, PBE; Troullier-Martins pseudopotentials, 70 Ry plane waves", "Kohn-Sham DFT, PBE; Troullier-Martins pseudopotentials, 71 Ry plane waves"),
    ("molecules, cell", "32, cubic 9.855 A", "32, cubic 9.939 A", "64, cubic 12.4 A"),
    ("density", "1.000 g/cm3", "0.975 g/cm3", "experimental ambient density"),
    ("temperature", "300 K", "350 K", "300 K"),
    ("dynamics", "Born-Oppenheimer with the ML potential, velocity Verlet, 0.25 fs", "Car-Parrinello, electron mass 600 a.u., 5 a.u. (0.12 fs) time step", "Car-Parrinello, electron mass 300 a.u., 4 a.u. (0.1 fs) time step"),
    ("nuclei", "classical, hydrogen mass", "classical, deuterium mass for hydrogen", "classical, deuterium mass for hydrogen"),
    ("thermostat", "stochastic velocity rescaling, tau 10 fs, global", "Nose-Hoover chains of length 4 on ions and electrons", "massive Nose-Hoover chains of length 4"),
    ("sampling", "5 trajectories, 20 to 50 ps each, 150 ps in total", "10 ps equilibration, 10 ps averages", "about 2 ps equilibration, at least 20 ps averages"),
    ("dispersion", "none beyond what the functionals contain", "none", "none in the plain PBE row"),
    ("periodic box treatment", "fixed cell, pressures between minus 8 and plus 7 kbar", "fixed cell", "fixed cell"),
]


def n_within(r, g, cutoff):
    rho = (N_O - 1) / CELL_A**3
    mask = r <= cutoff
    dr = r[1] - r[0]
    return float((4 * np.pi * r[mask] ** 2 * g[mask] * rho * dr).sum())


def fwhm(r, g):
    peak = g.argmax()
    half = g[peak] / 2
    left = peak
    while left > 0 and g[left] > half:
        left -= 1
    right = peak
    while right < len(g) - 1 and g[right] > half:
        right += 1
    return float(r[right] - r[left])


def main():
    conditions = json.load(open(ANALYSIS / "production_conditions.json"))
    table = np.genfromtxt(ANALYSIS / "curves" / "rdf_oo_20_50ps.tsv", names=True, delimiter="\t")
    r = table["r_A"]

    rows = []
    for key, label in OURS.items():
        c = conditions[key]
        g = table[f"{key}_mean"]
        seeds = [n_within(r, table[f"{key}_seed{k}"], DISTASIO_CUTOFF_A) for k in range(5)]
        rows.append({"label": label, "r1": c["late_first_peak_r_A"][0], "g1": c["late_first_peak_g"][0], "g1_spread": c["late_first_peak_g"][1], "fwhm": fwhm(r, g),
                     "rmin": c["late_first_min_r_A"][0], "gmin": c["late_first_min_g"][0], "gmin_spread": c["late_first_min_g"][1],
                     "r2": c["late_second_peak_r_A"][0], "g2": c["late_second_peak_g"][0], "n_3p28": float(np.mean(seeds)), "n_3p28_spread": float(np.std(seeds, ddof=1) / np.sqrt(5)),
                     "n_3p3": c["late_n_oo_common_radius"][0] if "late_n_oo_common_radius" in c else c["oo_n_oo_common_radius"][0]})

    lines = ["# Literature comparison of the O-O radial distribution functions", "",
             "Our values are the five-run means over 20 to 50 ps as stored by the production analysis (spread as given there); n_OO within 3.28 A is integrated here from the stored mean curves with the DiStasio cutoff and the analysis convention (unique pairs, density (N minus 1) over V). Literature values are copied from the papers' tables (Todorova Table 4 in pm, DiStasio Table I).", "",
             "| curve | first peak r, A | first peak g | FWHM, A | first minimum r, A | first minimum g | second peak r, A | second peak g | n_OO |", "|---|---|---|---|---|---|---|---|---|"]
    for o in rows:
        lines.append(f"| {o['label']}, 300 K | {o['r1']:.2f} | {o['g1']:.2f} ± {o['g1_spread']:.2f} | {o['fwhm']:.2f} | {o['rmin']:.2f} | {o['gmin']:.2f} ± {o['gmin_spread']:.2f} | {o['r2']:.2f} | {o['g2']:.2f} | {o['n_3p28']:.2f} ± {o['n_3p28_spread']:.2f} (3.28 A), {o['n_3p3']:.2f} (3.3 A) |")
    for label, v in LITERATURE.items():
        cells = [f"{v[k]:.2f}" if v[k] is not None else "" for k in ("r1", "g1", "fwhm", "rmin", "gmin", "r2", "g2")]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {v['n'] if v['n'] is not None else ''} |")
    lines += ["", "## Protocol differences", "", "| " + " | ".join(PROTOCOL[0]) + " |", "|---|---|---|---|"]
    for row in PROTOCOL[1:]:
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Reading", "",
              "- Qualitative consistency, not validation: the approximate LDA potential reproduces the direction and the size class of the LDA overstructuring reported at 350 K (first peak near 2.58 A with g about 5, first minimum near 0.2, narrow peak), and the approximate PBE potential sits between the two PBE references and above experiment, as PBE does in both papers.",
              "- The references were obtained at other temperatures (350 K for the LDA row), another density (0.975 g/cm3), with deuterium masses, Car-Parrinello dynamics and pseudopotential plane-wave DFT, so a quantitative match is not expected and would not be meaningful at the 0.1 to 0.3 level in g.",
              "- The approximate PBE first peak moved from 3.06 (10k correction) to 3.48 (18k correction) at identical MD protocol, which is larger than its distance to either PBE reference; the correction, not the physics, dominates that number.",
              "- Neither paper reports proton transfer or ionic defects for LDA water, so the coordination changes seen with the approximate LDA potential are not a known LDA feature and are the point most in need of the direct DFT check."]
    (TABLES / "literature_comparison.json").write_text(json.dumps({"ours": rows, "literature": LITERATURE}, indent=1) + "\n")
    (HERE / "REPORT_LITERATURE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
