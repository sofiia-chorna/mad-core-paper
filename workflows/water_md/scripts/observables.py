import numpy as np


KB_EV = 8.617333262e-5

# one radial grid for every pair and condition, up to half the 9.8555 A cell width of 4.928 A
RDF_RMAX = 4.92
RDF_BIN = 0.02
RDF_EDGES = np.arange(0.0, RDF_RMAX + 0.5 * RDF_BIN, RDF_BIN)
RDF_CENTRES = 0.5 * (RDF_EDGES[1:] + RDF_EDGES[:-1])

# per H, the distance to its nearest O and to its second-nearest O, wide enough for shared protons
NEAREST_OH_EDGES = np.arange(0.6, 1.8 + 1e-9, 0.0025)
SECOND_OH_EDGES = np.arange(0.9, 4.5 + 1e-9, 0.01)

# threshold crossings, frozen before production and reported apart from confirmed numerical failures
# the force limit is five times the largest pilot force at 1.0 g/cm3, 7.14 eV/A, rounded up
THRESHOLDS = {
    "min_pair_distance_A": 0.6,
    "max_atomic_force_eV_A": 36.0,
    "kinetic_temperature_K": 600.0,
}

# a conserved-quantity drift ten times the largest NVE pilot drift is reported as an integration warning
INTEGRATION_WARNING_MEV_PER_ATOM_PER_PS = 0.05

# structural events, reported and never used to truncate a trajectory
EVENTS = {"hh_close_A": 1.0, "oo_close_A": 2.0}


def mic_vectors(a, b, cell):
    inverse = np.linalg.inv(cell)
    fractional = (a[:, None, :] - b[None, :, :]) @ inverse
    fractional -= np.round(fractional)
    return fractional @ cell


def mic_distances(a, b, cell):
    return np.linalg.norm(mic_vectors(a, b, cell), axis=-1)


# kinetic temperature of a subset over its share of 3N minus 3, so that every subset averages to the bath temperature
def kinetic_temperatures(masses, velocities, symbols):
    kinetic = 0.5 * (masses[None, :, None] * velocities**2).sum(-1)
    total_mass = masses.sum()
    result = {}

    subsets = {"all": np.ones(len(symbols), dtype=bool), "H": symbols == "H", "O": symbols == "O"}
    for name, sel in subsets.items():
        dof = 3 * sel.sum() - 3 * masses[sel].sum() / total_mass
        result[name] = 2.0 * kinetic[:, sel].sum(1) / (dof * KB_EV)

    return result


# per-frame failure measures, molecular assignment, events and histograms
def frame_analysis(positions, cell, forces, symbols):
    o = np.where(symbols == "O")[0]
    h = np.where(symbols == "H")[0]
    initial_owner = 3 * (h // 3)

    d_all = mic_distances(positions, positions, cell)
    off = d_all[~np.eye(len(symbols), dtype=bool)]

    d_oo = d_all[np.ix_(o, o)][np.triu_indices(len(o), 1)]
    d_oh = d_all[np.ix_(o, h)].ravel()
    d_hh = d_all[np.ix_(h, h)][np.triu_indices(len(h), 1)]

    # each H belongs to its nearest O, the second-nearest O is its hydrogen-bond partner
    h_to_o = d_all[np.ix_(h, o)]
    order = np.argsort(h_to_o, axis=1)
    nearest = h_to_o[np.arange(len(h)), order[:, 0]]
    second = h_to_o[np.arange(len(h)), order[:, 1]]
    owner = o[order[:, 0]]
    hydrogens_per_o = np.bincount(order[:, 0], minlength=len(o))

    # H-O-H angle of every O that owns exactly two H
    angles = []
    for k in np.where(hydrogens_per_o == 2)[0]:
        pair = h[owner == o[k]]
        v = mic_vectors(positions[pair], positions[o[k]][None, :], cell)[:, 0, :]
        cosine = v[0] @ v[1] / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]))
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    return {
        "min_pair_distance_A": float(off.min()),
        "max_atomic_force_eV_A": float(np.linalg.norm(forces, axis=1).max()),
        "finite": bool(np.isfinite(positions).all() and np.isfinite(forces).all()),
        "non_water_oxygens": int((hydrogens_per_o != 2).sum()),
        "hydrogens_off_initial_oxygen": int((owner != initial_owner).sum()),
        "owner": owner,
        "min_hh_A": float(d_hh.min()),
        "min_oo_A": float(d_oo.min()),
        "nearest_oh_mean_A": float(nearest.mean()),
        "nearest_oh_max_A": float(nearest.max()),
        "hoh_angle_mean_deg": float(np.mean(angles)) if angles else np.nan,
        "hist_oo": np.histogram(d_oo, RDF_EDGES)[0],
        "hist_oh": np.histogram(d_oh, RDF_EDGES)[0],
        "hist_hh": np.histogram(d_hh, RDF_EDGES)[0],
        "hist_nearest_oh": np.histogram(nearest, NEAREST_OH_EDGES)[0],
        "hist_second_oh": np.histogram(second, SECOND_OH_EDGES)[0],
        "pairs": {"oo": d_oo.size, "oh": d_oh.size, "hh": d_hh.size},
    }


# g of r from summed histograms, ideal-gas normalisation at the cell density of each pair type
def rdf(hist_sum, frames, pairs, volume):
    shell = 4.0 / 3.0 * np.pi * (RDF_EDGES[1:] ** 3 - RDF_EDGES[:-1] ** 3)
    return hist_sum / (frames * pairs / volume * shell)


def density(hist_sum, edges):
    return hist_sum / (hist_sum.sum() * np.diff(edges))


# first peak, first minimum and second peak of an O-O g of r on the common grid
def oo_shells(g):
    r = RDF_CENTRES
    peak_window = (r >= 2.4) & (r <= 3.2)
    i1 = np.where(peak_window)[0][np.argmax(g[peak_window])]

    min_window = (r > r[i1]) & (r <= 4.2)
    i2 = np.where(min_window)[0][np.argmin(g[min_window])]

    second_window = (r > r[i2]) & (r <= RDF_RMAX)
    i3 = np.where(second_window)[0][np.argmax(g[second_window])]

    return {
        "first_peak_r_A": float(r[i1]),
        "first_peak_g": float(g[i1]),
        "first_min_r_A": float(r[i2]),
        "first_min_g": float(g[i2]),
        "second_peak_r_A": float(r[i3]),
        "second_peak_g": float(g[i3]),
        "peak_to_min_ratio": float(g[i1] / g[i2]) if g[i2] > 0 else float("inf"),
    }


# mean number of O neighbours of an O within radius r, from the O-O g of r over unique pairs
def coordination(g, n_o, volume, radius):
    r = RDF_CENTRES
    sel = r <= radius
    rho = (n_o - 1) / volume
    return float((4.0 * np.pi * rho * g[sel] * r[sel] ** 2 * RDF_BIN).sum())
