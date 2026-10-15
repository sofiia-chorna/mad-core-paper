import json

import numpy as np

from common_smap import (AUG_MODELS, AUG_SETS, LOGS, MAIN_MODELS, MAIN_SETS, POPULATION_FILE, REPORTS,
                         frame_order, load_errors, make_logger, read_tsv, write_tsv)

RANDOM_SETS = {name: MAIN_SETS[name] for name in ("random_test", "random_holdout")}
MADCORE_SETS = {name: MAIN_SETS[name] for name in ("madcore_holdout",)}

N_BINS = 64
MIN_COUNT = 10
SENSITIVITY = (5, 10, 15, 25, 50)
LIBRARY_CLIP = (0.001, 99.999)
PAD = 0.10
EVAL_CLIP = (0.0, 100.0)


def assemble(sets, models, coordinate_of):
    identifier, x, y, set_name, group = [], [], [], [], []
    errors = {name: {"energy": [], "force": []} for name in models}

    for name, (manifest, source) in sets.items():
        order = frame_order(source)
        position = {cid: i for i, cid in enumerate(order)}
        members = [r["configuration_id"] for r in read_tsv(manifest)]
        rows = np.array([position[cid] for cid in members])

        cached = {model: load_errors(stem, source) for model, stem in models.items()}
        for model in models:
            errors[model]["energy"].append(cached[model][0][rows])
            errors[model]["force"].append(cached[model][1][rows])

        identifier.extend(members)
        set_name.extend([name] * len(members))
        group.extend(cid.split(":")[0] for cid in members)
        x.extend(coordinate_of[cid][0] for cid in members)
        y.extend(coordinate_of[cid][1] for cid in members)

    return {"configuration_id": np.array(identifier), "x": np.array(x), "y": np.array(y),
            "set": np.array(set_name), "group": np.array(group),
            "errors": {m: {k: np.concatenate(v) for k, v in errors[m].items()} for m in models}}


def make_grid(library_xy, records):
    low = np.minimum(np.percentile(library_xy, LIBRARY_CLIP[0], axis=0),
                     np.percentile(np.stack([records["x"], records["y"]], 1), EVAL_CLIP[0], axis=0))
    high = np.maximum(np.percentile(library_xy, LIBRARY_CLIP[1], axis=0),
                      np.percentile(np.stack([records["x"], records["y"]], 1), EVAL_CLIP[1], axis=0))

    span = high - low
    low, high = low - PAD * span, high + PAD * span

    return {"x_edges": np.linspace(low[0], high[0], N_BINS + 1).tolist(),
            "y_edges": np.linspace(low[1], high[1], N_BINS + 1).tolist()}


def digitise(x, y, grid):
    x_edges, y_edges = np.asarray(grid["x_edges"]), np.asarray(grid["y_edges"])
    i = np.digitize(x, x_edges) - 1
    j = np.digitize(y, y_edges) - 1
    inside = (i >= 0) & (i < N_BINS) & (j >= 0) & (j < N_BINS)

    return i, j, inside


def bin_statistics(records, models, sets, grid):
    i, j, inside = digitise(records["x"], records["y"], grid)
    flat = i[inside] * N_BINS + j[inside]

    count = np.bincount(flat, minlength=N_BINS * N_BINS).reshape(N_BINS, N_BINS)
    composition = {}
    for name in sets:
        selected = records["set"][inside] == name
        composition[name] = np.bincount(flat[selected], minlength=N_BINS * N_BINS).reshape(N_BINS, N_BINS)

    statistics = {}
    for model in models:
        for kind in ("energy", "force"):
            values = records["errors"][model][kind][inside]
            total = np.bincount(flat, weights=values, minlength=N_BINS * N_BINS).reshape(N_BINS, N_BINS)
            statistics[f"{model}__{kind}_sum"] = total
            with np.errstate(invalid="ignore", divide="ignore"):
                statistics[f"{model}__{kind}_mean"] = np.where(count > 0, total / count, np.nan)

    return count, composition, statistics


def library_background(library_xy, population, grid):
    i, j, inside = digitise(library_xy[:, 0], library_xy[:, 1], grid)
    flat = i[inside] * N_BINS + j[inside]

    centres = np.bincount(flat, minlength=N_BINS * N_BINS).reshape(N_BINS, N_BINS)
    weighted = np.bincount(flat, weights=np.asarray(population)[inside].astype(np.float64),
                           minlength=N_BINS * N_BINS).reshape(N_BINS, N_BINS)

    return centres, weighted, int(inside.sum())


def family_table(records, models, min_structures=25):
    rows = []
    for group in sorted(set(records["group"].tolist())):
        selected = records["group"] == group
        if selected.sum() < min_structures:
            continue

        row = {"dataset_group": group, "n": int(selected.sum())}
        for model in models:
            row[f"{model}_E_mae"] = round(float(records["errors"][model]["energy"][selected].mean()), 2)
            row[f"{model}_F_mae"] = round(float(records["errors"][model]["force"][selected].mean()), 1)
        names = list(models)
        for kind, unit in (("energy", "E"), ("force", "F")):
            reference = records["errors"][names[0]][kind][selected].mean()
            for model in names[1:]:
                value = records["errors"][model][kind][selected].mean()
                row[f"log2_{model}_over_{names[0]}_{unit}"] = round(float(np.log2(value / reference)), 3)
        rows.append(row)

    return sorted(rows, key=lambda r: -r["n"])


def write_bins(path, count, composition, statistics, grid):
    x_edges, y_edges = np.asarray(grid["x_edges"]), np.asarray(grid["y_edges"])
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

    rows = []
    for i in range(N_BINS):
        for j in range(N_BINS):
            if count[i, j] == 0:
                continue
            row = {"bin_i": i, "bin_j": j, "x": round(float(x_centres[i]), 4), "y": round(float(y_centres[j]), 4),
                   "n": int(count[i, j])}
            row.update({f"n_{name}": int(values[i, j]) for name, values in composition.items()})
            row.update({key: ("" if np.isnan(values[i, j]) else round(float(values[i, j]), 4))
                        for key, values in statistics.items() if key.endswith("_mean")})
            rows.append(row)

    write_tsv(path, rows, list(rows[0]))

    return len(rows)


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    say = make_logger(LOGS / "bin_errors.log")

    stored = np.load(REPORTS / "coords_eval.npz", allow_pickle=True)
    embedding = stored["embedding"]
    coordinate_of = {cid: embedding[i, :2] for i, cid in enumerate(stored["configuration_id"].astype(str))}
    say(f"coordinates for {len(coordinate_of):,} frames, "
        f"{int(np.isnan(embedding[:, 0]).sum()):,} unresolved")

    library_xy = np.load(REPORTS / "cache" / "library_xy.npy")
    population = np.load(POPULATION_FILE, mmap_mode="r")

    summary = {"n_bins": N_BINS, "min_count": MIN_COUNT, "panels": {}}

    for label, sets, models in (("main", MAIN_SETS, MAIN_MODELS),
                                ("random_only", RANDOM_SETS, MAIN_MODELS),
                                ("madcore_only", MADCORE_SETS, MAIN_MODELS),
                                ("aug", AUG_SETS, AUG_MODELS)):
        records = assemble(sets, models, coordinate_of)
        say(f"\n[{label}] {len(records['configuration_id']):,} scored structures, "
            f"{len(set(records['group'].tolist()))} dataset groups, models {list(models)}")

        grid = make_grid(library_xy, records) if label == "main" else summary["grid"]
        if label == "main":
            summary["grid"] = grid

        _, _, inside = digitise(records["x"], records["y"], grid)
        say(f"[{label}] inside grid: {int(inside.sum()):,} / {len(inside):,} "
            f"({100 * inside.mean():.2f}%)")

        count, composition, statistics = bin_statistics(records, models, sets, grid)
        occupied = int((count > 0).sum())
        retained = {}
        for cut in SENSITIVITY:
            kept = count >= cut
            retained[str(cut)] = {
                "bins": int(kept.sum()),
                "structures": int(count[kept].sum()),
                "fraction": round(float(count[kept].sum() / count.sum()), 4),
                "per_set": {name: round(float(values[kept].sum() / max(values.sum(), 1)), 4)
                            for name, values in composition.items()}}
        say(f"[{label}] occupied bins {occupied} / {N_BINS ** 2}; retained: "
            + "  ".join(f"{k}:{v['bins']}bins/{v['fraction']:.1%}" for k, v in retained.items()))
        for name, fraction in retained[str(MIN_COUNT)]["per_set"].items():
            say(f"[{label}]   at min_count={MIN_COUNT}, {name}: {fraction:.1%} of its structures retained")

        written = write_bins(REPORTS / f"bins_{label}.tsv", count, composition, statistics, grid)
        families = family_table(records, models)
        write_tsv(REPORTS / f"family_{label}.tsv", families, list(families[0]))
        say(f"[{label}] wrote bins_{label}.tsv ({written} rows) and family_{label}.tsv ({len(families)} rows)")

        np.savez(REPORTS / "cache" / f"panels_{label}.npz",
                 count=count, **{f"set__{k}": v for k, v in composition.items()}, **statistics)

        summary["panels"][label] = {"structures": int(len(records["configuration_id"])),
                                    "inside_grid": int(inside.sum()), "occupied_bins": occupied,
                                    "models": list(models), "sets": {k: int((records["set"] == k).sum()) for k in sets},
                                    "inside_grid_per_set": {k: int((records["set"][inside] == k).sum()) for k in sets},
                                    "retained": retained}

    centres, weighted, inside_library = library_background(library_xy, population, summary["grid"])
    np.savez(REPORTS / "cache" / "panels_library.npz", centres=centres, population=weighted)
    summary["library"] = {"centres_inside_grid": inside_library,
                          "population_inside_grid": int(weighted.sum()),
                          "population_total": int(np.asarray(population).sum())}
    say(f"\nlibrary background: {inside_library:,} / {len(library_xy):,} centres inside the grid, "
        f"{int(weighted.sum()):,} of {int(np.asarray(population).sum()):,} structures")

    with open(REPORTS / "grid.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    say(f"wrote {REPORTS / 'grid.json'}")


if __name__ == "__main__":
    main()
