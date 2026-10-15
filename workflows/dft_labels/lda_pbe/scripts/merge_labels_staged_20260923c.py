import re
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SETS = ROOT / "sets"
COMPUTED = ROOT / "computed_20260923c"
OUT = ROOT / "labelled_20260923c"
REFERENCES = ROOT / "references"
FREE_ENERGIES = REFERENCES / "free_energies.tsv"
ENERGY_TOL = 1e-5

ARMS = ("madcore_25k", "random_25k", "voronoi_25k", "random_test_6k")
SPLITS = ("ttt", "surfaces", "fff")
BUCKETS = ("isolated_fff", "periodic_ttt", "surfaces_ttf")
FUNCTIONALS = {"pbe": ("pbe", "pbe"), "lda": ("pw-lda", "pwlda")}

TOKEN = re.compile(r'(\S+?)=(?:"([^"]*)"|(\S+))')
POSITION_TOL = 1e-5


def parse_comment(comment):
    info = {}
    for match in TOKEN.finditer(comment):
        info[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
    return info


def read_frames(path):
    frames = []
    with open(path) as fh:
        while True:
            head = fh.readline()
            if not head:
                break
            natoms = int(head)
            info = parse_comment(fh.readline().rstrip("\n"))
            lines = [fh.readline().rstrip("\n") for _ in range(natoms)]
            columns = [line.split() for line in lines]
            species = tuple(c[0] for c in columns)
            positions = np.array([c[1:4] for c in columns], dtype=np.float64)
            forces = np.array([c[4:7] for c in columns], dtype=np.float64) if len(columns[0]) >= 7 else None
            cell = np.array(info["Lattice"].split(), dtype=np.float64).reshape(3, 3) if "Lattice" in info else None
            frames.append(dict(info=info, species=species, positions=positions, forces=forces,
                               cell=cell, pbc=info["pbc"], lines=lines))
    return frames


def geometry_key(frame):
    rounded = np.round(frame["positions"], 4) + 0.0
    return frame["species"], rounded.tobytes()


def read_selection(arm):
    with open(SETS / arm / "selection.tsv") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        rows = [dict(zip(header, line.rstrip("\n").split("\t"))) for line in fh]
    return header, rows


def read_release_ids(arm):
    with open(SETS / arm / "release_ids.tsv") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        rows = [dict(zip(header, line.rstrip("\n").split("\t"))) for line in fh]
    return {row["configuration_id"]: row["release_configuration_id"] for row in rows}


def output_header(header):
    out = []
    for key in header:
        out.append(key)
        if key == "configuration_id":
            out.append("frozen_configuration_id")
    return out


def output_value(row, key):
    if key == "configuration_id":
        return row["release_configuration_id"]
    if key == "frozen_configuration_id":
        return row["configuration_id"]
    return row[key]


def load_arm_sets(arm):
    header, rows = read_selection(arm)
    frames = {split: read_frames(SETS / arm / f"{split}.extxyz") for split in SPLITS}
    release_ids = read_release_ids(arm)
    assert len(release_ids) == len(rows), (arm, len(release_ids), len(rows))

    split_counter = defaultdict(int)
    for row in rows:
        split = row["split_file"]
        row["split_row"] = split_counter[split]
        split_counter[split] += 1
        frame = frames[split][row["split_row"]]
        assert frame["info"]["configuration_id"] == row["configuration_id"], (arm, row["configuration_id"])
        row["frame"] = frame
        row["release_configuration_id"] = release_ids[row["configuration_id"]]

    return header, rows


def check_same_geometry(set_frame, computed, where):
    assert set_frame["species"] == computed["species"], where
    delta = set_frame["positions"] - computed["positions"]
    dev = np.abs(delta).max()

    if computed["cell"] is not None:
        assert set_frame["cell"] is not None, where
        assert np.abs(set_frame["cell"] - computed["cell"]).max() < POSITION_TOL, where
        if dev >= POSITION_TOL:
            fractional = np.linalg.solve(computed["cell"].T, delta.T).T
            assert np.abs(fractional - np.round(fractional)).max() < POSITION_TOL, (where, dev)
            return dev

    assert dev < POSITION_TOL, (where, dev)
    return 0.0


def match_by_geometry(arm, rows, computed_frames, bucket, functional):
    index = defaultdict(list)
    for row in rows:
        index[geometry_key(row["frame"])].append(row)

    matched = []
    for computed in computed_frames:
        candidates = index.get(geometry_key(computed), [])
        fid = int(computed["info"]["frame_id"])
        if len(candidates) > 1:
            candidates = [r for r in candidates if r["split_file"] == "ttt" and r["split_row"] == fid]
        assert len(candidates) == 1, (arm, bucket, functional, fid, len(candidates))
        row = candidates[0]
        assert check_same_geometry(row["frame"], computed, (arm, bucket, functional, fid)) == 0.0
        matched.append((row, computed))

    return matched


def assert_frame_id_rules(arm, rows, matched_by_bucket):
    n_ttt = sum(1 for r in rows if r["split_file"] == "ttt")
    ttf_rows = {r["split_row"] for r, _ in matched_by_bucket["surfaces_ttf"]}
    surfaces_rank = {}
    for r in rows:
        if r["split_file"] == "surfaces" and r["split_row"] not in ttf_rows:
            surfaces_rank[r["split_row"]] = len(surfaces_rank)

    for row, computed in matched_by_bucket["isolated_fff"]:
        assert row["split_file"] == "fff" and int(computed["info"]["frame_id"]) == row["split_row"], (arm, "isolated_fff")

    for row, computed in matched_by_bucket["periodic_ttt"]:
        fid = int(computed["info"]["frame_id"])
        if row["split_file"] == "ttt":
            assert fid == row["split_row"], (arm, "periodic_ttt", fid)
        else:
            assert row["split_file"] == "surfaces", (arm, "periodic_ttt", fid)
            assert fid == n_ttt + surfaces_rank[row["split_row"]], (arm, "periodic_ttt", fid)

    for row, _ in matched_by_bucket["surfaces_ttf"]:
        assert row["split_file"] == "surfaces", (arm, "surfaces_ttf")


def attach_label(row, functional, computed, bucket, source):
    labels = row.setdefault("labels", {})
    assert functional not in labels, (row["configuration_id"], functional)
    labels[functional] = dict(computed=computed, bucket=bucket, source=source,
                              frame_id=int(computed["info"]["frame_id"]),
                              energy=float(computed["info"]["energy"]), half_ts=None)


def read_free_energies():
    table = defaultdict(list)
    with open(FREE_ENERGIES) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            if row["converged"] == "1" and row["e_corrected"] and row["e_free"]:
                key = (row["arm"], row["kind"], int(row["frame"].split("_")[1]), row["functional"])
                table[key].append((float(row["e_corrected"]), float(row["e_free"])))
    return table


def swap_to_free_energy(arm, rows, free_energies):
    swapped = 0
    missing = []
    for row in rows:
        for functional, label in row.get("labels", {}).items():
            if label["source"] != arm:
                continue
            key = (arm, label["bucket"], label["frame_id"], FUNCTIONALS[functional][0])
            matches = sorted((abs(corrected - label["energy"]), free)
                             for corrected, free in free_energies.get(key, [])
                             if abs(corrected - label["energy"]) < ENERGY_TOL)
            if not matches or matches[-1][1] - matches[0][1] > ENERGY_TOL:
                missing.append((row["configuration_id"], functional, len(matches)))
                continue
            free = matches[0][1]
            label["half_ts"] = label["energy"] - free
            label["energy"] = free
            swapped += 1
    print(f"  {arm}: {swapped} labels switched from the corrected energy to the electronic free energy; "
          f"unmatched {len(missing)}")
    assert not missing, missing[:10]


def label_madcore(rows, functional):
    suffix = FUNCTIONALS[functional][1]
    computed_frames = read_frames(COMPUTED / f"madcore_25k-{suffix}.xyz")
    by_cid = {r["configuration_id"]: r for r in rows}
    wrapped = 0
    for computed in computed_frames:
        row = by_cid[computed["info"]["configuration_id"]]
        wrapped += check_same_geometry(row["frame"], computed, ("madcore_25k", functional)) >= POSITION_TOL
        attach_label(row, functional, computed, "madcore_25k", "madcore_25k")
    print(f"  madcore_25k {functional}: {len(computed_frames)} frames joined by configuration_id, {wrapped} stored wrapped into the cell")
    return computed_frames


def label_bucketed(arm, rows, functional):
    suffix = FUNCTIONALS[functional][0]
    matched_by_bucket = {}
    for bucket in BUCKETS:
        path = COMPUTED / f"{arm}_{bucket}_{suffix}.xyz"
        if not path.exists():
            matched_by_bucket[bucket] = []
            continue
        matched = match_by_geometry(arm, rows, read_frames(path), bucket, functional)
        matched_by_bucket[bucket] = matched
        print(f"  {arm} {functional} {bucket}: {len(matched)} frames joined by geometry")

    assert_frame_id_rules(arm, rows, matched_by_bucket)

    for bucket, matched in matched_by_bucket.items():
        for row, computed in matched:
            attach_label(row, functional, computed, bucket, arm)


def share_overlap_labels(voronoi_rows, madcore_frames):
    overlap = {line.strip() for line in open(SETS / "voronoi_25k" / "madcore_intersection.txt") if line.strip()}
    by_cid = {r["configuration_id"]: r for r in voronoi_rows}
    shared = defaultdict(int)
    agreement = []

    for functional, frames in madcore_frames.items():
        for computed in frames:
            cid = computed["info"]["configuration_id"]
            if cid not in overlap:
                continue
            row = by_cid[cid]
            own = row.get("labels", {}).get(functional)
            if own is None:
                check_same_geometry(row["frame"], computed, ("overlap", cid))
                attach_label(row, functional, computed, "madcore_25k", "madcore_25k")
                shared[functional] += 1
            else:
                per_atom = (float(computed["info"]["energy"]) - own["energy"]) / len(computed["species"])
                agreement.append((functional, cid, len(computed["species"]), per_atom * 1000))

    print(f"  voronoi_25k overlap: {len(overlap)} ids; labels shared from madcore_25k: {dict(shared)}")
    print("  overlap computed in both pipelines (madcore minus voronoi, meV/atom):")
    for functional, cid, natoms, diff in agreement:
        print(f"    {functional:4s} {cid:42s} natoms={natoms:3d} diff={diff:+.3f}")


def read_references(functional):
    path = REFERENCES / f"monomers_{functional}.xyz"
    if not path.exists():
        return None

    energies = {}
    for frame in read_frames(path):
        assert len(frame["species"]) == 1 and frame["info"]["subset"] == "monomers", path
        energies[frame["species"][0]] = float(frame["info"]["energy"])
    return energies


def atomization_energy(label, references):
    species = label["computed"]["species"]
    missing = set(species) - set(references)
    assert not missing, missing
    return label["energy"] - sum(references[s] for s in species)


def frame_comment(row, header, arm, functional, label, references):
    computed = label["computed"]
    info = computed["info"]
    tokens = []
    if "Lattice" in info:
        tokens.append(f'Lattice="{info["Lattice"]}"')

    for key in output_header(header):
        if key != "pbc":
            tokens.append(f'{key}="{output_value(row, key)}"')
    tokens.append(f'arm="{arm}"')
    tokens.append(f'subset="{arm}"')
    tokens.append(f'split_row="{row["split_row"]}"')
    tokens.append(f'dft_bucket="{label["bucket"]}"')
    tokens.append(f'frame_id="{label["frame_id"]}"')
    tokens.append(f'label_source="{label["source"]}"')
    tokens.append(f'functional="{functional}"')
    tokens.append(f'energy={label["energy"]!r}')
    tokens.append(f'free_energy={label["energy"]!r}')
    if references is not None:
        corrected = atomization_energy(label, references)
        tokens.append(f"atomization_energy={corrected!r}")
        tokens.append(f"corrected_energy={corrected!r}")
    if "stress" in info:
        tokens.append(f'stress="{info["stress"]}"')
    tokens.append("Properties=species:S:1:pos:R:3:forces:R:3")
    tokens.append(f'pbc="{info["pbc"]}"')
    return " ".join(tokens)


def write_arm(arm, header, rows, references):
    out_dir = OUT / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    for functional in FUNCTIONALS:
        with open(out_dir / f"{functional}.xyz", "w") as fh:
            for row in rows:
                label = row.get("labels", {}).get(functional)
                if label is None:
                    continue
                fh.write(f"{len(label['computed']['species'])}\n")
                fh.write(frame_comment(row, header, arm, functional, label, references[functional]) + "\n")
                fh.write("\n".join(label["computed"]["lines"]) + "\n")

    extra = ["split_row", "dft_bucket", "frame_id", "has_pbe", "has_lda", "label_source",
             "e_pbe", "e_lda", "half_ts_pbe", "half_ts_lda", "atomization_e_pbe", "atomization_e_lda",
             "corrected_e_pbe", "corrected_e_lda", "fmax_pbe", "fmax_lda"]
    with open(out_dir / "manifest.tsv", "w") as fh:
        fh.write("\t".join(output_header(header) + extra) + "\n")
        for row in rows:
            labels = row.get("labels", {})
            buckets = {l["bucket"] for l in labels.values()}
            frame_ids = {l["frame_id"] for l in labels.values()}
            sources = {l["source"] for l in labels.values()}
            assert len(buckets) <= 1 and len(frame_ids) <= 1 and len(sources) <= 1, row["configuration_id"]
            values = [output_value(row, key) for key in output_header(header)]
            values += [str(row["split_row"]), next(iter(buckets), ""), str(next(iter(frame_ids), "")),
                       str(int("pbe" in labels)), str(int("lda" in labels)), next(iter(sources), "")]
            for functional in FUNCTIONALS:
                values.append(repr(labels[functional]["energy"]) if functional in labels else "")
            for functional in FUNCTIONALS:
                half_ts = labels[functional]["half_ts"] if functional in labels else None
                values.append(f"{half_ts:.6f}" if half_ts is not None else "")
            corrected = {}
            for functional in FUNCTIONALS:
                has_both = functional in labels and references[functional] is not None
                corrected[functional] = repr(atomization_energy(labels[functional], references[functional])) if has_both else ""
            for functional in FUNCTIONALS:
                values.append(corrected[functional])
            for functional in FUNCTIONALS:
                values.append(corrected[functional])
            for functional in FUNCTIONALS:
                values.append(f'{np.linalg.norm(labels[functional]["computed"]["forces"], axis=1).max():.4f}' if functional in labels else "")
            fh.write("\t".join(values) + "\n")


def summarize(arm, rows):
    lines = []
    for split in SPLITS:
        subset = [r for r in rows if r["split_file"] == split]
        has = {f: sum(1 for r in subset if f in r.get("labels", {})) for f in FUNCTIONALS}
        both = sum(1 for r in subset if set(FUNCTIONALS) <= set(r.get("labels", {})))
        lines.append(f"| {arm} | {split} | {len(subset)} | {has['pbe']} | {has['lda']} | {both} |")
    has = {f: sum(1 for r in rows if f in r.get("labels", {})) for f in FUNCTIONALS}
    both = sum(1 for r in rows if set(FUNCTIONALS) <= set(r.get("labels", {})))
    lines.append(f"| {arm} | **all** | {len(rows)} | {has['pbe']} | {has['lda']} | {both} |")
    return lines


def main():
    summary = ["| arm | split | selected | pbe | lda | both |", "|---|---|---|---|---|---|"]
    madcore_frames = {}
    references = {functional: read_references(functional) for functional in FUNCTIONALS}
    for functional, energies in references.items():
        print(f"{functional}: isolated-atom references for {len(energies) if energies else 0} elements")
    free_energies = read_free_energies()
    print(f"free energies scanned from {len(free_energies)} (arm, kind, frame, functional) runs")

    for arm in ARMS:
        header, rows = load_arm_sets(arm)
        print(f"{arm}: {len(rows)} selected")
        for functional in FUNCTIONALS:
            if arm == "madcore_25k":
                madcore_frames[functional] = label_madcore(rows, functional)
            else:
                label_bucketed(arm, rows, functional)
        if arm != "madcore_25k":
            swap_to_free_energy(arm, rows, free_energies)
        if arm == "voronoi_25k":
            share_overlap_labels(rows, madcore_frames)
        write_arm(arm, header, rows, references)
        summary += summarize(arm, rows)

    print()
    print("\n".join(summary))


if __name__ == "__main__":
    main()
