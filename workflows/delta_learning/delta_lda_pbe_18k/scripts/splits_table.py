import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test_madcore", "test_random")


def main():
    s = json.load(open(D / "manifests" / "SETS.json"))
    c = s["composition"]
    bins = list(c["train"]["size_bins"])
    sources = sorted({k for name in SPLITS for k in c[name]["sources"]})

    head = "| quantity | " + " | ".join(SPLITS) + " |\n|---|" + "---|" * len(SPLITS)
    rows = [("structures", lambda x: f"{x['structures']:,}"), ("atoms", lambda x: f"{x['atoms']:,}"), ("distinct elements", lambda x: str(x["distinct_elements"])),
            ("median atoms", lambda x: f"{x['natoms_median']:.0f}"), ("max atoms", lambda x: str(x["natoms_max"]))]
    rows += [(f"source {src}", (lambda src: lambda x: str(x["sources"].get(src, 0)))(src)) for src in sources]
    rows += [(f"pbc {p}", (lambda p: lambda x: str(x["pbc"].get(p, 0)))(p)) for p in ("FFF", "TTT")]
    rows += [(f"size {b} atoms", (lambda b: lambda x: str(x["size_bins"][b]))(b)) for b in bins]
    rows.append(("nonperiodic molecules with 10 atoms or more", lambda x: str(x["nonperiodic_molecules_10_atoms_or_more"])))
    table = head + "\n" + "\n".join(f"| {label} | " + " | ".join(f(c[name]) for name in SPLITS) + " |" for label, f in rows)

    origins = sorted({k for name in SPLITS for k in c[name]["origins"]})
    rows_origin = [(f"origin {o}", (lambda o: lambda x: str(x["origins"].get(o, 0)))(o)) for o in origins]
    origin_table = head + "\n" + "\n".join(f"| {label} | " + " | ".join(f(c[name]) for name in SPLITS) + " |" for label, f in rows_origin)
    big = "\n".join(f"- {name}: " + (", ".join(c[name]["nonperiodic_molecules_10_atoms_or_more_ids"]) or "none") for name in SPLITS)
    singles = "\n".join(f"- {name}: " + (", ".join(c[name]["single_atom_nonperiodic"]) or "none") for name in SPLITS)
    heldout = "\n".join(f"- {name}: {len(v)}" + (" (" + ", ".join(v) + ")" if v else "") for name, v in s["mad16_heldout_geometries"].items())
    absent = "\n".join(f"- {name}: " + (", ".join(v) or "none") for name, v in s["elements_absent_from_training"].items())
    text = (f"# delta_lda_pbe_18k, split composition\n\nSeed {s['seed']}. Pool: 18,774 old training + 232 eligible old MAD-CORE validation + 1,875 old MAD-CORE test = "
            f"{c['pool']['structures']:,} structures, every one paired for r2SCAN/LDA and r2SCAN/PBE under the existing rules. Split rule: {s['split_rule']}. "
            f"The random test (973) is unchanged and separate.\n\n## Splits\n\n{table}\n\n## Previous membership of the structures\n\n{origin_table}\n\n"
            f"## Nonperiodic molecules with 10 atoms or more\n\nThe pool holds {c['pool']['nonperiodic_molecules_10_atoms_or_more']} such structures.\n\n{big}\n\n"
            f"## Single atom nonperiodic structures (eligible under the existing rules, no reference atoms added)\n\n{singles}\n\n"
            f"## MAD-1.6 validation or test geometries (r2SCAN label from those files, previously excluded from training by the training rule)\n\n{heldout}\n\n"
            f"## Elements absent from training\n\n{absent}\n")
    (D / "manifests" / "SPLITS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
