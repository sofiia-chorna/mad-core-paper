#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def chemical_symbol_for_species(sp: dict, skip_disordered: bool) -> Optional[str]:
    syms = sp.get("chemical_symbols") or []
    if not syms:
        return None
    if syms == ["vacancy"]:
        return None
    if len(syms) > 1:
        if skip_disordered:
            return None
        # take the highest-concentration component fall back to first
        concs = sp.get("concentration") or []
        if concs and len(concs) == len(syms):
            return syms[concs.index(max(concs))]
        return syms[0]
    return syms[0]


def jsonl_to_extxyz(
    input_path: Path,
    output_path: Path,
    max_atoms: Optional[int] = None,
    skip_disordered: bool = False,
    strict: bool = False,
    progress: bool = True,
) -> dict:
    n_in = n_out = 0
    n_skip_atoms = n_skip_disorder = n_skip_invalid = 0

    # pre-count lines for the progress bar (cheap on a local file)
    total = None
    if progress and tqdm is not None:
        with open(input_path, "rb") as f:
            total = sum(1 for _ in f)

    with open(input_path) as fin, open(output_path, "w", buffering=1024 * 1024) as fout:
        iterator = fin
        if progress and tqdm is not None:
            iterator = tqdm(fin, total=total, unit="struct", smoothing=0.1)

        for line in iterator:
            line = line.strip()
            if not line:
                continue
            n_in += 1

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                if strict:
                    raise
                n_skip_invalid += 1
                continue

            try:
                # nsites filter
                nsites = entry["nsites"]
                if max_atoms is not None and nsites > max_atoms:
                    n_skip_atoms += 1
                    continue

                positions = entry["cartesian_site_positions"]
                if len(positions) != nsites:
                    raise ValueError(
                        f"position/nsites mismatch: {len(positions)} vs {nsites}"
                    )

                # species lookup
                species_list = entry.get("species")
                site_names = entry["species_at_sites"]

                if species_list:
                    species_lookup = {s["name"]: s for s in species_list}
                else:
                    species_lookup = None  # signal direct-symbol mode

                disordered_hit = False
                symbols = []
                for name in site_names:
                    if species_lookup is None:
                        # direct element symbol mode
                        if name == "vacancy":
                            disordered_hit = True
                            break
                        symbols.append(name)
                        continue
                    sp = species_lookup.get(name)
                    if sp is None:
                        raise ValueError(f"unknown species '{name}'")
                    sym = chemical_symbol_for_species(sp, skip_disordered)
                    if sym is None:
                        # vacancy or disordered when skipping
                        disordered_hit = True
                        break
                    symbols.append(sym)
                if disordered_hit:
                    n_skip_disorder += 1
                    continue

                # PBC and lattice
                dim_types = entry.get("dimension_types") or [1, 1, 1]
                pbc_str = " ".join("T" if d else "F" for d in dim_types)

                lat = entry.get("lattice_vectors")
                if lat is None:
                    if any(dim_types):
                        raise ValueError("periodic structure with no lattice_vectors")
                    lat = [[0.0, 0.0, 0.0]] * 3
                lat_str = " ".join(f"{x:.10f}" for vec in lat for x in vec)

                # comment line metadata
                sid = entry.get("_optimade_id", "")
                formula = entry.get("chemical_formula_reduced", "")
                meta_kv = []
                if sid:
                    meta_kv.append(f"structure_id={sid}")
                if formula:
                    meta_kv.append(f"formula={formula}")
                meta_str = (" " + " ".join(meta_kv)) if meta_kv else ""

                # write frame
                fout.write(f"{nsites}\n")
                fout.write(
                    f'Lattice="{lat_str}" '
                    f'Properties=species:S:1:pos:R:3 '
                    f'pbc="{pbc_str}"'
                    f'{meta_str}\n'
                )
                for sym, pos in zip(symbols, positions):
                    fout.write(f"{sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n")
                n_out += 1

            except (KeyError, ValueError, TypeError) as exc:
                if strict:
                    print(
                        f"\n[FAIL] line {n_in}, id={entry.get('_optimade_id')}: {exc}",
                        file=sys.stderr,
                    )
                    raise
                n_skip_invalid += 1

    return {
        "n_in": n_in,
        "n_out": n_out,
        "skip_atoms": n_skip_atoms,
        "skip_disorder": n_skip_disorder,
        "skip_invalid": n_skip_invalid,
    }


def print_summary(stats: dict, output_path: Path) -> None:
    print(f"  Read:    {stats['n_in']:>10d} structures")
    print(f"  Wrote:   {stats['n_out']:>10d} structures")
    if stats["skip_atoms"]:
        print(f"  Skipped (size):       {stats['skip_atoms']:>10d}")
    if stats["skip_disorder"]:
        print(f"  Skipped (disorder):   {stats['skip_disorder']:>10d}")
    if stats["skip_invalid"]:
        print(f"  Skipped (invalid):    {stats['skip_invalid']:>10d}")
    size_gb = output_path.stat().st_size / 1e9 if output_path.exists() else 0.0
    print(f"  Output:  {output_path} ({size_gb:.2f} GB)")


def process_directory(
    input_dir: Path,
    output_dir: Optional[Path],
    overwrite: bool,
    max_atoms: Optional[int],
    skip_disordered: bool,
    strict: bool,
    progress: bool,
) -> None:
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl files found in {input_dir}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(files)} JSONL file(s) in {input_dir}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print()

    totals = {"n_in": 0, "n_out": 0,
              "skip_atoms": 0, "skip_disorder": 0, "skip_invalid": 0}
    n_done = n_skipped_existing = 0

    for i, jsonl_path in enumerate(files, 1):
        if output_dir is not None:
            xyz_path = output_dir / (jsonl_path.stem + ".xyz")
        else:
            xyz_path = jsonl_path.with_suffix(".xyz")

        print(f"[{i}/{len(files)}] {jsonl_path.name}")

        if xyz_path.exists() and not overwrite:
            print(f"  -> {xyz_path} already exists, skipping (use --overwrite to redo)")
            print()
            n_skipped_existing += 1
            continue

        stats = jsonl_to_extxyz(
            jsonl_path, xyz_path,
            max_atoms=max_atoms,
            skip_disordered=skip_disordered,
            strict=strict,
            progress=progress,
        )
        print_summary(stats, xyz_path)
        print()

        for k in totals:
            totals[k] += stats[k]
        n_done += 1

    # aggregate footer
    print("=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    print(f"Files processed:        {n_done}")
    if n_skipped_existing:
        print(f"Files skipped (exist):  {n_skipped_existing}")
    print(f"Total read:    {totals['n_in']:>12d} structures")
    print(f"Total written: {totals['n_out']:>12d} structures")
    if totals["skip_atoms"]:
        print(f"Total skipped (size):     {totals['skip_atoms']:>12d}")
    if totals["skip_disorder"]:
        print(f"Total skipped (disorder): {totals['skip_disorder']:>12d}")
    if totals["skip_invalid"]:
        print(f"Total skipped (invalid):  {totals['skip_invalid']:>12d}")


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("input", type=Path,
                   help="JSONL input file OR a directory of .jsonl files")
    p.add_argument("output", type=Path, nargs="?", default=None,
                   help="extxyz output (file mode only; "
                        "default: <input>.xyz alongside the JSONL)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (directory mode only; "
                        "default: place .xyz files alongside their .jsonl)")
    p.add_argument("--overwrite", action="store_true",
                   help="Reprocess files whose .xyz output already exists")
    p.add_argument("--max-atoms", type=int, default=None,
                   help="Skip structures with more than this many sites")
    p.add_argument("--skip-disordered", action="store_true",
                   help="Skip structures with mixed-occupancy sites")
    p.add_argument("--strict", action="store_true",
                   help="Abort on first malformed structure (default: skip)")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm progress bar")
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"Input does not exist: {args.input}")

    # directory mode
    if args.input.is_dir():
        if args.output is not None:
            sys.exit(
                "When input is a directory, do not pass a positional output. "
                "Use --output-dir instead."
            )
        process_directory(
            input_dir=args.input,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            max_atoms=args.max_atoms,
            skip_disordered=args.skip_disordered,
            strict=args.strict,
            progress=not args.no_progress,
        )
        return

    # single file mode
    if args.output_dir is not None:
        sys.exit("--output-dir only applies to directory input.")

    output = args.output or args.input.with_suffix(".xyz")

    if output.exists() and not args.overwrite:
        sys.exit(
            f"Output already exists: {output}\n"
            f"Pass --overwrite to replace it, or pick a different path."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    stats = jsonl_to_extxyz(
        args.input, output,
        max_atoms=args.max_atoms,
        skip_disordered=args.skip_disordered,
        strict=args.strict,
        progress=not args.no_progress,
    )

    print()
    print("=" * 60)
    print_summary(stats, output)


if __name__ == "__main__":
    main()
