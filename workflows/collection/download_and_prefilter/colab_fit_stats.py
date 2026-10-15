#!/usr/bin/env python3
"""
Quick statistics for downloaded ColabFit datasets.

Usage:
    pip install pyarrow pandas tabulate

    # Scan all downloaded datasets
    python colabfit_stats.py ./colabfit_data

    # Export to CSV
    python colabfit_stats.py ./colabfit_data --csv stats.csv
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq


def scan_dataset(dataset_dir):
    """Extract stats from a single dataset directory. Returns a dict."""
    dataset_dir = Path(dataset_dir)
    co_dir = dataset_dir / "co"
    ds_file = dataset_dir / "ds.parquet"

    name = dataset_dir.name
    # Strip colabfit_ prefix if downloaded via HF
    if name.startswith("colabfit_"):
        name = name[len("colabfit_"):]

    result = {"name": name, "path": str(dataset_dir)}

    # Read dataset metadata if available
    if ds_file.exists():
        try:
            ds = pq.read_table(ds_file).to_pydict()
            result["description"] = (ds.get("description") or [""])[0][:80]
        except Exception:
            pass

    if not co_dir.exists():
        result["error"] = "no co/ directory"
        return result

    try:
        # Read parquet metadata only (no data loaded into memory)
        parquet_files = sorted(co_dir.glob("co_*.parquet"))
        if not parquet_files:
            result["error"] = "no parquet files in co/"
            return result

        # Get schema from first file
        schema = pq.read_schema(parquet_files[0])
        available_props = []
        for col in ["energy", "atomic_forces", "cauchy_stress", "formation_energy",
                     "atomization_energy", "adsorption_energy", "electronic_band_gap",
                     "energy_above_hull"]:
            if col in schema.names:
                available_props.append(col)

        # Count rows and atoms using metadata (fast, no data read)
        total_configs = 0
        total_atoms = 0
        total_bytes = 0

        for pf in parquet_files:
            meta = pq.read_metadata(pf)
            total_configs += meta.num_rows
            total_bytes += pf.stat().st_size

        # Sample first batch to estimate atoms per config
        first_table = pq.read_table(
            parquet_files[0],
            columns=["atomic_numbers"],
        )
        first_batch = first_table.to_batches(max_chunksize=min(1000, first_table.num_rows))[0]
        d = first_batch.to_pydict()
        sample_natoms = [len(an) for an in d["atomic_numbers"]]
        avg_atoms = sum(sample_natoms) / len(sample_natoms) if sample_natoms else 0

        # For small datasets, compute exact total; for large ones, estimate
        if total_configs <= 100_000:
            # Exact count
            total_atoms = 0
            for pf in parquet_files:
                t = pq.read_table(pf, columns=["atomic_numbers"])
                for batch in t.to_batches():
                    bd = batch.to_pydict()
                    total_atoms += sum(len(an) for an in bd["atomic_numbers"])
            exact = True
        else:
            total_atoms = int(avg_atoms * total_configs)
            exact = False

        # Unique elements from sample
        elements = set()
        for an in d["atomic_numbers"]:
            elements.update(an)

        result.update({
            "n_configs": total_configs,
            "n_atoms": total_atoms,
            "atoms_exact": exact,
            "avg_atoms_per_config": round(avg_atoms, 1),
            "n_elements_sample": len(elements),
            "properties": ",".join(available_props),
            "n_files": len(parquet_files),
            "size_mb": round(total_bytes / 1e6, 1),
        })

    except Exception as e:
        result["error"] = str(e)

    return result


def find_datasets(root_dir):
    """Find all dataset directories (contain a co/ subdirectory)."""
    root = Path(root_dir)
    datasets = []
    # Check immediate children
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "co").exists():
            datasets.append(d)
    # Also check if root itself is a dataset
    if (root / "co").exists():
        datasets.append(root)
    return datasets


def main():
    parser = argparse.ArgumentParser(description="ColabFit dataset statistics")
    parser.add_argument("root_dir", help="Root directory containing downloaded datasets")
    parser.add_argument("--csv", type=str, default=None, help="Export results to CSV")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    datasets = find_datasets(args.root_dir)
    if not datasets:
        print(f"No datasets found in {args.root_dir}")
        print("Expected structure: root_dir/<dataset_name>/co/*.parquet")
        sys.exit(1)

    print(f"Found {len(datasets)} datasets, scanning...\n")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan_dataset, d): d for d in datasets}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if "error" not in r:
                approx = "~" if not r.get("atoms_exact", True) else ""
                print(f"  {r['name']}: {r['n_configs']:,} configs, "
                      f"{approx}{r['n_atoms']:,} atoms, {r['size_mb']} MB")
            else:
                print(f"  {r['name']}: ERROR - {r['error']}")

    results.sort(key=lambda r: r.get("n_configs", 0), reverse=True)

    # Summary
    total_configs = sum(r.get("n_configs", 0) for r in results)
    total_atoms = sum(r.get("n_atoms", 0) for r in results)
    total_size = sum(r.get("size_mb", 0) for r in results)
    n_ok = sum(1 for r in results if "error" not in r)

    print(f"\n{'='*70}")
    print(f"TOTAL: {n_ok} datasets, {total_configs:,} configs, "
          f"~{total_atoms:,} atoms, {total_size:,.0f} MB")
    print(f"{'='*70}\n")

    # Table output
    try:
        from tabulate import tabulate
        rows = []
        for r in results:
            if "error" in r:
                rows.append([r["name"], "ERROR", "", "", "", r["error"]])
            else:
                approx = "~" if not r.get("atoms_exact", True) else ""
                rows.append([
                    r["name"],
                    f"{r['n_configs']:,}",
                    f"{approx}{r['n_atoms']:,}",
                    r["avg_atoms_per_config"],
                    f"{r['size_mb']} MB",
                    r["properties"],
                ])
        print(tabulate(
            rows,
            headers=["Dataset", "Configs", "Atoms", "Avg atoms", "Size", "Properties"],
            tablefmt="simple",
        ))
    except ImportError:
        for r in results:
            if "error" not in r:
                approx = "~" if not r.get("atoms_exact", True) else ""
                print(f"{r['name']:40s} {r['n_configs']:>12,} configs  "
                      f"{approx}{r['n_atoms']:>14,} atoms  {r['size_mb']:>8} MB  "
                      f"{r['properties']}")

    # CSV export
    if args.csv:
        import csv
        fields = ["name", "n_configs", "n_atoms", "atoms_exact",
                  "avg_atoms_per_config", "n_elements_sample", "properties",
                  "n_files", "size_mb", "error"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nExported to {args.csv}")


if __name__ == "__main__":
    main()
