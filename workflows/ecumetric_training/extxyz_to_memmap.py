#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import ase.io
import numpy as np


def _fit_energy_baseline(structures, energies, energy_key: str) -> dict:
    """Fit per-element reference energies via least squares.

    Solves  A @ x ≈ e  (unconstrained), where A[i, z] = number of atoms of
    element z in structure i, and e[i] is the total energy of structure i.

    DFT/ML total energies are large negative numbers, so per-element reference
    energies must also be negative — a non-negativity constraint (NNLS) would
    return all-zero coefficients and is therefore not used here.

    :param structures: List of ``ase.Atoms`` objects.
    :param energies: 1-D numpy array of total energies, length ``len(structures)``.
    :param energy_key: Key used for energy (stored in JSON for reference).
    :returns: Dict with fit metadata and per-element reference energies.
    """
    from ase.data import chemical_symbols

    # Collect all element symbols in sorted order
    all_z = sorted({int(z) for s in structures for z in s.numbers})
    z_to_col = {z: j for j, z in enumerate(all_z)}
    n_elem = len(all_z)

    # Build composition matrix  A  (ns × n_elem)
    A = np.zeros((len(structures), n_elem), dtype=np.float64)
    for i, s in enumerate(structures):
        for z in s.numbers:
            A[i, z_to_col[int(z)]] += 1.0

    # Unconstrained least squares: allows negative reference energies
    coeffs, _, _, _ = np.linalg.lstsq(A, energies.astype(np.float64), rcond=None)

    residuals_per_atom = (A @ coeffs - energies) / A.sum(axis=1)
    result = {
        "energy_key": energy_key,
        "method": "lstsq",
        "rmse_per_atom_eV": float(np.sqrt(np.mean(residuals_per_atom ** 2))),
        "mae_per_atom_eV": float(np.mean(np.abs(residuals_per_atom))),
        "reference_energies": {
            chemical_symbols[z]: float(coeffs[j]) for z, j in z_to_col.items()
        },
    }
    return result

def apply_energy_baseline(structures, energies, baseline: dict) -> np.ndarray:
    """Subtract the per-element baseline from each structure energy.

    :param structures: List of ``ase.Atoms`` objects.
    :param energies: 1-D numpy array of total energies.
    :param baseline: Dict as returned by ``fit_energy_baseline`` or loaded from JSON.
    :returns: Array of residual energies (same shape as ``energies``).
    """
    from ase.data import atomic_numbers

    ref = baseline["reference_energies"]  # {symbol: eV}
    residuals = energies.copy().astype(np.float64)
    for i, s in enumerate(structures):
        for sym in s.get_chemical_symbols():
            residuals[i] -= ref.get(sym, 0.0)
    return residuals


def extxyz_to_memmap(
    input_path: str,
    output_dir: str,
    *,
    index: str = ":",
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    magmoms_key: str = "initial_magnetic_moments",
    has_forces: bool = True,
    has_stress: bool = True,
    has_magmoms: bool = True,
    fit_energy_baseline: bool = False,
    apply_baseline_from: str | None = None,
) -> None:
    """Convert an .extxyz file to a directory of MemmapDataset binary files.

    :param input_path: Path to the input .extxyz file.
    :param output_dir: Path to the output directory (created if needed).
    :param index: ASE index string, e.g. ``":"`` for all frames.
    :param energy_key: Key for energy in ``atoms.info`` (also the .bin filename).
    :param forces_key: Key for forces in ``atoms.arrays`` (also the .bin filename).
    :param stress_key: Key for stress in ``atoms.info`` (also the .bin filename).
    :param magmoms_key: Key for magnetic moments in ``atoms.arrays`` (also .bin filename).
    :param has_forces: Whether to write a forces file.
    :param has_stress: Whether to write a stress file.
    :param has_magmoms: Whether to write an initial_magnetic_moments file.
    :param fit_energy_baseline: If ``True``, fit per-element reference energies via NNLS,
        subtract them from the stored energies, and write the fit to
        ``<output_dir>/energy_baseline.json``.
    :param apply_baseline_from: Path to a JSON file produced by a previous run with
        ``fit_energy_baseline=True``.  The stored coefficients are subtracted from
        energies (use this for validation/test splits).
    """
    if fit_energy_baseline and apply_baseline_from is not None:
        raise ValueError(
            "--fit-energy-baseline and --apply-energy-baseline are mutually exclusive."
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path} (index='{index}') ...")
    structures = ase.io.read(input_path, index=index)
    if not isinstance(structures, list):
        structures = [structures]

    ns = len(structures)
    print(f"  {ns} structures found")

    # Cumulative atom counts: na[0]=0, na[i+1]-na[i] = n_atoms in structure i
    atom_counts = [len(s) for s in structures]
    na = np.zeros(ns + 1, dtype=np.int64)
    na[1:] = np.cumsum(atom_counts)
    total_atoms = int(na[-1])

    # -----------------------------------------------------------------------
    # Read all energies up front (needed for baseline fitting)
    # -----------------------------------------------------------------------
    raw_energies = np.zeros(ns, dtype=np.float64)
    for i, atoms in enumerate(structures):
        if energy_key in atoms.info:
            raw_energies[i] = float(atoms.info[energy_key])
        else:
            try:
                raw_energies[i] = float(atoms.get_potential_energy())
            except Exception:
                raise KeyError(
                    f"Structure {i}: energy not found under key '{energy_key}' "
                    f"in atoms.info and no calculator attached."
                )

    # -----------------------------------------------------------------------
    # Energy baseline
    # -----------------------------------------------------------------------
    baseline_info: dict | None = None

    if fit_energy_baseline:
        print("  Fitting per-element energy baseline (NNLS) ...")
        baseline_info = _fit_energy_baseline(
            structures, raw_energies, energy_key
        )
        baseline_path = output / "energy_baseline.json"
        baseline_path.write_text(json.dumps(baseline_info, indent=2))
        ref = baseline_info["reference_energies"]
        rmse = baseline_info["rmse_per_atom_eV"]
        print(f"  Baseline RMSE: {rmse*1000:.3f} meV/atom  → {baseline_path}")
        energies = apply_energy_baseline(structures, raw_energies, baseline_info)
    elif apply_baseline_from is not None:
        print(f"  Applying energy baseline from {apply_baseline_from} ...")
        baseline_info = json.loads(Path(apply_baseline_from).read_text())
        energies = apply_energy_baseline(structures, raw_energies, baseline_info)
    else:
        energies = raw_energies

    np.save(output / "ns.npy", ns)
    np.save(output / "na.npy", na)

    a_mm = np.memmap(output / "a.bin", dtype="int32", mode="w+", shape=(total_atoms,))
    x_mm = np.memmap(output / "x.bin", dtype="float32", mode="w+", shape=(total_atoms, 3))
    c_mm = np.memmap(output / "c.bin", dtype="float32", mode="w+", shape=(ns, 3, 3))

    e_mm = np.memmap(
        output / f"{energy_key}.bin", dtype="float32", mode="w+", shape=(ns, 1)
    )
    f_mm = (
        np.memmap(
            output / f"{forces_key}.bin",
            dtype="float32",
            mode="w+",
            shape=(total_atoms, 3),
        )
        if has_forces
        else None
    )
    s_mm = (
        np.memmap(
            output / f"{stress_key}.bin",
            dtype="float32",
            mode="w+",
            shape=(ns, 3, 3),
        )
        if has_stress
        else None
    )
    m_mm = (
        np.memmap(
            output / f"{magmoms_key}.bin",
            dtype="float32",
            mode="w+",
            shape=(total_atoms, 1),
        )
        if has_magmoms
        else None
    )

    warn_forces = 0
    warn_stress = 0
    warn_magmoms = 0

    for i, atoms in enumerate(structures):
        start, end = int(na[i]), int(na[i + 1])

        # Structural
        a_mm[start:end] = atoms.numbers.astype(np.int32)
        x_mm[start:end] = atoms.get_positions().astype(np.float32)
        c_mm[i] = atoms.get_cell()[:].astype(np.float32)

        # Energy (baseline already subtracted above)
        e_mm[i, 0] = float(energies[i])

        # Forces: raw forces (positive); MemmapDataset negates on read
        if f_mm is not None:
            if forces_key in atoms.arrays:
                f_mm[start:end] = atoms.arrays[forces_key].astype(np.float32)
            elif "forces" in atoms.arrays:
                f_mm[start:end] = atoms.arrays["forces"].astype(np.float32)
            else:
                try:
                    f_mm[start:end] = atoms.get_forces().astype(np.float32)
                except Exception:
                    warn_forces += 1
                    f_mm[start:end] = 0.0

        # Stress: stored as stress tensor (eV/Å³).
        # MemmapDataset reads: strain_gradient = stress_stored * V
        # Convention: stress_stored = -virial / V  so  strain_gradient = -virial
        if s_mm is not None:
            stress_3x3 = _get_stress_3x3(atoms, stress_key)
            if stress_3x3 is not None:
                s_mm[i] = stress_3x3.astype(np.float32)
            else:
                warn_stress += 1
                s_mm[i] = 0.0

        # Initial magnetic moments: per-atom scalar
        if m_mm is not None:
            if magmoms_key in atoms.arrays:
                m_mm[start:end, 0] = atoms.arrays[magmoms_key].astype(np.float32)
            elif "initial_magnetic_moments" in atoms.arrays:
                m_mm[start:end, 0] = atoms.arrays[
                    "initial_magnetic_moments"
                ].astype(np.float32)
            else:
                warn_magmoms += 1
                m_mm[start:end, 0] = 0.0

    a_mm.flush()
    x_mm.flush()
    c_mm.flush()
    e_mm.flush()
    if f_mm is not None:
        f_mm.flush()
    if s_mm is not None:
        s_mm.flush()
    if m_mm is not None:
        m_mm.flush()

    if warn_forces:
        print(f"  WARNING: {warn_forces}/{ns} structures missing forces → stored as 0")
    if warn_stress:
        print(f"  WARNING: {warn_stress}/{ns} structures missing stress → stored as 0")
    if warn_magmoms:
        print(
            f"  WARNING: {warn_magmoms}/{ns} structures missing '{magmoms_key}' "
            f"→ stored as 0"
        )

    files = ["ns.npy", "na.npy", "a.bin", "x.bin", "c.bin", f"{energy_key}.bin"]
    if f_mm is not None:
        files.append(f"{forces_key}.bin")
    if s_mm is not None:
        files.append(f"{stress_key}.bin")
    if m_mm is not None:
        files.append(f"{magmoms_key}.bin")
    if fit_energy_baseline:
        files.append("energy_baseline.json")

    print(f"Done.  Output: {output}")
    print(f"  structures: {ns},  atoms: {total_atoms}")
    print(f"  files: {', '.join(files)}")


def _get_stress_3x3(atoms, stress_key: str):
    """Return a (3, 3) stress tensor (eV/Å³) or None if unavailable.

    Priority:
    1. ``atoms.info[stress_key]`` — 6-element Voigt or (3,3) array
    2. ``atoms.info["virial"]`` — converted via  stress = -virial / volume
    3. Calculator ``results["stress"]`` — Voigt 6-vector
    """
    if stress_key in atoms.info:
        raw = np.asarray(atoms.info[stress_key], dtype=float)
        if raw.shape == (6,):
            return _voigt6_to_3x3(raw)
        if raw.shape == (3, 3):
            return raw
        # flat 9-element row-major
        if raw.size == 9:
            return raw.reshape(3, 3)

    if "virial" in atoms.info:
        virial = np.asarray(atoms.info["virial"], dtype=float)
        if virial.shape == (3, 3):
            vol = atoms.get_volume()
            return -virial / vol
        if virial.shape == (6,):
            return -_voigt6_to_3x3(virial) / atoms.get_volume()

    if atoms.calc is not None:
        try:
            results = atoms.calc.results
            if "stress" in results:
                return _voigt6_to_3x3(np.asarray(results["stress"], dtype=float))
        except Exception:
            pass

    return None


def _voigt6_to_3x3(v):
    """Convert a Voigt 6-vector (xx, yy, zz, yz, xz, xy) to a (3,3) matrix."""
    # ASE Voigt order: xx, yy, zz, yz, xz, xy
    return np.array(
        [
            [v[0], v[5], v[4]],
            [v[5], v[1], v[3]],
            [v[4], v[3], v[2]],
        ],
        dtype=float,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input .extxyz file")
    parser.add_argument("output", help="Output directory for memmap files")
    parser.add_argument(
        "--index",
        default=":",
        help="ASE index string (default: ':' = all frames)",
    )
    parser.add_argument(
        "--energy-key",
        default="energy",
        metavar="KEY",
        help="Key for energy in atoms.info (default: 'energy')",
    )
    parser.add_argument(
        "--forces-key",
        default="forces",
        metavar="KEY",
        help="Key for forces in atoms.arrays (default: 'forces')",
    )
    parser.add_argument(
        "--stress-key",
        default="stress",
        metavar="KEY",
        help="Key for stress tensor in atoms.info (default: 'stress')",
    )
    parser.add_argument(
        "--magmoms-key",
        default="initial_magnetic_moments",
        metavar="KEY",
        help="Key for per-atom magnetic moments in atoms.arrays "
        "(default: 'initial_magnetic_moments')",
    )
    parser.add_argument("--no-forces", action="store_true", help="Skip forces")
    parser.add_argument("--no-stress", action="store_true", help="Skip stress")
    parser.add_argument(
        "--no-magmoms",
        action="store_true",
        help="Skip initial_magnetic_moments",
    )

    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--fit-energy-baseline",
        action="store_true",
        help=(
            "Fit per-element reference energies (NNLS) and subtract them from the "
            "stored energies.  Coefficients are written to "
            "<output>/energy_baseline.json."
        ),
    )
    baseline_group.add_argument(
        "--apply-energy-baseline",
        metavar="JSON",
        default=None,
        help=(
            "Path to energy_baseline.json produced by --fit-energy-baseline on the "
            "training split.  Subtracts the same reference energies from this split."
        ),
    )

    args = parser.parse_args()

    extxyz_to_memmap(
        args.input,
        args.output,
        index=args.index,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        magmoms_key=args.magmoms_key,
        has_forces=not args.no_forces,
        has_stress=not args.no_stress,
        has_magmoms=not args.no_magmoms,
        fit_energy_baseline=args.fit_energy_baseline,
        apply_baseline_from=args.apply_energy_baseline,
    )


if __name__ == "__main__":
    main()

