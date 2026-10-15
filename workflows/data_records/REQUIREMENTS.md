first 

# Release verification requirements

Applied tier by tier. A tier passes only if every requirement below holds; each
check is written so that a failure names the file and the offending rows.

## A. Manifest / file integrity

- A1 every file present on disk is listed in MANIFEST.json, and every file listed
  in MANIFEST.json exists on disk (MANIFEST.json and subdirectories excluded)
- A2 on-disk byte size == MANIFEST `bytes`
- A3 sha256 of the file == MANIFEST `sha256`
- A4 MANIFEST `bytes` == prod(shape) * itemsize(dtype) — the size is derivable
  from the declared shape, so truncation is detectable from the size alone
- A5 for `.npy` files, the header dtype and shape equal the MANIFEST entry
- A6 MANIFEST carries tier, format_version, ns, n_atoms, row_order, built_at

## B. Row and atom counts

- B1 `ns.npy` is a single int64 equal to MANIFEST `ns`
- B2 `na.npy` has ns+1 entries, na[0] == 0, strictly increasing (no empty structure)
- B3 na[-1] == MANIFEST `n_atoms`
- B4 every per-structure file has exactly ns rows, every per-atom file exactly na[-1]

## C. FPS ordering (coreset tiers)

- C1 `fps_order.npy` == arange(ns) (identity by construction)
- C2 `k_set.npy` non-decreasing, and the number of rows with k_set <= k equals
  min(2^k, ns) — the first 2^k rows are exactly the level-k coreset
- C3 `epsilon_sq.npy` non-increasing (the FPS separation can only shrink);
  values >= 0 up to float round-off, first entry may be +inf

## D. Identity

- D1 `configuration_id` non-empty on every row, no leading/trailing whitespace,
  max byte length == the declared S<W></w> width (W is not over-padded)
- D2 configuration_id unique across the tier
- D3 `dataset_group` and `dataset_id` non-empty; dataset_id is consistent with
  dataset_group on every row (group is the prefix of the id, or the documented map)
- D4 the id prefix agrees with the group on every row
- D5 the set of groups/ids matches what FORMAT.md and dataset_metadata.tsv declare

## E. Geometry

- E1 atomic numbers within 1..118
- E2 positions finite (no NaN/inf), and within a sane magnitude
- E3 cells finite
- E4 pbc consistent with the cell: an all-zero cell must have pbc all False, and a
  non-zero cell must have at least one pbc True
- E5 for periodic rows, cell volume != 0 (non-degenerate lattice)

## F. Predictions and features

- F1 `ecumetric_energy` finite
- F2 `ecumetric_nc_forces` finite, shape (na[-1], 3, 1)
- F3 `ecumetric_features` finite, no all-zero rows, shape (ns, 512)
- F4 the ecumetric scaler applies: `mean` has 512 entries, `scale` is a positive
  scalar, and standardised features have a sane spread

## G. Diagnostics

- G1 zbl_max_force / zbl_energy_per_atom / zbl_max_atomic_energy finite
- G2 the tier contains only rows that passed the filter (no zbl-rejected rows)

## H. Voronoi

- H1 `voronoi_a[:, k]` in [0, min(2^k, 16777216)) for every level k
- H2 `voronoi_d` finite and >= 0 up to round-off
- H3 `voronoi_d[:, k]` non-increasing in k (adding centers cannot move the
  nearest one further away)
- H4 `voronoi_a[:, k]` is stable: if the distance did not change from level k to
  k+1, the assigned center index did not change either
- H5 spot-check by recomputation: for a sample of rows, the claimed center is the
  true argmin over the first 2^k coreset224 rows in the scaled ecumetric metric

## I. extxyz mirror

- I1 the extxyz MANIFEST lists every file present, chunk row ranges tile [0, ns)
  exactly with no gap or overlap
- I2 sha256 and byte size of every chunk match its MANIFEST entry
- I3 the frame count of each chunk equals its declared row range
- I4 frame i carries the same natoms as na[i+1]-na[i]
- I5 frame i carries the same configuration_id / dataset_group / dataset_id as the
  binary row i (same order, no reshuffle)
- I6 the info/array keys present are exactly those declared in the extxyz MANIFEST
- I7 energies, positions, forces, cell and pbc in the extxyz agree with the binary
  files within float32 print precision
- I8 the extxyz reflects the CURRENT binary columns — in particular any dataset_id
  rename applied to the binary must also be applied here

## J. Cross-tier and documentation

- J1 FORMAT.md describes every file that exists in the tier, and every file
  FORMAT.md promises exists
- J2 the counts quoted in FORMAT.md (ns per tier, group counts, sums) match the
  data
- J3 top-level files FORMAT.md promises (ecumetric_scaler.npz,
  dataset_metadata.tsv) are present in the release root
- J4 claims FORMAT.md makes about the tier relative to other tiers are true
  (e.g. "mad_coreset rows are not coreset224 centers")
- J5 every dataset_id present in the tier appears in dataset_metadata.tsv
