## geometry

| file    | shape       | dtype   | content                                                                                 |
| ------- | ----------- | ------- | --------------------------------------------------------------------------------------- |
| ns.npy  | (1,)        | int64   | number of structures in this directory                                                  |
| na.npy  | (ns+1,)     | int64   | cumulative atom counts; atoms of structure i live at [na[i], na[i+1]) in per-atom files |
| a.bin   | (na[-1],)   | int32   | atomic numbers, all structures concatenated                                             |
| x.bin   | (na[-1], 3) | float32 | positions, Å                                                                           |
| c.bin   | (ns, 3, 3)  | float32 | cell matrices, Å (zeros where non-periodic)                                            |
| pbc.npy | (ns, 3)     | bool    | periodic boundary flags (our extension; needed for 2D)                                  |

### predictions and features

| file                    | shape          | dtype   | content                                       |
| ----------------------- | -------------- | ------- | --------------------------------------------- |
| ecumetric_energy.bin    | (ns, 1)        | float32 | predicted energy                              |
| ecumetric_nc_forces.bin | (na[-1], 3, 1) | float32 | predicted non-conservative forces             |
| ecumetric_features.bin  | (ns, 512)      | float32 | 512-d structure feature (256 mean ⊕ 256 std) |

### identity and diagnostics

| file                         | shape      | dtype   | content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------- | ---------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| configuration_id.npy         | (ns,)      | S\<W\>  | content-addressed structure id, fixed width W bytes, zero-padded. W is per directory — the max id byte length within that directory, recorded in the MANIFEST and in the .npy header itself (numpy compares mixed-width S arrays transparently)                                                                                                                                                                                                                                                                |
| dataset_group.npy            | (ns,)      | S\<Wg\> | dataset group = folder name in the store tree; coreset tiers only (in collection directories the group IS the directory name, so no file)                                                                                                                                                                                                                                                                                                                                                                       |
| dataset_id.npy               | (ns,)      | S\<Wd\> | source dataset, as a readable name — this is what identifies the split a structure came from (e.g.`colabfit_xxMD-DFT_train`, `colabfit_OMat24_train_rattled_300_subsampled`, `Alexandria_pbesol`, `ODAC_train`). ColabFit-sourced datasets carry the `colabfit_` prefix — the prefix is the provenance marker; other ingests keep their own name. The original ColabFit DS-ids are preserved in `dataset_metadata.tsv` (column `colabfit_ds_id`), so the mapping back to ColabFit is never lost |
| zbl_max_force.npy            | (ns,)      | float32 | ZBL diagnostic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| zbl_energy_per_atom.npy      | (ns,)      | float32 | ZBL diagnostic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| zbl_max_atomic_energy.npy    | (ns,)      | float32 | ZBL diagnostic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| source_reference_offsets.npy | (ns + 1,)  | int64   | byte span of row i's reference:`[off[i], off[i+1])`                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| source_reference_bytes.bin   | (off[-1],) | uint8   | where row i came from in its ORIGINAL source — see below. collection and mad-core only                                                                                                                                                                                                                                                                                                                                                                                                                         |

### split (mad-next-coreset only)

| file      | shape | dtype | content                                                                                 |
| --------- | ----- | ----- | --------------------------------------------------------------------------------------- |
| split.npy | (ns,) | S5    | `train` / `val` / `test` where an official upstream split exists, empty otherwise |

Coreset tiers only (mad-next-coreset, mad-core)

| file                   | shape | dtype   | content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------- | ----- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| fps_order.npy          | (ns,) | int64   | explicit pick index (== the row index; redundant by construction, kept so any row subset retains its provenance)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| epsilon_sq.npy         | (ns,) | float64 | mad-next-coreset: squared separation distance at pick time. mad-core: squared distance from row i to its nearest earlier row (minimum over j < i) in the scaled ecumetric metric, recomputed in float64 from the released features, +inf at row 0. The smallest epsilon_sq among the first n rows is the squared separation of the first n picks. Inside the accrete-anchor block (rows < 856,012) the sequence is not monotone, each accretion level opens with anchor points admitted farthest-first, then plain FPS. Not an interatomic distance. The previous mad-core column (per-level proxy, float32 FPS values beyond row 856,012) is kept in mad-core/_backup_epsilon_sq_2026-09-28 |
| levels.json            |       |         | mad-core only. Per-level table for the sizes 2^1..2^24 and the 856,012-row anchor: size, anchor members, exact separation, coverage over the 184,180,658-row accrete pool (run-time float32 value from the construction log, its reconstruction from the published margin, the exact lower bound with its witness row, exact upper bound pending), status of the separation-coverage check. The condition sep(S_k) >= cov_X(S_k) refers to the accrete pool X, not to the full collection |
| k_set.npy              | (ns,) | int32   | level at which the pick entered (NOT derivable from the row index for the accrete-anchor block)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| voronoi_population.npy | (ns,) | int64   | mad-core only — number of released structures in this pick's level-24 Voronoi cell. There is no such file in mad-next-coreset because population is defined per center and mad-next-coreset is not the set of centers: 340,555 of its 856,012 rows are also mad-core rows (dataset_group`mad-next` there, verified id-for-id), the other 515,457 are not centers at all and would have no value. The population of a MAD row that IS a center is read from mad-core at its mad-core row index. Sums to 1,694,502,355 = every release row: collection 1,693,646,343 + mad-next-coreset 856,012. MAD is counted once, through the mad-next-coreset tier |

### Row-level cross-tier pointers (mad-core only)

| file                         | shape  | dtype  | content                                                                                            |
| ---------------------------- | ------ | ------ | -------------------------------------------------------------------------------------------------- |
| release_directory.npy        | (ns,)  | int16  | which release folder holds this row's full record, as an index into`release_directory_names.npy` |
| release_row_in_directory.npy | (ns,)  | int32  | which row inside that folder                                                                       |
| release_directory_names.npy  | (455,) | S\<W\> | the folder names in index order (`collection/3BPA`, …, `mad-next-coreset`)                    |

### (collection/<dataset_group>/)

| file             | shape | dtype | content                                                                                                     |
| ---------------- | ----- | ----- | ----------------------------------------------------------------------------------------------------------- |
| filter_code.npy  | (ns,) | int8  | 0 ok, 1 zbl_rejected, 2 nan_features, 3 nomad_excluded (mapping repeated in MANIFEST.json)                  |
| is_duplicate.npy | (ns,) | bool  | true on any repeat of a configuration_id already counted once (global, first occurrence in scan order wins) |

### Voronoi assignment (collection groups AND mad-next-coreset)

| file          | shape    | dtype   | content                                                                                     |
| ------------- | -------- | ------- | ------------------------------------------------------------------------------------------- |
| voronoi_a.bin | (ns, 25) | int32   | column k = row index (= fps_order) of the nearest mad-core center among the first 2^k picks |
| voronoi_d.bin | (ns, 25) | float32 | column k = distance to that center in the scaled ecumetric metric                           |


### Within-cell FPS orderings (mad-core/cell_fps/)

Every release row was assigned to its nearest mad-core center at level 24
(16,777,216 cells, `voronoi_a[:, 24]`). `mad-core/cell_fps/` holds, for each cell,
its members ordered by farthest-point sampling *within* the cell, seeded at the
cell's own center. Taking the first m of every cell yields a balanced, maximally
diverse refinement of the collection below the last coreset level. The full
uncapped ordering is shipped; users choose where to stop (`epsilon_sq` says how
far each pick was from everything chosen before it).

Rows are **cell-major**: all of cell 0, then all of cell 1, … Within a cell rows
are ordered by `fps_order_in_cell`, best-spread first, unordered members last.
`n = 1,694,502,355` — every release row (454 collection folders plus
mad-next-coreset), so `diff(center_offsets)` equals `voronoi_population.npy` cell
for cell. Of those, 1,641,975,594 entered the FPS pool (release `filter_code == 0`
and not `is_duplicate`, plus the mad-next-coreset tier); the rest are present and
counted but carry no ordering of their own.

| file                         | shape          | dtype   | content                                                                                                                                                                                                                              |
| ---------------------------- | -------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| center_offsets.npy           | (2^24 + 1,)    | int64   | cell c owns rows`[center_offsets[c], center_offsets[c+1])` in every per-row file                                                                                                                                                   |
| center_configuration_id.npy  | (2^24,)        | S\<W\>  | the id of each cell's center; equals`mad-core/configuration_id.npy` row-for-row                                                                                                                                                    |
| fps_order_in_cell.npy        | (n,)           | int32   | this member's position in its own cell's FPS ordering: 0 is the cell's center, 1 the member farthest from it, and so on.`-1` means no ordering (below)                                                                             |
| epsilon_sq.npy               | (n,)           | float32 | squared separation distance at pick time, in the scaled ecumetric metric, measured WITHIN the cell. Non-increasing along a cell up to isolated ~1e-6 steps at checkpoint-resume boundaries.`NaN` where `fps_order_in_cell == -1` |
| configuration_id_offsets.npy | (n + 1,)       | int64   | byte span of row i's id                                                                                                                                                                                                              |
| configuration_id_bytes.bin   | (offsets[-1],) | uint8   | the configuration_id, concatenated. The durable identity — content-addressed, stays correct no matter how the release is reorganised                                                                                                |
| release_directory.npy        | (n,)           | int16   | which release folder this row lives in, as an index into`release_directory_names.npy`                                                                                                                                              |
| release_row_in_directory.npy | (n,)           | int32   | which row inside that folder                                                                                                                                                                                                         |
| release_directory_names.npy  | (455,)         | S\<W\>  | the folder names in index order (`collection/3BPA`, …, `mad-next-coreset`)                                                                                                                                                      |
| release_manifest_sha256.npy  | (455,)         | S64     | sha256 of each folder's MANIFEST.json when this was built                                                                                                                                                                            |

`fps_order_in_cell` is a property of the STRUCTURE, not of the row: it is assigned
by `configuration_id`. Duplicates repeat their twin's value (a duplicate IS the
same structure, so FPS is not re-run for it — within a cell the value is unique
among distinct ids, not among rows), and their `epsilon_sq` repeats too. `-1`
means the row has no ordering of its own in this cell: 486,653 rows — filtered
rows with no clean twin (473,416), duplicates whose only other copies were also
filtered (12,940), and duplicates whose twin landed in a different cell through
~1e-7 featurisation jitter of the duplicated store rows (297).

The pointer pair (`release_directory`, `release_row_in_directory`) is positional
and therefore version-bound: if a folder is ever added or removed, indices shift.
`release_manifest_sha256` makes that detectable — check it against the release you
are reading; if it disagrees, resolve through `configuration_id` instead. The id
is never wrong.

Retrieving a cell (one seek per file, every column contiguous per cell):

```python
D = "mad-core/cell_fps/"
lo, hi = center_offsets[c], center_offsets[c + 1]
k = min(100_000, hi - lo)

order = np.load(D + "fps_order_in_cell.npy", mmap_mode="r")[lo:lo + k]
eps   = np.load(D + "epsilon_sq.npy",        mmap_mode="r")[lo:lo + k]
folder = np.load(D + "release_directory.npy",        mmap_mode="r")[lo:lo + k]
row    = np.load(D + "release_row_in_directory.npy", mmap_mode="r")[lo:lo + k]
```

Most cells are small: the median level-24 population is 3 and 99% hold fewer than
256 members; the largest cell holds 114,996,707. A size-m refinement of the whole
collection is one contiguous slice per cell:

```python
off = np.load(D + "center_offsets.npy")
take = [(folder[a:min(a + m, b)], row[a:min(a + m, b)])
        for a, b in zip(off[:-1], off[1:])]
```

`dataset_group`, `dataset_id`, geometry, features and diagnostics are not
duplicated here — `release_directory_names[release_directory[i]]` already names
the folder, and everything else is one indexed read into it.

### Per directory: MANIFEST.json

Self-contained integrity document: tier, format version, release version,
row-order definition, ns/atom counts, file list with shape/dtype/byte size/sha256,
build date, filter-code mapping (collection). String widths are not separate
fields — each file's S\<W\> dtype is in its entry (and in the .npy header itself).
No internal paths — build provenance (source paths, git commit, zbl-fill
accounting) is kept repo-side in `release/build/provenance/`. A released directory
is immutable; any fix = full rebuild with a new MANIFEST. `format_version: 2`
marks the four directories that carry image-flag sidecars (below).
`format_version: 3` (mad-core, 2026-09-28) marks the one in-place amendment: epsilon_sq.npy
replaced by the exact float64 column and levels.json added, every other file unchanged,
the previous column and MANIFEST kept in `mad-core/_backup_epsilon_sq_2026-09-28/`.

### extxyz mirror

The coreset tiers also ship an `extxyz/` subdirectory holding the same rows in
plain extended-XYZ, for tools that do not read the binary layout. It carries the
geometry, `ecumetric_energy`, `ecumetric_nc_forces`, the identity columns,
`fps_order` (= the row index, kept explicit so shuffled or subset frames retain
their pick index) and (in mad-next-coreset) `split`; the features, ZBL diagnostics
and Voronoi assignment are binary-only. Its own MANIFEST.json records frame count,
keys and sha256.

- `mad-next-coreset/extxyz/mad-next-coreset.extxyz` — one file, 856,012 frames, 3.0 GB.
  Frame i is row i. `split` is omitted on rows with no official split.
- `mad-core/extxyz/frames_rSTART-END.extxyz` — 256 chunks of 65,536 frames,
  27 GB total, each file named after the inclusive row range it holds. Because rows
  are FPS-ordered and a chunk is 2^16 frames, the first 2^(k-16) files together are
  exactly the level-k coreset — the first file alone is the 2^16 coreset. Chunked
  because a single 27 GB extxyz is impractical to open and cannot be read in
  parallel; the chunks are plain concatenated frames, so `cat` produces a valid
  single file if one is wanted.

### Image flags (folded coordinates)

All released coordinates are folded into the unit cell. About 70k structures
(29 in mad-core, the rest in the nomad and two mcloudarchive collection groups)
came from their source with unwrapped coordinates, some as far as 10^10 Å from the
origin. For those rows the fold is recorded LAMMPS-style: `image_flag_rows.npy`
lists the rows, `image_flags.bin` holds the per-atom integer image flags, and
`image_flag_cells.bin` holds their cells in float64. The original coordinates are
`x[row] + flags @ cell`. All stored predictions and features for these rows were
computed on the original coordinates and are exactly reproducible from them.
Featurizing the folded coordinates directly gives correct but different values.
Background and verification live in `release/build/` (far-from-origin writeup).

## Integrity contract

- every per-structure file is exactly ns × itemsize bytes, every per-atom file
  exactly na[-1] × itemsize — a wrong or truncated file is detectable from size
  alone
- per-directory MANIFEST sha256 covers every file; directories are versioned as a
  unit; the root MANIFEST.json pins every directory manifest and top-level file
- coreset rows are semantically pinned (row i = i-th FPS pick), so alignment is
  independently checkable against the internal store by configuration_id

## How to use it

Plain numpy — no metatensor/metatrain needed:

```python
import numpy, json

d = "mad-core/"
na = numpy.load(d + "na.npy")
a = numpy.memmap(d + "a.bin", dtype=numpy.int32, mode="r")
x = numpy.memmap(d + "x.bin", dtype=numpy.float32, mode="r").reshape(-1, 3)
cid = numpy.load(d + "configuration_id.npy", mmap_mode="r")

i = 5_000_000
atoms_Z = a[na[i]:na[i+1]]
atoms_xyz = x[na[i]:na[i+1]]
structure_id = cid[i].decode()
```

Level-k coreset = the first 2^k rows:

```python
features_k16 = numpy.memmap(d + "ecumetric_features.bin", dtype=numpy.float32,
                            mode="r").reshape(-1, 512)[:2**16]
```

Everything from one source dataset:

```python
group = numpy.load(d + "dataset_group.npy", mmap_mode="r")
rows = numpy.flatnonzero(group[:] == b"Alexandria")
```

ASE Atoms:

```python
from ase import Atoms
c = numpy.memmap(d + "c.bin", dtype=numpy.float32, mode="r").reshape(-1, 3, 3)
pbc = numpy.load(d + "pbc.npy", mmap_mode="r")
atoms = Atoms(numbers=atoms_Z, positions=atoms_xyz, cell=c[i], pbc=pbc[i])
```

Voronoi navigation (Tier 1 → coreset):

```python
va = numpy.memmap("collection/Alexandria/voronoi_a.bin", dtype=numpy.int32,
                  mode="r").reshape(-1, 25)
center_row = va[j, 24]
center_id = numpy.load("mad-core/configuration_id.npy", mmap_mode="r")[center_row].decode()
```

Training in metatrain — the directory IS a MemmapDataset:

```yaml
training_set:
  systems:
    read_from: mad-core/
  targets:
    energy:
      key: ecumetric_energy
      forces:
        key: ecumetric_nc_forces
```

(metatrain reads the files it knows — ns/na/a/x/c and configured keys — and
ignores the rest; the id/dataset/diagnostic files are read with numpy as above.)

## Sizes

| tier             | structures    | total                                                                                                                               |
| ---------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| mad-next-coreset | 856,012       | 5.7 GB (2.7 GB binary + 3.0 GB extxyz mirror)                                                                                       |
| mad-core         | 16,777,216    | ~77 GB binary + extxyz (features 34 GB, extxyz 27 GB), plus ~90 GB cell_fps                                                         |
| collection       | 1,693,646,343 | ~6.0 TB (features 3.5 TB, positions+forces 1.7 TB, voronoi 340 GB, atomic numbers 290 GB, ids and diagnostics ~200 GB; 71.4B atoms) |
