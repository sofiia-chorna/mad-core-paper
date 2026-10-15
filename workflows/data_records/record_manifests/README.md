# MAD-COSMO

MAD-COSMO is the collection of around 1.7 billion structures from publicly available
atomistic datasets that MAD-CORE was selected from. Every structure is assigned to its
nearest MAD-CORE structure, and the record follows the MAD-CORE order. We call the
MAD-COSMO structures assigned to one MAD-CORE structure its cell. Inside a cell
structures come in farthest point sampling order. The record contains geometries,
model-predicted energies and forces, features, ids, metadata and the filter diagnostics.
Rejected structures and duplicates are also kept and flagged.

This record for now includes the first three intervals, cells `[0, 2^20)`. The remaining
cell intervals follow in later uploads.

## Licence

The record is published under CC-BY-4.0. The structures come from source datasets, each
with its own licence and citation, listed per `dataset_id` in `dataset_metadata.tsv`.
Please cite the source datasets. See `LICENSES.md` for detailed information on used
licences.

## Files

Files are organised by intervals of MAD-CORE rows, like the MAD-CORE record: `0-2pow18`,
`2pow18-2pow19`, and so on up to `2pow23-2pow24`. Large intervals will be probably split
into numbered parts, e.g. `madcosmo_cells_2pow20-2pow21_part0001-of-0002_data.h5`.
Files:

* `madcosmo_cells_X-Y_data.h5` - structures, predictions, ids, metadata, cell
  assignments and diagnostics, HDF5
* `madcosmo_cells_X-Y_features.h5` - features of the same structures, float32 (rows,
  512), with the `mean` vector and `scale` value that define the metric, see Features
  below
* `dataset_metadata.tsv` - source datasets keyed by `dataset_id`: title, `upstream_url`
  where the data lives, `source_publication` the paper that introduced it, licence.
  `rows` is how many structures a dataset contributes to MAD-COSMO. `source_kind` says
  where a dataset was obtained (colabfit,  optimade, nomad record, materials cloud
  archive record, standalone dataset, or mad-next). It also has one row per NOMAD
  upload, see `LICENSES.md`
* `LICENSES.md` - licence info
* `example.py` - an example of how to read the files and turn a structure into xyz

Row `i` of the data file and row `i` of the features file are the same structure.

## Structures

One structure per row. Per-atom arrays are flat and cut with `geometry/atom_offsets`:
the atoms of row `r` are `atom_offsets[r]:atom_offsets[r + 1]`. Everything is
zero-based.

* `geometry/numbers`, `geometry/positions` - per atom
* `geometry/cell`, `geometry/pbc` - per structure
* `predictions/ecumetric_energy` - per structure
* `predictions/ecumetric_nc_forces` - per atom

Energies are in eV, forces in eV/A, coordinates in A. Every energy and force is
predicted by ecumetric used for the selection, the record contains no DFT reference
values.

Coordinates are folded into the cell. Structures that were stored unfolded in their
source are listed in `images/rows`, with `images/atom_offsets`, `images/flags` and
`images/original_cell`. Original coordinates are `positions + flags @ original_cell`.

## MAD-CORE order and cells

Every MAD-COSMO structure is assigned to its nearest MAD-CORE structure. The structures
assigned to MAD-CORE structure number `c` form cell, where `c` is the `fps_order` of
that structure in the MAD-CORE record. The file stores cell 0 first, then cell 1, and so
on, so the cells follow the MAD-CORE order.

* `index/cell_offsets` - where each cell starts. Cell `c` is rows
  `cell_offsets[c]:cell_offsets[c + 1]`
* `index/center_configuration_id` - configuration id of the MAD-CORE structure at the
  centre of each cell (to which this MAD-COSMO structure is assigned)
* `ordering/fps_order_in_cell` - rank of the row inside its cell. Rows that were not
  ranked have -1 and sit at the end of the cell
* `ordering/epsilon_sq` - squared distance to the nearest earlier row of the cell, in
  the scaled feature metric
* `assignments/center_index` - (rows, 25), column `k` is the nearest structure among the
  first 2^k MAD-CORE structures. Column 24 is the cell we run FPS for within-cell ordering
* `assignments/distance` - the corresponding distances in the scaled feature metric

## Filtering

Rejected structures and duplicates are kept in the files. A structure passed the filters
when `diagnostics/filter_code` is 0 and `diagnostics/is_duplicate` is false.

* 0 - passed
* 1 - ZBL-rejected
* 3 - excluded (see paper)

Structures whose features are not finite are not in the record.

## By dataset

Rows are in cell order, not grouped by dataset, but selecting a dataset is quick.
`dictionaries/dataset_id` is the list of dataset names. Instead of repeating the name on
every row, `identity/dataset_id_code` stores one number per structure, the position of
its dataset in that list. So all structures of one dataset carry the same number, and
you find them by looking up the position of the name and then the rows with that number:

```python
names = [name.decode() for name in data["dictionaries/dataset_id"][:]]
code = names.index("ODAC_train")
codes = data["identity/dataset_id_code"][:]
rows = np.where(codes == code)[0]
```

`rows` are the file rows of that dataset, can be used like
`data["predictions/ecumetric_energy"][rows]`.

## Features

`ecumetric_features` in `madcosmo_cells_X-Y_features.h5` is float32 `(rows, 512)`, raw
model output, 256 mean components followed by 256 std components. `scaler/mean` (512,)
and the scalar `scaler/scale` are in the same file. The metric is the Euclidean distance
on `(ecumetric_features - mean) / scale`.

## Metadata

* `identity/configuration_id` - the configuration id of each structure
* `identity/dataset_id_code`, `identity/dataset_group_code` - position of the dataset and
  of the dataset group in the name lists `dictionaries/dataset_id` and
  `dictionaries/dataset_group`. `dataset_id` is the key into `dataset_metadata.tsv`
* `diagnostics/zbl_max_force`, `diagnostics/zbl_energy_per_atom`,
  `diagnostics/zbl_max_atomic_energy` - ZBL diagnostics used for filtering
* `identity/source_reference_bytes`, `identity/source_reference_offsets` - id of the
  structure in its source dataset, as text bytes cut with the offsets like the atoms.
  Empty for the MAD-NEXT structures, which have no source dataset

## Reading

See `example.py` for an example.
