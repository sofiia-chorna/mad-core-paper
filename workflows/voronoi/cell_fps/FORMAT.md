# Binary format of the within-cell FPS orderings

Destination: `aa002/mad-core/voronoi/cell_fps/`. One column = one flat array file,
aligned by row, readable with plain numpy — the same conventions as
`release/FORMAT.md`.

Every structure of the collection was assigned to its nearest coreset224 centre at
level 24 (16,777,216 cells). This directory holds, for each cell, its members
ordered by farthest-point sampling *within* the cell, seeded at the cell's own
centre. Taking the first m of every cell yields a balanced, maximally diverse
refinement of the collection below the last coreset level.

## A cell IS a coreset point

Cell c is the structure at row c of `release/coreset224/` — the c-th FPS pick of the
production coreset. A release row belongs to cell c iff `voronoi_a[row, 24] == c`.

## Rows

Rows are **cell-major**: all of cell 0, then all of cell 1, … Within a cell they are
ordered by `fps_order_in_cell`, best-spread first, with the unordered members last.

`n = 1,694,502,355` — **every release row**: the 454 collection folders plus
`mad_coreset`. Nothing is filtered out, so a cell's row count here is its full
population and `diff(center_offsets)` equals
`release/coreset224/voronoi_population.npy` cell for cell.

Of those, 1,641,975,594 actually entered the FPS pool (release `filter_code == 0`
and not `is_duplicate`, plus the mad_coreset tier). The rest are present and counted
but carry no ordering of their own — see `fps_order_in_cell` below.

## Files

### The cell index

| file | shape | dtype | content |
|---|---|---|---|
| center_offsets.npy | (2^24 + 1,) | int64 | cell c owns rows `[center_offsets[c], center_offsets[c+1])` in every per-row file. `[-1] == n` |
| center_configuration_id.npy | (2^24,) | S\<W\> | the configuration_id of each cell's CENTRE, so a cell can be named without opening coreset224. Equals `release/coreset224/configuration_id.npy` |

### Per-row: the ordering

| file | shape | dtype | content |
|---|---|---|---|
| fps_order_in_cell.npy | (n,) | int32 | this member's position in its own cell's FPS ordering: `0` is the cell's centre, `1` the member farthest from it, and so on. `-1` means no ordering (below). Named `_in_cell` because the release already uses `fps_order` for position in the CORESET ordering — a different quantity |
| epsilon_sq.npy | (n,) | float32 | squared separation distance at pick time, in the scaled ecumetric metric. Same quantity as release `epsilon_sq.npy` but measured WITHIN the cell. Non-increasing along a cell (see METHODS.md on resume boundaries). `NaN` where `fps_order_in_cell == -1` |

### Why fps_order_in_cell can be -1, and why duplicates repeat

`fps_order_in_cell` is a property of the STRUCTURE, not of the row: it is assigned by
`configuration_id`, which is content-addressed, so every row carrying an id that was
ordered gets that id's position.

- **duplicates repeat their twin's value** — a duplicate *is* the same structure, so
  FPS is not re-run for it. Within a cell the value is therefore unique among
  distinct ids, not among rows.
- **`-1` means this row has no ordering of its own in this cell.** The test is on
  the id, not the row: a filtered row whose structure also exists as a clean row in
  the same cell carries that structure's order, while a structure whose every copy
  was filtered gets `-1` on all of them.

Measured breakdown of the 486,653 rows carrying `-1`:

| | rows |
|---|---|
| `filter_code != 0` and not a duplicate | 473,416 |
| duplicates whose only other copies were also filtered | 12,940 |
| duplicates whose twin landed in a DIFFERENT cell | 297 |

The last group exists because duplicated store rows carry ~1e-7 of featurisation
jitter, which is occasionally enough to change which centre is nearest. Their twin's
`fps_order_in_cell` is a position within a different cell and would be meaningless
here, so `-1` is the correct value rather than a missed match.

### Per-row: which structure this is

Two independent ways of saying it, with different guarantees. Use the id when you
need certainty, the pointer when you need speed.

| file | shape | dtype | content |
|---|---|---|---|
| configuration_id_offsets.npy | (n + 1,) | int64 | byte span of row i's id |
| configuration_id_bytes.bin | (offsets[-1],) | uint8 | the configuration_id, concatenated. **The durable identity**: content-addressed, so it stays correct no matter how the release is reorganised |
| release_directory.npy | (n,) | int16 | which release folder this row lives in, as an index into `release_directory_names.npy` |
| release_row_in_directory.npy | (n,) | int32 | which row inside that folder |
| release_directory_names.npy | (455,) | S\<W\> | the folder names in index order: `collection/3BPA`, …, `collection/gdb13`, …, `mad_coreset` |
| release_manifest_sha256.npy | (455,) | S64 | sha256 of each folder's MANIFEST.json when this was built |

Worked example for one row:

| file | value | meaning |
|---|---|---|
| `release_directory[i]` | `180` | → `collection/gdb13` |
| `release_row_in_directory[i]` | `462894103` | row 462,894,103 of that folder |
| `configuration_id` of row i | `gdb13:CC1N(C=O)C2=NOC(C(N)=N)=C12` | the structure itself |

```python
name  = release_directory_names[release_directory[i]].decode()
local = release_row_in_directory[i]
path  = f"release/{name}/"        # every release array indexes at `local`
```

**The pointer is positional and therefore version-bound.** If a folder is ever added
to or removed from the release, the indices shift and the pointer silently refers to
the wrong structure. `release_manifest_sha256` exists to make that detectable: check
it against the release you are reading, and if it disagrees, ignore the pointer and
resolve through `configuration_id` instead. The id is never wrong.

## Retrieving a cell

Every column is contiguous for a given cell, so one seek per file:

```python
lo, hi = center_offsets[c], center_offsets[c + 1]
k = min(100_000, hi - lo)                       # a cell may hold fewer

order = np.load(D + "fps_order_in_cell.npy", mmap_mode="r")[lo:lo + k]
eps   = np.load(D + "epsilon_sq.npy",        mmap_mode="r")[lo:lo + k]
folder = np.load(D + "release_directory.npy",         mmap_mode="r")[lo:lo + k]
row    = np.load(D + "release_row_in_directory.npy",  mmap_mode="r")[lo:lo + k]

off  = np.load(D + "configuration_id_offsets.npy", mmap_mode="r")[lo:lo + k + 1]
blob = np.memmap(D + "configuration_id_bytes.bin", dtype=np.uint8,
                 mode="r")[off[0]:off[-1]]
start = off - off[0]
ids = [bytes(blob[start[j]:start[j + 1]]).decode() for j in range(k)]
```

`ids[0]` is the cell's own centre (`fps_order_in_cell == 0`), and `eps` says how far
each pick was from everything chosen before it — use it to decide where to stop.

Most cells are small: the median level-24 population is 3 and 99% hold fewer than
256, so a 100k slice only exists in the largest cells (the biggest, cell 16,531,443,
holds 114,996,707).

## A size-m refinement of the whole collection

```python
off = np.load(D + "center_offsets.npy")
folder = np.memmap(D + "release_directory.npy", dtype=np.int16, mode="r")
row    = np.memmap(D + "release_row_in_directory.npy", dtype=np.int32, mode="r")
take = [(folder[a:min(a + m, b)], row[a:min(a + m, b)])
        for a, b in zip(off[:-1], off[1:])]
```

One contiguous slice per cell — no scan, no index build.

## What is NOT stored, and why

`dataset_group` is the folder name, so `release_directory_names[release_directory[i]]`
already gives it. `dataset_id`, geometry, features, energies, zbl diagnostics and
`source_reference` are all one indexed read into that folder. Duplicating them here
would add ~65 GB to restate what a single indexed read already provides.

## Integrity contract

- every per-row file is exactly `n * itemsize` bytes
- `center_offsets` is non-decreasing, `[0] == 0`, `[-1] == n`
- `diff(center_offsets)` equals `release/coreset224/voronoi_population.npy` cell for
  cell, and the per-cell counts tallied independently during the release scan
- within each cell, the `fps_order_in_cell` values of rows carrying distinct
  configuration_ids are exactly `0 .. (ordered members - 1)`, each once
- `epsilon_sq` non-increasing within a cell, except at documented resume boundaries
- the configuration_id of row i equals `configuration_id.npy[release_row_in_directory[i]]`
  of the folder `release_directory[i]` names — a round trip through both identity systems
- SHA256SUMS covers every file; the directory is versioned as a unit

## What this is not

Not part of the public release tier tree: the release has its own binary contract
(`release/FORMAT.md`) with per-directory MANIFEST.json.
