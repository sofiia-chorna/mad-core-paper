# MAD-CORE matcloud record, extxyz + features, full 2^24

Rebuild of the Materials Cloud record (draft `${MC_DRAFT_MADCORE}`, DOI 10.24435/materialscloud:rs-tt,
unpublished, 50 GB quota) and extension from the first 2^23 rows to the full production coreset.

`EXPORT_VERSION = "xyz-2pow24-v1"`

- build tree `${DEV_ROOT}/phace-atomic/release/build/matcloud/xyz-2pow24-v1`
- built files `${DEV_ROOT}/phace-atomic/matcloud-upload/xyz-2pow24-v1/files`
- source, read only `${STORE_ROOT}/mad-core/release/mad-core`

2^24 = 16,777,216 structures, 216,185,078 atoms. MAD-COSMO, the 1.7 billion structure release they
were selected from, is not published yet.

## What the record is

Two formats, two roles.

```
madcore_X-Y.extxyz.gz    structures, and everything known about each structure
madcore_features_X-Y.h5  the 512-dimensional feature vectors used for the selection, and the scaler
```

plus `image_flags.npz`, `dataset_metadata.csv`, `LICENSES.md`, `README.md`, `example.py`,
`index.json`.

Seven row ranges: `0-2pow18`, `2pow18-2pow19`, `2pow19-2pow20`, `2pow20-2pow21`, `2pow21-2pow22`,
`2pow22-2pow23`, `2pow23-2pow24`. Each file covers the rows its name says. Frame `i` of an interval
and row `i` of the matching features file are the same structure.

Expected size, from measurements on the build that already ran:

| | GB |
| --- | --- |
| extxyz.gz, metadata in the frame headers | ~10.7 |
| features h5, gzip-4 + shuffle | 28.97 |
| everything else | ~0.25 |
| total | **~40** of 50 |

### extxyz frame

Per atom `species`, `pos`, `ecumetric_nc_forces`.

Per frame, as the release wrote them: `configuration_id`, `dataset_id`, `dataset_group`,
`ecumetric_energy`, `fps_order`, `pbc`, `Lattice`.

Per frame, appended by this export: `epsilon_sq`, `k_set`, `voronoi_population`, `zbl_max_force`,
`zbl_energy_per_atom`, `zbl_max_atomic_energy`, `release_row_in_group`, `source_reference`.

### features file

`features` float32 `(rows, 512)`, chunks `(256, 512)`, gzip-4 with the shuffle filter. `mean`
`(512,)` and scalar `scale` in the same file. The metric is the Euclidean distance on
`(features - mean) / scale`.

### image_flags.npz

29 structures ship folded like everything else, but their original coordinates reach 1.1e10 A.
Arrays `rows` int64 (29,), `offsets` int64 (30,), `flags` int64 (469, 3), `cells` float64
(29, 3, 3). Reconstruction is `positions + flags @ cell`. float64 is required: at 1e10 A the
float32 spacing is 1024 A. int64 flags are required: the largest is 2,340,920,911, past int32.

## Settled, with the measurement behind each

1. **extxyz is the only structure format.** Positions round-trip against the release binary to
   5.0e-9 A, measured on 40,000 frames. DFT geometry convergence is ~1e-4 A.
2. **All per-structure metadata goes in the frame headers.** Measured on 131,072 real frames: the
   eight appended keys cost +1.29 GB compressed over the whole record. Every frame is then
   self-contained and the record has no third file kind.
3. **Features stay a separate binary matrix.** 16.7M x 512 float32 is 34.36 GB raw; as text it
   would be unshippable. gzip-4 + shuffle gives 28.97 GB (1.18x; plain deflate gives only 1.08x,
   the shuffle filter is what makes float32 compress).
4. **Containers: `.extxyz.gz`, not `.zip`.** Text compresses 3.49x, and uncompressed text plus
   features is ~63 GB, over quota, so compression is not optional. gzip is one file rather than a
   container holding one file, and ASE reads `.gz` directly, tested. Deterministic with `mtime=0`
   for a fixed zlib version.
5. **`release_directory` dropped.** Record-wide it is 1:1 with `dataset_group`: 297 distinct values
   each way, `"collection/" + dataset_group` for 96.8% of rows, one exception,
   `mad-next-coreset` <-> `mad-next`, 267,720 rows. `release_directory_names.npy` goes with it.
6. **`release_row_in_directory` renamed `release_row_in_group`.** The only stable join to
   MAD-COSMO, since ids were re-minted.
7. **`k_set` kept** although `fps_order` determines it: it is the smallest k with
   `fps_order < 2^k`, that is `max(1, ceil(log2(fps_order + 1)))`, checked against the release
   array on all 2^24 rows, zero mismatches. An earlier version of this plan and of the
   record README said plain `ceil(log2(fps_order + 1))`, which is wrong at row 0 only, where
   the release says 1 because level 1 is the first two structures.
8. **dtype rule: record dtype = release dtype**, unless something requires otherwise. `epsilon_sq`
   stays float64. The record ships no DFT reference labels, every energy and force in it is a model
   prediction, so float32 predictions need no defence.
9. **Feature values ship as they are, for the folded coordinates.** Closed, not revisited.
10. **`ns.npy` is moot.** It existed only in the metatrain memmap layout, which this record no
    longer ships.
11. **No metatrain converter ships.** Dropped on request.
12. **aa002 is read-only input.** The release is validated and frozen and its MANIFEST hashes are
    pinned by the Voronoi and cell_fps artifacts. Everything this build writes goes to scratch.

## Decisions that were reversed, and what that costs

Earlier versions of this plan kept the frame headers exactly as the release wrote them and put the
metadata in a separate table, first parquet, then inside the features h5. That is reversed: all
metadata now goes in the headers. Two consequences worth stating plainly.

- **Byte-identity with the release is gone.** Interval boundaries are multiples of the release's
  65,536-row extxyz chunks, so the previous build concatenated whole chunk files byte for byte and
  validation was one SHA-256 comparison against the release manifest. Rewriting every header means
  every frame is re-emitted. Validation now has to reconstruct the original line by stripping the
  appended keys and hash that instead. Atom lines are still copied verbatim, so coordinates and
  forces are untouched.
- **Metadata queries need a parse.** "Which rows are OMat24", or "the 10k largest
  voronoi_population", means reading the text. The README ships a snippet that converts the frame
  headers to a table once.

## Blocker for publication, not for the build

Ten structures carry `cc-by-sa-4.0` (`mcloudarchive__hyjhn-8em96`) while the record states
CC-BY-4.0. Six are already in the current draft, so this is pre-existing; 2^24 makes it ten. They
cannot simply be removed: `fps_order == row` and the nesting property depend on contiguity, so
removal would mean ten documented holes. `LICENSES.md` names the source and states its licence.
Michele is being asked whether that suffices, or whether the record should also carry
`cc-by-sa-4.0` as a second rights entry. `gdb13` grows from 146 to 1,720 rows and is covered by the
written permission recorded in the licence audit.

## Build state, 2026-09-10 18:50

Built and validated. Seven power-of-two ranges, confirmed: the 2^20 file cap was considered and
declined, a 14.5 GB features file is acceptable.

```
7 x madcore_<range>.extxyz.gz     11.01 GB   structures and all per-structure metadata
7 x madcore_features_<range>.h5   28.97 GB   features and the scaler
    image_flags.npz, dataset_metadata.csv, README.md, LICENSES.md, example.py, index.json
                                  -------
                                  39.96 GB = 37.22 GiB, 20 files
```

Validation passed on all seven intervals with zero failures. It is exact, not sampled: for every
one of the 16,777,216 frames it rebuilds the release chunk, appends the intended keys and requires
the shipped bytes to match; and it compares every feature block against the release binary.

`index.json` records, per file, `sha256` of the distributed file and `content_sha256` of the
uncompressed content, plus a `build` block with zlib, numpy, h5py, platform, gzip level and the
HDF5 filter settings. Recompressing a file that had been gunzipped reproduced its `sha256` exactly,
so the build is byte-reproducible on this machine, and content-reproducible anywhere.

Measured, for the README and for anyone judging the format choice:

- `ase.io.read(path, index=":100")` on a gzipped file: 0.09 s, no unpacking. A plain integer index
  makes ASE read the whole file first, so the docs say to use a slice.
- `gunzip` on the largest file, 6.03 GB to 18.42 GB: 1 min 33 s. The whole set is about 3 minutes.
- uncompressed the record would be about 61 GB, over quota. That is why the text is gzipped.

## Uploaded 2026-09-10 19:28

Draft cleared (22 old files deleted) and the new record uploaded: 20 files,
39,961,977,490 bytes = 37.22 GiB, verified 20/20, remote total matches the local total exactly.
The 14.46 GB features file went through in 407 s with the streaming PUT. `default_preview` is
README.md. Not published.

Remaining: the record description still says "first 2^23" and "numpy arrays"; the aa002 mirror
still holds the old layout; not committed.

## Chemiscope candidates uploaded 2026-09-10 23:20

Four sketch-map files uploaded to the draft for testing in the Materials Cloud viewer, which
runs chemiscope **v0.8.6**, not the 1.0.3 the build writes with. Only one will stay in the
record; `index.json` and the README are deliberately not updated until that choice is made.

```
madcore_0-2pow16.chemiscope.json.gz    16.7 MB     65,536 structures
madcore_0-2pow17.chemiscope.json.gz    33.1 MB    131,072 structures
madcore_0-2pow18.chemiscope.json.gz    66.6 MB    262,144 structures
madcore_0-2pow20.chemiscope.json.gz   286.4 MB  1,048,576 structures
```

Draft now 24 files, 40.36 GB of 50. Every upload verified by md5 and size.

15 properties: `smap 1`..`smap 8`, `k_set`, `fps_order`, `log10 voronoi_population`,
`dataset_group`, `ecumetric_energy per atom`, `configuration_id`, `chemical_formula`.
Colour and symbol are `dataset_group`, size is `log10 voronoi_population`.

Settings are the intersection of what 0.8.6 and 1.0.3 accept: `labelsProperty` dropped, and
`useLOD` emitted only above 2^18, so the small files raise no warning on the old viewer.

Open: whether 0.8.6 accepts a string property for colour. If not, colour goes to `k_set` and
symbol stays on `dataset_group`. Also 21 dataset groups is more than plotly has distinct
markers, so symbols repeat.

Coordinates come from `ecumetric_sketchmap.pt`, verified byte-identical in spec to
`skmatter-run/sketchmap/projector/projector_d8_p20d8_silu.json`. The fitted 2^20 embedding
was purged from scratch; the projector reproduces it at mean R^2 0.9982. The projector is not
in the record, so `smap` is the only thing a user cannot recompute from what we ship.

`default_preview` is `None` as of this upload; it was `README.md` after the 19:28 upload.
Not changed here.

Build cost, after making the exporter read and project once and emit every size from
prefixes: 126 s for all four files, 58 s to validate all four. The first version re-read and
re-projected per size and used gzip 9; that took about 22 minutes.

## Steps

1. `export_xyz.py` — build all seven intervals.
   *verify*: 20 files; `index.json` row ranges cover [0, 16777216) with no gap or overlap; total
   under 50 GB.
2. `validate_xyz.py` — per interval: frame count equals the row range; stripping the appended keys
   reproduces the release chunk bytes exactly, checked by SHA-256 against the release manifest;
   every appended value equals the release array at that row, on a sample; features shape
   `(rows, 512)` float32 and its bytes round-trip to the release block; every file SHA-256 matches
   `index.json`.
   *verify*: zero failures, report written to `validation_report.json`.
3. `example.py` — read frames with ASE, read features, print one structure.
   *verify*: runs against the built files.
4. `refresh_docs.py` — copy `README.md`, `LICENSES.md`, `example.py` into the record, re-pin their
   hashes in `index.json`, write `SHA256SUMS`.
   *verify*: no placeholders left; no markdown tables, Materials Cloud does not render them.
5. **STOP for review.** Nothing uploaded, nothing synced, nothing committed.

## After approval, in order

6. Delete from the draft everything not in the new layout: six `madcore_data_*.zip` and six
   `madcore_features_*.zip`, 24 GB. `upload_draft.py` adds and replaces but never deletes, and old
   plus new would exceed the 50 GB quota, so this precedes the upload. It is one-way on the draft,
   so the local and aa002 copies of the old layout stay until the new upload verifies.
7. Upload, roughly 40 GB, about 20 minutes at the rate measured on the last upload.
8. `sync_record.sh` to `aa002/mad-core/matcloud/record/`. This is `rsync --delete`, so it runs only
   after the upload verifies and only with explicit approval.
9. She commits.

## Worth a second opinion

- Is "extxyz with all metadata in the frame headers, plus one HDF5 feature matrix" the right shape
  for a 16.7 million structure public record, or is a separate table worth being a third file kind?
- Is giving up byte-identity with the source release an acceptable price for self-contained frames?
- `.extxyz.gz` versus `.zip` versus uncompressed, for a long-lived archive. Materials Cloud stores
  plain files, so anything clever has to live inside the file itself.
- HDF5 gzip-4 + shuffle for a 29 GB float32 matrix: better chunking, or a filter that does not add
  a dependency, would be worth knowing about.
- Anything about ten-year readability that this misses.
