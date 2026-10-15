# MAD-COSMO structures outside the 2^24 coverage radius

Scope: the 0.83 % of MAD-COSMO that the manuscript's 99.17 % coverage statement leaves outside the
level-24 (2^24) MAD-CORE coverage radius. Existing release outputs only. No featurization or
tessellation was rerun. The manuscript is unchanged.

## Conventions

- Radius r = 0.2179152404598418, the exact certified level-24 coverage of the 184,180,658-row
  selection pool. This is the value behind the 99.17 %, not the rounded 0.218.
- Stored Voronoi distances and r are plain distances in the scaled ecumetric metric, not squares.
  A direct float64 recomputation on sampled rows agrees with the stored float32 values to about 5e-5.
- Population: all 1,694,502,355 release rows, flagged rows and duplicates included. The 67,975 rows
  with non-finite distance count as outside, as in the 99.17 %. The scan reproduces the reported
  within count exactly (1,680,369,825).

## 1. How many lie outside

| | rows | % of release |
|---|---|---|
| outside r, total | 14,132,530 | 0.834 |
| with finite distance | 14,064,555 | 0.830 |
| with non-finite distance | 67,975 | 0.004 |

With the float32 construction-log radius (0.2179064) 6,754 more rows lie outside. In the published
record, which drops the 67,975 non-finite rows, coverage is 99.170 %.

## 2. Which datasets

Share is of all 14,132,530 outside rows. The last column is the fraction of that dataset's own rows
that lie outside.

| dataset | outside | share of outside | % of dataset outside |
|---|---|---|---|
| OMat24 | 13,523,917 | 95.69 % | 12.78 % |
| nomad (67,927 of them non-finite) | 311,706 | 2.21 % | 5.76 % |
| Alexandria | 216,541 | 1.53 % | 0.18 % |
| OMol25 | 20,337 | 0.14 % | 0.014 % |
| OC20 S2EF | 19,629 | 0.14 % | 0.015 % |
| ANI-1 | 18,813 | 0.13 % | 0.076 % |
| OC22 S2EF, OC22 IS2RE, OC20 IS2RES | 12,345 | 0.09 % | 0.02 to 0.07 % |
| all other datasets | 9,242 | 0.07 % | |

The top three datasets account for 99.43 % of the outside rows. 78 of the 455 release folders
contribute any. gdb13 contributes 132 rows out of 934 million.

Within OMat24, 20.4 % of the rattled subsets lie outside (23 to 25 % of every rattled 300, 500
and 1000 subset), against 6.0 % of the AIMD subsets (about 12 % at 3000 K, 4 % at 1000 K).

## 3. Distance over radius

Statistics are over the 14,064,555 rows with finite distance.

| | median | p90 | p99 | max |
|---|---|---|---|---|
| all finite outside rows | 1.052 | 1.125 | 2.74 | 55,204 |
| non-finite rows counted as infinitely far | 1.052 | 1.127 | 6.18 | |
| unflagged rows (13,700,804) | 1.050 | 1.119 | 1.21 | 1.84 |
| flagged rows (363,751) | 2.17 | 15.1 | 419 | 55,204 |

| outside rows within | cumulative share |
|---|---|
| 1.01 r | 11.2 % |
| 1.05 r | 48.4 % |
| 1.10 r | 79.8 % |
| 1.25 r | 97.1 % |
| 2 r | 98.6 % |

## 4. Filtering flags and structure size

- 431,726 of the outside rows (3.1 %) carry a filter flag: 140,703 rejected by the ZBL check,
  223,048 excluded NOMAD entries, and all 67,975 non-finite rows.
- The flags account for the whole distant tail. 99.3 % of the rows beyond 1.5 r and 100 % of
  the 201,291 rows beyond 2 r are flagged.
- Flagged rows are rarely covered: 93 % of ZBL-rejected rows and 85 % of excluded NOMAD rows lie
  outside.
- Large structures do not explain the tail. Outside rows are smaller than average: median 14
  atoms against 27 for the whole release, p99 88 against 315. Only 537 outside rows (0.004 %) have
  more than 315 atoms, against 1.0 % of all rows.

## Answer: just beyond, with a separate flagged tail

### Measured

- 97.4 % of the finite outside rows are unflagged, and none of them lies beyond 1.84 r.
- All but one of these rows belong to the ten largest source datasets and lie outside the
  10-million-row per-dataset FPS prefix through which those datasets entered the selection pool.
  No unflagged row from any of the other 444 datasets lies outside.
- Each of these large datasets stays below the bound set by its own FPS radius at 10 million picks,
  1 + r_dataset / r. For OMat24, r_dataset = 0.249, bound 2.14, observed max 1.84. For Alexandria,
  0.064, 1.29 and 1.25. For OMol25, 0.098, 1.45 and 1.22. For gdb13, 0.086, 1.40 and 1.04.
- The one selection-pool row outside r lies 4.7e-7 beyond it (ratio 1.0000022). This is float32
  rounding of the stored distance, not a coverage violation.
- The ten most distant rows (38,000 to 55,000 r) are ZBL-rejected NOMAD structures with 3 to 5
  atoms in cells 0.2 to 0.53 Å wide, with nearest atoms 0.17 to 0.26 Å apart.

### Possible explanations, not tested

- OMat24 was never fully covered. Its per-dataset FPS stopped at the 10-million cap at radius 0.249,
  above the final 0.218. The rattled subsets are the most spread out, which fits their higher
  outside fraction.
- The other large datasets had per-dataset radii below 0.1 at the cap, so only a thin shell of
  their rows can exceed r.
- Flagged rows were excluded from selection by design and are often non-physical geometries.
  Their descriptors can land arbitrarily far away, so their large ratios more likely reflect broken
  inputs than gaps in the coverage of physical structures.

## Sources

Scripts are in `paper/claims/`.

- `outside_coverage_scan.py`, a single pass over the release with built-in checks against the
  99.17 % run.
- `submit_outside_coverage_scan.sh`, the job script. This run was made on a login node with
  12 workers in about 13 min because the cluster queue was full.
- `outside_coverage_report.py`, which builds the tables.

Outputs are in `paper/claims/out/`.

- `outside_coverage.json` holds every number quoted here.
- `outside_coverage_scan.json` holds counts per folder and per dataset.
- `outside_coverage_rows.npz` holds one entry per outside row (370 MB, not for git).
