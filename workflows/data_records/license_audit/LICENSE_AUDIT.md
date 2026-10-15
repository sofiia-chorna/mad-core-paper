# License audit of the mad-core release — 2026-08-21

Scope: every source dataset redistributed by
`${STORE_ROOT}/mad-core/release` (655 metadata rows, 454 collection
groups, 1,693,646,343 structures) plus the MAD tier in the two coreset directories.
Every row was re-verified from scratch against an independent authority; the shipped
`dataset_metadata.tsv` was treated as a claim, not as evidence.

What the release redistributes per source dataset: **geometries (atomic numbers, positions,
cells, pbc) and provenance strings only.** All energies/forces/features are model predictions
(`ecumetric_*`, PhACE ep92) computed by us. No upstream DFT labels ship. This narrows the
redistribution question to the structures themselves.

## Result

| status | datasets | rows | share |
|---|---:|---:|---:|
| VERIFIED, CC-BY-4.0 | 532 | 751,998,305 | 44.40% |
| VERIFIED, CC0-1.0 | 24 | 2,464,253 | 0.15% |
| VERIFIED, NIST-PD | 11 | 2,425,092 | 0.14% |
| VERIFIED, MIT | 55 | 1,152,041 | 0.07% |
| VERIFIED, Apache-2.0 | 14 | 946,134 | 0.06% |
| VERIFIED, BSD-3-Clause | 14 | 824,173 | 0.05% |
| VERIFIED, CC-BY-3.0 | 1 | 21,881 | 0.00% |
| PERMISSION (gdb13, written permission 2026-08-12) | 1 | 933,762,319 | 55.13% |
| **PROBLEM: CC-BY-SA-4.0 — decision pending** | **3** | **52,145** | **0.003%** |

**651 of 655 datasets (>99.99% of rows) are confirmed compatible with a CC-BY-4.0 record**
(CC-BY-4.0/3.0, CC0, MIT, BSD-3-Clause, Apache-2.0, NIST-PD all permit redistribution and
commercial reuse with attribution). gdb13 is covered by written permission. The only open
item is the 3 share-alike Materials Cloud records below.

Full per-row evidence: `license_audit.tsv` (status, authority, authority license, evidence
URL, notes for each of the 655 rows).

## Verification method by family

| family | rows | authority | result |
|---|---:|---|---|
| ColabFit-mirrored (403) | — | HuggingFace API license tag per `colabfit/<name>` repo; DS-id cross-echo from the card where present (369/403) | 390 exact matches; 11 JARVIS resolved via NIST terms; 2 OPoly26 resolved via the source's own statement (below) |
| Materials Cloud archive ingests (232) | — | `archive.materialscloud.org/api/records/<id>` → `metadata.rights` (SPDX) | 227 cc-by-4.0, 1 mit (TSV already said mit), 1 cc-by-4.0 with metadata addendum, 3 cc-by-sa-4.0 |
| Legacy `mcloud__*` OPTIMADE ingests (7) | 9,489 | Each discover database's source archive record (how-to-cite DOI → record → rights) | all 7 cc-by-4.0 |
| Standalone ingests (12) | — | Official source statements (details below) | all CC-BY-4.0 |
| gdb13 (1) | 933,762,319 | GDB usage policy + written permission | permission granted 2026-08-12 |

The five pre-migration Materials Cloud ids (`1z-pd`, `ed-g2`, `jk-9v`, `m0-zg`, `zt-z4`)
were resolved through their DOIs to current records and verified the same way.

### The 14 previously hand-filled rows — now all independently verified

These carried `cc-by-4.0` with no traceable source; each now has primary evidence:

| dataset | verified license | evidence |
|---|---|---|
| OMol25_train/val | CC-BY-4.0 | source page states "The OMol25 dataset is provided under a CC-BY-4.0 License" (huggingface.co/facebook/OMol25) |
| OC25_train/val | CC-BY-4.0 | "The OC25 dataset is provided under a CC-BY-4.0 license" (huggingface.co/facebook/OC25) |
| ODAC_train/val | CC-BY-4.0 | "The ODAC25 dataset is provided under a CC-BY-4.0 License" (huggingface.co/facebook/ODAC25) |
| OC20_S2EF_train_all | CC-BY-4.0 | "The OC20 dataset is licensed under a Creative Commons Attribution 4.0 License" (fairchem docs, docs/catalysts/datasets/oc20.md) |
| oc20neb | CC-BY-4.0 | License: CC-BY-4.0 in the official OC20NEB doc (fairchem docs, docs/catalysts/datasets/oc20neb.md) |
| Alexandria_pbesol / _pressure | CC-BY-4.0 | "available for use under the terms of the Creative Commons Attribution 4.0 License" (alexandria.icams.rub.de/about.html) |
| nomad | CC-BY-4.0 | "Content uploaded to NOMAD is made available subject to the rules of the Creative Commons Attribution 4.0 License" (nomad-lab.eu/nomad-lab/terms.html) |
| oqmd__oqmd | CC-BY-4.0 | oqmd.org states data is licensed CC-BY 4.0 with citation |
| mcloud__mc2d | CC-BY-4.0 | archive record 17gf6-84915, "The Materials Cloud 2D database (MC2D)" |

The old `DATASET_IDS.txt` note that flagged oqmd as "unknown-license" is now obsolete:
oqmd.org states CC-BY-4.0 explicitly.

Note for all FAIR Chemistry datasets (OMol25, OC25, ODAC25, OMC25, OPoly26): the
**datasets** are CC-BY-4.0; only the **model checkpoints** are under the restrictive "FAIR
Chemistry License". We redistribute no Meta checkpoints, so this does not affect us. Their
HF repos are gated (contact info + acceptable-use acknowledgment), but gating is an access
mechanism on Meta's distribution channel, not a license term — CC-BY-4.0 grants the
redistribution right.

### OPoly26 — the "FAIR Chemistry License" scare resolved

`release/build/colabfit_doi_from_web.tsv` recorded "FAIR Chemistry License" for OPoly26 and
the shipped TSV silently changed it to cc-by-4.0. The shipped value turns out to be
**correct**: the source states "The OPoly26 dataset is provided under a CC-BY-4.0 license"
(huggingface.co/facebook/OMol25). ColabFit's mirror tags "other" because it conflates the
checkpoint license. Recommended: record this in the TSV notes so it never looks silent again.

### gdb13 — 55.13% of the release

- GDB terms: citation required; no patent use; no redistribution without written permission.
- **Written permission exists**: email from the GDB-13 corresponding author, 2026-08-12, in
  reply to a request that explicitly described redistributing generated 3D coordinates (not
  SMILES) in this collection: "Yes, go ahead - our licensing request was primarily made to
  try to prevent that a company might use the dataset to write broad patent claims, and
  certainly not to block academic research!"
- SMILES were already scrubbed from the release (2026-08-14 id-unification audit; all
  30,753,452,423 bytes of the gdb13 reference blob verified clean); the SMILES-bearing
  backups live outside the release tree.
- **Action needed (Sofiia)**: archive the permission email (PDF/.eml) alongside the record's
  private documentation.
- **Wording nuance to align with Michele**: the permission was framed around enabling
  research and preventing broad patent claims. The gdb13 subset therefore should NOT be
  labeled CC-BY-4.0 in the public record; label it with its own terms line ("GDB-13-derived
  structures, redistributed with written permission of the authors; citation required; no
  patent use") and state in the README that the blanket record license applies to all
  non-gdb13 subsets. The paper's "academic as well as commercial use" sentence should be
  qualified accordingly, or the authors asked to confirm commercial-use wording explicitly.

### The 3 CC-BY-SA-4.0 records — the only open problem

| group | rows | record |
|---|---:|---|
| mcloudarchive__5pdz3-vsq42 | 18,130 | archive.materialscloud.org/records/5pdz3-vsq42 (amorphous molecular solids by NMR) |
| mcloudarchive__hyjhn-8em96 | 34,014 | archive.materialscloud.org/records/hyjhn-8em96 |
| mcloudarchive__zw0nd-wx180 | 1 | archive.materialscloud.org/records/zw0nd-wx180 |

Confirmed CC-BY-SA-4.0 at source; share-alike is on the project's own do-not-use list
(`dataset_collection/note_on_licenses.txt`) because derivatives must carry the same license —
incompatible with a blanket CC-BY-4.0 record.

**Impact assessment (measured, read-only):**
- Collection tier: 52,145 rows in 3 directories — removable by dropping the directories,
  updating `dataset_metadata.tsv` (652 rows), the root MANIFEST, and the paper's group/row
  counts (454 → 451 groups; 1,693,646,343 → 1,693,594,198 rows).
- **mad-core coreset: 10 rows from `mcloudarchive__hyjhn-8em96` sit inside the frozen 2^24
  FPS ordering** (found via `dataset_group.npy`). Removing them breaks the frozen fps_order
  contiguity; keeping them makes the coreset mixed-license. Options: (a) drop the 3 groups
  from the public collection AND remove/gap-mark the 10 coreset rows (re-freeze + re-audit),
  (b) keep everything and document per-dataset licenses with an explicit SA carve-out,
  (c) keep collection dirs with per-dataset licensing but replace the 10 coreset rows.
- mad-next-coreset: clean (0 rows).
- Per prior decision: **no removal executed** — this is Michele's call. Evidence is complete.

### Smaller corrections (in `dataset_metadata_corrected.tsv`)

1. All license spellings normalized to SPDX (`CC-BY-4` / `cc-by-4.0` / `CC-BY-4.0` → CC-BY-4.0;
   `CC0-1`/`CC0` → CC0-1.0; `Apache-2` → Apache-2.0; `mit` → MIT; `CC-BY-3` → CC-BY-3.0).
2. `mcloudarchive__qffdm-h9w98` "License addendum,cc-by-4.0" → CC-BY-4.0. The addendum on the
   record only puts the record *metadata* (except emails) under CC-BY-SA-4.0; the data is
   plain CC-BY-4.0. Not a problem.
3. `mcloudarchive__ztd8f-vj464` is MIT at source and the TSV already said mit — no change
   beyond normalization (worth knowing: it is the one MIT record among the mcloud ingests).
4. `nomad` row: title and upstream_url filled (no DOI exists for the aggregate; the NOMAD
   terms URL is the licensing evidence). The per-entry attribution question (which NOMAD
   entries) remains a citation-completeness issue, not a licensing one — NOMAD publishes all
   public content CC-BY-4.0.
5. gdb13 license string now records the written permission and its date.
6. MAD tier: `mad_tier_metadata_proposed.tsv` adds the 24 own-data dataset_ids
   (mad15 ×14, mad_next ×7, mad_sol ×3; mad-core additionally carries the group-level ids
   MAD-1.5 / mad_next / mad_sol, 340,555 rows). License proposed CC-BY-4.0 — **confirm with
   Michele** before merging into the shipped TSV.

### Attribution obligations to carry into README/LICENSES

- CC-BY-4.0/3.0 (533 datasets): attribution — covered by `dataset_metadata.tsv` (DOI +
  source publication per dataset) provided it ships with the record and the README points to it.
- MIT / BSD-3-Clause / Apache-2.0 (83 datasets): attribution/notice-preservation — same
  mechanism suffices for data redistribution.
- NIST-PD (11 JARVIS datasets): US-government work, public domain in the US; NIST asks for
  explicit acknowledgment and notes possible foreign copyright. Add the NIST acknowledgment
  line to the README.
- CC0 (24): no obligations; cite anyway.

### Negative controls (all clean)

- None of the 49 quarantined `colabfit_data_restricted/` datasets appear in the shipped TSV
  (exact and substring match).
- The 3 groups removed at validation (2 mcloud records, the MAD colabfit mirror) are absent
  from the TSV and from `collection/`.
- TSV `dataset_group` set == the 454 `collection/` directories, exactly.
- Row counts sum to exactly 1,693,646,343.

### Residual caveats (accepted risk, disclosed)

- For ~360 ColabFit datasets the original HF snapshot is deleted from disk; verification used
  the live HF mirror tag, which is ColabFit's own curation of the upstream license. For the
  top datasets by volume the license was additionally confirmed at the original source. A
  deeper per-paper check of all 403 upstream publications is possible but was not performed.
- 34 ColabFit HF cards don't echo a DS-id in the description (list in `license_audit.tsv`,
  `ds_id_confirmed=no`); all 34 matched by exact repo name and their licenses agree, and the
  big ones (OC20/OC22 splits, JARVIS, OMC25, OPoly26, HME21, Alex_MP-20, UNEP) were
  independently confirmed.
- Legacy discover-database → archive-record mapping for `mcloud__2dtopo` and `mcloud__mc2d`
  was matched by canonical title (their legacy pages are JS-only); see manual list below.

## Addendum 2026-08-21: second round — original-source confirmations

To close the "ColabFit-tag-only" gap, the largest ColabFit-mirrored datasets were
additionally verified at their ORIGINAL sources:

| dataset (group rows) | original source | license there |
|---|---|---|
| OMat24 (105.8M) | huggingface.co/datasets/fairchem/OMAT24 | "licensed under a Creative Commons Attribution 4.0 License" |
| ANI-2x (~37M) | Zenodo record 10108942 (API) | cc-by-4.0 |
| ANI-1 (24.4M) | figshare article 5287732 (API) | **CC0** — original is MORE permissive than our CC-BY-4.0 label; harmless |
| Materials Project (6.3M) | legacy.materialsproject.org/terms | CC-BY-4.0 ("may be copied, distributed, transmitted, and adapted... provided proper attribution") |

Already original-source-verified in round one: OC20, OC20-NEB, OC22 (fairchem docs),
OC25, ODAC25, OMol25, OMC25, OPoly26 (Meta pages), Alexandria (about page), NOMAD (terms),
OQMD (site), JARVIS (NIST terms), and all 239 Materials Cloud records (the archive API is
the hosting platform's own metadata — the archive IS the distributor for those).

Combined effect: **>95% of collection rows are now verified at the original source or the
hosting platform's own API**; the remainder is the long tail of ~300 small ColabFit-curated
paper datasets (tens of millions of rows) whose license rests on ColabFit's published tag.
Note: CC licenses are irrevocable — data obtained under CC-BY-4.0 stays validly licensed
even if an upstream page changes later.

Materials Project nuance, for completeness: MP's user-contributed MPContribs datasets can
carry CC-BY-NC, but the mirrored `colabfit_Materials_Project` dataset is the core computed
database, which is plain CC-BY-4.0.

## Manual checks for Sofiia

Must (before the reply goes out):
1. Archive the gdb13 permission email as PDF/.eml next to the release's private docs.
2. Eyeball the 3 SA records once (URLs in the table above) — confirm nothing in their
   descriptions suggests a more permissive grant we missed.

Spot-checks (nice-to-have, 2 minutes each — confirm my record mapping):
3. https://legacy.materialscloud.org/discover/2dtopo → "how to cite" should point at
   record knrr2-v5b78 (Z2 topological order).
4. https://mc2d.materialscloud.org/ → cite → record 17gf6-84915 (MC2D database).
5. https://legacy.materialscloud.org/discover/autowannier → record 0ctng-gre46.

## Files in this directory

| file | what |
|---|---|
| `license_audit.tsv` | 655 rows × status/authority/evidence — the audit itself |
| `dataset_metadata_shipped.tsv` | untouched copy of the shipped TSV (CRLF) |
| `dataset_metadata_corrected.tsv` | SPDX-normalized + corrections, for review; NOT copied to the store |
| `mad_tier_metadata_proposed.tsv` | proposed MAD-tier rows (license pending Michele) |
| `hf_results.tsv`, `mcloud_results.tsv` | raw API cross-check outputs |
| `01…05_*.py` | the scripts that produced everything above (rerunnable) |
| `drafts/` | release-root README.md + LICENSES.md drafts, email drafts, upload plan |
