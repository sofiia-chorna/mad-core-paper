# Handover — mechanical follow-ups to the license audit

Self-contained tasks for a follow-up session. Context: the license audit in this directory
is COMPLETE (see LICENSE_AUDIT.md); these are rendering/tooling chores only. Rules: work
only inside `${DEV_ROOT}/phace-atomic/release/build/license_audit/`;
NEVER write anywhere under `${STORE_ROOT}/`; never `git add/commit/push`.
Python: plain `python3` on the login node (stdlib only; note: no `str.removeprefix`, python < 3.9).

## Task 1 — generate drafts/release_ATTRIBUTION_draft.md

Input: `dataset_metadata_corrected.tsv` (tab-separated, CRLF, 655 data rows, columns:
dataset_id, dataset_group, colabfit_ds_id, rows, source_kind, store_subfolder, title, doi,
upstream_url, source_publication, license).

Output: a markdown file listing every source dataset for the public record:
- Sections by license, descending total rows (CC-BY-4.0 first; the gdb13 row is its own
  section titled "GDB terms + written permission").
- Within a section, one bullet per dataset: `title (dataset_id)` — DOI as a link when
  present, else upstream_url; append the source_publication link when present.
- Section headers include dataset count + total rows.
- End with the NIST acknowledgment sentence for the NIST-PD section (copy it from
  drafts/release_LICENSES_draft.md).

Verify: bullet count == 655; row totals match the table in drafts/release_LICENSES_draft.md.

## Task 2 — write 06_lint_metadata.py (upload gate)

A standalone assert script for any future dataset_metadata.tsv candidate (argument: path).
Checks, hard-failing on violation:
1. Exactly 655 data rows (or the updated expected count passed as second argument).
2. `rows` column sums to 1,693,646,343 (ditto).
3. Every license value ∈ {CC-BY-4.0, CC-BY-3.0, CC0-1.0, MIT, BSD-3-Clause, Apache-2.0,
   NIST-PD, CC-BY-SA-4.0, the gdb13 permission string} — no legacy spellings.
4. No dataset_id or dataset_group matches (exact or substring) any directory name in
   `${STORE_ROOT}/dataset_collection/colabfit_data_restricted/` (read-only
   listing is fine), nor the removed groups: mcloudarchive__1j65w-wy733,
   mcloudarchive__88sbv-m7172, Massive_Atomic_Diversity_MAD.
5. dataset_group set equals the directory names under
   `${STORE_ROOT}/mad-core/release/collection/` (read-only).
6. doi or upstream_url non-empty for every row.

Run it on `dataset_metadata_corrected.tsv` and paste the passing output at the bottom of
this file.

## Task 3 (optional, separate thread) — chemiscope production re-bundle

Only if asked. The shared chemiscope bundle is the stale 150-iteration sketch-map fit; the
production rebuild procedure is specified in
`${DEV_ROOT}/phace-atomic/HANDOVER_voronoi_sketchmap.md`
(`make_bundle.py --tag p20d8 --d 8` + chemiscope rebuild). Follow that document; it is
self-contained.

## Done criteria

Tasks 1-2 produce files in this directory + a short note here listing what was created and
the verification outputs. Flag Sofiia to review; she commits.
