# Paper claims

Each number stated in the manuscript gets a script here that reproduces it from the released
files in `${STORE_ROOT}/mad-core/release`. Scripts only read the release.
Results are written to `out/` and committed, so every number in the text has a source.

| claim in the text | where in the text | script | result |
|---|---|---|---|
| X% of the 1,694,502,355 released records lie within the 2^24 coverage radius of the candidate pool | `text/body.tex` (99.17%), `text/body_updated.tex` and `04_conclusions.tex` (99.23%) | `collection_within_coverage.py`, run with `submit_collection_within_coverage.sh` | `out/collection_within_coverage.json` |
| every structure of the 184,180,658-row candidate pool lies within the coverage radius of each level | appendix, conclusions | `phace-atomic/accrete/runs/coreset224/coverage_cert/` (stages A, B and C) | `coverage_cert/out/stageB/coverage_exact.json` |
| MAD-COSMO populations behind the counts, filters, duplicates, the 99.17% coverage, the 117-cell and 115-million-cell statistics and the map weights | `text/body.tex` Atomistic datasets, Voronoi, training sections | `madcosmo_populations.py`, run with `submit_madcosmo_populations.sh` (also reads the coverage certificate and the pre-release population arrays in `phace-atomic/voronoi/out/analysis`) | `out/madcosmo_populations.json`, read in `../MAD_CORE_Overleaf_update/AUDIT_2026-10-09_madcosmo_populations.md` |
| half of the records fall into 117 of the 2^24 cells, the largest cell holds 115 million records around one 26-atom organic molecule from GDB-13 | `text/body.tex` Voronoi tessellation | `largest_voronoi_cell.py`, about 6 s on a login node (`python3 -I largest_voronoi_cell.py`) | `out/largest_voronoi_cell.json` |
| every record with non-finite ecumetric descriptors has a broken source geometry (unwrapped coordinates 10^9 to 10^10 Å from the origin, or two atoms at the same position) | `text/body.tex` Atomistic datasets, filter paragraph | `nonfinite_descriptors.py`, about 4 min on a login node (`python3 -I nonfinite_descriptors.py`) | `out/nonfinite_descriptors.json` |
| the 556 NOMAD dataset citations are exactly the NOMAD DOIs that contain released records, 32 of them are nested in another cited DOI | `text/body.tex` mass cite, `biblio/nomad.bib` | `nomad_citations.py`, queries the NOMAD API once (about 25 min, cached in `out/nomad_doi_entries.json.gz`), reruns offline in about 1 min | `out/nomad_citations.json` |
| every released NOMAD record has its NOMAD entry id and upload id | data-collection appendix, release | `nomad_provenance.py`, queries the NOMAD API once (about 2 h at 4 threads, resumable) | the caches `out/nomad_provenance/entry_metadata_cache.tsv.gz` and `upload_metadata_cache.json.gz`, read by `nomad_attribution.py`, which writes the final tables |
| every NOMAD record carries the authors, references and comment of its entry, uncapped, and all 562 dataset DOIs holding released records are mapped to their citation or to the cited DOI that contains them | data-collection appendix, release README | `nomad_credit.py` (API sweep per upload, resumable, `--batched` for small uploads, cached in `out/nomad_provenance/credit_cache/`), then `nomad_attribution.py` (offline, about 6 min) | `out/nomad_attribution.json`, `out/nomad_provenance/{nomad_records.tsv.gz,nomad_credit_variants.tsv,nomad_uploads.tsv,nomad_dois.tsv}` |
| the 379,008 NOMAD structures of MAD-CORE (491 Materials Project) carry the same credit columns | MAD-CORE record | `nomad_attribution_madcore.py` (offline, about 6 min, reads the tables of `nomad_attribution.py`) | `out/nomad_attribution_madcore.json`, `out/nomad_provenance/madcore/` |
| every DOI, arXiv id and link of the bibliography resolves to the cited work (title, authors, year) | `biblio/*.bib` | `bib_verify.py [files]` (no arguments: colabfit, optimade, extra_datasets; e.g. `bib_verify.py others.bib optimade.bib`) via Crossref, DataCite, arXiv and the ColabFit pages | `out/bib_verify.json`, `out/bib_verify_<files>.json`; flagged names are LaTeX accents, see the run notes |

## nomad_credit.py and nomad_attribution.py

`nomad_credit.py` pages through every entry of each of the 6,225 uploads and keeps, for the released entries, the
authors, references, comment, `external_db`, `external_id` and today's dataset DOIs, with no cap. `--batched` puts
small uploads into shared queries. `nomad_attribution.py` reads only the caches and writes the final tables, which
supersede those of `nomad_provenance.py`. Entries of one upload with different credit metadata get different
credit variants. Two 2015 datasets carry short DOIs (`10/f2696t`, `10/z9c`), mapped to the DOIs they resolve to.
The summary reconciles the records outside cited datasets with the uploads that hold no cited dataset.

## nomad_provenance.py

Gives every row of `collection/nomad` its NOMAD entry id, from the source reference or, for the 1,204 rows
without one, from the comment line of the filtered source file. Asks the NOMAD API for the upload id, upload
name, main author, dataset DOIs and references of every entry in batches, caching each batch so an
interrupted run resumes where it stopped. Writes one line per released record (`nomad_records.tsv.gz`, with
the bib keys of its cited datasets and the entry link) and one line per upload (`nomad_uploads.tsv`).

## nomad_citations.py

Takes the 597 DOIs that the NOMAD download query aggregated (`dataset_collection/download_nomad/doi.txt`) and the
DOIs of `biblio/nomad.bib`, and asks the NOMAD API for every entry carrying each DOI. Released entry ids come
from the source references of `collection/nomad`, and the 1,204 rows whose reference has no entry id are
resolved from the comment lines of their filtered source file. Reports the DOIs with released entries, the
cited DOIs without any, the DOIs whose released entries all lie in another DOI, and a minimal citation set that
keeps the largest DOIs first and covers the same released records.

## nonfinite_descriptors.py

A row has non-finite descriptors when its level-24 distance in `voronoi_d.bin` is not finite. In
`collection/nomad` the script reconstructs the source coordinates of the image-flag rows
(`x + flags @ cell`) and compares them with the non-finite rows. Outside nomad it reads only the
ZBL-rejected rows, since the full scan in `out/madcosmo_populations.json` puts every other
non-finite row there, and reports image-flag membership and the smallest interatomic distance.
Asserts that the per-folder counts add up to the release total of 67,975.

## largest_voronoi_cell.py

Reads `mad-core/voronoi_population.npy` (level 24, asserted to sum to 1,694,502,355) and counts the
largest cells needed to hold half of all records. Identifies the centre of the largest cell
through the `mad-core` pointers (`release_directory`, `release_row_in_directory`) and reads its
geometry, id, dataset and source reference from that folder. Takes the cell's members from the
cell-major `mad-core/cell_fps/` files and counts them by release folder. Also lists the dataset
groups of the centres of the half-holding cells.

## collection_within_coverage.py

Reads the level-24 radius from the coverage certificate (`cov_exact_direct`, release features)
and the run-time float32 radius from `release/mad-core/levels.json`. Counts the released rows
whose distance to their assigned MAD-CORE structure (`voronoi_d.bin`, level 24) is within each
radius, over the 454 collection folders and `mad-next-coreset`. The row total is asserted to be
1,694,502,355.
