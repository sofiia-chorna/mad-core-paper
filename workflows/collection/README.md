How the structures were gathered. download_and_prefilter has the downloads, from ColabFit
(Hugging Face mirror), NOMAD (REST API, cursor pagination, the dataset DOIs with
collect_dois.py), OPTIMADE providers and the Open Catalyst and OMol families, the license
policy (note_on_licenses.txt) and the close contact filter: a structure is rejected when two
atoms are closer than scale times the sum of their covalent radii (pyykko_filter_parquet.py,
run_filter_all.py). The collection keeps what passes at scale 0.35, plus what fails at 0.35
and passes at 0.25. What fails at 0.25 was not featurised and is not in MAD-COSMO. The paper
does not mention this step.

ingest converts the sources that did not come through ColabFit to one parquet schema with
content addressed ids (convert.py, ids.py, one reader per format in readers/). GDB-13 comes
from SMILES through RDKit, ETKDGv3 with seed 0xf00d, then MMFF94, molecules that fail to
embed dropped (preprocess/smiles2extxyz.py). runs/ has the submit script and manifest of
every source. The Materials Cloud archive records were fetched with
runs/mcloud_archive_tarzip/download_extract.sh. These runs also computed features with an
earlier model, which were replaced by the featurisation in featurize_collection.
