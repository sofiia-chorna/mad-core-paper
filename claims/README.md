Each script computes numbers stated in the text from the released data and writes a json to out/.
They read the aa002 release (--release-root) or the store (--store-root).

    largest_voronoi_cell.py        117 cells hold half of the records, the largest cell, 6 s
    nonfinite_descriptors.py       the 67,975 rows with non finite descriptors, 4 min
    element_coverage.py            all 102 elements within 2,158 MAD-CORE and 480 MAD-NEXT picks, 1 min
    collection_within_coverage.py  99.17 percent of the collection within the level 24 radius, Slurm
    outside_coverage_scan.py       distances of all rows to MAD-CORE, Slurm, then outside_coverage_maxima.py and outside_coverage_report.py
    madcosmo_populations.py        counts, filters, duplicates and cell statistics, Slurm

nomad_and_bibliography/ holds the NOMAD provenance and citation checks and the bibliography checks.
