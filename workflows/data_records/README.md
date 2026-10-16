How the release and the two Materials Cloud records were built. The release on the
project store keeps one folder per dataset group with flat binary files, atomic numbers,
positions, cells, the 512 features, the ecumetric energy and forces, the ids, the ZBL
numbers, a filter code (0 passed, 1 ZBL rejected, 2 non finite features, 3 NOMAD excluded),
a duplicate flag and the nearest MAD-CORE structure at every level with its distance. The
mad-core folder adds fps_order, k_set, the cell populations and levels.json. The claim scripts
read this layout.

build_dataset_metadata.py and the check scripts validate it, matcloud/export_xyz.py and
export_chemiscope.py write the MAD-CORE record, matcloud/mad-cosmo-v1 the MAD-COSMO cells,
license_audit checks the license of every source dataset (651 of 655 compatible with CC BY
4.0, GDB-13 by written permission, three CC BY-SA records). record_manifests holds the
README, LICENSES and example of both records as uploaded, with the file list and checksums
of MAD-CORE.
