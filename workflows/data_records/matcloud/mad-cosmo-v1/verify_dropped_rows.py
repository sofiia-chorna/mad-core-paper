import json
from pathlib import Path

import h5py
import numpy as np


OLD = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1/uploaded/files')
NEW = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1/rebuild_0-2pow18')
STEM = 'madcosmo_cells_0-2pow18'
BLOCK_ROWS = 1 << 18


def names(handle):
    found = []
    handle.visititems(lambda name, item: found.append(name) if isinstance(item, h5py.Dataset) else None)
    return sorted(found)


# the only allowed difference is a chunk cut to the length of a shrunk array
def same_storage(old, new):
    chunks = None if old.chunks is None else tuple(min(chunk, size) for chunk, size in zip(old.chunks, new.shape))
    return (old.dtype == new.dtype and new.chunks == chunks and old.compression == new.compression
            and old.compression_opts == new.compression_opts and old.shuffle == new.shuffle and old.fletcher32 == new.fletcher32)


# the new array must be the old one with [start, stop) removed along the first axis
def equal_without(old, new, start, stop):
    if new.shape != (old.shape[0] - (stop - start),) + old.shape[1:]:
        return False
    for begin in range(0, new.shape[0], BLOCK_ROWS):
        end = min(begin + BLOCK_ROWS, new.shape[0])
        source = np.concatenate([old[begin:min(end, start)] if begin < start else old[0:0],
                                 old[max(begin, start) + (stop - start):end + (stop - start)] if end > start else old[0:0]])
        if not np.array_equal(new[begin:end], source, equal_nan=new.dtype.kind == 'f'):
            return False
    return True


def main():
    summary = json.loads((NEW / 'drop_summary.json').read_text())
    cut_start, cut_stop = summary['cut']
    removed = cut_stop - cut_start
    report = {}

    with h5py.File(OLD / f'{STEM}_data.h5', 'r') as old, h5py.File(NEW / f'{STEM}_data.h5', 'r') as new:
        report['same dataset list'] = names(old) == names(new)
        report['same storage settings'] = all(same_storage(old[name], new[name]) for name in names(old))
        old_atoms, new_atoms = old['geometry/atom_offsets'][:], new['geometry/atom_offsets'][:]
        atom_start, atom_stop = int(old_atoms[cut_start]), int(old_atoms[cut_stop])
        old_refs = old['identity/source_reference_offsets'][:]
        reference_start, reference_stop = int(old_refs[cut_start]), int(old_refs[cut_stop])

        per_row = [name for name in names(old) if old[name].shape[:1] == old['identity/configuration_id'].shape and name not in ('geometry/atom_offsets', 'identity/source_reference_offsets')]
        per_atom = [name for name in names(old) if old[name].shape[:1] == (int(old_atoms[-1]),)]
        report['rows removed from every per-row array'] = {name: equal_without(old[name], new[name], cut_start, cut_stop) for name in per_row}
        report['atoms removed from every per-atom array'] = {name: equal_without(old[name], new[name], atom_start, atom_stop) for name in per_atom}
        report['source reference bytes'] = equal_without(old['identity/source_reference_bytes'], new['identity/source_reference_bytes'], reference_start, reference_stop)

        # offsets of a kept row point to the same atoms and the same reference text as before
        keep = np.concatenate([np.arange(cut_start), np.arange(cut_stop, len(old_atoms) - 1)])
        report['atom counts per row kept'] = bool(np.array_equal(np.diff(new_atoms), np.diff(old_atoms)[keep]))
        new_refs = new['identity/source_reference_offsets'][:]
        report['reference lengths per row kept'] = bool(np.array_equal(np.diff(new_refs), np.diff(old_refs)[keep]))

        old_cells, new_cells = old['index/cell_offsets'][:], new['index/cell_offsets'][:]
        sizes_old, sizes_new = np.diff(old_cells), np.diff(new_cells)
        report['cell 0 size'] = [int(sizes_old[0]), int(sizes_new[0])]
        report['every other cell size unchanged'] = bool(np.array_equal(sizes_old[1:], sizes_new[1:]))
        centre = new['assignments/center_index'][:, 24]
        cells = np.repeat(np.arange(len(sizes_new)), sizes_new.astype(np.int64))
        report['every row sits in the cell it is assigned to'] = bool(np.array_equal(centre, cells))
        report['no non-finite distance left'] = bool(np.isfinite(new['assignments/distance'][:]).all())
        report['no filter code 2 left'] = int((new['diagnostics/filter_code'][:] == 2).sum()) == 0
        report['cell 0 rows all ranked or flagged as before'] = bool(np.array_equal(new['ordering/fps_order_in_cell'][:cut_start], old['ordering/fps_order_in_cell'][:cut_start]))

        old_rows, new_rows = old['images/rows'][:], new['images/rows'][:]
        kept = (old_rows < cut_start) | (old_rows >= cut_stop)
        expected_rows = np.where(old_rows[kept] >= cut_stop, old_rows[kept] - removed, old_rows[kept])
        report['image rows shifted'] = bool(np.array_equal(new_rows, expected_rows))
        report['image cells kept'] = bool(np.array_equal(new['images/original_cell'][:], old['images/original_cell'][:][kept]))
        old_flags, new_flags = old['images/flags'][:], new['images/flags'][:]
        old_image_offsets, new_image_offsets = old['images/atom_offsets'][:], new['images/atom_offsets'][:]
        report['image flags kept'] = all(np.array_equal(new_flags[new_image_offsets[n]:new_image_offsets[n + 1]], old_flags[old_image_offsets[i]:old_image_offsets[i + 1]])
                                         for n, i in enumerate(np.flatnonzero(kept)))
        for name in ('dictionaries/dataset_group', 'dictionaries/dataset_id', 'index/center_configuration_id'):
            report[f'{name} unchanged'] = bool(np.array_equal(old[name][:], new[name][:]))
        attrs_old, attrs_new = dict(old.attrs), dict(new.attrs)
        report['attrs'] = dict(structures=[int(attrs_old['structures']), int(attrs_new['structures'])], atoms=[int(attrs_old['atoms']), int(attrs_new['atoms'])],
                               others_unchanged=all(np.array_equal(attrs_old[key], attrs_new[key]) for key in attrs_old if key not in ('structures', 'atoms')),
                               structures_match=int(attrs_new['structures']) == new['identity/configuration_id'].shape[0],
                               atoms_match=int(attrs_new['atoms']) == int(new_atoms[-1]))

    with h5py.File(OLD / f'{STEM}_features.h5', 'r') as old, h5py.File(NEW / f'{STEM}_features.h5', 'r') as new:
        report['features same dataset list and storage'] = names(old) == names(new) and all(same_storage(old[name], new[name]) for name in names(old))
        report['features rows removed'] = equal_without(old['ecumetric_features'], new['ecumetric_features'], cut_start, cut_stop)
        report['scaler unchanged'] = all(np.array_equal(old[name][()], new[name][()]) for name in ('scaler/mean', 'scaler/scale'))
        finite = True
        for begin in range(0, new['ecumetric_features'].shape[0], BLOCK_ROWS):
            finite &= bool(np.isfinite(new['ecumetric_features'][begin:begin + BLOCK_ROWS]).all())
        report['all kept features finite'] = finite
        removed_finite = np.isfinite(old['ecumetric_features'][cut_start:cut_stop]).all(axis=1)
        report['removed rows with non-finite features'] = int((~removed_finite).sum())
        report['features attrs'] = dict(structures=int(new.attrs['structures']), atoms=int(new.attrs['atoms']))

    (NEW / 'verify_report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
