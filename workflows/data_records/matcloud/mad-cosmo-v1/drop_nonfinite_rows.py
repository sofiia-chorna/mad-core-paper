import csv
import json
from pathlib import Path

import h5py
import numpy as np


SOURCE = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1/uploaded/files')
OUT = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1/rebuild_0-2pow18')
STEM = 'madcosmo_cells_0-2pow18'
LEVELS = 25
BLOCK_BYTES = 1 << 28

STRUCTURE_ARRAYS = ('assignments/center_index', 'assignments/distance', 'diagnostics/filter_code', 'diagnostics/is_duplicate',
                    'diagnostics/zbl_energy_per_atom', 'diagnostics/zbl_max_atomic_energy', 'diagnostics/zbl_max_force',
                    'geometry/cell', 'geometry/pbc', 'identity/configuration_id', 'identity/dataset_group_code',
                    'identity/dataset_id_code', 'ordering/epsilon_sq', 'ordering/fps_order_in_cell', 'predictions/ecumetric_energy')
ATOM_ARRAYS = ('geometry/numbers', 'geometry/positions', 'predictions/ecumetric_nc_forces')
UNCHANGED = ('dictionaries/dataset_group', 'dictionaries/dataset_id', 'index/center_configuration_id')


# same storage as the source, a chunk is at most as long as the array, as publish.py chose it
def create_like(target, source, name, shape):
    dataset = source[name]
    if dataset.chunks is None:
        return target.create_dataset(name, shape=shape, dtype=dataset.dtype)
    chunks = tuple(min(chunk, size) for chunk, size in zip(dataset.chunks, shape))
    return target.create_dataset(name, shape=shape, dtype=dataset.dtype, chunks=chunks,
                                 compression=dataset.compression, compression_opts=dataset.compression_opts,
                                 shuffle=dataset.shuffle, fletcher32=dataset.fletcher32, fillvalue=dataset.fillvalue)


# copy everything outside [cut_start, cut_stop) along the first axis, in blocks
def copy_without(target, source, name, cut_start, cut_stop):
    dataset = source[name]
    length = dataset.shape[0] - (cut_stop - cut_start)
    out = create_like(target, source, name, (length,) + dataset.shape[1:])
    step = max(1, BLOCK_BYTES // (dataset.dtype.itemsize * int(np.prod(dataset.shape[1:]))))
    written = 0
    for begin, end in ((0, cut_start), (cut_stop, dataset.shape[0])):
        for start in range(begin, end, step):
            stop = min(start + step, end)
            out[written:written + stop - start] = dataset[start:stop]
            written += stop - start
    assert written == length, name


def write_values(target, source, name, values):
    out = create_like(target, source, name, values.shape)
    out[...] = values


# the offsets of the kept rows, the ones after the cut move down by what was removed
def offsets_without(offsets, cut_start, cut_stop):
    removed = offsets[cut_stop] - offsets[cut_start]
    return np.concatenate([offsets[:cut_start + 1], offsets[cut_stop + 1:] - removed]).astype(offsets.dtype)


def copy_attrs(source, target, rows, atoms):
    for key, value in source.attrs.items():
        target.attrs[key] = value
    target.attrs['structures'] = np.int64(rows)
    if 'atoms' in source.attrs:
        target.attrs['atoms'] = np.int64(atoms)


def main():
    OUT.mkdir(exist_ok=True)
    data_path, features_path = SOURCE / f'{STEM}_data.h5', SOURCE / f'{STEM}_features.h5'

    with h5py.File(data_path, 'r') as data:
        rows = data['identity/configuration_id'].shape[0]
        cell_offsets = data['index/cell_offsets'][:]
        distance = data['assignments/distance'][:]
        bad = np.flatnonzero(~np.isfinite(distance[:, LEVELS - 1]))
        cut_start, cut_stop = int(bad[0]), int(bad[-1]) + 1

        # the rows to drop are exactly the unranked tail of cell 0 with no distance and no centre at any level
        assert np.array_equal(bad, np.arange(cut_start, cut_stop))
        assert cut_stop == cell_offsets[1] and cut_start > cell_offsets[0]
        assert (~np.isfinite(distance[cut_start:cut_stop])).all() and np.isfinite(distance[:cut_start]).all() and np.isfinite(distance[cut_stop:]).all()
        assert (data['assignments/center_index'][cut_start:cut_stop] == 0).all()
        assert (data['ordering/fps_order_in_cell'][cut_start:cut_stop] == -1).all()
        assert set(np.unique(data['diagnostics/filter_code'][cut_start:cut_stop]).tolist()) <= {1, 2}
        del distance

        atom_offsets = data['geometry/atom_offsets'][:]
        reference_offsets = data['identity/source_reference_offsets'][:]
        image_rows = data['images/rows'][:]
        image_offsets = data['images/atom_offsets'][:]
        assert np.all(np.diff(image_rows.astype(np.int64)) > 0)

        names = [name.decode() for name in data['dictionaries/dataset_id'][:]]
        columns = zip(data['identity/configuration_id'][cut_start:cut_stop], data['identity/dataset_id_code'][cut_start:cut_stop],
                      data['diagnostics/filter_code'][cut_start:cut_stop], data['diagnostics/is_duplicate'][cut_start:cut_stop])
        dropped = [dict(configuration_id=identifier.decode(), dataset_id=names[code], filter_code=int(flag), is_duplicate=bool(duplicate))
                   for identifier, code, flag, duplicate in columns]

        atom_start, atom_stop = int(atom_offsets[cut_start]), int(atom_offsets[cut_stop])
        reference_start, reference_stop = int(reference_offsets[cut_start]), int(reference_offsets[cut_stop])
        keep_image = (image_rows < cut_start) | (image_rows >= cut_stop)
        kept_rows = np.where(image_rows[keep_image] >= cut_stop, image_rows[keep_image] - (cut_stop - cut_start), image_rows[keep_image]).astype(image_rows.dtype)
        image_counts = np.diff(image_offsets.astype(np.int64))[keep_image]
        kept_image_offsets = np.concatenate([[0], np.cumsum(image_counts)]).astype(image_offsets.dtype)
        new_cells = cell_offsets.copy()
        new_cells[1:] -= cut_stop - cut_start

        new_rows = rows - (cut_stop - cut_start)
        new_atoms = int(atom_offsets[-1]) - (atom_stop - atom_start)
        with h5py.File(OUT / data_path.name, 'x') as target:
            copy_attrs(data, target, new_rows, new_atoms)
            for name in STRUCTURE_ARRAYS:
                copy_without(target, data, name, cut_start, cut_stop)
            for name in ATOM_ARRAYS:
                copy_without(target, data, name, atom_start, atom_stop)
            copy_without(target, data, 'identity/source_reference_bytes', reference_start, reference_stop)
            write_values(target, data, 'geometry/atom_offsets', offsets_without(atom_offsets, cut_start, cut_stop))
            write_values(target, data, 'identity/source_reference_offsets', offsets_without(reference_offsets, cut_start, cut_stop))
            write_values(target, data, 'index/cell_offsets', new_cells)
            for name in UNCHANGED:
                write_values(target, data, name, data[name][:])
            write_values(target, data, 'images/rows', kept_rows)
            write_values(target, data, 'images/atom_offsets', kept_image_offsets)
            write_values(target, data, 'images/original_cell', data['images/original_cell'][:][keep_image])
            flags = data['images/flags'][:]
            write_values(target, data, 'images/flags', np.concatenate([flags[image_offsets[i]:image_offsets[i + 1]] for i in np.flatnonzero(keep_image)]))

    with h5py.File(features_path, 'r') as features, h5py.File(OUT / features_path.name, 'x') as target:
        assert features['ecumetric_features'].shape[0] == rows
        copy_attrs(features, target, new_rows, new_atoms)
        copy_without(target, features, 'ecumetric_features', cut_start, cut_stop)
        for name in ('scaler/mean', 'scaler/scale'):
            write_values(target, features, name, features[name][()])

    with open(OUT / 'dropped_rows.tsv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dropped[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(dropped)
    summary = dict(cut=[cut_start, cut_stop], rows_before=rows, rows_after=new_rows, atoms_removed=atom_stop - atom_start,
                   cell_0_before=int(cell_offsets[1]), cell_0_after=int(new_cells[1]), images_removed=int((~keep_image).sum()))
    (OUT / 'drop_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
