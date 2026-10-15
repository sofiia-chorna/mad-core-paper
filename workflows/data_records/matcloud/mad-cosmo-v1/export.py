import argparse
import hashlib
import json
import mmap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from preflight import INDEX, ROOT, sha256


def packed(values):
    values = [bytes(value) for value in values]
    offsets = np.empty(len(values) + 1, dtype='<u8')
    offsets[0] = 0
    np.cumsum([len(value) for value in values], out=offsets[1:])
    return offsets, np.frombuffer(b''.join(values), dtype='u1')


def expanded_indices(starts, counts):
    counts = np.asarray(counts, dtype=np.int64)
    cumulative = np.r_[0, np.cumsum(counts)]
    return np.repeat(np.asarray(starts, dtype=np.int64) - cumulative[:-1], counts) + np.arange(cumulative[-1])


def output_label(start, stop):
    def bound(value):
        if value > 1 and value & (value - 1) == 0:
            return f'2pow{value.bit_length()-1}'
        return str(value)
    return f'madcosmo_cells_{bound(start)}-{bound(stop)}'


def report_label(report):
    if 'part_number' in report:
        interval = output_label(report['interval_cell_start'], report['interval_cell_stop'])
        return f"{interval}_part{report['part_number']:04d}-of-{report['part_count']:04d}"
    return output_label(report['cell_start'], report['cell_stop'])


def local_cell_offsets(centers, report):
    selected = centers[report['cell_start']:report['cell_stop']+1]
    return np.clip(selected, report['member_start'], report['member_stop']) - report['member_start']


def stage(report, staging, workers=1, metadata_staging=None, resume=False, reader=None, checkpoint_hook=None):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return stage_arrays(report, staging, workers, executor, metadata_staging, resume, reader, checkpoint_hook)


def stage_arrays(report, staging, workers, executor, metadata_staging, resume, reader=None, checkpoint_hook=None):
    existing = staging.exists()
    if existing and not resume:
        raise FileExistsError(staging)
    staging.mkdir(parents=True, exist_ok=resume)
    progress_path = staging / 'stage-progress.json'
    completed = set()
    source_positions = {}
    if existing:
        progress = json.loads(progress_path.read_text())
        assert progress['report'] == report
        completed = set(progress['completed_sources'])
        source_positions = progress.get('source_positions', {})
    n = report['structures']
    a = report['atoms']
    lo, hi = report['member_start'], report['member_stop']
    names = np.load(INDEX / 'release_directory_names.npy')
    source = np.load(INDEX / 'release_directory.npy', mmap_mode='r')[lo:hi]
    rows = np.load(INDEX / 'release_row_in_directory.npy', mmap_mode='r')[lo:hi]
    arrays = {}

    def source_mapping(array):
        if workers > 1:
            array._mmap.madvise(mmap.MADV_RANDOM)
        return array

    def gather(array, indices):
        if reader is not None:
            return reader.read(array, indices)
        if workers == 1 or len(indices) < workers:
            return array[indices]
        chunks = np.array_split(indices, workers)
        return np.concatenate(list(executor.map(lambda chunk: array[chunk], chunks)))

    def create(name, dtype, shape, value=None):
        path = staging / (name.replace('/', '__') + '.npy')
        if existing and path.exists():
            array = np.lib.format.open_memmap(path, mode='r+')
            assert array.dtype == np.dtype(dtype) and array.shape == tuple(shape), name
            arrays[name] = array
            return array
        array = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
        if value is not None:
            array[:] = value
        arrays[name] = array
        return array

    create('record_id', '<u8', (n,), (source.astype(np.uint64) << np.uint64(32)) | rows.astype(np.uint64))
    create('provenance/source_directory_code', '<u2', (n,), source)
    create('provenance/source_row', '<u4', (n,), rows)
    create('ordering/fps_order_in_cell', '<i4', (n,), np.load(INDEX / 'fps_order_in_cell.npy', mmap_mode='r')[lo:hi])
    create('ordering/epsilon_sq', '<f4', (n,), np.load(INDEX / 'epsilon_sq.npy', mmap_mode='r')[lo:hi])
    centers = np.load(INDEX / 'center_offsets.npy', mmap_mode='r')
    offsets = local_cell_offsets(centers, report)
    create('index/cell_offsets', '<u8', offsets.shape, offsets)
    if 'part_number' in report:
        global_offsets = centers[report['cell_start']:report['cell_stop']+1]
        create('index/global_cell_offsets', '<u8', global_offsets.shape, global_offsets)
    create('ordering/cell_id', '<i4', (n,), np.repeat(np.arange(report['cell_start'], report['cell_stop'], dtype=np.int32), np.diff(offsets)))
    center_ids = np.load(INDEX / 'center_configuration_id.npy', mmap_mode='r')[report['cell_start']:report['cell_stop']]
    create('index/center_configuration_id', center_ids.dtype, center_ids.shape, center_ids)
    id_offsets = np.load(INDEX / 'configuration_id_offsets.npy', mmap_mode='r')[lo:hi+1]
    id_bytes = np.memmap(INDEX / 'configuration_id_bytes.bin', mode='r', dtype='u1')
    create('identity/configuration_id_offsets', '<u8', (n+1,), id_offsets-id_offsets[0])
    create('identity/configuration_id_bytes', 'u1', (int(id_offsets[-1]-id_offsets[0]),), id_bytes[id_offsets[0]:id_offsets[-1]])
    counts = np.empty(n, dtype=np.int64)
    ref_lengths = np.zeros(n, dtype=np.int64)
    if metadata_staging is not None:
        previous_ids = np.load(metadata_staging / 'record_id.npy', mmap_mode='r')
        assert np.array_equal(previous_ids, arrays['record_id'])
        previous_atoms = np.load(metadata_staging / 'geometry__atom_offsets.npy', mmap_mode='r')
        previous_refs = np.load(metadata_staging / 'identity__source_reference_offsets.npy', mmap_mode='r')
        assert previous_atoms.shape == previous_refs.shape == (n + 1,)
        assert previous_atoms[0] == previous_refs[0] == 0
        counts[:] = np.diff(previous_atoms.astype(np.int64))
        ref_lengths[:] = np.diff(previous_refs.astype(np.int64))
        assert np.all(counts > 0) and np.all(ref_lengths >= 0)
        assert int(counts.sum()) == a
        assert int(ref_lengths.sum()) == report['source_reference_bytes']
        print(json.dumps({'reused_offsets_from': str(metadata_staging), 'structures': n}), flush=True)
    dataset_width = 1
    group_width = 1
    source_manifests = {}
    for item in report['sources']:
        code = item['code']
        directory = ROOT / item['directory']
        manifest = json.loads((directory / 'MANIFEST.json').read_text())
        source_manifests[item['directory']] = manifest
        dataset_width = max(dataset_width, np.dtype(manifest['files']['dataset_id.npy']['dtype']).itemsize)
        group_width = max(group_width, len(directory.name))
        if 'dataset_group.npy' in manifest['files']:
            group_width = max(group_width, np.dtype(manifest['files']['dataset_group.npy']['dtype']).itemsize)
        selected = np.flatnonzero(source == code)
        if metadata_staging is not None:
            assert int(counts[selected].sum()) == item['atoms']
            continue
        selected = selected[np.argsort(rows[selected], kind='stable')]
        source_rows = rows[selected].astype(np.int64)
        na = source_mapping(np.load(directory / 'na.npy', mmap_mode='r'))
        counts[selected] = gather(na, source_rows+1) - gather(na, source_rows)
        if (directory / 'source_reference_offsets.npy').exists():
            ro = source_mapping(np.load(directory / 'source_reference_offsets.npy', mmap_mode='r'))
            ref_lengths[selected] = gather(ro, source_rows+1) - gather(ro, source_rows)
    assert int(counts.sum()) == a
    atom_offsets = create('geometry/atom_offsets', '<u8', (n+1,))
    if not existing:
        atom_offsets[0] = 0
        np.cumsum(counts, out=atom_offsets[1:])
    else:
        assert np.array_equal(np.diff(atom_offsets.astype(np.int64)), counts)
    ref_offsets = create('identity/source_reference_offsets', '<u8', (n+1,))
    if not existing:
        ref_offsets[0] = 0
        np.cumsum(ref_lengths, out=ref_offsets[1:])
    else:
        assert np.array_equal(np.diff(ref_offsets.astype(np.int64)), ref_lengths)
    create('identity/source_reference_bytes', 'u1', (int(ref_offsets[-1]),))
    create('identity/source_reference_present', '?', (n,), False)
    create('geometry/numbers', 'u1', (a,))
    create('geometry/positions', '<f4', (a,3))
    create('geometry/cell', '<f4', (n,3,3))
    create('geometry/pbc', '?', (n,3))
    create('predictions/ecumetric_energy', '<f4', (n,))
    create('predictions/ecumetric_nc_forces', '<f4', (a,3))
    create('ecumetric_features', '<f4', (n,512))
    create('assignments/center_index', '<i4', (n,25))
    create('assignments/distance', '<f4', (n,25))
    create('diagnostics/collection_flags_present', '?', (n,), False)
    create('diagnostics/filter_code', 'i1', (n,), -1)
    create('diagnostics/is_duplicate', '?', (n,), False)
    for field in ('zbl_energy_per_atom', 'zbl_max_atomic_energy', 'zbl_max_force'):
        create('diagnostics/' + field, '<f4', (n,))
    dataset_ids = create('_work/dataset_ids', f'S{dataset_width}', (n,))
    groups = create('_work/dataset_groups', f'S{group_width}', (n,))
    create('mad_next/present', '?', (n,), False)
    create('mad_next/fps_order', '<i8', (n,), -1)
    create('mad_next/epsilon_sq', '<f8', (n,), np.nan)
    create('mad_next/k_set', '<i4', (n,), -1)
    create('mad_next/split', 'S5', (n,), b'')

    def checkpoint(force=False):
        for array in arrays.values():
            array.flush()
        progress = dict(report=report, completed_sources=sorted(completed), source_positions=source_positions)
        temporary = progress_path.with_suffix('.json.partial')
        temporary.write_text(json.dumps(progress, indent=2) + '\n')
        temporary.replace(progress_path)
        if checkpoint_hook is not None:
            checkpoint_hook(staging, progress, force=force)

    if not existing:
        checkpoint()
    images = []
    for item in report['sources']:
        directory = ROOT / item['directory']
        code = item['code']
        selected = np.flatnonzero(source == code)
        selected = selected[np.argsort(rows[selected], kind='stable')]
        all_rows = rows[selected].astype(np.int64)
        source_arrays = {}

        def npy(name):
            if name not in source_arrays:
                source_arrays[name] = source_mapping(np.load(directory / (name + '.npy'), mmap_mode='r'))
            return source_arrays[name]

        def binary(name, dtype, shape):
            if name not in source_arrays:
                source_arrays[name] = source_mapping(np.memmap(directory / (name + '.bin'), mode='r', dtype=dtype).reshape(shape))
            return source_arrays[name]

        na = npy('na')
        batch_start = len(selected) if int(code) in completed else int(source_positions.get(str(code), 0))
        assert 0 <= batch_start <= len(selected)
        cumulative = np.r_[0, np.cumsum(counts[selected])]
        while batch_start < len(selected):
            end = min(len(selected), batch_start+4096,
                      max(batch_start+1, int(np.searchsorted(cumulative, cumulative[batch_start]+262144, side='right')-1)))
            dest = selected[batch_start:end]
            src = all_rows[batch_start:end]
            natoms = counts[dest]
            source_atom_starts = gather(na, src)
            assert np.array_equal(gather(na, src+1) - source_atom_starts, natoms)
            src_atoms = expanded_indices(source_atom_starts, natoms)
            dest_atoms = expanded_indices(atom_offsets[dest], natoms)
            numbers = gather(binary('a', '<i4', (-1,)), src_atoms)
            assert np.all((numbers >= 0) & (numbers <= 118))
            arrays['geometry/numbers'][dest_atoms] = numbers
            arrays['geometry/positions'][dest_atoms] = gather(binary('x', '<f4', (-1,3)), src_atoms)
            arrays['predictions/ecumetric_nc_forces'][dest_atoms] = gather(binary('ecumetric_nc_forces', '<f4', (-1,3)), src_atoms)
            arrays['geometry/cell'][dest] = gather(binary('c', '<f4', (-1,3,3)), src)
            arrays['geometry/pbc'][dest] = gather(npy('pbc'), src)
            arrays['predictions/ecumetric_energy'][dest] = gather(binary('ecumetric_energy', '<f4', (-1,)), src)
            arrays['ecumetric_features'][dest] = gather(binary('ecumetric_features', '<f4', (-1,512)), src)
            assignments = gather(binary('voronoi_a', '<i4', (-1,25)), src)
            assert np.array_equal(assignments[:,24], arrays['ordering/cell_id'][dest])
            arrays['assignments/center_index'][dest] = assignments
            arrays['assignments/distance'][dest] = gather(binary('voronoi_d', '<f4', (-1,25)), src)
            for field in ('zbl_energy_per_atom', 'zbl_max_atomic_energy', 'zbl_max_force'):
                arrays['diagnostics/' + field][dest] = gather(npy(field), src)
            dataset_ids[dest] = gather(npy('dataset_id'), src)
            if item['directory'] == 'mad-next-coreset':
                groups[dest] = gather(npy('dataset_group'), src)
                arrays['mad_next/present'][dest] = True
                for field in ('fps_order', 'epsilon_sq', 'k_set', 'split'):
                    arrays['mad_next/' + field][dest] = gather(npy(field), src)
            else:
                groups[dest] = directory.name.encode()
                arrays['diagnostics/collection_flags_present'][dest] = True
                for field in ('filter_code', 'is_duplicate'):
                    arrays['diagnostics/' + field][dest] = gather(npy(field), src)
            if (directory / 'source_reference_offsets.npy').exists():
                source_offsets = npy('source_reference_offsets')
                ref_data = binary('source_reference_bytes', 'u1', (-1,))
                source_ref_starts = gather(source_offsets, src)
                assert np.array_equal(gather(source_offsets, src+1) - source_ref_starts, ref_lengths[dest])
                src_bytes = expanded_indices(source_ref_starts, ref_lengths[dest])
                dest_bytes = expanded_indices(ref_offsets[dest], ref_lengths[dest])
                arrays['identity/source_reference_bytes'][dest_bytes] = gather(ref_data, src_bytes)
                arrays['identity/source_reference_present'][dest] = True
            expected_ids = gather(npy('configuration_id'), src)
            local_id_offsets = arrays['identity/configuration_id_offsets']
            local_id_bytes = arrays['identity/configuration_id_bytes']
            for target, expected in zip(dest, expected_ids):
                assert local_id_bytes[local_id_offsets[target]:local_id_offsets[target+1]].tobytes() == bytes(expected)
            batch_start = end
            source_positions[str(code)] = end
            if checkpoint_hook is not None:
                checkpoint()
            if workers > 1:
                print(json.dumps({'copy_source': item['directory'], 'copied': end, 'total': len(selected)}), flush=True)
        if (directory / 'image_flag_rows.npy').exists():
            image_rows = npy('image_flag_rows')
            image_offsets = np.r_[0, np.cumsum(na[image_rows+1]-na[image_rows])]
            matched, source_match, image_match = np.intersect1d(all_rows, image_rows, return_indices=True)
            for source_pos, image_pos in zip(source_match, image_match):
                image_a, image_b = image_offsets[image_pos:image_pos+2]
                flags = binary('image_flags', '<i8', (-1,3))[image_a:image_b].copy()
                cell = binary('image_flag_cells', '<f8', (-1,3,3))[image_pos].copy()
                images.append((int(selected[source_pos]), flags, cell))
        print(json.dumps({'staged': item['directory'], 'structures': len(selected), 'atoms': item['atoms']}), flush=True)
        completed.add(int(code))
        checkpoint()
    if checkpoint_hook is not None:
        checkpoint(force=True)
    arrays.pop('_work/dataset_ids')
    arrays.pop('_work/dataset_groups')
    for field, values in (('dataset_id', dataset_ids), ('dataset_group', groups)):
        dictionary, codes = np.unique(values, return_inverse=True)
        assert len(dictionary) <= 65536
        create('identity/' + field + '_code', '<u2', (n,), codes)
        create('dictionaries/' + field, dictionary.dtype, dictionary.shape, dictionary)
    create('dictionaries/source_directory', names.dtype, names.shape, names)
    hashes = np.load(INDEX / 'release_manifest_sha256.npy')
    create('dictionaries/source_manifest_sha256', hashes.dtype, hashes.shape, hashes)
    images.sort(key=lambda item: item[0])
    image_rows = np.array([item[0] for item in images], dtype='<u8')
    image_counts = np.array([len(item[1]) for item in images], dtype='<u8')
    image_offsets = np.r_[np.uint64(0), np.cumsum(image_counts, dtype=np.uint64)]
    image_flags = np.concatenate([item[1] for item in images]) if images else np.empty((0,3), dtype='<i8')
    image_cells = np.array([item[2] for item in images], dtype='<f8').reshape(-1,3,3)
    for name, array in (('rows', image_rows), ('atom_offsets', image_offsets), ('flags', image_flags), ('original_cell', image_cells)):
        create('images/' + name, array.dtype, array.shape, array)
    assert len(images) == report['image_flag_structures']
    for array in arrays.values():
        array.flush()
    state = dict(report=report, arrays={name: path.name for name in arrays for path in [staging / (name.replace('/', '__') + '.npy')]}, source_manifests=source_manifests)
    state_path = staging / 'stage.json.partial'
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    state_path.replace(staging / 'stage.json')
    return state


def encode(state, staging, output):
    report = state['report']
    output.mkdir(parents=True, exist_ok=True)
    label = report_label(report)
    results = {}
    for kind in ('data', 'features'):
        path = output / f'{label}_{kind}.h5'
        temporary = path.with_suffix('.h5.partial')
        if path.exists() or temporary.exists():
            raise FileExistsError(path)
        field_hashes = {}
        with h5py.File(temporary, 'x') as handle:
            handle.attrs.update(schema='madcosmo-hdf5-v1', kind=kind,
                                cell_start=report['cell_start'], cell_stop=report['cell_stop'],
                                member_start=report['member_start'], member_stop=report['member_stop'],
                                structures=report['structures'], atoms=report['atoms'],
                                root_manifest_sha256=report['root_manifest_sha256'],
                                index_manifest_sha256=report['index_manifest_sha256'],
                                ordering='cell-major in MAD-CORE order; existing within-cell FPS order; unranked tail retained',
                                record_id_encoding='(source_directory_code << 32) | source_row',
                                feature_dimension=512)
            if 'part_number' in report:
                handle.attrs.update({key: report[key] for key in
                                     ('interval_cell_start', 'interval_cell_stop', 'part_number', 'part_count')})
                handle.attrs['partial_cells'] = 'cell_offsets are local to this part; global_cell_offsets give full-cell source bounds'
            for name, filename in state['arrays'].items():
                wanted = name in ('record_id', 'ecumetric_features') if kind == 'features' else name != 'ecumetric_features'
                if not wanted:
                    continue
                array = np.load(staging / filename, mmap_mode='r')
                options = {}
                row_bytes = array.dtype.itemsize * int(np.prod(array.shape[1:]))
                if len(array):
                    chunk_rows = min(len(array), 256 if name == 'ecumetric_features' else max(1, (256 << 10) // row_bytes))
                    options = dict(chunks=(chunk_rows,) + array.shape[1:], compression='gzip', compression_opts=4, shuffle=True, fletcher32=True)
                dataset = handle.create_dataset(name, shape=array.shape, dtype=array.dtype, **options)
                digest = hashlib.sha256()
                batch = max(1, (16 << 20) // row_bytes)
                for start in range(0, len(array), batch):
                    block = np.asarray(array[start:start+batch])
                    dataset[start:start+batch] = block
                    digest.update(block.tobytes())
                field_hashes[name] = digest.hexdigest()
                dataset.attrs['decoded_sha256'] = digest.hexdigest()
                print(f'encoded {kind}: {name} {array.shape}', flush=True)
            if kind == 'features':
                with np.load(ROOT / 'ecumetric_scaler.npz') as scaler:
                    for name in scaler.files:
                        handle.create_dataset('scaler/' + name, data=scaler[name])
        with h5py.File(temporary, 'r') as handle:
            for name, expected in field_hashes.items():
                dataset = handle[name]
                digest = hashlib.sha256()
                row_bytes = dataset.dtype.itemsize * int(np.prod(dataset.shape[1:]))
                for start in range(0, len(dataset), max(1, (16 << 20) // row_bytes)):
                    digest.update(dataset[start:start+max(1, (16 << 20) // row_bytes)].tobytes())
                assert digest.hexdigest() == expected, name
        temporary.rename(path)
        results[path.name] = dict(bytes=path.stat().st_size, sha256=sha256(path), decoded_fields=field_hashes)
        print(f'finished {path.name}: {path.stat().st_size} bytes', flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', required=True, type=Path)
    parser.add_argument('--staging', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--encode-existing', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--metadata-staging', type=Path)
    parser.add_argument('--resume-staging', action='store_true')
    args = parser.parse_args()
    report = json.loads(args.preflight.read_text())
    assert sha256(ROOT / 'MANIFEST.json') == report['root_manifest_sha256']
    assert sha256(INDEX / 'MANIFEST.json') == report['index_manifest_sha256']
    for item in report['sources']:
        assert sha256(ROOT / item['directory'] / 'MANIFEST.json') == item['manifest_sha256']
    started = time.time()
    assert 1 <= args.workers <= 32
    state = json.loads((args.staging / 'stage.json').read_text()) if args.encode_existing else stage(report, args.staging, args.workers, args.metadata_staging, args.resume_staging)
    assert state['report'] == report
    results = encode(state, args.staging, args.output)
    assert sha256(ROOT / 'MANIFEST.json') == report['root_manifest_sha256']
    manifest = dict(schema='madcosmo-hdf5-v1', range=report, files=results, seconds=time.time()-started,
                    validation='compression round-trip verified; independent source validation required before upload')
    manifest_path = args.output.parent / (report_label(report) + '_build.json')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'manifest': str(manifest_path), 'seconds': manifest['seconds']}), flush=True)


if __name__ == '__main__':
    main()
