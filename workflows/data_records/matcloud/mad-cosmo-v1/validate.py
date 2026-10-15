import argparse
import hashlib
import json
import mmap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import h5py
import numpy as np

from preflight import INDEX, ROOT, sha256
from export import expanded_indices


def equal(actual, expected, name):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.tobytes() != expected.tobytes():
        raise AssertionError(f'Bitwise mismatch: {name}; {actual.shape}/{actual.dtype}, {expected.shape}/{expected.dtype}')


def validate(staging, output, manifest_path, workers=1, checkpoint_dir=None, source_chunk_rows=0, reader=None):
    state = json.loads((staging / 'stage.json').read_text())
    report = state['report']
    build = json.loads(manifest_path.read_text())
    assert report == build['range']
    arrays = {name: np.load(staging / filename, mmap_mode='r') for name, filename in state['arrays'].items()}
    for filename, entry in build['files'].items():
        path = output / filename
        assert path.stat().st_size == entry['bytes']
        assert sha256(path) == entry['sha256']
        with h5py.File(path, 'r') as handle:
            assert handle.attrs['root_manifest_sha256'] == sha256(ROOT / 'MANIFEST.json')
            assert handle.attrs['structures'] == report['structures']
            for key in ('cell_start', 'cell_stop', 'member_start', 'member_stop', 'index_manifest_sha256'):
                assert handle.attrs[key] == report[key], key
            if 'part_number' in report:
                for key in ('interval_cell_start', 'interval_cell_stop', 'part_number', 'part_count'):
                    assert handle.attrs[key] == report[key], key
            for name in entry['decoded_fields']:
                dataset = handle[name]
                expected = arrays[name]
                assert dataset.shape == expected.shape and dataset.dtype == expected.dtype
                block_size = max(1, (8 << 20) // max(1, expected.dtype.itemsize * int(np.prod(expected.shape[1:]))))
                for start in range(0, len(dataset), block_size):
                    equal(dataset[start:start+block_size], expected[start:start+block_size], filename + ':' + name)
            if handle.attrs['kind'] == 'features':
                with np.load(ROOT / 'ecumetric_scaler.npz') as scaler:
                    assert set(handle['scaler']) == set(scaler.files)
                    for name in scaler.files:
                        equal(handle['scaler/' + name][()], scaler[name], 'scaler/' + name)
    lo, hi = report['member_start'], report['member_stop']
    source_names = np.load(INDEX / 'release_directory_names.npy')
    source_codes = np.load(INDEX / 'release_directory.npy', mmap_mode='r')[lo:hi]
    source_rows = np.load(INDEX / 'release_row_in_directory.npy', mmap_mode='r')[lo:hi]
    equal(arrays['provenance/source_directory_code'], source_codes.astype('<u2'), 'source directory pointers')
    equal(arrays['provenance/source_row'], source_rows.astype('<u4'), 'source row pointers')
    expected_ids = (source_codes.astype(np.uint64) << np.uint64(32)) | source_rows.astype(np.uint64)
    equal(arrays['record_id'], expected_ids, 'occurrence identity')
    assert len(np.unique(expected_ids)) == len(expected_ids)
    for target, original in (('ordering/fps_order_in_cell', 'fps_order_in_cell'), ('ordering/epsilon_sq', 'epsilon_sq')):
        equal(arrays[target], np.load(INDEX / (original + '.npy'), mmap_mode='r')[lo:hi], target)
    centers = np.load(INDEX / 'center_offsets.npy', mmap_mode='r')
    global_offsets = centers[report['cell_start']:report['cell_stop']+1]
    offsets = np.maximum(np.minimum(global_offsets, hi), lo) - lo
    assert offsets[0] == 0 and offsets[-1] == report['structures']
    equal(arrays['index/cell_offsets'], offsets.astype('<u8'), 'cell offsets')
    if 'part_number' in report:
        equal(arrays['index/global_cell_offsets'], global_offsets.astype('<u8'), 'global cell offsets')
    center_ids = np.load(INDEX / 'center_configuration_id.npy', mmap_mode='r')[report['cell_start']:report['cell_stop']]
    equal(arrays['index/center_configuration_id'], center_ids, 'center identities')
    equal(arrays['dictionaries/source_directory'], source_names, 'source dictionary')
    equal(arrays['dictionaries/source_manifest_sha256'], np.load(INDEX / 'release_manifest_sha256.npy'), 'source hashes')
    atom_offsets = arrays['geometry/atom_offsets']
    atom_counts = np.diff(atom_offsets.astype(np.int64))
    assert atom_offsets[0] == 0 and atom_offsets[-1] == report['atoms']
    assert np.all(atom_counts > 0)
    for name in ('configuration_id', 'source_reference'):
        off = arrays['identity/' + name + '_offsets']
        assert off[0] == 0 and off[-1] == len(arrays['identity/' + name + '_bytes'])
        assert np.all(np.diff(off.astype(np.int64)) >= 0)
    for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:]), start=report['cell_start']):
        assert np.all(arrays['ordering/cell_id'][a:b] == cell)
        rank = arrays['ordering/fps_order_in_cell'][a:b]
        ranked = rank[rank >= 0]
        assert np.all(np.diff(ranked.astype(np.int64)) >= 0)
        assert np.all(rank[len(ranked):] == -1)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        header = dict(build_sha256=sha256(manifest_path), stage_sha256=sha256(staging / 'stage.json'),
                      validator_sha256=sha256(Path(__file__)), source_chunk_rows=source_chunk_rows)
        header_path = checkpoint_dir / 'header.json'
        if header_path.exists():
            assert json.loads(header_path.read_text()) == header
        else:
            header_path.write_text(json.dumps(header, indent=2) + '\n')
    image_lookup = {}
    image_lock = Lock()

    def gather(array, indices):
        return reader.read(array, indices) if reader is not None else array[indices]

    def verify_source(code, targets, chunk):
        seen_images = []
        checked = 0
        name = bytes(source_names[code]).decode()
        directory = ROOT / name
        assert sha256(directory / 'MANIFEST.json') == bytes(arrays['dictionaries/source_manifest_sha256'][code]).decode()
        cache_path = checkpoint_dir / f'{code}-{chunk}.json' if checkpoint_dir is not None else None
        if cache_path is not None and cache_path.exists():
            cached = json.loads(cache_path.read_text())
            assert cached['name'] == name and cached['checked'] == len(targets)
            assert cached['first_target'] == int(targets[0]) and cached['last_target'] == int(targets[-1])
            return name, cached['checked'], cached['images']
        npys = {}
        bins = {}

        def load(name):
            if name not in npys:
                npys[name] = np.load(directory / (name + '.npy'), mmap_mode='r')
                if workers > 1:
                    npys[name]._mmap.madvise(mmap.MADV_RANDOM)
            return npys[name]

        def binary(name, dtype, shape):
            if name not in bins:
                bins[name] = np.memmap(directory / (name + '.bin'), mode='r', dtype=dtype).reshape(shape)
                if workers > 1:
                    bins[name]._mmap.madvise(mmap.MADV_RANDOM)
            return bins[name]

        na = load('na')
        for begin in range(0, len(targets), 512):
            dest = targets[begin:begin+512]
            src = source_rows[dest].astype(np.int64)
            source_starts = gather(na, src)
            counts = gather(na, src + 1) - source_starts
            equal(atom_counts[dest], counts, name + ':atom counts')
            numeric = [('geometry/cell', 'c', '<f4', (-1,3,3)),
                       ('predictions/ecumetric_energy', 'ecumetric_energy', '<f4', (-1,)),
                       ('ecumetric_features', 'ecumetric_features', '<f4', (-1,512)),
                       ('assignments/center_index', 'voronoi_a', '<i4', (-1,25)),
                       ('assignments/distance', 'voronoi_d', '<f4', (-1,25))]
            for target, filename, dtype, shape in numeric:
                equal(arrays[target][dest], gather(binary(filename, dtype, shape), src), name + ':' + target)
            equal(arrays['assignments/center_index'][dest,24], arrays['ordering/cell_id'][dest], name + ':assignment')
            equal(arrays['geometry/pbc'][dest], gather(load('pbc'), src), name + ':pbc')
            for field in ('zbl_energy_per_atom', 'zbl_max_atomic_energy', 'zbl_max_force'):
                equal(arrays['diagnostics/' + field][dest], gather(load(field), src), name + ':' + field)
            is_mad = name == 'mad-next-coreset'
            assert np.all(arrays['mad_next/present'][dest] == is_mad)
            assert np.all(arrays['diagnostics/collection_flags_present'][dest] == (not is_mad))
            for target, original in (('dataset_id', 'dataset_id'), ('dataset_group', 'dataset_group')):
                expected = gather(load(original), src) if original == 'dataset_id' or is_mad else np.full(len(src), directory.name.encode())
                actual = arrays['dictionaries/' + target][arrays['identity/' + target + '_code'][dest]]
                assert np.array_equal(actual, expected), name + ':' + target
            if is_mad:
                for field in ('fps_order', 'epsilon_sq', 'k_set', 'split'):
                    equal(arrays['mad_next/' + field][dest], gather(load(field), src), name + ':' + field)
                assert np.all(arrays['diagnostics/filter_code'][dest] == -1)
            else:
                for field in ('filter_code', 'is_duplicate'):
                    equal(arrays['diagnostics/' + field][dest], gather(load(field), src), name + ':' + field)
            has_reference = (directory / 'source_reference_offsets.npy').exists()
            assert np.all(arrays['identity/source_reference_present'][dest] == has_reference)
            expected_ids = gather(load('configuration_id'), src)
            if reader is not None:
                source_atoms = expanded_indices(source_starts, counts)
                target_atoms = expanded_indices(atom_offsets[dest], counts)
                for target, original, dtype, shape in (
                    ('geometry/numbers', 'a', '<i4', (-1,)),
                    ('geometry/positions', 'x', '<f4', (-1, 3)),
                    ('predictions/ecumetric_nc_forces', 'ecumetric_nc_forces', '<f4', (-1, 3)),
                ):
                    actual = arrays[target][target_atoms].astype(dtype, copy=False)
                    equal(actual, gather(binary(original, dtype, shape), source_atoms), name + ':' + target)
                ref_offsets = arrays['identity/source_reference_offsets']
                lengths = ref_offsets[dest + 1] - ref_offsets[dest]
                if has_reference:
                    starts = gather(load('source_reference_offsets'), src)
                    source_lengths = gather(load('source_reference_offsets'), src + 1) - starts
                    equal(lengths, source_lengths.astype(lengths.dtype), name + ':reference lengths')
                    original_bytes = expanded_indices(starts, lengths)
                    target_bytes = expanded_indices(ref_offsets[dest], lengths)
                    equal(arrays['identity/source_reference_bytes'][target_bytes],
                          gather(binary('source_reference_bytes', 'u1', (-1,)), original_bytes), name + ':reference')
                else:
                    assert np.all(lengths == 0)
            for target_row, original_row, expected_id in zip(dest, src, expected_ids):
                ia, ib = arrays['identity/configuration_id_offsets'][target_row:target_row+2]
                assert arrays['identity/configuration_id_bytes'][ia:ib].tobytes() == bytes(expected_id)
                if reader is None:
                    a, b = map(int, atom_offsets[target_row:target_row+2])
                    sa, sb = map(int, na[original_row:original_row+2])
                    equal(arrays['geometry/numbers'][a:b].astype('<i4'), binary('a', '<i4', (-1,))[sa:sb], name + ':numbers')
                    equal(arrays['geometry/positions'][a:b], binary('x', '<f4', (-1,3))[sa:sb], name + ':positions')
                    equal(arrays['predictions/ecumetric_nc_forces'][a:b], binary('ecumetric_nc_forces', '<f4', (-1,3))[sa:sb], name + ':forces')
                    ra, rb = arrays['identity/source_reference_offsets'][target_row:target_row+2]
                    if has_reference:
                        sa, sb = load('source_reference_offsets')[original_row:original_row+2]
                        equal(arrays['identity/source_reference_bytes'][ra:rb], binary('source_reference_bytes', 'u1', (-1,))[sa:sb], name + ':reference')
                    else:
                        assert ra == rb
            checked += len(dest)
            if checked % 32768 == 0:
                print(json.dumps({'validating_source': name, 'checked': checked, 'total': len(targets)}), flush=True)
        if (directory / 'image_flag_rows.npy').exists():
            with image_lock:
                if int(code) not in image_lookup:
                    image_rows = load('image_flag_rows')
                    image_lookup[int(code)] = image_rows, np.r_[0, np.cumsum(na[image_rows+1]-na[image_rows])]
                image_rows, image_offsets = image_lookup[int(code)]
            for target in targets:
                row = source_rows[target]
                image_pos = int(np.searchsorted(image_rows, row))
                if image_pos == len(image_rows) or image_rows[image_pos] != row:
                    continue
                output_pos = int(np.searchsorted(arrays['images/rows'], target))
                assert output_pos < len(arrays['images/rows']) and arrays['images/rows'][output_pos] == target
                a, b = arrays['images/atom_offsets'][output_pos:output_pos+2]
                sa, sb = image_offsets[image_pos:image_pos+2]
                equal(arrays['images/flags'][a:b], binary('image_flags', '<i8', (-1,3))[sa:sb], name + ':image flags')
                equal(arrays['images/original_cell'][output_pos], binary('image_flag_cells', '<f8', (-1,3,3))[image_pos], name + ':image cell')
                assert b-a == atom_offsets[target+1]-atom_offsets[target]
                seen_images.append(int(target))
        if cache_path is not None:
            temporary = cache_path.with_suffix('.json.partial')
            temporary.write_text(json.dumps(dict(name=name, checked=checked, images=seen_images,
                                                 first_target=int(targets[0]), last_target=int(targets[-1]))) + '\n')
            temporary.replace(cache_path)
        return name, checked, seen_images

    seen_images = []
    checked = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for code in np.unique(source_codes):
            targets = np.flatnonzero(source_codes == code)
            targets = targets[np.argsort(source_rows[targets], kind='stable')]
            size = source_chunk_rows or len(targets)
            for chunk, begin in enumerate(range(0, len(targets), size)):
                futures.append(executor.submit(verify_source, code, targets[begin:begin+size], chunk))
        for future in as_completed(futures):
            name, count, images = future.result()
            checked += count
            seen_images.extend(images)
            print(json.dumps({'verified_source': name, 'structures': count, 'checked_total': checked}), flush=True)
    assert checked == report['structures']
    assert sorted(seen_images) == arrays['images/rows'].tolist()
    assert sha256(ROOT / 'MANIFEST.json') == report['root_manifest_sha256']
    return dict(status='PASS', structures=checked, atoms=report['atoms'], image_flag_structures=len(seen_images),
                scope='Every exported structure, atom, feature, prediction, assignment, identity, source reference and optional sidecar compared to source; all HDF5 fields compared bitwise to validated staging',
                files={name: {'bytes': entry['bytes'], 'sha256': entry['sha256']} for name, entry in build['files'].items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--staging', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--source-chunk-rows', type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    assert 1 <= args.workers <= 32
    assert args.source_chunk_rows >= 0
    result = validate(args.staging, args.output, args.manifest, args.workers, args.checkpoint_dir, args.source_chunk_rows)
    result['seconds'] = time.time() - started
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
