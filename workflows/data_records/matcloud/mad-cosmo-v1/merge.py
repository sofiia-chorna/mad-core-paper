import argparse
import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np


ROOT = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1/to_upload')
INDEX = Path('${STORE_ROOT}/mad-core/release/mad-core/cell_fps')
BLOCK_BYTES = 64 << 20
PART_ATTRS = ('interval_cell_start', 'interval_cell_stop', 'part_number', 'part_count', 'partial_cells')
FILTERS = dict(compression='gzip', compression_opts=4, shuffle=True, fletcher32=True)
SMALL = ('dictionaries/', 'index/', 'images/')


def chunk_rows(length, dtype, shape_tail, features=False):
    row_bytes = np.dtype(dtype).itemsize * int(np.prod(shape_tail))
    return min(length, 256 if features else max(1, (256 << 10) // row_bytes))


def create(group, name, shape, dtype, features=False):
    if shape[0] == 0:
        return group.create_dataset(name, shape=shape, dtype=dtype)
    chunks = (chunk_rows(shape[0], dtype, shape[1:], features),) + tuple(shape[1:])
    return group.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks, **FILTERS)


def write_small(group, name, values):
    dataset = create(group, name, values.shape, values.dtype)
    if len(values):
        dataset[...] = values


def kind_of(name):
    if name in ('geometry/numbers', 'geometry/positions', 'predictions/ecumetric_nc_forces'):
        return 'atoms'
    if name == 'identity/source_reference_bytes':
        return 'refs'
    if name == 'geometry/atom_offsets':
        return 'atom_offsets'
    if name == 'identity/source_reference_offsets':
        return 'ref_offsets'
    if name == 'identity/configuration_id':
        return 'string'
    if name in ('identity/dataset_id_code', 'identity/dataset_group_code'):
        return 'codes'
    return 'rows'


def dataset_names(handle):
    names = []
    handle.visititems(lambda name, item: names.append(name) if isinstance(item, h5py.Dataset) else None)
    return sorted(names)


def load_plan(interval):
    pattern = re.compile(rf'madcosmo_cells_{re.escape(interval)}_part(\d{{4}})-of-(\d{{4}})_data\.h5')
    found = []
    for path in (ROOT / 'files').iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), int(match.group(2)), path))
    found.sort()
    count = found[0][1]
    assert [number for number, _, _ in found] == list(range(1, count + 1)), 'parts are missing'
    assert all(total == count for _, total, _ in found)

    parts = []
    base = dict(rows=0, atoms=0, refs=0, images=0, image_atoms=0)
    for number, _, data_path in found:
        features_path = data_path.with_name(data_path.name.replace('_data.h5', '_features.h5'))
        with h5py.File(data_path, 'r') as data, h5py.File(features_path, 'r') as features:
            attrs = dict(data.attrs)
            image_offsets = data['images/atom_offsets'][:]
            part = dict(
                number=number, data=str(data_path), features=str(features_path), attrs=attrs,
                features_attrs=dict(features.attrs), names=dataset_names(data),
                rows=int(attrs['structures']), atoms=int(attrs['atoms']),
                refs=int(data['identity/source_reference_bytes'].shape[0]),
                images=int(data['images/rows'].shape[0]), image_atoms=int(image_offsets[-1]),
                base=dict(base),
            )
            assert data['geometry/atom_offsets'][-1] == part['atoms']
            assert features['ecumetric_features'].shape[0] == part['rows']
        parts.append(part)
        for key in ('rows', 'atoms', 'refs', 'images', 'image_atoms'):
            base[key] += part[key]

    first = parts[0]['attrs']
    for previous, part in zip(parts, parts[1:]):
        assert part['attrs']['member_start'] == previous['attrs']['member_stop']
        assert part['names'] == parts[0]['names']
        for key in ('interval_cell_start', 'interval_cell_stop', 'schema', 'kind', 'ordering', 'feature_dimension'):
            assert part['attrs'][key] == first[key], key

    attrs = {key: value for key, value in first.items() if key not in PART_ATTRS}
    attrs.update(cell_start=first['interval_cell_start'], cell_stop=first['interval_cell_stop'],
                 member_start=first['member_start'], member_stop=parts[-1]['attrs']['member_stop'],
                 structures=np.int64(base['rows']), atoms=np.int64(base['atoms']))
    assert attrs['member_stop'] - attrs['member_start'] == base['rows']
    features_attrs = {key: value for key, value in parts[0]['features_attrs'].items() if key not in PART_ATTRS}
    features_attrs.update({key: attrs[key] for key in ('cell_start', 'cell_stop', 'member_start', 'member_stop', 'structures', 'atoms')})
    return dict(interval=interval, parts=parts, total=base, attrs=attrs, features_attrs=features_attrs)


def merged_dictionaries(plan):
    result = {}
    for field in ('dataset_id', 'dataset_group'):
        per_part = []
        for part in plan['parts']:
            with h5py.File(part['data'], 'r') as data:
                per_part.append(data['dictionaries/' + field][:])
        merged = np.unique(np.concatenate(per_part))
        lookups = []
        for names in per_part:
            lookup = np.searchsorted(merged, names.astype(merged.dtype))
            assert np.array_equal(merged[lookup], names)
            lookups.append(lookup.astype(np.uint16))
        assert len(merged) <= 65536
        result[field] = dict(merged=merged, per_part=per_part, lookups=lookups)
    return result


def small_datasets(plan):
    cell_start = int(plan['attrs']['cell_start'])
    cell_stop = int(plan['attrs']['cell_stop'])
    cells = cell_stop - cell_start
    global_offsets = np.full(cells + 1, -1, dtype=np.int64)
    widths = []
    centers = [None] * cells
    images = dict(rows=[], atom_offsets=[np.zeros(1, dtype=np.uint64)], flags=[], original_cell=[])
    for part in plan['parts']:
        with h5py.File(part['data'], 'r') as data:
            first = int(part['attrs']['cell_start']) - cell_start
            last = int(part['attrs']['cell_stop']) - cell_start
            offsets = data['index/global_cell_offsets'][:].astype(np.int64)
            seen = global_offsets[first:last + 1]
            known = seen >= 0
            assert np.array_equal(seen[known], offsets[known]), 'cell offsets disagree between parts'
            global_offsets[first:last + 1] = offsets

            values = data['index/center_configuration_id'][:]
            widths.append(values.dtype.itemsize)
            for cell, value in zip(range(first, last), values):
                assert centers[cell] is None or centers[cell] == bytes(value), 'center ids disagree between parts'
                centers[cell] = bytes(value)

            if part['images']:
                images['rows'].append(data['images/rows'][:] + np.uint64(part['base']['rows']))
                images['atom_offsets'].append(data['images/atom_offsets'][1:] + np.uint64(part['base']['image_atoms']))
                images['flags'].append(data['images/flags'][:])
                images['original_cell'].append(data['images/original_cell'][:])
            image_dtypes = {key: data['images/' + key].dtype for key in ('rows', 'atom_offsets', 'flags', 'original_cell')}

    assert np.all(global_offsets >= 0) and all(value is not None for value in centers)
    member_start = int(plan['attrs']['member_start'])
    cell_offsets = (global_offsets - member_start).astype(np.uint64)
    assert cell_offsets[0] == 0 and cell_offsets[-1] == plan['total']['rows']
    result = {
        'index/cell_offsets': cell_offsets,
        'index/center_configuration_id': np.array(centers, dtype=f'S{max(widths)}'),
        'images/rows': np.concatenate(images['rows']) if images['rows'] else np.empty(0, image_dtypes['rows']),
        'images/atom_offsets': np.concatenate(images['atom_offsets']).astype(image_dtypes['atom_offsets']),
        'images/flags': np.concatenate(images['flags']) if images['flags'] else np.empty((0, 3), image_dtypes['flags']),
        'images/original_cell': np.concatenate(images['original_cell']) if images['original_cell'] else np.empty((0, 3, 3), image_dtypes['original_cell']),
    }
    assert len(result['images/rows']) == plan['total']['images']
    assert result['images/atom_offsets'][-1] == plan['total']['image_atoms']
    return result


def column_task(plan, name, dictionaries):
    kind = kind_of(name)
    length_key = {'atoms': 'atoms', 'refs': 'refs'}.get(kind, 'rows')
    with h5py.File(plan['parts'][0]['data'], 'r') as data:
        source = data[name]
        dtype = source.dtype
        tail = source.shape[1:]
    if kind == 'string':
        widths = []
        for part in plan['parts']:
            with h5py.File(part['data'], 'r') as data:
                widths.append(data[name].dtype.itemsize)
        dtype = np.dtype(f'S{max(widths)}')
    total = plan['total'][length_key] + (1 if kind.endswith('offsets') else 0)
    element_total = plan['total']['atoms' if kind == 'atom_offsets' else 'refs'] if kind.endswith('offsets') else None
    task = dict(name=name, kind=kind, length_key=length_key, dtype=dtype.str, tail=tail, total=total, element_total=element_total,
                parts=[dict(data=part['data'], length=part[length_key], base=part['base']) for part in plan['parts']])
    if kind == 'codes':
        field = name.split('/')[1].removesuffix('_code')
        task['merged'] = dictionaries[field]['merged']
        task['per_part'] = dictionaries[field]['per_part']
        task['lookups'] = dictionaries[field]['lookups']
    return task


def raw_copy_possible(task):
    if task['kind'] not in ('rows', 'atoms', 'refs'):
        return False
    rows = chunk_rows(task['total'], task['dtype'], task['tail'])
    for index, part in enumerate(task['parts']):
        with h5py.File(part['data'], 'r') as data:
            source = data[task['name']]
            if source.dtype.str != task['dtype'] or source.chunks is None or source.chunks[0] != rows:
                return False
            if (source.compression, source.compression_opts, source.shuffle, source.fletcher32) != ('gzip', 4, True, True):
                return False
        if index < len(task['parts']) - 1 and part['length'] % rows:
            return False
    return True


def transform(task, index, block):
    kind = task['kind']
    if kind == 'string':
        return block.astype(task['dtype'])
    if kind == 'codes':
        return task['lookups'][index][block]
    if kind == 'atom_offsets':
        return block + np.uint64(task['parts'][index]['base']['atoms'])
    if kind == 'ref_offsets':
        return block + np.uint64(task['parts'][index]['base']['refs'])
    return block


# rows are queued and written only in whole chunks, so no chunk is written twice
class ChunkWriter:
    def __init__(self, target, start, rows_per_write):
        self.target = target
        self.position = start
        self.rows_per_write = rows_per_write
        self.pending = []
        self.pending_rows = 0

    def add(self, block):
        self.pending.append(block)
        self.pending_rows += len(block)
        if self.pending_rows >= self.rows_per_write:
            self.flush(whole_chunks=True)

    def flush(self, whole_chunks=False):
        if not self.pending_rows:
            return
        block = np.concatenate(self.pending)
        size = len(block)
        if whole_chunks:
            chunk = self.target.chunks[0]
            size = (self.position + size) // chunk * chunk - self.position
            if size <= 0:
                return
        self.target[self.position:self.position + size] = block[:size]
        self.position += size
        rest = block[size:]
        self.pending = [rest] if len(rest) else []
        self.pending_rows = len(rest)


def output_offset(task, part):
    base = part['base'][task['length_key']]
    return base + 1 if task['kind'].endswith('offsets') else base


def write_column(task, path):
    started = time.time()
    raw = raw_copy_possible(task)
    with h5py.File(path, 'w') as out:
        target = create(out, task['name'], (task['total'],) + tuple(task['tail']), np.dtype(task['dtype']))
        if task['kind'].endswith('offsets'):
            target[0:1] = np.zeros(1, dtype=target.dtype)
        rows_per_write = target.chunks[0] * max(1, BLOCK_BYTES // (target.chunks[0] * target.dtype.itemsize * int(np.prod(task['tail']))))
        writer = ChunkWriter(target, 1 if task['kind'].endswith('offsets') else 0, rows_per_write)
        for index, part in enumerate(task['parts']):
            with h5py.File(part['data'], 'r') as data:
                source = data[task['name']]
                if raw:
                    base = output_offset(task, part)
                    for chunk_index in range(source.id.get_num_chunks()):
                        info = source.id.get_chunk_info(chunk_index)
                        mask, chunk = source.id.read_direct_chunk(info.chunk_offset)
                        target.id.write_direct_chunk((info.chunk_offset[0] + base,) + info.chunk_offset[1:], chunk, mask)
                    continue
                first = 1 if task['kind'].endswith('offsets') else 0
                stop = part['length'] + first
                step = max(1, BLOCK_BYTES // (source.dtype.itemsize * int(np.prod(source.shape[1:]))))
                for start in range(first, stop, step):
                    writer.add(transform(task, index, source[start:min(start + step, stop)]))
        writer.flush()
    return dict(name=task['name'], raw_copy=raw, seconds=time.time() - started)


def verify_column(task, path, raw):
    with h5py.File(path, 'r') as out:
        target = out[task['name']]
        assert target.shape == (task['total'],) + tuple(task['tail']) and target.dtype.str == task['dtype']
        if task['kind'].endswith('offsets'):
            assert target[0] == 0 and target[-1] == task['element_total']
        for index, part in enumerate(task['parts']):
            with h5py.File(part['data'], 'r') as data:
                source = data[task['name']]
                base = output_offset(task, part)
                if raw:
                    for chunk_index in range(source.id.get_num_chunks()):
                        info = source.id.get_chunk_info(chunk_index)
                        _, chunk = source.id.read_direct_chunk(info.chunk_offset)
                        _, copied = target.id.read_direct_chunk((info.chunk_offset[0] + base,) + info.chunk_offset[1:])
                        assert chunk == copied, (task['name'], index, chunk_index)
                    continue
                step = max(1, BLOCK_BYTES // (source.dtype.itemsize * int(np.prod(source.shape[1:]))))
                kind = task['kind']
                if kind.endswith('offsets'):
                    element_base = part['base']['atoms' if kind == 'atom_offsets' else 'refs']
                    assert target[base - 1] == element_base, (task['name'], index)
                    for start in range(0, part['length'], step):
                        stop = min(start + step, part['length'])
                        expected = np.diff(source[start:stop + 1].astype(np.int64))
                        got = np.diff(target[base - 1 + start:base + stop].astype(np.int64))
                        assert np.array_equal(expected, got), (task['name'], index, start)
                    continue
                for start in range(0, part['length'], step):
                    stop = min(start + step, part['length'])
                    expected = source[start:stop]
                    got = target[base + start:base + stop]
                    if kind == 'codes':
                        assert np.array_equal(task['merged'][got], task['per_part'][index][expected]), (task['name'], index, start)
                    elif kind == 'string':
                        assert np.array_equal(got, expected), (task['name'], index, start)
                    else:
                        assert got.dtype == expected.dtype and got.tobytes() == expected.tobytes(), (task['name'], index, start)


def run_column(task, work):
    marker = work / (task['name'].replace('/', '__') + '.json')
    path = work / (task['name'].replace('/', '__') + '.h5')
    if marker.exists():
        return json.loads(marker.read_text())
    if path.exists():
        path.unlink()
    report = write_column(task, path)
    started = time.time()
    verify_column(task, path, report['raw_copy'])
    report['verify_seconds'] = time.time() - started
    report['status'] = 'PASS'
    marker.write_text(json.dumps(report) + '\n')
    print(json.dumps(report), flush=True)
    return report


def run_features(plan, final):
    marker = final.with_suffix('.json')
    if marker.exists() and final.exists():
        return json.loads(marker.read_text())
    started = time.time()
    partial = final.with_suffix('.h5.partial')
    if partial.exists():
        partial.unlink()
    parts = plan['parts']
    with h5py.File(parts[0]['features'], 'r') as first:
        mean = first['scaler/mean'][()]
        scale = first['scaler/scale'][()]
        width = first['ecumetric_features'].shape[1]
    with h5py.File(partial, 'x') as out:
        out.attrs.update(plan['features_attrs'])
        target = create(out, 'ecumetric_features', (plan['total']['rows'], width), np.float32, features=True)
        out.create_dataset('scaler/mean', data=mean)
        out.create_dataset('scaler/scale', data=scale)
        for index, part in enumerate(parts):
            with h5py.File(part['features'], 'r') as data:
                source = data['ecumetric_features']
                assert np.array_equal(data['scaler/mean'][()], mean) and data['scaler/scale'][()] == scale
                assert source.chunks == target.chunks and source.dtype == target.dtype
                assert (source.compression, source.compression_opts, source.shuffle, source.fletcher32) == ('gzip', 4, True, True)
                assert part['base']['rows'] % target.chunks[0] == 0
                for chunk_index in range(source.id.get_num_chunks()):
                    info = source.id.get_chunk_info(chunk_index)
                    mask, chunk = source.id.read_direct_chunk(info.chunk_offset)
                    target.id.write_direct_chunk((info.chunk_offset[0] + part['base']['rows'], 0), chunk, mask)
    written = time.time() - started
    with h5py.File(partial, 'r') as out:
        target = out['ecumetric_features']
        for part in parts:
            with h5py.File(part['features'], 'r') as data:
                source = data['ecumetric_features']
                for chunk_index in range(source.id.get_num_chunks()):
                    info = source.id.get_chunk_info(chunk_index)
                    _, chunk = source.id.read_direct_chunk(info.chunk_offset)
                    _, copied = target.id.read_direct_chunk((info.chunk_offset[0] + part['base']['rows'], 0))
                    assert chunk == copied
        assert target.id.get_num_chunks() == -(-target.shape[0] // target.chunks[0])
    partial.rename(final)
    report = dict(name=final.name, status='PASS', seconds=written, verify_seconds=time.time() - started - written)
    marker.write_text(json.dumps(report) + '\n')
    print(json.dumps(report), flush=True)
    return report


# chunks are visited by their position, a lookup by chunk number gets slow on millions of chunks
def compare_copied_column(source_path, target_path, name):
    with h5py.File(source_path, 'r') as source, h5py.File(target_path, 'r') as target:
        original, copied = source[name], target[name]
        assert copied.shape == original.shape and copied.dtype == original.dtype and copied.chunks == original.chunks
        rows = original.chunks[0]
        grid = -(-original.shape[0] // rows)
        assert original.id.get_num_chunks() == copied.id.get_num_chunks() == grid, name
        tail = (0,) * (original.ndim - 1)
        for chunk in range(grid):
            offset = (chunk * rows,) + tail
            assert original.id.read_direct_chunk(offset) == copied.id.read_direct_chunk(offset), (name, chunk)
    return name


def assemble(plan, work, final, columns, small, workers):
    partial = final.with_suffix('.h5.partial')
    if partial.exists():
        partial.unlink()
    with h5py.File(partial, 'x') as out:
        out.attrs.update(plan['attrs'])
        for name in columns:
            source_path = work / (name.replace('/', '__') + '.h5')
            parent, leaf = name.rsplit('/', 1) if '/' in name else ('/', name)
            with h5py.File(source_path, 'r') as source:
                source.copy(source[name], out.require_group(parent), name=leaf)
        for name, values in small.items():
            parent, leaf = name.rsplit('/', 1)
            write_small(out.require_group(parent), leaf, values)

    source_paths = [work / (name.replace('/', '__') + '.h5') for name in columns]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        compared = list(pool.map(compare_copied_column, source_paths, [partial] * len(columns), columns))
    assert compared == columns

    with h5py.File(partial, 'r') as out:
        for name, values in small.items():
            assert np.array_equal(out[name][()], values), name
    partial.rename(final)


def cross_check_release(plan, final):
    cell_start, cell_stop = int(plan['attrs']['cell_start']), int(plan['attrs']['cell_stop'])
    member_start, member_stop = int(plan['attrs']['member_start']), int(plan['attrs']['member_stop'])
    centers = np.load(INDEX / 'center_offsets.npy', mmap_mode='r')
    center_ids = np.load(INDEX / 'center_configuration_id.npy', mmap_mode='r')
    fps = np.load(INDEX / 'fps_order_in_cell.npy', mmap_mode='r')
    epsilon = np.load(INDEX / 'epsilon_sq.npy', mmap_mode='r')
    with h5py.File(final, 'r') as data:
        assert int(centers[cell_start]) == member_start and int(centers[cell_stop]) == member_stop
        assert np.array_equal(data['index/cell_offsets'][()], (centers[cell_start:cell_stop + 1] - member_start).astype(np.uint64))
        assert np.array_equal(data['index/center_configuration_id'][()], center_ids[cell_start:cell_stop])
        step = 1 << 24
        for start in range(0, member_stop - member_start, step):
            stop = min(start + step, member_stop - member_start)
            assert np.array_equal(data['ordering/fps_order_in_cell'][start:stop], fps[member_start + start:member_start + stop])
            assert data['ordering/epsilon_sq'][start:stop].tobytes() == np.asarray(epsilon[member_start + start:member_start + stop]).tobytes()


def file_hashes(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(64 << 20), b''):
            sha.update(block)
            md5.update(block)
    return dict(bytes=os.path.getsize(path), sha256=sha.hexdigest(), md5=md5.hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('interval')
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    started = time.time()

    work = ROOT / 'merge_work' / args.interval
    columns_dir = work / 'columns'
    columns_dir.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'internal' / 'merge_reports' / f'{args.interval}_merge.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    plan = load_plan(args.interval)
    dictionaries = merged_dictionaries(plan)
    small = small_datasets(plan)
    small['dictionaries/dataset_id'] = dictionaries['dataset_id']['merged']
    small['dictionaries/dataset_group'] = dictionaries['dataset_group']['merged']
    names = plan['parts'][0]['names']
    columns = [name for name in names if not name.startswith(SMALL)]
    assert set(columns) | set(small) == set(names) - {'index/global_cell_offsets'}
    print(json.dumps(dict(interval=args.interval, parts=len(plan['parts']), structures=plan['total']['rows'],
                          atoms=plan['total']['atoms'], columns=len(columns))), flush=True)

    label = f'madcosmo_cells_{args.interval}'
    data_final = work / f'{label}_data.h5'
    features_final = work / f'{label}_features.h5'
    tasks = sorted((column_task(plan, name, dictionaries) for name in columns),
                   key=lambda task: -task['total'] * np.dtype(task['dtype']).itemsize * int(np.prod(task['tail'])))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        features_future = pool.submit(run_features, plan, features_final)
        column_futures = [pool.submit(run_column, task, columns_dir) for task in tasks]
        column_reports = [future.result() for future in column_futures]
        features_report = features_future.result()

    assemble(plan, columns_dir, data_final, columns, small, args.workers)
    cross_check_release(plan, data_final)

    with ProcessPoolExecutor(max_workers=2) as pool:
        hashes = dict(zip((data_final.name, features_final.name), pool.map(file_hashes, (data_final, features_final))))

    report = dict(status='PASS', interval=args.interval, parts_merged=[Path(part['data']).name for part in plan['parts']],
                  structures=plan['total']['rows'], atoms=plan['total']['atoms'], files=hashes,
                  columns=column_reports, features=features_report, seconds=time.time() - started,
                  checks='every column compared with the parts it came from, chunk copies compared byte for byte, '
                         'codes compared by name, offsets by differences, cell index and FPS ordering compared with the aa002 release index')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    shutil.rmtree(columns_dir)
    print(json.dumps(dict(status='PASS', interval=args.interval, files=hashes, seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
