import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


SCHEMA = 'madcosmo-hdf5-public-v1'
BLOCK_BYTES = 16 << 20

DROP_DATA = (
    'record_id',
    'ordering/cell_id',
    'diagnostics/collection_flags_present',
    'identity/source_reference_present',
    'identity/configuration_id_bytes',
    'identity/configuration_id_offsets',
    'provenance/source_directory_code',
    'provenance/source_row',
    'dictionaries/source_directory',
    'dictionaries/source_manifest_sha256',
    'mad_next/present',
    'mad_next/fps_order',
    'mad_next/epsilon_sq',
    'mad_next/k_set',
    'mad_next/split',
)
DROP_FEATURES = ('record_id',)
DROP_ATTRS = ('record_id_encoding', 'root_manifest_sha256', 'index_manifest_sha256')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dataset_names(handle):
    names = []
    handle.visititems(lambda name, item: names.append(name) if isinstance(item, h5py.Dataset) else None)
    return sorted(names)


def block_rows(dataset):
    row_bytes = dataset.dtype.itemsize * int(np.prod(dataset.shape[1:]))
    return max(1, BLOCK_BYTES // row_bytes)


# copy one dataset in blocks with the same chunking and compression, values replaces the content when given
def copy_dataset(source, target, name, values=None):
    dataset = source[name]

    if dataset.chunks is None:
        target.create_dataset(name, data=dataset[()] if values is None else values)
        return

    out = target.create_dataset(name, shape=dataset.shape, dtype=dataset.dtype, chunks=dataset.chunks,
                                compression=dataset.compression, compression_opts=dataset.compression_opts,
                                shuffle=dataset.shuffle, fletcher32=dataset.fletcher32)
    step = block_rows(dataset)
    for start in range(0, len(dataset), step):
        if values is None:
            out[start:start + step] = dataset[start:start + step]
        else:
            out[start:start + step] = values[start:start + step]


def publish(source_path, output_path, drop, replacements, additions):
    with h5py.File(source_path, 'r') as source, h5py.File(output_path, 'x') as target:
        for key, value in source.attrs.items():
            if key not in DROP_ATTRS:
                target.attrs[key] = value
        target.attrs['schema'] = SCHEMA

        for name in dataset_names(source):
            if name not in drop:
                copy_dataset(source, target, name, replacements.get(name))

        for name, values in additions.items():
            target.create_dataset(name, data=values, chunks=(min(len(values), max(1, (256 << 10) // values.dtype.itemsize)),),
                                  compression='gzip', compression_opts=4, shuffle=True, fletcher32=True)


def verify(source_path, output_path, drop, replacements, additions):
    with h5py.File(source_path, 'r') as source, h5py.File(output_path, 'r') as target:
        kept = [name for name in dataset_names(source) if name not in drop]
        assert dataset_names(target) == sorted(kept + list(additions)), 'dataset list differs'

        for name, values in additions.items():
            copy = target[name]
            assert copy.shape == values.shape and copy.dtype == values.dtype, name
            assert copy[()].tobytes() == values.tobytes(), name

        for key, value in source.attrs.items():
            if key in DROP_ATTRS:
                assert key not in target.attrs, key
            elif key == 'schema':
                assert target.attrs[key] == SCHEMA
            else:
                assert target.attrs[key] == value, key

        for name in kept:
            original = source[name]
            copy = target[name]
            assert original.shape == copy.shape and original.dtype == copy.dtype, name
            assert len(copy.attrs) == 0, name

            if original.chunks is None:
                expected = original[()] if name not in replacements else replacements[name]
                assert np.asarray(copy[()]).tobytes() == np.asarray(expected).tobytes(), name
                continue

            step = block_rows(original)
            for start in range(0, len(original), step):
                if name in replacements:
                    expected = replacements[name][start:start + step]
                else:
                    expected = original[start:start + step]
                assert copy[start:start + step].tobytes() == expected.tobytes(), (name, start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('features', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replace-into', type=Path, help='after a passing verification move the two files into this directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # MAD-NEXT structures were never filtered, in the published files they count as passed
    with h5py.File(args.data, 'r') as data:
        mad_next = data['mad_next/present'][:]
        filter_code = data['diagnostics/filter_code'][:]
        assert np.all(filter_code[mad_next] == -1)
        offsets = data['identity/configuration_id_offsets'][:].astype(np.int64)
        raw = data['identity/configuration_id_bytes'][:]
    replacements = {'diagnostics/filter_code': np.where(mad_next, 0, filter_code).astype(filter_code.dtype)}

    # the configuration id becomes one fixed width string column instead of bytes plus offsets
    lengths = np.diff(offsets)
    assert lengths.min() > 0
    configuration_id = np.zeros(len(lengths), dtype=f'S{int(lengths.max())}')
    for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        configuration_id[index] = raw[start:stop].tobytes()
    additions = {'identity/configuration_id': configuration_id}

    data_out = args.output / args.data.name
    features_out = args.output / args.features.name
    publish(args.data, data_out, DROP_DATA, replacements, additions)
    publish(args.features, features_out, DROP_FEATURES, {}, {})
    verify(args.data, data_out, DROP_DATA, replacements, additions)
    verify(args.features, features_out, DROP_FEATURES, {}, {})

    report = dict(status='PASS', schema=SCHEMA,
                  source={name: str(path) for name, path in (('data', args.data), ('features', args.features))},
                  dropped=dict(data=list(DROP_DATA), features=list(DROP_FEATURES), attributes=list(DROP_ATTRS)),
                  mad_next_rows_set_to_filter_code_0=int(mad_next.sum()), structures=int(len(filter_code)),
                  files={path.name: dict(bytes=path.stat().st_size, sha256=sha256(path)) for path in (data_out, features_out)})
    if args.replace_into is not None:
        for path in (data_out, features_out):
            os.replace(path, args.replace_into / path.name)
        report['replaced_into'] = str(args.replace_into)

    label = args.data.name.removesuffix('_data.h5')
    (args.output / (label + '_publish.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
