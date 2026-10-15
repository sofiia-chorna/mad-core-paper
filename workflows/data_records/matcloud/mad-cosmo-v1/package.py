import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from preflight import INDEX, ROOT, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--staging', required=True, type=Path)
    args = parser.parse_args()
    output = args.build / 'files'
    internal = args.build / 'internal'
    internal.mkdir(parents=True, exist_ok=True)
    if (output / 'madcosmo_manifest.json').exists():
        raise FileExistsError('Package is already finalized')
    validation = json.loads((args.build / 'validation.json').read_text())
    if validation['status'] != 'PASS':
        raise SystemExit('Independent source validation has not passed')
    state = json.loads((args.staging / 'stage.json').read_text())
    report = state['report']
    for name, detail in validation['files'].items():
        assert (output / name).stat().st_size == detail['bytes']
        assert sha256(output / name) == detail['sha256']
    scripts = Path(__file__).resolve().parent
    for name in ('MAD_COSMO_HDF5.md',):
        destination = output / name
        if destination.exists():
            assert sha256(destination) == sha256(scripts / name), name
        else:
            shutil.copyfile(scripts / name, destination)
    for source, name in ((ROOT / 'dataset_metadata.tsv', 'dataset_metadata.tsv'),):
        if (output / name).exists():
            assert sha256(output / name) == sha256(source), name
        else:
            shutil.copyfile(source, output / name)
    with (ROOT / 'dataset_metadata.tsv').open(newline='') as stream:
        catalog = {row['dataset_id']: row for row in csv.DictReader(stream, delimiter='\t')}
    dictionary = np.load(args.staging / state['arrays']['dictionaries/dataset_id'])
    codes = np.load(args.staging / state['arrays']['identity/dataset_id_code'], mmap_mode='r')
    counts = np.bincount(codes, minlength=len(dictionary))
    with (internal / 'madcosmo_dataset_counts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset_id', 'metadata_dataset_id', 'exported_structures', 'license', 'upstream_url', 'source_publication'])
        writer.writeheader()
        for dataset, count in zip(dictionary, counts):
            name = bytes(dataset).decode()
            catalog_name = name
            if name not in catalog:
                family, separator, subset = name.partition(':')
                assert separator and subset and family in ('mad15', 'mad_next', 'mad_sol'), name
                catalog_name = {'mad15': 'MAD-1.5', 'mad_next': 'mad_next', 'mad_sol': 'mad_sol'}[family]
            metadata = catalog[catalog_name]
            writer.writerow(dict(dataset_id=name, metadata_dataset_id=catalog_name, exported_structures=int(count),
                                 **{key: metadata[key] for key in ('license', 'upstream_url', 'source_publication')}))
    source_manifests = dict(root=json.loads((ROOT / 'MANIFEST.json').read_text()),
                            cell_index=json.loads((INDEX / 'MANIFEST.json').read_text()),
                            exported_sources=state['source_manifests'])
    source_path = internal / 'madcosmo_source_manifests.json'
    if source_path.exists():
        assert json.loads(source_path.read_text()) == source_manifests
    else:
        with source_path.open('x') as stream:
            json.dump(source_manifests, stream, indent=2)
            stream.write('\n')
    files = {path.name: dict(bytes=path.stat().st_size, sha256=sha256(path)) for path in sorted(output.iterdir()) if path.is_file()}
    assert not any(name.endswith('.partial') for name in files)
    allowed = set(validation['files']) | {'MAD_COSMO_HDF5.md', 'dataset_metadata.tsv'}
    assert set(files) == allowed, 'Unexpected files in the public upload directory'
    interval = {key: report[key] for key in ('cell_start', 'cell_stop', 'member_start', 'member_stop', 'structures', 'atoms')}
    interval['data_file'] = next(name for name in validation['files'] if name.endswith('_data.h5'))
    interval['features_file'] = next(name for name in validation['files'] if name.endswith('_features.h5'))
    manifest = dict(schema='madcosmo-hdf5-v1', status='incremental export; not the complete collection',
                    total_release_structures=1694502355, total_release_cells=16777216,
                    root_manifest_sha256=report['root_manifest_sha256'],
                    index_manifest_sha256=report['index_manifest_sha256'],
                    available_intervals=[interval], files=files,
                    population='collection plus mad-next-coreset; all indexed occurrences retained',
                    ordering='MAD-CORE cell order, existing within-cell FPS order, unranked tail retained')
    with (output / 'madcosmo_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'files': len(files)+1, 'bytes': sum(item['bytes'] for item in files.values()) + (output / 'madcosmo_manifest.json').stat().st_size,
                      'interval': interval}, indent=2))


if __name__ == '__main__':
    main()
