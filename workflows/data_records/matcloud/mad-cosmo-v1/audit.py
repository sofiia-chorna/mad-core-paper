import csv
import json
from pathlib import Path

import h5py
import numpy as np

from preflight import INDEX, ROOT, sha256
from upload import BASE, call, hashes


WORKSPACE = Path('${DEV_ROOT}/phace-atomic')
RUN = WORKSPACE / 'matcloud-upload/mad-cosmo-v1'
STORE = Path('${STORE_ROOT}/mad-core/matcloud/mad-cosmo-v1')


def main():
    local_files = RUN / 'first/files'
    stored_files = STORE / 'files'
    internal = RUN / 'first/internal'
    archived_manifest = internal / 'uploaded_manifest_20261003.json'
    manifest_path = archived_manifest if archived_manifest.exists() else local_files / 'madcosmo_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    validation = json.loads((RUN / 'first/validation.json').read_text())
    assert validation['status'] == 'PASS'
    assert manifest['root_manifest_sha256'] == sha256(ROOT / 'MANIFEST.json')
    assert manifest['index_manifest_sha256'] == sha256(INDEX / 'MANIFEST.json')
    assert len(manifest['available_intervals']) == 1
    interval = manifest['available_intervals'][0]
    assert interval['cell_start'] == 0 and interval['cell_stop'] == 262144
    assert interval['structures'] == validation['structures'] == 2357504
    assert interval['atoms'] == validation['atoms'] == 53973063
    expected = dict(manifest['files'])
    expected['madcosmo_manifest.json'] = dict(bytes=manifest_path.stat().st_size, sha256=sha256(manifest_path))
    local_md5 = {}
    for name, detail in expected.items():
        assert Path(name).name == name
        local_path = local_files / name
        if archived_manifest.exists():
            if name == 'madcosmo_manifest.json':
                local_path = archived_manifest
            elif name == 'MAD_COSMO_HDF5.md':
                local_path = internal / 'uploaded_MAD_COSMO_HDF5_20261003.md'
            elif (internal / name).exists():
                local_path = internal / name
        for path in (local_path, stored_files / name):
            assert path.stat().st_size == detail['bytes'], str(path)
            digest, md5 = hashes(path)
            assert digest == detail['sha256'], str(path)
            local_md5[name] = md5
    source_offsets = np.load(INDEX / 'center_offsets.npy', mmap_mode='r')[:262145]
    with h5py.File(stored_files / interval['data_file'], 'r') as data, h5py.File(stored_files / interval['features_file'], 'r') as features:
        assert np.array_equal(data['index/cell_offsets'][:], source_offsets)
        assert data['record_id'].shape == features['record_id'].shape == (2357504,)
        assert features['ecumetric_features'].shape == (2357504,512)
        assert data['geometry/positions'].shape == data['predictions/ecumetric_nc_forces'].shape == (53973063,3)
        assert data['geometry/numbers'].shape == (53973063,)
        assert data['geometry/atom_offsets'][0] == 0
        assert data['geometry/atom_offsets'][-1] == 53973063
        assert data['assignments/center_index'].shape == data['assignments/distance'].shape == (2357504,25)
        for key in ('root_manifest_sha256', 'index_manifest_sha256', 'cell_start', 'cell_stop', 'member_start', 'member_stop'):
            assert data.attrs[key] == features.attrs[key]
        for start in range(0, 2357504, 65536):
            stop = min(start+65536, 2357504)
            record_ids = data['record_id'][start:stop]
            assert np.array_equal(record_ids, features['record_id'][start:stop])
            source_codes = data['provenance/source_directory_code'][start:stop].astype(np.uint64)
            source_rows = data['provenance/source_row'][start:stop].astype(np.uint64)
            assert np.array_equal(record_ids, (source_codes << np.uint64(32)) | source_rows)
        dictionary = {value.decode() for value in data['dictionaries/dataset_id'][:]}
    with (stored_files / 'madcosmo_dataset_counts.csv').open(newline='') as stream:
        counts = list(csv.DictReader(stream))
    assert sum(int(row['exported_structures']) for row in counts) == 2357504
    assert {row['dataset_id'] for row in counts} == dictionary
    token_path = Path('${HOME}/.mcloud_token')
    assert not token_path.stat().st_mode & 0o077
    token = token_path.read_text().strip()
    draft = call('GET', BASE, token)
    assert draft['is_published'] is False and draft['metadata']['title'] == 'MAD-COSMO'
    remote = {entry['key']: entry for entry in call('GET', BASE + '/files', token)['entries']}
    for name, detail in expected.items():
        entry = remote[name]
        assert entry['status'] == 'completed', name
        assert entry['size'] == detail['bytes'], name
        assert entry['checksum'] == 'md5:' + local_md5[name], name
    report = dict(status='PASS', structures=2357504, atoms=53973063, cell_start=0, cell_stop=262144,
                  verified_files=len(expected), verified_bytes=sum(item['bytes'] for item in expected.values()),
                  record='${MC_DRAFT_MADCOSMO}', record_is_published=False,
                  checks=['uploaded local artifacts (including archived metadata) and store copies match every uploaded-manifest SHA-256 and size',
                          'full data/features occurrence alignment and cell offsets verified',
                          'independent exhaustive source validation report checked',
                          'dataset count and attribution mapping checked',
                          'fresh remote inventory matches every local MD5 and size'])
    (RUN / 'completion_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
