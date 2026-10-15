import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


BASE = 'https://archive.materialscloud.org/api/records/${MC_DRAFT_MADCOSMO}/draft'


def call(method, url, token, data=None, length=None):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    if length is not None:
        headers.update({'Content-Type': 'application/octet-stream', 'Content-Length': str(length)})
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=3600 if length is not None else 60) as response:
            payload = response.read()
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'{method} {url}: HTTP {error.code}: {error.read(2000).decode(errors="replace")}') from None


def hashes(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def approved_names(approval, manifest):
    if approval.get('record') != '${MC_DRAFT_MADCOSMO}':
        raise ValueError('Approval must identify the MAD-COSMO draft')
    approved = approval.get('files')
    if not isinstance(approved, dict) or not approved:
        raise ValueError('An explicitly approved, nonempty file list is required')
    known = set(manifest['files']) | {'madcosmo_manifest.json'}
    for name, detail in approved.items():
        if Path(name).name != name or name not in known:
            raise ValueError('File is not in the reviewed package: ' + name)
        if not isinstance(detail, dict) or set(detail) != {'bytes', 'sha256'}:
            raise ValueError('Approval must pin size and SHA-256: ' + name)
        if not isinstance(detail['bytes'], int) or detail['bytes'] < 0:
            raise ValueError('Invalid approved size: ' + name)
        if not isinstance(detail['sha256'], str) or len(detail['sha256']) != 64:
            raise ValueError('Invalid approved SHA-256: ' + name)
    return sorted(approved, key=lambda name: (0 if name.endswith('_data.h5') else 1 if name.endswith('_features.h5') else 2, name))


def projected_record_bytes(entries, local_files, limit):
    remote = {entry['key']: entry for entry in entries}
    total = 0
    for name, entry in remote.items():
        if entry.get('status') == 'completed':
            total += entry['size']
        elif name not in local_files:
            raise ValueError('Unreviewed pending file prevents a reliable capacity check: ' + name)
    for name, (sha, md5, size) in local_files.items():
        entry = remote.get(name)
        if entry is not None and entry.get('status') == 'completed':
            if entry.get('size') != size or entry.get('checksum') != 'md5:' + md5:
                raise ValueError('Refusing to replace an existing completed file: ' + name)
        else:
            total += size
    if total > limit:
        raise ValueError(f'Approved upload would reach {total / 2**30:.3f} GiB, above the {limit / 2**30:.3f} GiB cap')
    return total


class ProgressReader:
    def __init__(self, stream, name):
        self.stream = stream
        self.name = name
        self.total = 0
        self.next_report = 128 << 20

    def read(self, size=-1):
        block = self.stream.read(size)
        self.total += len(block)
        if self.total >= self.next_report:
            print(f'uploading {self.name}: {self.total / 1e9:.2f} GB sent', flush=True)
            self.next_report += 128 << 20
        return block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--approved-files', required=True, type=Path,
                        help='Exact record, filenames, sizes and SHA-256 hashes approved by the user before upload')
    parser.add_argument('--validation-report', required=True, type=Path,
                        help='Internal source-validation report; it is not added to the upload list')
    parser.add_argument('--max-record-gib', type=float, default=50,
                        help='Local safety cap for the entire draft; raise only after the user confirms increased capacity')
    parser.add_argument('--recover-pending')
    parser.add_argument('--recover-created')
    args = parser.parse_args()
    if not 0 < args.max_record_gib < float('inf'):
        parser.error('--max-record-gib must be a positive finite value')
    approval = json.loads(args.approved_files.read_text())
    manifest = json.loads((args.files / 'madcosmo_manifest.json').read_text())
    names = approved_names(approval, manifest)
    token_path = Path('${HOME}/.mcloud_token')
    if token_path.stat().st_mode & 0o077:
        raise SystemExit('Token permissions are not private')
    token = token_path.read_text().strip()
    draft = call('GET', BASE, token)
    if draft.get('is_published') is not False or draft.get('metadata', {}).get('title') != 'MAD-COSMO':
        raise SystemExit('Refusing to upload: destination is not the unpublished MAD-COSMO draft')
    validation = json.loads(args.validation_report.read_text())
    assert validation['status'] == 'PASS'
    state = json.loads(args.state.read_text()) if args.state.exists() else {'record': '${MC_DRAFT_MADCOSMO}', 'files': {}}
    assert state['record'] == '${MC_DRAFT_MADCOSMO}'
    local_files = {}
    for name in names:
        if Path(name).name != name:
            raise ValueError(name)
        path = args.files / name
        sha, md5 = hashes(path)
        size = path.stat().st_size
        if approval['files'][name] != {'bytes': size, 'sha256': sha}:
            raise SystemExit('File changed after approval; ask for approval again: ' + name)
        if name != 'madcosmo_manifest.json':
            assert manifest['files'][name] == {'bytes': size, 'sha256': sha}, name
        if name.endswith('.h5'):
            assert validation['files'][name] == {'bytes': size, 'sha256': sha}, name
        local_files[name] = (sha, md5, size)
    state['last_approved_files'] = approval['files']
    projected = projected_record_bytes(call('GET', BASE + '/files', token).get('entries', []),
                                       local_files, int(args.max_record_gib * 2**30))
    print(json.dumps({'preflight': 'PASS', 'files': len(names), 'bytes': sum(detail[2] for detail in local_files.values()),
                      'projected_record_gib': projected / 2**30, 'record_cap_gib': args.max_record_gib}), flush=True)
    for name in names:
        path = args.files / name
        sha, md5, size = local_files[name]
        remote = {entry['key']: entry for entry in call('GET', BASE + '/files', token).get('entries', [])}
        projected_record_bytes(list(remote.values()), local_files, int(args.max_record_gib * 2**30))
        existing = remote.get(name)
        if existing and existing.get('status') == 'completed':
            if existing.get('size') != size or existing.get('checksum') != 'md5:' + md5:
                raise SystemExit(f'Refusing to overwrite existing completed file: {name}')
            state['files'][name] = {'status': 'verified', 'bytes': size, 'sha256': sha, 'md5': md5,
                                    'remote_id': existing.get('file_id'), 'remote_created': existing['created']}
            args.state.write_text(json.dumps(state, indent=2) + '\n')
            print(f'already verified: {name}', flush=True)
            continue
        if existing:
            prior = state['files'].get(name, {})
            recognized = prior.get('remote_created') == existing.get('created') and prior.get('sha256') == sha
            if not recognized:
                recover = (name == args.recover_pending and existing.get('created') == args.recover_created
                           and existing.get('status') == 'pending' and existing.get('size', 0) == 0
                           and not existing.get('checksum'))
                if not recover:
                    raise SystemExit(f'Refusing to touch an unrecognized pending file: {name}')
                print(f'Recovering the explicitly identified empty pending entry: {name}', flush=True)
        else:
            call('POST', BASE + '/files', token, json.dumps([{'key': name}]).encode())
            remote = {entry['key']: entry for entry in call('GET', BASE + '/files', token).get('entries', [])}
            existing = remote[name]
        state['files'][name] = {'status': 'uploading', 'bytes': size, 'sha256': sha, 'md5': md5,
                                'remote_created': existing['created']}
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(state, indent=2) + '\n')
        url = BASE + '/files/' + urllib.parse.quote(name, safe='')
        started = time.time()
        with path.open('rb') as stream:
            call('PUT', url + '/content', token, ProgressReader(stream, name), length=size)
        call('POST', url + '/commit', token)
        remote = {entry['key']: entry for entry in call('GET', BASE + '/files', token).get('entries', [])}
        entry = remote[name]
        if entry.get('status') != 'completed' or entry.get('size') != size or entry.get('checksum') != 'md5:' + md5:
            raise RuntimeError(f'Remote verification failed: {name}')
        state['files'][name]['status'] = 'verified'
        args.state.write_text(json.dumps(state, indent=2) + '\n')
        print(f'verified remote: {name}, {size} bytes, {time.time()-started:.1f} seconds', flush=True)
    draft = call('GET', BASE, token)
    assert draft.get('is_published') is False
    remote = {entry['key']: entry for entry in call('GET', BASE + '/files', token).get('entries', [])}
    for name, (sha, md5, size) in local_files.items():
        entry = remote[name]
        if entry.get('status') != 'completed' or entry.get('checksum') != 'md5:' + md5 or entry.get('size') != size:
            raise RuntimeError('Final remote inventory mismatch: ' + name)
    state['status'] = 'complete'
    state['record_is_published'] = False
    args.state.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'record': state['record'], 'verified_files': len(names), 'published': False}), flush=True)


if __name__ == '__main__':
    main()
