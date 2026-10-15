import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock


RECORD = '${MC_DRAFT_MADCOSMO}'
BASE = f'https://archive.materialscloud.org/api/records/{RECORD}/draft'
FILES = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1/uploaded/files')
STATE = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1/uploaded/internal/reupload_state.json')
TOKEN = Path('${HOME}/.mcloud_token')

state_lock = Lock()


def call(method, url, token, data=None, length=None):
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    if length is not None:
        headers.update({'Content-Type': 'application/octet-stream', 'Content-Length': str(length)})
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=3600 if length is not None else 120) as response:
            payload = response.read()
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'{method} {url}: HTTP {error.code}: {error.read(2000).decode(errors="replace")}') from None


def md5sum(path):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def remote_files(token):
    return {entry['key']: entry for entry in call('GET', BASE + '/files', token).get('entries', [])}


def matches(entry, size, md5):
    return entry.get('status') == 'completed' and entry.get('size') == size and entry.get('checksum') == 'md5:' + md5


class ProgressReader:
    def __init__(self, stream, name):
        self.stream = stream
        self.name = name
        self.total = 0
        self.next_report = 256 << 20

    def read(self, size=-1):
        block = self.stream.read(size)
        self.total += len(block)
        if self.total >= self.next_report:
            print(f'uploading {self.name}: {self.total / 1e9:.2f} GB sent', flush=True)
            self.next_report += 256 << 20
        return block


def save_state(state):
    with state_lock:
        temporary = STATE.with_suffix('.json.partial')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(temporary, STATE)


# a dropped connection should not abort the run, retry the file from scratch a few times
def upload(name, size, md5, token, state):
    for attempt in range(1, 4):
        try:
            return upload_once(name, size, md5, token, state)
        except Exception as error:
            print(f'attempt {attempt} for {name} failed: {error!r}'[:400], flush=True)
            entry = recover(name, size, md5, token)
            if entry is not None:
                state['files'][name] = dict(status='verified', bytes=size, md5=md5, remote_created=entry['created'])
                save_state(state)
                print(f'recovered {name}: the server finished the file after the error', flush=True)
                return
            if attempt == 3:
                raise
            if name in remote_files(token):
                call('DELETE', BASE + '/files/' + urllib.parse.quote(name, safe=''), token)


# after a gateway timeout the body is usually fully sent, give the server a few minutes to finish the file
def recover(name, size, md5, token):
    url = BASE + '/files/' + urllib.parse.quote(name, safe='')
    for _ in range(10):
        time.sleep(30)
        entry = remote_files(token).get(name)
        if entry is None:
            return None
        if matches(entry, size, md5):
            return entry
        if entry.get('status') == 'pending' and entry.get('size') == size and entry.get('checksum') == 'md5:' + md5:
            call('POST', url + '/commit', token)
    return None


def upload_once(name, size, md5, token, state):
    url = BASE + '/files/' + urllib.parse.quote(name, safe='')
    started = time.time()

    call('POST', BASE + '/files', token, json.dumps([{'key': name}]).encode())
    state['files'][name] = dict(status='uploading', bytes=size, md5=md5)
    save_state(state)

    with (FILES / name).open('rb') as stream:
        call('PUT', url + '/content', token, ProgressReader(stream, name), length=size)
    call('POST', url + '/commit', token)

    entry = remote_files(token)[name]
    if not matches(entry, size, md5):
        raise RuntimeError(f'remote verification failed for {name}: {entry}')
    state['files'][name] = dict(status='verified', bytes=size, md5=md5, remote_created=entry['created'])
    save_state(state)
    print(f'verified {name}: {size:,} bytes in {time.time() - started:.0f} s', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='without it only the plan is printed')
    parser.add_argument('--parallel', type=int, default=3)
    args = parser.parse_args()

    if TOKEN.stat().st_mode & 0o077:
        raise SystemExit('token permissions are not private')
    token = TOKEN.read_text().strip()

    draft = call('GET', BASE, token)
    if draft.get('is_published') is not False or draft.get('metadata', {}).get('title') != 'MAD-COSMO':
        raise SystemExit('destination is not the unpublished MAD-COSMO draft')

    local = {}
    for path in sorted(FILES.iterdir()):
        local[path.name] = (path.stat().st_size, md5sum(path))
    remote = remote_files(token)

    # delete every remote file that is not in the package or differs from it, upload what is missing
    delete = [name for name, entry in remote.items() if name not in local or not matches(entry, *local[name])]
    keep = [name for name in local if name in remote and matches(remote[name], *local[name])]
    put = [name for name in local if name not in keep]
    put.sort(key=lambda name: -local[name][0])

    print(json.dumps(dict(record=RECORD, remote_files=len(remote), remote_bytes=sum(e.get('size') or 0 for e in remote.values()),
                          delete=delete, keep=keep, upload=put,
                          upload_bytes=sum(local[name][0] for name in put)), indent=2), flush=True)
    if not args.apply:
        print('dry run, nothing changed', flush=True)
        return

    state = dict(record=RECORD, files={name: dict(status='kept', bytes=local[name][0], md5=local[name][1]) for name in keep})
    save_state(state)

    for name in delete:
        call('DELETE', BASE + '/files/' + urllib.parse.quote(name, safe=''), token)
        print(f'deleted remote {name}', flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [executor.submit(upload, name, *local[name], token, state) for name in put]
        for future in futures:
            future.result()

    remote = remote_files(token)
    assert set(remote) == set(local), (sorted(remote), sorted(local))
    for name, (size, md5) in local.items():
        assert matches(remote[name], size, md5), name
    assert call('GET', BASE, token).get('is_published') is False

    state['status'] = 'complete'
    state['record_is_published'] = False
    state['remote_bytes'] = sum(entry['size'] for entry in remote.values())
    save_state(state)
    print(json.dumps(dict(status='complete', record=RECORD, files=len(remote), remote_bytes=state['remote_bytes'])), flush=True)


if __name__ == '__main__':
    main()
