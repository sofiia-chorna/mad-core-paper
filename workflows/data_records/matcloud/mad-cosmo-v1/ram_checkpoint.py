import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def durable_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_checked(source, destination):
    digest = hashlib.sha256()
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        for block in iter(lambda: incoming.read(8 << 20), b''):
            digest.update(block)
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    return dict(bytes=destination.stat().st_size, sha256=digest.hexdigest())


class RamCheckpoint:
    def __init__(self, root, binding, interval=300):
        self.root = root
        self.binding = binding
        self.interval = interval
        self.last = time.monotonic()
        root.mkdir(parents=True, exist_ok=True)

    def __call__(self, staging, progress, force=False):
        if not force and time.monotonic() - self.last < self.interval:
            return False
        generation = self.root / ('generation-' + str(time.time_ns()))
        generation.mkdir()
        files = {}
        for path in sorted(staging.glob('*.npy')):
            files[path.name] = copy_checked(path, generation / path.name)
        durable_json(generation / 'checkpoint.json', dict(binding=self.binding, progress=progress, files=files))
        manifest_sha = hashlib.sha256((generation / 'checkpoint.json').read_bytes()).hexdigest()
        pointer = self.root / 'CURRENT.json'
        previous = json.loads(pointer.read_text()) if pointer.exists() else {}
        current = dict(generation=generation.name, manifest_sha256=manifest_sha)
        durable_json(pointer, dict(current=current, previous=previous.get('current')))
        old = previous.get('previous')
        if old:
            obsolete = self.root / old['generation']
            assert obsolete.parent == self.root and obsolete.name.startswith('generation-')
            shutil.rmtree(obsolete)
        self.last = time.monotonic()
        print(json.dumps(dict(checkpoint=generation.name, bytes=sum(row['bytes'] for row in files.values()))), flush=True)
        return True

    def restore(self, destination):
        pointer = json.loads((self.root / 'CURRENT.json').read_text())['current']
        generation = self.root / pointer['generation']
        assert generation.parent == self.root and generation.name.startswith('generation-')
        manifest = generation / 'checkpoint.json'
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == pointer['manifest_sha256']
        saved = json.loads(manifest.read_text())
        assert saved['binding'] == self.binding
        destination.mkdir(exist_ok=False)
        for name, expected in saved['files'].items():
            assert Path(name).name == name and name.endswith('.npy')
            assert copy_checked(generation / name, destination / name) == expected, name
        durable_json(destination / 'stage-progress.json', saved['progress'])
        return saved['progress']
