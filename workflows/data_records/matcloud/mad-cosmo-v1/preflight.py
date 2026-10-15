import argparse
import hashlib
import json
import mmap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path('${STORE_ROOT}/mad-core/release')
INDEX = ROOT / 'mad-core/cell_fps'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect(start, stop, workers=1, metadata=None, member_start=None, member_stop=None):
    root_manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    index_manifest = json.loads((INDEX / 'MANIFEST.json').read_text())
    assert sha256(INDEX / 'MANIFEST.json') == root_manifest['directories']['mad-core/cell_fps']
    for name in ('center_offsets.npy', 'release_directory_names.npy', 'release_manifest_sha256.npy'):
        assert sha256(INDEX / name) == index_manifest['files'][name]['sha256'], name
    offsets = np.load(INDEX / 'center_offsets.npy', mmap_mode='r')
    assert 0 <= start < stop < len(offsets)
    lo, hi = map(int, offsets[[start, stop]])
    if member_start is not None or member_stop is not None:
        assert member_start is not None and member_stop is not None
        assert lo <= member_start < member_stop <= hi
        lo, hi = member_start, member_stop
    directories = np.load(INDEX / 'release_directory.npy', mmap_mode='r')[lo:hi]
    rows = np.load(INDEX / 'release_row_in_directory.npy', mmap_mode='r')[lo:hi]
    names = np.load(INDEX / 'release_directory_names.npy')
    hashes = np.load(INDEX / 'release_manifest_sha256.npy')
    report = dict(cell_start=start, cell_stop=stop, member_start=lo, member_stop=hi,
                  structures=hi-lo, root_manifest_sha256=sha256(ROOT / 'MANIFEST.json'),
                  index_manifest_sha256=sha256(INDEX / 'MANIFEST.json'), sources=[])
    counts = np.empty(hi-lo, dtype=np.int64)
    refs = np.zeros(hi-lo, dtype=np.int64)
    def inspect_source(code):
        name = bytes(names[code]).decode()
        directory = ROOT / name
        digest = sha256(directory / 'MANIFEST.json')
        assert digest == bytes(hashes[code]).decode() == root_manifest['directories'][name], name
        manifest = json.loads((directory / 'MANIFEST.json').read_text())
        for filename, detail in manifest['files'].items():
            assert (directory / filename).stat().st_size == detail['bytes'], (name, filename)
        positions = np.flatnonzero(directories == code)
        positions = positions[np.argsort(rows[positions], kind='stable')]
        selected = np.asarray(rows[positions], dtype=np.int64)
        assert selected.min() >= 0 and selected.max() < manifest['ns']
        assert len(np.unique(selected)) == len(selected), 'Repeated source occurrence'
        atom_offsets = np.load(directory / 'na.npy', mmap_mode='r')
        if workers > 1:
            atom_offsets._mmap.madvise(mmap.MADV_RANDOM)
        counts[positions] = atom_offsets[selected+1] - atom_offsets[selected]
        assert np.all(counts[positions] > 0)
        if (directory / 'source_reference_offsets.npy').exists():
            ref_offsets = np.load(directory / 'source_reference_offsets.npy', mmap_mode='r')
            if workers > 1:
                ref_offsets._mmap.madvise(mmap.MADV_RANDOM)
            refs[positions] = ref_offsets[selected+1] - ref_offsets[selected]
        image_count = 0
        if (directory / 'image_flag_rows.npy').exists():
            image_rows = np.load(directory / 'image_flag_rows.npy')
            image_count = int(np.isin(selected, image_rows).sum())
        item = dict(code=int(code), directory=name, structures=len(selected),
                    atoms=int(counts[positions].sum()), image_flag_structures=image_count,
                    manifest_sha256=digest)
        print(json.dumps(item), flush=True)
        return item

    with ThreadPoolExecutor(max_workers=workers) as executor:
        report['sources'] = list(executor.map(inspect_source, np.unique(directories)))
    report.update(atoms=int(counts.sum()), max_atoms=int(counts.max()),
                  source_reference_bytes=int(refs.sum()), source_groups=len(report['sources']),
                  raw_feature_bytes=(hi-lo)*512*4,
                  image_flag_structures=sum(s['image_flag_structures'] for s in report['sources']))
    if metadata is not None:
        metadata.mkdir(parents=True, exist_ok=False)
        np.save(metadata / 'record_id.npy', (directories.astype(np.uint64) << np.uint64(32)) | rows.astype(np.uint64))
        np.save(metadata / 'geometry__atom_offsets.npy', np.r_[np.uint64(0), np.cumsum(counts, dtype=np.uint64)])
        np.save(metadata / 'identity__source_reference_offsets.npy', np.r_[np.uint64(0), np.cumsum(refs, dtype=np.uint64)])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell-start', type=int, default=0)
    parser.add_argument('--cell-stop', type=int, default=2**18)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--metadata', type=Path)
    args = parser.parse_args()
    started = time.time()
    assert 1 <= args.workers <= 32
    report = inspect(args.cell_start, args.cell_stop, args.workers, args.metadata)
    report['seconds'] = time.time() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
