import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


UPLOAD = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1')
STORE = Path('${STORE_ROOT}/mad-core/matcloud/mad-cosmo-v1')
NEW = STORE / 'files_new'
MERGED_INTERVALS = ('2pow20-2pow21', '2pow21-2pow22', '2pow22-2pow23', '2pow23-2pow24')

# the pilot copy of 2026-10-03 that this copy replaces
PILOT_FILES = {'dataset_metadata.tsv', 'madcosmo_cells_0-2pow18_data.h5', 'madcosmo_cells_0-2pow18_features.h5',
               'madcosmo_dataset_counts.csv', 'MAD_COSMO_HDF5.md', 'madcosmo_manifest.json', 'madcosmo.py',
               'madcosmo_source_manifests.json', 'madcosmo_validation.json'}
PILOT_BUILD = {'madcosmo_cells_0-2pow18_build.json', 'preflight.json', 'validation.json'}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 26), b''):
            digest.update(block)
    return digest.hexdigest()


def copy_one(source, target, expected):
    started = time.time()
    if not (target.exists() and target.stat().st_size == source.stat().st_size):
        shutil.copyfile(source, target)
    copied = time.time() - started
    expected = expected or sha256_file(source)
    actual = sha256_file(target)
    print(f'{target.name}: {"ok" if actual == expected else "MISMATCH"}, copy {copied:.0f} s, total {time.time() - started:.0f} s', flush=True)
    return target.name, dict(bytes=target.stat().st_size, sha256=actual, expected=expected, ok=actual == expected)


def main():
    assert {path.name for path in (STORE / 'files').iterdir()} == PILOT_FILES
    assert {path.name for path in (STORE / 'build').iterdir()} == PILOT_BUILD
    NEW.mkdir(exist_ok=True)

    jobs = []
    for interval in MERGED_INTERVALS:
        report = json.loads((UPLOAD / 'to_upload' / 'internal' / 'merge_reports' / f'{interval}_merge.json').read_text())
        assert report['status'] == 'PASS'
        for name, entry in report['files'].items():
            source = UPLOAD / 'to_upload' / 'files' / name
            assert source.stat().st_size == entry['bytes'], name
            jobs.append((source, NEW / name, entry['sha256']))
    for source in sorted((UPLOAD / 'uploaded' / 'files').iterdir()):
        jobs.append((source, NEW / source.name, None))

    # the largest files first, so the long copies start at once
    jobs.sort(key=lambda job: -job[0].stat().st_size)
    with ProcessPoolExecutor(len(jobs)) as pool:
        results = dict(pool.map(copy_one, *zip(*jobs)))

    report = dict(files=results, all_ok=all(entry['ok'] for entry in results.values()),
                  bytes=sum(entry['bytes'] for entry in results.values()))
    (UPLOAD / 'to_upload' / 'internal' / 'store_copy_report.json').write_text(json.dumps(report, indent=1) + '\n')
    if not report['all_ok']:
        raise SystemExit('checksum mismatch, the pilot copy is left in place')

    # every copy matches, so the pilot copy goes and the new files take its place
    for name in PILOT_FILES:
        (STORE / 'files' / name).unlink()
    (STORE / 'files').rmdir()
    NEW.rename(STORE / 'files')
    for name in PILOT_BUILD:
        (STORE / 'build' / name).unlink()
    for source in sorted((UPLOAD / 'to_upload' / 'internal' / 'merge_reports').iterdir()):
        shutil.copyfile(source, STORE / 'build' / source.name)

    with open(STORE / 'SHA256SUMS', 'w') as stream:
        for name in sorted(results):
            stream.write(f'{results[name]["sha256"]}  files/{name}\n')
    print(f'store copy done: {len(results)} files, {report["bytes"] / 2 ** 40:.3f} TiB, pilot copy replaced', flush=True)


if __name__ == '__main__':
    main()
