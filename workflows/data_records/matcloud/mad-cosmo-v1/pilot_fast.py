import argparse
import json
import os
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

from block_reader import BlockReader
from export import encode, stage
from preflight import INDEX, ROOT, inspect, sha256
from ram_checkpoint import RamCheckpoint, durable_json
from validate import validate


SCRIPTS = Path(__file__).resolve().parent
RUN = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1')
WORK = RUN / 'pilot-fast-v1'
BASELINE = RUN / 'remaining-v2'


def event(name, **values):
    record = dict(phase=name, time=time.time(), job_id=os.environ.get('SLURM_JOB_ID'),
                  host=socket.gethostname(), **values)
    durable_json(WORK / 'status.json', record)
    print(json.dumps(record), flush=True)


def compare_encoded(files, output, golden_build):
    golden = json.loads(golden_build.read_text())
    assert set(files) == set(golden['files'])
    for name, detail in files.items():
        assert detail['decoded_fields'] == golden['files'][name]['decoded_fields'], name
        with h5py.File(output / name, 'r') as actual, h5py.File(golden_build.parent / 'files' / name, 'r') as expected:
            assert set(actual.attrs) == set(expected.attrs)
            for key in actual.attrs:
                assert np.array_equal(actual.attrs[key], expected.attrs[key]), (name, key)
            for field in detail['decoded_fields']:
                assert actual[field].shape == expected[field].shape, (name, field)
                assert actual[field].dtype == expected[field].dtype, (name, field)


def smoke(binding):
    root = WORK / 'smoke'
    root.mkdir()
    report = inspect(0, 1, workers=8, metadata=root / 'metadata')
    checkpoint = RamCheckpoint(root / 'checkpoints', binding, interval=0)
    interrupted = False

    def interrupt(staging, progress, force=False):
        if any(progress['source_positions'].values()):
            checkpoint(staging, progress, force=True)
            raise InterruptedError('intentional interruption after durable batch checkpoint')

    with tempfile.TemporaryDirectory(prefix='madcosmo-fast-interrupt-', dir='/tmp') as temporary, BlockReader() as reader:
        try:
            stage(report, Path(temporary) / 'staging', workers=8, metadata_staging=root / 'metadata',
                  reader=reader, checkpoint_hook=interrupt)
        except InterruptedError:
            interrupted = True
    assert interrupted
    with tempfile.TemporaryDirectory(prefix='madcosmo-fast-resume-', dir='/tmp') as temporary, BlockReader() as reader:
        staging = Path(temporary) / 'staging'
        checkpoint.restore(staging)
        state = stage(report, staging, workers=8, metadata_staging=root / 'metadata', resume=True, reader=reader)
        files = encode(state, staging, root / 'hdf5')
        compare_encoded(files, root / 'hdf5', RUN / 'pilot/madcosmo_cells_0-1_build.json')
        manifest = root / 'build.json'
        durable_json(manifest, dict(range=report, files=files))
        result = validate(staging, root / 'hdf5', manifest, workers=8, reader=reader)
        assert result['status'] == 'PASS' and result['image_flag_structures'] > 60000
        durable_json(root / 'validation.json', result)
        subprocess.run([sys.executable, str(BASELINE / 'code/validate.py'), '--staging', str(staging),
                        '--output', str(root / 'hdf5'), '--manifest', str(manifest), '--report',
                        str(root / 'legacy-validation.json'), '--workers', '8'], check=True)
    durable_json(WORK / 'SMOKE_PASS.json', dict(binding=binding, status='PASS',
                 checks=['restart from durable batch cursor into new RAM directory',
                         'all decoded field hashes and HDF5 metadata match golden cell',
                         'new full validator', 'frozen original full validator', 'image sidecars']))


def run(plan, resume=False):
    binding = dict(plan_sha256=sha256(WORK / 'plan.json'))
    for name, expected in plan['code_sha256'].items():
        assert sha256(SCRIPTS / name) == expected, name
    assert sha256(ROOT / 'MANIFEST.json') == plan['root_manifest_sha256']
    assert sha256(INDEX / 'MANIFEST.json') == plan['index_manifest_sha256']
    golden = Path(plan['golden_build'])
    assert sha256(golden) == plan['golden_build_sha256']
    assert not (WORK / 'PILOT_PASS.json').exists(), 'Pilot already passed'
    started = time.monotonic()
    if resume:
        previous = json.loads((WORK / 'SMOKE_PASS.json').read_text())
        assert previous['status'] == 'PASS' and previous['binding'] == binding
    else:
        event('smoke')
        smoke(binding)
    event('smoke_passed', seconds=time.monotonic() - started)
    part = plan['part']
    root = WORK / 'full'
    if resume:
        report = json.loads((root / 'preflight.json').read_text())
        for key, value in part.items():
            if key != 'label':
                assert report[key] == value, key
    else:
        root.mkdir()
        report = inspect(part['cell_start'], part['cell_stop'], workers=8, metadata=root / 'metadata',
                         member_start=part['member_start'], member_stop=part['member_stop'])
        report.update({key: part[key] for key in ('interval_cell_start', 'interval_cell_stop', 'part_number', 'part_count')})
        durable_json(root / 'preflight.json', report)
    reference_staging = golden.parent / 'staging'
    required = sum(path.stat().st_size for path in reference_staging.glob('*.npy'))
    assert required < 64 << 30, 'Pilot RAM budget exceeded'
    assert shutil.disk_usage('/tmp').free > required * 2
    checkpoint = RamCheckpoint(root / 'checkpoints', binding, interval=300)
    with tempfile.TemporaryDirectory(prefix='madcosmo-fast-full-', dir='/tmp') as temporary, BlockReader() as reader:
        staging = Path(temporary) / 'staging'
        if resume:
            restored = checkpoint.restore(staging)
            assert restored['report'] == report
        phase_start = time.monotonic()
        event('copying', structures=report['structures'], reference_staging_bytes=required)
        state = stage(report, staging, workers=8, metadata_staging=root / 'metadata',
                      reader=reader, checkpoint_hook=checkpoint, resume=resume)
        copy_seconds = time.monotonic() - phase_start
        event('encoding', copy_seconds=copy_seconds)
        phase_start = time.monotonic()
        if resume:
            for name in ('hdf5', 'validation-checkpoints'):
                path = root / name
                if path.exists():
                    path.rename(root / (name + '-preserved-' + str(time.time_ns())))
        files = encode(state, staging, root / 'hdf5')
        encode_seconds = time.monotonic() - phase_start
        compare_encoded(files, root / 'hdf5', golden)
        manifest = root / 'build.json'
        durable_json(manifest, dict(range=report, files=files))
        event('validating', copy_seconds=copy_seconds, encode_seconds=encode_seconds)
        phase_start = time.monotonic()
        result = validate(staging, root / 'hdf5', manifest, workers=8,
                          checkpoint_dir=root / 'validation-checkpoints', source_chunk_rows=65536, reader=reader)
        validation_seconds = time.monotonic() - phase_start
        assert result['status'] == 'PASS'
        durable_json(root / 'validation.json', result)
        names = sorted(files)
        subprocess.run([sys.executable, str(SCRIPTS / 'test_reader.py'),
                        str(root / 'hdf5' / names[0]), str(root / 'hdf5' / names[1])], check=True)
    result = dict(status='PASS', structures=report['structures'], binding=binding, files=files,
                  copy_seconds=copy_seconds, encode_seconds=encode_seconds, validation_seconds=validation_seconds,
                  total_seconds=time.monotonic() - started, maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  golden_decoded_fields_match=True, production_released=False, uploaded=False)
    durable_json(WORK / 'PILOT_PASS.json', result)
    event('passed', **result)


def prepare():
    original = json.loads((BASELINE / 'plan.json').read_text())
    part = original['parts'][0]
    golden = BASELINE / part['label'] / (part['label'] + '_build.json')
    WORK.mkdir(exist_ok=False)
    snapshot = WORK / 'code'
    snapshot.mkdir()
    names = ['pilot_fast.py', 'block_reader.py', 'ram_checkpoint.py', 'export.py', 'validate.py',
             'preflight.py', 'madcosmo.py', 'test_reader.py', 'test_fast_io.py']
    for name in names:
        shutil.copyfile(SCRIPTS / name, snapshot / name)
    shutil.copyfile(SCRIPTS / 'slurm/fast/submit.sh', snapshot / 'submit.sh')
    durable_json(WORK / 'plan.json', dict(part=part, root_manifest_sha256=original['root_manifest_sha256'],
                 index_manifest_sha256=original['index_manifest_sha256'], golden_build=str(golden),
                 golden_build_sha256=sha256(golden), maximum_node_hours=2, automatic_retry=False,
                 code_sha256={path.name: sha256(path) for path in snapshot.iterdir()}))
    print(str(WORK / 'plan.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    try:
        run(json.loads((WORK / 'plan.json').read_text()), resume=args.resume)
    except BaseException as error:
        event('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
