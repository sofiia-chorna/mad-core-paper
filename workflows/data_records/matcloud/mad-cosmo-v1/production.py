import argparse
import fcntl
import json
import os
import resource
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np

from block_reader import BlockReader
from export import encode, report_label, stage
from parts import verify_parts
from preflight import INDEX, ROOT, inspect, sha256
from ram_checkpoint import RamCheckpoint, copy_checked, durable_json
from validate import validate


SCRIPTS = Path(__file__).resolve().parent
RUN = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1')
WORK = RUN / 'production-v2'
BASELINE = RUN / 'remaining-v2'
PILOT = RUN / 'pilot-fast-v1'


def read_json(path):
    return json.loads(path.read_text())


def check_part(report, part, plan):
    for key, value in part.items():
        if key != 'label':
            assert report[key] == value, key
    assert report_label(part) == part['label']
    for key in ('root_manifest_sha256', 'index_manifest_sha256'):
        assert report[key] == plan[key], key


def check_plan(work):
    plan = read_json(work / 'plan.json')
    assert sha256(BASELINE / 'plan.json') == plan['baseline_plan_sha256']
    assert read_json(BASELINE / 'plan.json')['parts'] == plan['parts']
    for name, digest in plan['code_sha256'].items():
        assert Path(name).name == name
        assert sha256(SCRIPTS / name) == digest, name
    for key, path in (('root_manifest_sha256', ROOT / 'MANIFEST.json'),
                      ('index_manifest_sha256', INDEX / 'MANIFEST.json'),
                      ('pilot_pass_sha256', PILOT / 'PILOT_PASS.json'),
                      ('pilot_plan_sha256', PILOT / 'plan.json'),
                      ('pilot_validation_sha256', PILOT / 'full/validation.json')):
        assert sha256(path) == plan[key], key
    if 'reuse' in plan:
        check_previous(plan)
    return plan


def check_previous(plan):
    reuse = plan['reuse']
    previous = Path(reuse['directory'])
    assert previous == RUN / 'production-v1'
    assert sha256(previous / 'plan.json') == reuse['plan_sha256']
    original = read_json(previous / 'plan.json')
    for key in ('parts', 'baseline_plan_sha256', 'root_manifest_sha256',
                'index_manifest_sha256', 'pilot_pass_sha256', 'pilot_plan_sha256',
                'pilot_validation_sha256', 'maximum_rows_per_part'):
        assert original[key] == plan[key], key
    for name, digest in original['code_sha256'].items():
        assert Path(name).name == name
        assert sha256(previous / 'code' / name) == digest, name
    return previous


def staging_bound(report):
    dataset_width = 1
    group_width = 1
    for source in report['sources']:
        directory = ROOT / source['directory']
        manifest = read_json(directory / 'MANIFEST.json')
        assert sha256(directory / 'MANIFEST.json') == source['manifest_sha256']
        dataset_width = max(dataset_width, np.dtype(manifest['files']['dataset_id.npy']['dtype']).itemsize)
        group_width = max(group_width, len(directory.name))
        if 'dataset_group.npy' in manifest['files']:
            group_width = max(group_width, np.dtype(manifest['files']['dataset_group.npy']['dtype']).itemsize)
    offsets = np.load(INDEX / 'configuration_id_offsets.npy', mmap_mode='r')
    id_bytes = int(offsets[report['member_stop']] - offsets[report['member_start']])
    center_ids = np.load(INDEX / 'center_configuration_id.npy', mmap_mode='r')
    cells = report['cell_stop'] - report['cell_start']
    atom_bytes = 49 if report['image_flag_structures'] else 25
    return (report['structures'] * (2700 + dataset_width + group_width)
            + report['atoms'] * atom_bytes + id_bytes + report['source_reference_bytes']
            + (cells + 1) * (16 + center_ids.dtype.itemsize)
            + 88 * report['image_flag_structures'] + (16 << 20))


def check_memory(report, limit, legacy_staging=None):
    required = staging_bound(report)
    if legacy_staging is not None:
        required = max(required, sum(path.stat().st_size for path in legacy_staging.glob('*.npy')))
    assert required <= limit, f'Staging bound {required} exceeds RAM staging cap {limit}'
    assert shutil.disk_usage('/tmp').free >= 2 * required, 'Insufficient node-local space'
    return required


def verify_ready(root, part, binding, hash_files=False):
    ready = read_json(root / 'READY.json')
    result = read_json(root / 'validation.json')
    build = read_json(root / 'build.json')
    assert ready['status'] == result['status'] == 'PASS'
    assert ready['binding'] == binding and ready['part'] == part
    assert ready['validation_sha256'] == sha256(root / 'validation.json')
    assert ready['build_sha256'] == sha256(root / 'build.json')
    assert ready['structures'] == result['structures'] == part['structures']
    expected = {name: {key: detail[key] for key in ('bytes', 'sha256')}
                for name, detail in build['files'].items()}
    assert ready['files'] == result['files'] == expected
    assert set(expected) == {part['label'] + '_data.h5', part['label'] + '_features.h5'}
    for name, detail in expected.items():
        assert Path(name).name == name
        path = root / 'files' / name
        assert path.stat().st_size == detail['bytes']
        if hash_files:
            assert sha256(path) == detail['sha256'], name
    return ready


def gate_indices(plan, index):
    if index in plan['canary_indices']:
        return []
    if index in plan['ramp_indices']:
        return plan['canary_indices']
    return plan['canary_indices'] + plan['ramp_indices']


def verify_gates(work, plan, indices):
    binding = dict(plan_sha256=sha256(work / 'plan.json'))
    for index in indices:
        part = plan['parts'][index]
        ready = verify_ready(work / part['label'], part, binding)
        if ready['mode'] == 'validated_reuse':
            previous = check_previous(plan) / part['label']
            digest = plan['reuse']['ready_sha256'][str(index)]
            assert sha256(previous / 'READY.json') == digest
            expected = dict(directory=str(previous), ready_sha256=digest,
                            plan_sha256=plan['reuse']['plan_sha256'])
            assert ready['previous_validation'] == expected
            ready = verify_ready(previous, part, dict(plan_sha256=expected['plan_sha256']))
        assert ready['mode'] == 'new_export', 'Scaling gate must measure a new full export'
        normalized = ready['seconds'] * plan['maximum_rows_per_part'] / part['structures']
        assert normalized <= plan['gate_seconds_per_full_part'], (index, normalized)


def prepare_gate(work, plan):
    assert not list((work / 'failures').glob('*.json'))
    previous = check_previous(plan)
    binding = dict(plan_sha256=sha256(work / 'plan.json'))
    original_binding = dict(plan_sha256=plan['reuse']['plan_sha256'])
    indices = plan['canary_indices'] + plan['ramp_indices']
    assert set(map(int, plan['reuse']['ready_sha256'])) == set(indices)
    with (work / 'gate.lock').open('a') as gate_lock:
        fcntl.flock(gate_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index in indices:
            part = plan['parts'][index]
            source = previous / part['label']
            root = work / part['label']
            root.mkdir(exist_ok=True)
            with (source / 'run.lock').open('a') as old_lock, (root / 'run.lock').open('a') as new_lock:
                fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(new_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                digest = plan['reuse']['ready_sha256'][str(index)]
                assert sha256(source / 'READY.json') == digest
                started = time.monotonic()
                ready = verify_ready(source, part, original_binding, hash_files=True)
                assert ready['mode'] == 'new_export'
                build = read_json(source / 'build.json')
                check_part(build['range'], part, plan)
                if (root / 'READY.json').exists():
                    verify_ready(root, part, binding, hash_files=True)
                    continue
                subprocess.run([sys.executable, str(SCRIPTS / 'test_reader.py'),
                                str(source / 'files' / (part['label'] + '_data.h5')),
                                str(source / 'files' / (part['label'] + '_features.h5'))], check=True)
                durable_json(root / 'build.json', build)
                publish(root, source / 'files', part, binding, read_json(source / 'validation.json'),
                        started, 'validated_reuse',
                        previous_validation=dict(directory=str(source), ready_sha256=digest,
                                                 plan_sha256=original_binding['plan_sha256']))
        verify_gates(work, plan, indices)
        durable_json(work / 'GATE_PASS.json', dict(status='PASS', binding=binding,
                     gate_seconds_per_full_part=plan['gate_seconds_per_full_part'],
                     ready_sha256={str(index): sha256(work / plan['parts'][index]['label'] / 'READY.json')
                                   for index in indices}))


def verify_gate_pass(work, plan):
    if not plan.get('require_gate_pass'):
        return
    passed = read_json(work / 'GATE_PASS.json')
    assert passed['status'] == 'PASS'
    assert passed['binding'] == dict(plan_sha256=sha256(work / 'plan.json'))
    assert passed['gate_seconds_per_full_part'] == plan['gate_seconds_per_full_part']
    indices = plan['canary_indices'] + plan['ramp_indices']
    assert passed['ready_sha256'] == {
        str(index): sha256(work / plan['parts'][index]['label'] / 'READY.json') for index in indices}


def publish(root, output, part, binding, result, started, mode, **metrics):
    assert result['status'] == 'PASS' and result['structures'] == part['structures']
    assert set(result['files']) == {part['label'] + '_data.h5', part['label'] + '_features.h5'}
    files = root / 'files'
    files.mkdir(exist_ok=True)
    for name, detail in result['files'].items():
        assert Path(name).name == name
        source = output / name
        destination = files / name
        if output != files:
            if destination.exists():
                assert destination.samefile(source), name
            else:
                os.link(source, destination)
        assert destination.stat().st_size == detail['bytes']
        with destination.open('rb') as stream:
            os.fsync(stream.fileno())
    descriptor = os.open(files, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    durable_json(root / 'validation.json', result)
    ready = dict(status='PASS', binding=binding, part=part, structures=result['structures'],
                 files=result['files'], build_sha256=sha256(root / 'build.json'),
                 validation_sha256=sha256(root / 'validation.json'), mode=mode,
                 seconds=time.monotonic() - started,
                 maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 job_id=os.environ['SLURM_JOB_ID'], uploaded=False, **metrics)
    durable_json(root / 'READY.json', ready)
    return ready


def run_part(work, plan, index, resume=False):
    assert 0 <= index < len(plan['parts'])
    failures = [path for path in (work / 'failures').glob('*.json')
                if not (resume and path.name == f'{index}.json')]
    assert not failures, 'Earlier production failure requires review before more parts start'
    part = plan['parts'][index]
    root = work / part['label']
    legacy = BASELINE / part['label']
    binding = dict(plan_sha256=sha256(work / 'plan.json'))
    root.mkdir(exist_ok=True)

    def event(phase, **values):
        record = dict(phase=phase, time=time.time(), job_id=os.environ['SLURM_JOB_ID'],
                      host=socket.gethostname(), index=index)
        record.update(values)
        durable_json(root / 'status.json', record)
        print(json.dumps(record), flush=True)

    with ExitStack() as stack:
        for path in (root / 'run.lock', legacy / 'run.lock'):
            if path.parent.exists():
                lock = stack.enter_context(path.open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify_gate_pass(work, plan)
            verify_gates(work, plan, gate_indices(plan, index))
            if (root / 'READY.json').exists():
                verify_ready(root, part, binding, hash_files=True)
                event('already_ready')
                return
            assert resume or not (root / 'status.json').exists(), 'Explicit --resume required for an earlier attempt'
            started = time.monotonic()
            event('starting')
            legacy_build = legacy / (part['label'] + '_build.json')
            if index == 0 and legacy_build.exists():
                original = read_json(legacy_build)
                check_part(original['range'], part, plan)
                pilot = read_json(PILOT / 'PILOT_PASS.json')
                pilot_plan = read_json(PILOT / 'plan.json')
                assert pilot_plan['part'] == part
                assert pilot['status'] == 'PASS' and pilot['golden_decoded_fields_match']
                assert pilot['binding']['plan_sha256'] == plan['pilot_plan_sha256']
                result = read_json(PILOT / 'full/validation.json')
                for name, detail in original['files'].items():
                    assert detail == pilot['files'][name]
                    assert result['files'][name] == {key: detail[key] for key in ('bytes', 'sha256')}
                    assert sha256(legacy / 'files' / name) == detail['sha256']
                durable_json(root / 'build.json', original)
                ready = publish(root, legacy / 'files', part, binding, result, started, 'pilot_reuse')
                event('ready', **ready)
                return
            internal = root / 'internal'
            internal.mkdir(exist_ok=True)
            attempt = str(time.time_ns())
            if legacy_build.exists():
                build = read_json(legacy_build)
                report = build['range']
                check_part(report, part, plan)
                assert read_json(legacy / 'staging/stage.json')['report'] == report
                durable_json(root / 'build.json', build)
                mode = 'existing_export'
            elif (root / 'preflight.json').exists():
                report = read_json(root / 'preflight.json')
                mode = 'resumed_export'
            elif (legacy / 'staging/stage-progress.json').exists():
                report = read_json(legacy / 'staging/stage-progress.json')['report']
                assert report == read_json(legacy / 'preflight.json')
                durable_json(root / 'preflight.json', report)
                mode = 'legacy_resume'
            else:
                event('preflight')
                metadata = internal / ('metadata-' + attempt)
                report = inspect(part['cell_start'], part['cell_stop'], workers=plan['workers'],
                                 metadata=metadata, member_start=part['member_start'], member_stop=part['member_stop'])
                report.update({key: part[key] for key in ('interval_cell_start', 'interval_cell_stop', 'part_number', 'part_count')})
                report['metadata_directory'] = str(metadata)
                durable_json(root / 'preflight.json', report)
                mode = 'new_export'
            check_part(report, part, plan)
            required = check_memory(report, plan['maximum_staging_bytes'], legacy / 'staging')
            event('copying', mode=mode, staging_bound_bytes=required)
            checkpoint = RamCheckpoint(internal / 'checkpoints', binding, interval=plan['checkpoint_seconds'])
            phase_start = time.monotonic()
            with tempfile.TemporaryDirectory(prefix='madcosmo-production-', dir='/tmp') as temporary, BlockReader(plan['workers']) as reader:
                staging = Path(temporary) / 'staging'
                restored = False
                if (checkpoint.root / 'CURRENT.json').exists():
                    assert resume
                    assert checkpoint.restore(staging)['report'] == report
                    restored = True
                elif (legacy / 'staging/stage-progress.json').exists():
                    staging.mkdir()
                    for path in sorted((legacy / 'staging').glob('*.npy')):
                        copy_checked(path, staging / path.name)
                    for name in ('stage-progress.json', 'stage.json'):
                        path = legacy / 'staging' / name
                        if path.exists():
                            copy_checked(path, staging / name)
                    restored = True
                if legacy_build.exists():
                    assert read_json(staging / 'stage.json')['report'] == report
                    output = legacy / 'files'
                else:
                    state = stage(report, staging, workers=plan['workers'],
                                  metadata_staging=Path(report['metadata_directory']), resume=restored,
                                  reader=reader, checkpoint_hook=checkpoint)
                    output = root / 'files'
                    if not (root / 'build.json').exists():
                        if output.exists():
                            output.rename(internal / ('partial-files-' + attempt))
                        event('encoding', mode=mode)
                        files = encode(state, staging, output)
                        durable_json(root / 'build.json', dict(schema='madcosmo-hdf5-v1', range=report, files=files))
                prepare_seconds = time.monotonic() - phase_start
                event('validating', mode=mode, prepare_seconds=prepare_seconds)
                phase_start = time.monotonic()
                result = validate(staging, output, root / 'build.json', workers=plan['workers'],
                                  checkpoint_dir=internal / ('validation-' + attempt),
                                  source_chunk_rows=65536, reader=reader)
                validation_seconds = time.monotonic() - phase_start
                subprocess.run([sys.executable, str(SCRIPTS / 'test_reader.py'),
                                str(output / (part['label'] + '_data.h5')),
                                str(output / (part['label'] + '_features.h5'))], check=True)
            ready = publish(root, output, part, binding, result, started, mode,
                            prepare_seconds=prepare_seconds, validation_seconds=validation_seconds,
                            staging_bound_bytes=required)
            event('ready', **ready)
        except BaseException as error:
            event('failed', error=repr(error))
            failures_root = work / 'failures'
            failures_root.mkdir(exist_ok=True)
            durable_json(failures_root / f'{index}.json', dict(binding=binding, index=index,
                         error=repr(error), job_id=os.environ['SLURM_JOB_ID']))
            array_job = os.environ.get('SLURM_ARRAY_JOB_ID')
            if array_job and array_job.isdigit():
                held = subprocess.run(['scontrol', 'hold', array_job], text=True, capture_output=True)
                print(json.dumps(dict(hold_array=array_job, returncode=held.returncode,
                                      stdout=held.stdout, stderr=held.stderr)), flush=True)
            raise


def audit(work, plan):
    centers = np.load(INDEX / 'center_offsets.npy', mmap_mode='r')
    assert sha256(INDEX / 'center_offsets.npy') == read_json(INDEX / 'MANIFEST.json')['files']['center_offsets.npy']['sha256']
    verify_parts(centers, plan['parts'])
    binding = dict(plan_sha256=sha256(work / 'plan.json'))
    entries = []
    for part in plan['parts']:
        root = work / part['label']
        ready = verify_ready(root, part, binding)
        check_part(read_json(root / 'build.json')['range'], part, plan)
        for name in ready['files']:
            with h5py.File(root / 'files' / name, 'r') as handle:
                check_part(dict(handle.attrs), part, plan)
        entries.append(dict(part=part, files=ready['files'], directory=str(root / 'files')))
    durable_json(work / 'READY.json', dict(status='PASS', binding=binding, parts=entries,
                 structures=sum(part['structures'] for part in plan['parts']),
                 scope='remaining intervals only; first three intervals retain their existing validation',
                 audit='coverage, readiness, validation and build hashes, file sizes and HDF5 attributes; no upload',
                 uploaded=False))


def main():
    if sys.flags.optimize:
        raise RuntimeError('Python assertions must be enabled')
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['part', 'gate', 'audit'])
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Run the production worker on an allocated node'
    plan = check_plan(args.work)
    if 'cluster' in plan:
        assert os.environ.get('SLURM_CLUSTER_NAME') == plan['cluster']
        assert os.environ.get('SLURM_JOB_ACCOUNT') == plan['account']
    if args.phase == 'audit':
        verify_gate_pass(args.work, plan)
        audit(args.work, plan)
    elif args.phase == 'gate':
        prepare_gate(args.work, plan)
    else:
        run_part(args.work, plan, int(os.environ['SLURM_ARRAY_TASK_ID']), args.resume)


if __name__ == '__main__':
    main()
