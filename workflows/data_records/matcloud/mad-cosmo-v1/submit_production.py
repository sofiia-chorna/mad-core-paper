import argparse
import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from parts import verify_parts
from preflight import INDEX, sha256
from production import RUN, WORK, check_previous, read_json, verify_ready
from ram_checkpoint import copy_checked, durable_json


SCRIPTS = Path(__file__).resolve().parent
PREVIOUS = RUN / 'production-v1'
PREVIOUS_SHA = 'a35b971ac57e169a2c72a6472702751c80338290861586685505b9b78a36055b'
SNAPSHOT_FILES = ('production.py', 'submit_production.py', 'block_reader.py', 'ram_checkpoint.py',
                  'export.py', 'validate.py', 'preflight.py', 'parts.py', 'madcosmo.py', 'test_reader.py')


def waves(plan):
    reused = set(map(int, plan['reuse']['ready_sha256']))
    assert reused == set(plan['canary_indices'] + plan['ramp_indices'])
    assert len(reused) == 6
    assert all(0 <= index < len(plan['parts']) for index in reused)
    remaining = [index for index in range(len(plan['parts'])) if index not in reused]
    return [('gate', [], 1), ('bulk', remaining, plan['maximum_parallel_parts']), ('audit', [], 1)]


def prepare(work, maximum_nodes):
    assert maximum_nodes in (16, 32)
    if (work / 'plan.json').exists():
        plan = read_json(work / 'plan.json')
        assert plan['maximum_parallel_parts'] == maximum_nodes
        assert plan['account'] == 'lp133' and plan['cluster'] == 'alps-daint'
        check_previous(plan)
        for name, digest in plan['code_sha256'].items():
            assert Path(name).name == name
            assert sha256(work / 'code' / name) == digest, name
        return plan
    assert sha256(PREVIOUS / 'plan.json') == PREVIOUS_SHA
    original = read_json(PREVIOUS / 'plan.json')
    plan = dict(original, account='lp133', cluster='alps-daint', maximum_parallel_parts=maximum_nodes,
                time_limit_hours=4, gate_seconds_per_full_part=10800, require_gate_pass=True,
                gate_time_limit_hours=1, audit_time_limit_hours=1,
                reuse=dict(directory=str(PREVIOUS), plan_sha256=PREVIOUS_SHA, ready_sha256={}))
    check_previous(plan)
    assert not list((PREVIOUS / 'failures').glob('*.json'))
    for index in original['canary_indices'] + original['ramp_indices']:
        part = plan['parts'][index]
        root = PREVIOUS / part['label']
        ready = verify_ready(root, part, dict(plan_sha256=PREVIOUS_SHA))
        assert ready['mode'] == 'new_export'
        assert ready['seconds'] * plan['maximum_rows_per_part'] / part['structures'] <= 10800
        plan['reuse']['ready_sha256'][str(index)] = sha256(root / 'READY.json')
    assert sha256(INDEX / 'center_offsets.npy') == read_json(INDEX / 'MANIFEST.json')['files']['center_offsets.npy']['sha256']
    verify_parts(np.load(INDEX / 'center_offsets.npy', mmap_mode='r'), plan['parts'])
    waves(plan)
    work.mkdir(exist_ok=True)
    snapshot = work / 'code'
    snapshot.mkdir(exist_ok=False)
    for name in SNAPSHOT_FILES:
        copy_checked(SCRIPTS / name, snapshot / name)
    copy_checked(SCRIPTS / 'slurm/production/submit.sh', snapshot / 'submit.sh')
    plan['code_sha256'] = {path.name: sha256(path) for path in sorted(snapshot.iterdir())}
    durable_json(work / 'plan.json', plan)
    return plan


def check_scheduler(known_jobs=()):
    result = subprocess.run(['scontrol', 'show', 'config'], text=True, capture_output=True, check=True)
    cluster = [line.split('=', 1)[1].strip() for line in result.stdout.splitlines()
               if line.strip().startswith('ClusterName')]
    assert cluster == ['alps-daint'], cluster
    result = subprocess.run(['squeue', '--noheader', '--user=' + str(os.getuid()),
                             '--format=%i|%j'], text=True, capture_output=True, check=True)
    for line in result.stdout.splitlines():
        job, name = line.strip().split('|', 1)
        if name.startswith(('cosmo-rest-', 'cosmo-retry-', 'cosmo-fast-')):
            assert job.split('_', 1)[0] in known_jobs, 'Another production run is still queued'


def submit(work, plan):
    state_path = work / 'submission.json'
    state = read_json(state_path) if state_path.exists() else dict(jobs={}, plan_sha256=sha256(work / 'plan.json'))
    assert state['plan_sha256'] == sha256(work / 'plan.json')
    assert 'submitting' not in state, 'Unknown submission outcome; inspect Slurm before retrying'
    check_scheduler(state['jobs'].values())
    assert shutil.disk_usage(work).free >= 20 * 10**12, 'Need 20 TB free for outputs and retained checkpoints'
    previous = None
    for name, indices, concurrency in waves(plan):
        if name in state['jobs']:
            previous = state['jobs'][name]
            continue
        hours = plan['time_limit_hours'] if indices else plan[name + '_time_limit_hours']
        phase = 'part' if indices else name
        options = ['--account=' + plan['account'], '--job-name=cosmo-fast-v2-' + name,
                   f'--time={hours:02d}:00:00', '--kill-on-invalid-dep=yes']
        if indices:
            options.append('--array=' + ','.join(map(str, indices)) + '%' + str(concurrency))
        if previous:
            options.append('--dependency=afterok:' + previous)
        command = ['sbatch', '--parsable', *options, str(work / 'code/submit.sh'), str(work), phase]
        state['submitting'] = dict(wave=name, command=command)
        durable_json(state_path, state)
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr + '\nSubmission state retained for manual reconciliation')
        job = result.stdout.strip().split(';')[0]
        assert job.isdigit(), result.stdout
        state['jobs'][name] = job
        del state['submitting']
        durable_json(state_path, state)
        print(dict(submitted=name, job_id=job, tasks=len(indices) or 1,
                   maximum_nodes=concurrency, account=plan['account']), flush=True)
        previous = job


def main():
    if sys.flags.optimize:
        raise RuntimeError('Python assertions must be enabled')
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--max-nodes', type=int, default=32, choices=(16, 32))
    args = parser.parse_args()
    if not (args.prepare or args.submit):
        parser.error('Choose --prepare or --submit')
    with (RUN / 'production-submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(WORK, args.max_nodes)
        remaining = len(waves(plan)[1][1])
        print(dict(parts=remaining, reused_parts=len(plan['reuse']['ready_sha256']),
                   maximum_parallel_parts=args.max_nodes, account=plan['account'],
                   total_node_hour_cap=remaining * plan['time_limit_hours'] +
                   plan['gate_time_limit_hours'] + plan['audit_time_limit_hours'], work=str(WORK)), flush=True)
        if args.submit:
            submit(WORK, plan)


if __name__ == '__main__':
    main()
