import argparse
import fcntl
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import production
from preflight import sha256
from production import WORK, read_json, verify_ready
from ram_checkpoint import durable_json


BULK = '4982420'
AUDIT = '4982421'
PLAN_SHA256 = '97cea7b7c56c83936a491cb5677d6bf5ce96c5a443a82b36fe28364d9e01612f'
MAX_RETRIES = 2
MAX_SUBMISSIONS = 40
ACTIVE = {'PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED'}


def command(*arguments):
    result = subprocess.run(arguments, text=True, capture_output=True, timeout=60)
    if result.returncode:
        messages = (result.stdout + result.stderr).strip().splitlines()
        finished_members = (arguments[:2] in (('scontrol', 'hold'), ('scontrol', 'release'), ('scontrol', 'update'))
                            and messages and all(line.endswith((': Job has already finished', ': No error'))
                                                 for line in messages))
        if not finished_members:
            raise RuntimeError(f'{arguments}: {result.returncode}: {result.stdout}{result.stderr}')
    return result.stdout.strip()


def retry_reason(state, log, root):
    assertions = re.findall(r'^AssertionError: (.*)$', log, flags=re.MULTILINE)
    blocked = 'Earlier production failure requires review before more parts start'
    if assertions:
        assert state == 'FAILED' and assertions == [blocked], 'Validation or safety assertion failed'
        return 'blocked_by_reviewed_failure'
    if state == 'TIMEOUT':
        return 'time_limit'
    assert state == 'FAILED', f'Unexpected terminal state {state}'
    errors = re.findall(r"^PermissionError: \[Errno 13\] Permission denied: '([^']+)'$", log, re.MULTILINE)
    assert len(errors) == 1 and 'temporary.write_text(' in log, 'Unrecognized worker failure'
    path = Path(errors[0])
    assert path.parent.parent == root / 'internal', 'Permission failure outside validation checkpoints'
    assert re.fullmatch(r'validation-\d+', path.parent.name)
    assert re.fullmatch(r'\d+-\d+\.json\.partial', path.name)
    return 'validation_checkpoint_permission'


def log_tail(path):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - (2 << 20)))
        return stream.read().decode(errors='replace')


def resume_evidence(root, part):
    if not (root / 'build.json').exists():
        return
    build = read_json(root / 'build.json')
    for name, detail in build['files'].items():
        assert Path(name).name == name
        assert (root / 'files' / name).stat().st_size == detail['bytes']
    checkpoint = root / 'internal/checkpoints'
    pointer = read_json(checkpoint / 'CURRENT.json')['current']
    generation = checkpoint / pointer['generation']
    assert generation.parent == checkpoint and generation.name.startswith('generation-')
    manifest = generation / 'checkpoint.json'
    assert sha256(manifest) == pointer['manifest_sha256']
    saved = read_json(manifest)
    assert saved['binding'] == dict(plan_sha256=PLAN_SHA256)
    assert saved['progress']['report'] == build['range']
    for key, value in part.items():
        if key != 'label':
            assert build['range'][key] == value


def scheduler(state):
    jobs = {BULK, AUDIT}
    jobs.update(row['job_id'] for rows in state['attempts'].values() for row in rows)
    queued = {}
    text = command('squeue', '--noheader', '--array', '--user=' + str(os.getuid()), '--format=%i|%T|%R|%j')
    for line in text.splitlines():
        job, status, reason, name = line.split('|', 3)
        if job.split('_')[0] in jobs:
            queued[job] = dict(state=status, reason=reason.strip('()'))
        elif name.startswith('cosmo-fast-'):
            raise RuntimeError('Untracked production job ' + job)
    accounting = {}
    text = command('sacct', '-X', '--noheader', '--parsable2', '--array', '--jobs=' + ','.join(sorted(jobs)),
                   '--format=JobID%40,State%30')
    for line in text.splitlines():
        job, status = line.split('|')[:2]
        accounting[job] = status.split()[0]
    return queued, accounting


def hold(queued):
    arrays = {job.split('_')[0] for job, row in queued.items()
              if row['state'] == 'PENDING' and row.get('reason') != 'JobHeldUser'}
    for job in sorted(arrays):
        command('scontrol', 'hold', job)


def recovery_capacity(bulk_running, live_retries):
    return max(0, min(8 - live_retries, 32 - bulk_running - live_retries))


def cycle(state, directory, plan):
    assert 'submitting' not in state, 'Unknown submission outcome needs manual reconciliation'
    queued, accounting = scheduler(state)
    state['queued'] = queued
    failures = sorted((WORK / 'failures').glob('*.json'))
    if failures:
        hold(queued)
    ready = set()
    candidates = []
    reviewed = {}
    for index, part in enumerate(plan['parts']):
        root = WORK / part['label']
        attempts = state['attempts'].get(str(index), [])
        job = attempts[-1]['job_id'] if attempts else BULK
        task = f'{job}_{index}'
        if (root / 'READY.json').exists():
            verify_ready(root, part, dict(plan_sha256=PLAN_SHA256))
            ready.add(index)
            continue
        if task in queued or accounting.get(task) in ACTIVE or task not in accounting:
            continue
        status = accounting[task]
        name = 'cosmo-fast-v2-recover' if attempts else 'cosmo-fast-v2-bulk'
        log = WORK.parent / 'logs' / f'production-{name}-{task}.log'
        reason = retry_reason(status, log_tail(log), root)
        assert len(attempts) < MAX_RETRIES, f'Retry limit reached for part {index}'
        resume_evidence(root, part)
        reviewed[index] = reason
        candidates.append((index, reason))
    for path in failures:
        failure = read_json(path)
        index = failure['index']
        assert path.name == f'{index}.json'
        assert failure['binding'] == dict(plan_sha256=PLAN_SHA256)
        assert failure['error'] == "PermissionError(13, 'Permission denied')", failure
        root = WORK / plan['parts'][index]['label']
        assert read_json(root / 'status.json')['job_id'] == failure['job_id']
        if index not in reviewed:
            assert index not in ready, 'Unreviewed failure after READY'
            state.update(status='WAITING_FOR_FAILED_WORKER', time=time.time())
            durable_json(directory / 'state.json', state)
            return
        assert reviewed[index] == 'validation_checkpoint_permission', failure
    for path in failures:
        failure = read_json(path)
        destination = directory / 'reviewed-failures' / f"{failure['index']}-{failure['job_id']}.json"
        assert not destination.exists()
        path.rename(destination)
    retry_jobs = {row['job_id'] for rows in state['attempts'].values() for row in rows}
    live_retries = [task for task in queued if task.split('_')[0] in retry_jobs]
    bulk_running = sum(task.startswith(BULK + '_') and row['state'] != 'PENDING'
                       for task, row in queued.items())
    bulk_pending = any(task.startswith(BULK + '_') and row['state'] == 'PENDING'
                       for task, row in queued.items())
    throttle = 24 if candidates or live_retries else 32
    if bulk_pending:
        command('scontrol', 'update', 'JobId=' + BULK, 'ArrayTaskThrottle=' + str(throttle))
    slots = recovery_capacity(bulk_running, len(live_retries))
    for index, reason in candidates[:slots]:
        assert sum(map(len, state['attempts'].values())) < MAX_SUBMISSIONS, 'Recovery submission cap reached'
        arguments = ['sbatch', '--parsable', '--account=lp133', '--partition=normal', '--time=08:00:00',
                     '--array=' + str(index), '--job-name=cosmo-fast-v2-recover', '--kill-on-invalid-dep=yes',
                     str(WORK / 'code/submit.sh'), str(WORK), 'part', '--resume']
        state['submitting'] = dict(index=index, command=arguments, reason=reason)
        durable_json(directory / 'state.json', state)
        job = command(*arguments).split(';')[0]
        assert job.isdigit(), job
        state['attempts'].setdefault(str(index), []).append(dict(job_id=job, reason=reason, time=time.time()))
        del state['submitting']
        durable_json(directory / 'state.json', state)
        print(json.dumps(dict(submitted=job, index=index, reason=reason)), flush=True)
    assert not list((WORK / 'failures').glob('*.json')), 'New failure occurred during recovery'
    for job in sorted({task.split('_')[0] for task, row in queued.items()
                       if row['state'] == 'PENDING' and row['reason'] == 'JobHeldUser' and task != AUDIT}):
        command('scontrol', 'release', job)
    state.update(ready_parts=len(ready), time=time.time(), status='MONITORING')
    state.pop('error', None)
    if len(ready) == len(plan['parts']) and not [job for job in queued if job != AUDIT]:
        assert AUDIT in queued and queued[AUDIT]['state'] == 'PENDING'
        command('scontrol', 'release', AUDIT)
        state['status'] = 'AUDIT_RELEASED'
    durable_json(directory / 'state.json', state)
    print(json.dumps(dict(status=state['status'], ready=len(ready), retry_candidates=len(candidates),
                         active_recovery_tasks=len(live_retries), bulk_running=bulk_running)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    assert not sys.flags.optimize, 'Python assertions must be enabled'
    assert sha256(WORK / 'plan.json') == PLAN_SHA256
    plan = read_json(WORK / 'plan.json')
    assert sha256(Path(production.__file__)) == plan['code_sha256']['production.py']
    production.SCRIPTS = WORK / 'code'
    production.check_plan(WORK)
    production.verify_gate_pass(WORK, plan)
    configuration = command('scontrol', 'show', 'config')
    cluster = [line.split('=', 1)[1].strip() for line in configuration.splitlines()
               if line.strip().startswith('ClusterName')]
    assert cluster == ['alps-daint'], cluster
    if not args.run:
        print('Plan, frozen code and gate verified; no changes made')
        return
    directory = WORK / 'recovery-20261005'
    directory.mkdir(exist_ok=True)
    (directory / 'reviewed-failures').mkdir(exist_ok=True)
    with (WORK / 'recovery-submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / 'state.json'
        state = read_json(path) if path.exists() else dict(plan_sha256=PLAN_SHA256, attempts={}, started=time.time())
        assert state['plan_sha256'] == PLAN_SHA256
        state.update(host=socket.gethostname(), pid=os.getpid(), controller_sha256=sha256(Path(__file__)))
        state['policy'] = dict(account='lp133', maximum_nodes=32, maximum_recovery_tasks=8,
                               maximum_retries_per_part=MAX_RETRIES, maximum_submissions=MAX_SUBMISSIONS,
                               retry_time_limit_hours=8, controller_time_limit_hours=48,
                               frozen_worker_and_plan_unchanged=True)
        try:
            while state.get('status') != 'AUDIT_RELEASED':
                assert time.time() - state['started'] < 48 * 3600, 'Controller reached 48-hour limit'
                cycle(state, directory, plan)
                if state.get('status') != 'AUDIT_RELEASED':
                    time.sleep(45)
        except BaseException as error:
            print(repr(error), flush=True)
            state.update(status='STOPPED_FOR_REVIEW', error=repr(error), time=time.time())
            durable_json(path, state)
            queued, _ = scheduler(state)
            hold(queued)
            raise


if __name__ == '__main__':
    main()
