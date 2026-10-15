import argparse
import fcntl
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from preflight import sha256


WORKSPACE = Path('${DEV_ROOT}/phace-atomic')
RUN = WORKSPACE / 'matcloud-upload/mad-cosmo-v1'
SCRIPTS = WORKSPACE / 'release/build/matcloud/mad-cosmo-v1'
STORE = Path('${STORE_ROOT}/mad-core/matcloud/mad-cosmo-v1')
PYTHON = '/usr/bin/python3.11'


def run(recovery, approved_files):
    validation_path = RUN / 'first/validation.json'
    if not validation_path.exists():
        process = json.loads((RUN / 'logs/first-process.json').read_text())
        if process['host'] != socket.gethostname():
            raise SystemExit('Run the waiting workflow on the exporter host: ' + process['host'])
    while not validation_path.exists():
        command_path = Path(f'/proc/{process["pid"]}/cmdline')
        if not command_path.exists():
            raise SystemExit('Exporter exited without a validation report; refusing to stage or upload')
        command = command_path.read_bytes().split(b'\0')
        if str(SCRIPTS / 'submit.sh').encode() not in command or b'first' not in command:
            raise SystemExit('Exporter PID no longer identifies the expected process')
        print('Waiting for the live first-interval export and exhaustive validation', flush=True)
        time.sleep(30)
    validation = json.loads(validation_path.read_text())
    if validation['status'] != 'PASS' or validation['structures'] != 2357504:
        raise SystemExit('First-interval validation did not pass with the expected row count')
    files = RUN / 'first/files'
    pair = [files / 'madcosmo_cells_0-2pow18_data.h5', files / 'madcosmo_cells_0-2pow18_features.h5']
    subprocess.run([PYTHON, str(SCRIPTS / 'test_reader.py'), *map(str, pair)], check=True)
    if not (files / 'madcosmo_manifest.json').exists():
        subprocess.run([PYTHON, str(SCRIPTS / 'package.py'), '--build', str(RUN / 'first'), '--staging', str(RUN / 'first-staging')], check=True)
    manifest = json.loads((files / 'madcosmo_manifest.json').read_text())
    expected = manifest['files'].copy()
    expected['madcosmo_manifest.json'] = dict(bytes=(files / 'madcosmo_manifest.json').stat().st_size,
                                              sha256=sha256(files / 'madcosmo_manifest.json'))
    for name, detail in expected.items():
        if Path(name).name != name or (files / name).stat().st_size != detail['bytes'] or sha256(files / name) != detail['sha256']:
            raise SystemExit('Local package checksum failed: ' + name)
    STORE.joinpath('files').mkdir(parents=True, exist_ok=True)
    STORE.joinpath('build').mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '--ignore-existing', str(files) + '/', str(STORE / 'files') + '/'], check=True)
    for name, detail in expected.items():
        destination = STORE / 'files' / name
        if destination.stat().st_size != detail['bytes'] or sha256(destination) != detail['sha256']:
            raise SystemExit('Store copy checksum failed: ' + name)
    for source in (validation_path, RUN / 'preflight.json', RUN / 'first/madcosmo_cells_0-2pow18_build.json'):
        destination = STORE / 'build' / source.name
        if destination.exists():
            if sha256(destination) != sha256(source):
                raise SystemExit('Refusing to overwrite a different provenance file: ' + str(destination))
        else:
            shutil.copyfile(source, destination)
    print('Verified the separate aa002 staging copy; starting additive MAD-COSMO draft upload', flush=True)
    subprocess.run([PYTHON, str(SCRIPTS / 'upload.py'), '--files', str(STORE / 'files'),
                    '--state', str(RUN / 'upload_state.json'), '--approved-files', str(approved_files),
                    '--validation-report', str(validation_path), *recovery], check=True)
    approval = json.loads(approved_files.read_text())
    with (RUN / 'completion.json').open('x') as stream:
        json.dump(dict(status='complete', store=str(STORE), draft='${MC_DRAFT_MADCOSMO}', published=False,
                       structures=validation['structures'], verified_files=len(approval['files'])), stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--approved-files', required=True, type=Path)
    parser.add_argument('--recover-pending')
    parser.add_argument('--recover-created')
    args = parser.parse_args()
    if bool(args.recover_pending) != bool(args.recover_created):
        parser.error('Both recovery arguments are required together')
    recovery = ['--recover-pending', args.recover_pending, '--recover-created', args.recover_created] if args.recover_pending else []
    if args.run:
        with (RUN / 'finish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(recovery, args.approved_files.resolve())
        return
    log = RUN / 'logs' / f'finish-{int(time.time())}.log'
    with log.open('xb') as stream:
        process = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--run',
                                    '--approved-files', str(args.approved_files.resolve()), *recovery],
                                   cwd=WORKSPACE, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
    result = dict(pid=process.pid, host=socket.gethostname(), log=str(log))
    (RUN / 'logs/finish-process.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
