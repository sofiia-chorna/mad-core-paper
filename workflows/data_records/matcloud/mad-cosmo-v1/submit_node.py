import argparse
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path


WORKSPACE = Path('${DEV_ROOT}/phace-atomic')
RUN = WORKSPACE / 'matcloud-upload/mad-cosmo-v1'
SCRIPTS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-existing', action='store_true')
    args = parser.parse_args()
    with (RUN / 'node-submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = RUN / 'logs/node-process.json'
        if record.exists():
            raise SystemExit('A node submission is already recorded; inspect it before resubmitting')
        if (RUN / 'first/validation.json').exists():
            raise SystemExit('The login-node export already finished; inspect it before moving')
        if args.validate_existing:
            manifest = json.loads((RUN / 'first/madcosmo_cells_0-2pow18_build.json').read_text())
            for name, detail in manifest['files'].items():
                assert (RUN / 'first/files' / name).stat().st_size == detail['bytes']
        snapshot = RUN / ('node-code-' + str(int(time.time())))
        shutil.copytree(SCRIPTS, snapshot, ignore=shutil.ignore_patterns('__pycache__'))
        script = 'slurm/validate/submit.sh' if args.validate_existing else 'slurm/submit.sh'
        result = subprocess.run(['sbatch', '--parsable', '--export=ALL,MADCOSMO_CODE=' + str(snapshot),
                                 str(snapshot / script)], check=True, text=True, capture_output=True)
        job = result.stdout.strip().split(';')[0]
        assert job.isdigit(), result.stdout
        details = dict(job_id=job, account='ab010', partition='normal', nodes=1, cpus=16,
                       workers=8, time_limit='12:00:00', snapshot=str(snapshot),
                       build=str(RUN / ('first-node-' + job)),
                       staging=str(RUN / ('first-node-' + job + '-staging')),
                       log=str(RUN / 'logs' / ('node-' + job + '.log')))
        if args.validate_existing:
            details.update(purpose='validate_existing', build=str(RUN / 'first'),
                           staging=str(RUN / 'first-staging'),
                           log=str(RUN / 'logs' / ('validation-node-' + job + '.log')))
        with record.open('x') as stream:
            json.dump(details, stream, indent=2)
            stream.write('\n')
        print(json.dumps(details, indent=2), flush=True)


if __name__ == '__main__':
    main()
