import json
import os
import socket
import time
from pathlib import Path

from validate import validate


RUN = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1')


def main():
    job = os.environ['SLURM_JOB_ID']
    status_path = RUN / 'logs' / ('validation-node-' + job + '.json')

    def status(phase, **extra):
        result = dict(job_id=job, host=socket.gethostname(), phase=phase, time=time.time(), workers=8, **extra)
        temporary = status_path.with_suffix('.json.partial')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(status_path)
        print(json.dumps(result), flush=True)

    started = time.time()
    try:
        status('pilot_validation')
        baseline = json.loads((RUN / 'pilot/validation.json').read_text())
        pilot = validate(RUN / 'pilot-staging', RUN / 'pilot/files',
                         RUN / 'pilot/madcosmo_cells_0-1_build.json', workers=8)
        for key in ('status', 'structures', 'atoms', 'image_flag_structures', 'files'):
            assert pilot[key] == baseline[key], key
        status('pilot_passed', structures=pilot['structures'])
        status('full_source_validation')
        result = validate(RUN / 'first-staging', RUN / 'first/files',
                          RUN / 'first/madcosmo_cells_0-2pow18_build.json', workers=8)
        result.update(seconds=time.time()-started, slurm_job_id=job, validation_workers=8)
        report_path = RUN / 'first' / ('validation-node-' + job + '.json')
        with report_path.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        canonical = RUN / 'first/validation.json'
        try:
            os.link(report_path, canonical)
        except FileExistsError:
            previous = json.loads(canonical.read_text())
            for key in ('status', 'structures', 'atoms', 'image_flag_structures', 'files'):
                assert previous[key] == result[key], key
        status('PASS', report=str(report_path), structures=result['structures'])
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
