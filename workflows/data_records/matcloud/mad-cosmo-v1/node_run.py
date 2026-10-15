import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from export import stage
from preflight import INDEX, ROOT, sha256
from validate import equal


WORKSPACE = Path('${DEV_ROOT}/phace-atomic')
RUN = WORKSPACE / 'matcloud-upload/mad-cosmo-v1'
SCRIPTS = Path(__file__).resolve().parent


def main():
    job = os.environ['SLURM_JOB_ID']
    build = RUN / ('first-node-' + job)
    staging = RUN / ('first-node-' + job + '-staging')
    build.mkdir(exist_ok=False)

    def status(phase, **extra):
        result = dict(job_id=job, host=socket.gethostname(), phase=phase, time=time.time(),
                      workers=8, build=str(build), staging=str(staging), **extra)
        temporary = build / 'node-status.json.partial'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(build / 'node-status.json')
        print(json.dumps(result), flush=True)

    try:
        status('pilot_check', python=sys.version, numpy=np.__version__, h5py=h5py.__version__)
        pilot_report = json.loads((RUN / 'cell0-preflight.json').read_text())
        assert sha256(ROOT / 'MANIFEST.json') == pilot_report['root_manifest_sha256']
        assert sha256(INDEX / 'MANIFEST.json') == pilot_report['index_manifest_sha256']
        reference = RUN / 'pilot-staging'
        reference_state = json.loads((reference / 'stage.json').read_text())
        assert json.loads((RUN / 'pilot/validation.json').read_text())['status'] == 'PASS'
        pilot_staging = RUN / ('pilot-node-' + job + '-staging')
        state = stage(pilot_report, pilot_staging, workers=8, metadata_staging=reference)
        assert state == reference_state
        for name, filename in state['arrays'].items():
            actual = np.load(pilot_staging / filename, mmap_mode='r')
            expected = np.load(reference / filename, mmap_mode='r')
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            for begin in range(0, len(actual), 8192):
                equal(actual[begin:begin+8192], expected[begin:begin+8192], 'parallel pilot:' + name)
        status('pilot_passed', structures=pilot_report['structures'], fields=len(state['arrays']))
        status('exporting')
        subprocess.run([sys.executable, str(SCRIPTS / 'export.py'), '--preflight', str(RUN / 'preflight.json'),
                        '--staging', str(staging), '--output', str(build / 'files'), '--workers', '8',
                        '--metadata-staging', str(RUN / 'first-staging')], check=True)
        status('validating')
        subprocess.run([sys.executable, str(SCRIPTS / 'validate.py'), '--staging', str(staging),
                        '--output', str(build / 'files'), '--manifest', str(build / 'madcosmo_cells_0-2pow18_build.json'),
                        '--report', str(build / 'validation.json'), '--workers', '8'], check=True)
        status('validated')
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
