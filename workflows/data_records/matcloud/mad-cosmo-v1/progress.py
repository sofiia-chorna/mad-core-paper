import json
from pathlib import Path

import numpy as np


def main():
    run = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1')
    staging = run / 'first-staging'
    marked = 0
    for name in ('diagnostics__collection_flags_present.npy', 'mad_next__present.npy'):
        path = staging / name
        if not path.exists():
            print(json.dumps({'stage': 'initial indexing'}))
            return
        marked += int(np.count_nonzero(np.load(path, mmap_mode='r')))
    report = dict(rows_marked_during_copy=marked, total_rows=2357504,
                  percent_marked=round(100*marked/2357504, 2),
                  note='Live copy counter, not a validation result; the current batch may still be in flight')
    files = run / 'first/files'
    report['hdf5_files'] = {path.name: path.stat().st_size for path in sorted(files.glob('*.h5*'))} if files.exists() else {}
    report['source_validation_report_exists'] = (run / 'first/validation.json').exists()
    report['completion_report_exists'] = (run / 'completion.json').exists()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
