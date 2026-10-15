import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np


RELEASE = Path('${STORE_ROOT}/mad-core/release')
REMOVED = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1/rebuild_0-2pow18')
ANALYSIS = Path('${DEV_ROOT}/phace-atomic/voronoi/out/analysis')
OUTPUT = ANALYSIS / 'populations_level20_record.npy'
JULY = ANALYSIS / 'populations_level20.npy'
RECORD_LEVEL24 = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/xyz-2pow24-v2/voronoi_population_record.npy')
LEVELS = 25
CHUNK_ROWS = 20_000_000
WORKERS = 64


# level 20 and level 24 centre counts of one slice of one release folder, only the centres it uses
def count_slice(folder, start, stop):
    assignments = np.memmap(RELEASE / folder / 'voronoi_a.bin', dtype=np.int32, mode='r').reshape(-1, LEVELS)
    block = np.asarray(assignments[start:stop])
    counts = []
    for level in (20, 24):
        centres, number = np.unique(block[:, level], return_counts=True)
        assert centres[0] >= 0 and centres[-1] < 1 << level, (folder, level)
        counts.append((centres, number))
    return counts


def main():
    folders = [name.decode() for name in np.load(RELEASE / 'mad-core' / 'release_directory_names.npy')]
    jobs = []
    for folder in folders:
        rows = (RELEASE / folder / 'voronoi_a.bin').stat().st_size // (4 * LEVELS)
        count = np.load(RELEASE / folder / 'ns.npy')
        assert count.shape == (1,) and rows == int(count[0]), folder
        jobs += [(folder, start, min(start + CHUNK_ROWS, rows)) for start in range(0, rows, CHUNK_ROWS)]

    level20 = np.zeros(1 << 20, dtype=np.int64)
    level24 = np.zeros(1 << 24, dtype=np.int64)
    with ProcessPoolExecutor(WORKERS) as pool:
        for (centres20, number20), (centres24, number24) in pool.map(count_slice, *zip(*jobs)):
            level20[centres20] += number20
            level24[centres24] += number24

    # the scan saw every release row once when its level 24 counts are the released populations
    released = np.load(RELEASE / 'mad-core' / 'voronoi_population.npy')
    assert np.array_equal(level24, released), 'level 24 counts differ from the release'

    # the rows removed from the record were assigned to centre 0 at every level
    summary = json.loads((REMOVED / 'drop_summary.json').read_text())
    removed = summary['cut'][1] - summary['cut'][0]
    with h5py.File(REMOVED / 'before' / 'madcosmo_cells_0-2pow18_data.h5', 'r') as before:
        centres = before['assignments/center_index'][summary['cut'][0]:summary['cut'][1]]
    assert (centres == 0).all() and removed == 67_975
    level20[0] -= removed
    level24[0] -= removed
    assert np.array_equal(level24, np.load(RECORD_LEVEL24))

    assert int(level20.sum()) == 1_694_434_380 == int(level24.sum())
    assert not OUTPUT.exists()
    np.save(OUTPUT, level20)

    july = np.load(JULY)
    report = dict(path=str(OUTPUT), shape=list(level20.shape), dtype=str(level20.dtype), sum=int(level20.sum()), row_0=int(level20[0]),
                  rows_differing_from_july=int((level20 != july).sum()), july_sum=int(july.sum()), july_row_0=int(july[0]),
                  level24_matches_record_file=True)
    (ANALYSIS / 'populations_level20_record.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
