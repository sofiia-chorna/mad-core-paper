import argparse
from pathlib import Path

import numpy as np

from madcosmo import MADCosmo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('features', type=Path)
    args = parser.parse_args()
    with MADCosmo(args.data, args.features) as release:
        first = int(release.data.attrs['cell_start'])
        last = int(release.data.attrs['cell_stop']) - 1
        candidates = {0, len(release)-1}
        for cell in (first, last):
            rows = release.cell_rows(cell)
            if len(rows):
                candidates.update((rows.start, rows.stop-1))
        if len(release.image_rows):
            candidates.add(int(release.image_rows[0]))
            candidates.add(int(release.image_rows[-1]))
        for row in sorted(candidates):
            frame = release.read(row)
            assert len(frame['numbers']) == len(frame['positions']) == len(frame['ecumetric_nc_forces'])
            assert frame['ecumetric_features'].shape == (512,)
            assert frame['voronoi_a'].shape == frame['voronoi_d'].shape == (25,)
            assert frame['voronoi_a'][24] == frame['cell_id']
            assert row in release.cell_rows(frame['cell_id'])
            assert frame['record_id'] == int(release.features['record_id'][row])
            if frame['image_flags'] is not None:
                unfolded = release.read(row, original_coordinates=True)
                expected = frame['positions'].astype(np.float64) + frame['image_flags'] @ frame['original_cell']
                assert unfolded['positions'].tobytes() == expected.tobytes()
                assert unfolded['cell'].tobytes() == frame['original_cell'].tobytes()
        for bad in (-1, len(release)):
            try:
                release.read(bad)
            except IndexError:
                pass
            else:
                raise AssertionError('Out-of-range row accepted')
        for bad in (first-1, last+1):
            try:
                release.cell_rows(bad)
            except IndexError:
                pass
            else:
                raise AssertionError('Out-of-range cell accepted')
    with MADCosmo(args.data) as release:
        assert 'ecumetric_features' not in release.read(0)
    try:
        with MADCosmo(args.data, args.data):
            pass
    except ValueError:
        pass
    else:
        raise AssertionError('Data file accepted as features file')
    print(f'PASS: reader boundaries, image reconstruction, optional features, pair checks; {len(candidates)} representative rows')


if __name__ == '__main__':
    main()
