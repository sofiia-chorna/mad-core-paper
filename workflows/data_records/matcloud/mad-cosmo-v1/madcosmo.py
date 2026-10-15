import argparse
from itertools import islice
from pathlib import Path

import h5py
import numpy as np

try:
    from ase import Atoms
    from ase.io import write
except ImportError:
    Atoms = None
    write = None


PAIR_KEYS = ('schema', 'cell_start', 'cell_stop', 'member_start', 'member_stop', 'root_manifest_sha256')
FILTER_CODES = (0, 1, 2, 3)


class MADCosmo:
    def __init__(self, data_path, features_path=None):
        self.data = h5py.File(data_path, 'r')
        self.features = None

        if self.data.attrs['schema'] != 'madcosmo-hdf5-v1' or self.data.attrs['kind'] != 'data':
            self.data.close()
            raise ValueError('not a MAD-COSMO data file')

        if features_path is not None:
            self.features = h5py.File(features_path, 'r')
            mismatch = self.features.attrs['kind'] != 'features' or any(
                self.features.attrs[key] != self.data.attrs[key] for key in PAIR_KEYS)
            if mismatch:
                self.close()
                raise ValueError('features file does not match the data file')

        self.cell_start = int(self.data.attrs['cell_start'])
        self.cell_stop = int(self.data.attrs['cell_stop'])
        self.image_rows = self.data['images/rows'][:]

    def close(self):
        self.data.close()
        if self.features is not None:
            self.features.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        return int(self.data.attrs['structures'])

    # file rows of one cell, including rejected and duplicate rows
    def cell_rows(self, cell_id):
        if not self.cell_start <= cell_id < self.cell_stop:
            raise IndexError(f'cell {cell_id} is outside [{self.cell_start}, {self.cell_stop})')

        local = cell_id - self.cell_start
        first, last = self.data['index/cell_offsets'][local:local + 2]
        return range(int(first), int(last))

    # rows that pass the collection filters, plus MAD-NEXT rows unless excluded
    def iter_rows(self, cell_id=None, include_discarded=False, include_mad_next=True, block_rows=65536):
        rows = self.cell_rows(cell_id) if cell_id is not None else range(len(self))

        for start in range(rows.start, rows.stop, block_rows):
            stop = min(start + block_rows, rows.stop)
            mad_next = self.data['mad_next/present'][start:stop]

            if include_discarded:
                selected = np.ones(stop - start, dtype=bool)
            else:
                flagged = self.data['diagnostics/collection_flags_present'][start:stop]
                codes = self.data['diagnostics/filter_code'][start:stop]
                duplicate = self.data['diagnostics/is_duplicate'][start:stop]
                if np.any(~flagged & ~mad_next):
                    raise ValueError('Missing filter flags on a row outside MAD-NEXT')
                if np.any(flagged & ~np.isin(codes, FILTER_CODES)):
                    raise ValueError('Unknown collection filter code')
                selected = (flagged & (codes == 0) & ~duplicate) | mad_next

            if not include_mad_next:
                selected &= ~mad_next

            for row in np.flatnonzero(selected):
                yield start + int(row)

    def text(self, name, row):
        first, last = self.data[f'identity/{name}_offsets'][row:row + 2]
        return self.data[f'identity/{name}_bytes'][int(first):int(last)].tobytes().decode()

    def read(self, row, original_coordinates=False):
        if not 0 <= row < len(self):
            raise IndexError(row)

        d = self.data
        first, last = map(int, d['geometry/atom_offsets'][row:row + 2])
        flagged = bool(d['diagnostics/collection_flags_present'][row])

        frame = dict(
            record_id=int(d['record_id'][row]),
            configuration_id=self.text('configuration_id', row),
            source_reference=self.text('source_reference', row) if d['identity/source_reference_present'][row] else None,
            source_row=int(d['provenance/source_row'][row]),
            numbers=d['geometry/numbers'][first:last],
            positions=d['geometry/positions'][first:last],
            cell=d['geometry/cell'][row],
            pbc=d['geometry/pbc'][row],
            ecumetric_energy=d['predictions/ecumetric_energy'][row],
            ecumetric_nc_forces=d['predictions/ecumetric_nc_forces'][first:last],
            cell_id=int(d['ordering/cell_id'][row]),
            fps_order_in_cell=int(d['ordering/fps_order_in_cell'][row]),
            epsilon_sq_in_cell=d['ordering/epsilon_sq'][row],
            voronoi_a=d['assignments/center_index'][row],
            voronoi_d=d['assignments/distance'][row],
            filter_code=int(d['diagnostics/filter_code'][row]) if flagged else None,
            is_duplicate=bool(d['diagnostics/is_duplicate'][row]) if flagged else None,
            mad_next=None,
            image_flags=None,
            original_cell=None,
        )

        for name in ('dataset_id', 'dataset_group'):
            code = int(d[f'identity/{name}_code'][row])
            frame[name] = d[f'dictionaries/{name}'][code].decode()
        code = int(d['provenance/source_directory_code'][row])
        frame['source_directory'] = d['dictionaries/source_directory'][code].decode()

        for name in ('zbl_energy_per_atom', 'zbl_max_atomic_energy', 'zbl_max_force'):
            frame[name] = d[f'diagnostics/{name}'][row]

        if d['mad_next/present'][row]:
            frame['mad_next'] = {name: d[f'mad_next/{name}'][row] for name in ('fps_order', 'epsilon_sq', 'k_set', 'split')}
            frame['mad_next']['split'] = frame['mad_next']['split'].decode()

        position = int(np.searchsorted(self.image_rows, row))
        if position < len(self.image_rows) and self.image_rows[position] == row:
            image_first, image_last = map(int, d['images/atom_offsets'][position:position + 2])
            frame['image_flags'] = d['images/flags'][image_first:image_last]
            frame['original_cell'] = d['images/original_cell'][position]
            if original_coordinates:
                frame['positions'] = frame['positions'].astype(np.float64) + frame['image_flags'] @ frame['original_cell']
                frame['cell'] = frame['original_cell']

        if self.features is not None:
            if int(self.features['record_id'][row]) != frame['record_id']:
                raise ValueError(f'feature record_id mismatch at row {row}')
            frame['ecumetric_features'] = self.features['ecumetric_features'][row]

        return frame


def write_xyz(release, rows, path, original_coordinates):
    if Atoms is None:
        raise ImportError('install ase to write extended xyz')

    count = 0
    with path.open('x') as stream:
        for row in rows:
            frame = release.read(row, original_coordinates)
            atoms = Atoms(numbers=frame['numbers'], positions=frame['positions'], cell=frame['cell'], pbc=frame['pbc'])
            for name in ('configuration_id', 'dataset_id', 'dataset_group', 'ecumetric_energy', 'cell_id', 'fps_order_in_cell'):
                atoms.info[name] = frame[name]
            atoms.arrays['ecumetric_nc_forces'] = frame['ecumetric_nc_forces']
            write(stream, atoms, format='extxyz')
            count += 1

    print(f'wrote {count} structures to {path}')


def print_table(release, rows, original_coordinates, with_features):
    columns = ['row', 'configuration_id', 'atoms', 'fps_rank', 'filter_code', 'duplicate', 'mad_next']
    if with_features:
        columns.append('feature_dimensions')
    print('\t'.join(columns))

    for row in rows:
        frame = release.read(row, original_coordinates)
        values = [row, frame['configuration_id'], len(frame['numbers']), frame['fps_order_in_cell'],
                  frame['filter_code'], frame['is_duplicate'], frame['mad_next'] is not None]
        if with_features:
            values.append(len(frame['ecumetric_features']))
        print('\t'.join(map(str, values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--cell', type=int, help='defaults to the first cell in the file')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--xyz', type=Path, help='write the selected structures to a new extended xyz file')
    parser.add_argument('--original-coordinates', action='store_true')
    parser.add_argument('--include-discarded', action='store_true', help='also show rejected rows and duplicates')
    parser.add_argument('--exclude-mad-next', action='store_true')
    args = parser.parse_args()

    with MADCosmo(args.data, args.features) as release:
        cell_id = release.cell_start if args.cell is None else args.cell
        stored = release.cell_rows(cell_id)
        print(f'cell {cell_id}: {len(stored)} stored structures, file rows [{stored.start}, {stored.stop})')

        rows = release.iter_rows(cell_id, include_discarded=args.include_discarded,
                                 include_mad_next=not args.exclude_mad_next)
        rows = islice(rows, args.limit)

        if args.xyz:
            write_xyz(release, rows, args.xyz, args.original_coordinates)
        else:
            print_table(release, rows, args.original_coordinates, args.features is not None)


if __name__ == '__main__':
    main()
