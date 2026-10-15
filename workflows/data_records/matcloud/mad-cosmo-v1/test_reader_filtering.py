import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from itertools import islice
from pathlib import Path
from unittest.mock import call, patch

import h5py
import numpy as np

from madcosmo import MADCosmo, main


class ReaderFilteringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / 'data.h5'
        with h5py.File(self.path, 'w') as data:
            data.attrs.update(schema='madcosmo-hdf5-v1', kind='data', structures=8,
                              cell_start=10, cell_stop=13)
            data['images/rows'] = np.array([], dtype=np.uint64)
            data['index/cell_offsets'] = [0, 6, 8, 8]
            data['diagnostics/collection_flags_present'] = [True] * 5 + [False, True, True]
            data['diagnostics/filter_code'] = [0, 1, 2, 3, 0, -1, 0, 1]
            data['diagnostics/is_duplicate'] = [False, False, False, False, True, False, False, True]
            data['mad_next/present'] = [False] * 5 + [True, False, False]

    def tearDown(self):
        self.temporary.cleanup()

    def test_default_excludes_every_rejection_code_and_duplicates(self):
        with MADCosmo(self.path) as release:
            self.assertEqual(list(release.iter_rows(block_rows=2)), [0, 5, 6])

    def test_all_rows_require_explicit_opt_in(self):
        with MADCosmo(self.path) as release:
            self.assertEqual(list(release.iter_rows(include_discarded=True)), list(range(8)))

    def test_mad_next_can_be_excluded_explicitly(self):
        with MADCosmo(self.path) as release:
            self.assertEqual(list(release.iter_rows(include_mad_next=False)), [0, 6])
            self.assertEqual(list(release.iter_rows(include_discarded=True, include_mad_next=False)),
                             [0, 1, 2, 3, 4, 6, 7])

    def test_cell_bounds_and_original_row_indices_are_preserved(self):
        with MADCosmo(self.path) as release:
            self.assertEqual(list(release.iter_rows(10, block_rows=1)), [0, 5])
            self.assertEqual(list(release.iter_rows(11)), [6])
            self.assertEqual(list(release.iter_rows(12)), [])
            self.assertEqual(list(release.cell_rows(11)), [6, 7])
            self.assertEqual(len(release), 8)

    def test_limit_applies_after_filtering(self):
        with MADCosmo(self.path) as release:
            self.assertEqual(list(islice(release.iter_rows(), 2)), [0, 5])

    def test_unknown_unflagged_rows_are_not_silently_kept(self):
        with h5py.File(self.path, 'r+') as data:
            data['mad_next/present'][5] = False
        with MADCosmo(self.path) as release:
            with self.assertRaisesRegex(ValueError, 'Missing filter flags'):
                list(release.iter_rows())

    def test_unknown_filter_codes_are_not_silently_kept(self):
        with h5py.File(self.path, 'r+') as data:
            data['diagnostics/filter_code'][0] = 99
        with MADCosmo(self.path) as release:
            with self.assertRaisesRegex(ValueError, 'Unknown collection filter code'):
                list(release.iter_rows())

    def test_invalid_cells_fail(self):
        with MADCosmo(self.path) as release:
            for cell in [9, 13]:
                with self.assertRaises(IndexError):
                    list(release.iter_rows(cell))

    def run_cli(self, *arguments):
        frame = dict(configuration_id='test', numbers=[1], fps_order_in_cell=0,
                     filter_code=0, is_duplicate=False, mad_next=None)
        output = StringIO()
        with patch('sys.argv', ['madcosmo.py', str(self.path), *arguments]):
            with patch.object(MADCosmo, 'read', return_value=frame) as read:
                with redirect_stdout(output):
                    main()
        return output.getvalue(), read.call_args_list

    def test_cli_defaults_to_first_cell_and_limits_selected_rows(self):
        output, calls = self.run_cli('--limit', '2')
        self.assertIn('cell 10: 6 stored structures', output)
        self.assertIn('filter_code\tduplicate\tmad_next', output)
        self.assertEqual(calls, [call(0, False), call(5, False)])

    def test_cli_discarded_rows_require_opt_in(self):
        output, calls = self.run_cli('--cell', '11', '--include-discarded', '--limit', '2')
        self.assertIn('cell 11: 2 stored structures', output)
        self.assertEqual(calls, [call(6, False), call(7, False)])

    def test_cli_zero_limit_does_not_decode_structures(self):
        output, calls = self.run_cli('--limit', '0')
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
