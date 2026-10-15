import copy
import unittest

import numpy as np

from export import local_cell_offsets, report_label
from parts import plan_interval, verify_parts


class PartsTests(unittest.TestCase):
    def check_partition(self, counts, maximum_rows):
        centers = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
        parts = plan_interval(centers, 0, len(counts), maximum_rows)
        verify_parts(centers, parts)
        expected = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
        chunks = []
        for part in parts:
            offsets = local_cell_offsets(centers, part)
            self.assertEqual(offsets[0], 0)
            self.assertEqual(offsets[-1], part['structures'])
            self.assertLessEqual(part['structures'], maximum_rows)
            ids = np.repeat(np.arange(part['cell_start'], part['cell_stop'], dtype=np.int32), np.diff(offsets))
            np.testing.assert_array_equal(ids, expected[part['member_start']:part['member_stop']])
            chunks.append(ids)
        np.testing.assert_array_equal(np.concatenate(chunks), expected)
        return centers, parts

    def test_giant_cell_spans_parts(self):
        centers, parts = self.check_partition([2, 31, 1], 5)
        self.assertEqual(len(parts), 7)
        self.assertEqual(parts[2]['cell_start'], parts[3]['cell_start'])

    def test_empty_cells_at_edges_and_boundaries(self):
        self.check_partition([0, 0, 3, 0, 2, 0, 0], 3)

    def test_whole_interval(self):
        centers, parts = self.check_partition([2, 0, 3], 100)
        self.assertEqual(len(parts), 1)
        np.testing.assert_array_equal(local_cell_offsets(centers, parts[0]), centers)

    def test_every_possible_cut(self):
        for maximum_rows in range(1, 20):
            self.check_partition([0, 3, 0, 11, 0, 2, 0], maximum_rows)

    def test_randomized_boundaries(self):
        generator = np.random.default_rng(7351)
        for unused in range(100):
            counts = generator.integers(0, 40, 30)
            self.check_partition(counts, int(generator.integers(1, 120)))

    def test_gap_rejected(self):
        centers, parts = self.check_partition([5, 10], 5)
        changed = copy.deepcopy(parts)
        changed[1]['member_start'] += 1
        with self.assertRaises(AssertionError):
            verify_parts(centers, changed)

    def test_overlap_rejected(self):
        centers, parts = self.check_partition([5, 10], 5)
        changed = copy.deepcopy(parts)
        changed[1]['member_start'] -= 1
        with self.assertRaises(AssertionError):
            verify_parts(centers, changed)

    def test_missing_part_rejected(self):
        centers, parts = self.check_partition([5, 10], 5)
        with self.assertRaises(AssertionError):
            verify_parts(centers, parts[:-1])

    def test_old_label_unchanged(self):
        self.assertEqual(report_label(dict(cell_start=0, cell_stop=2**18)), 'madcosmo_cells_0-2pow18')

    def test_doubled_parts_preserve_adjacent_ranges(self):
        centers = np.array([0, 0, 5, 6, 29, 29, 40], dtype=np.int64)
        small = plan_interval(centers, 0, len(centers)-1, 6)
        large = plan_interval(centers, 0, len(centers)-1, 12)
        verify_parts(centers, large)
        self.assertEqual(len(large), (len(small)+1)//2)
        for index, part in enumerate(large):
            self.assertEqual(part['member_start'], small[index*2]['member_start'])
            self.assertEqual(part['member_stop'], small[min(index*2+1, len(small)-1)]['member_stop'])


if __name__ == '__main__':
    unittest.main()
