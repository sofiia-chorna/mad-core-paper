import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from block_reader import BlockReader
from ram_checkpoint import RamCheckpoint


class BlockReaderTests(unittest.TestCase):
    def test_binary_and_npy_bitwise(self):
        generator = np.random.default_rng(54211)
        with tempfile.TemporaryDirectory(prefix='madcosmo-io-test-') as folder, BlockReader(4) as reader:
            for width in (1, 3, 25, 512):
                bits = generator.integers(0, 2**32, (5000, width), dtype=np.uint32)
                bits.flat[:6] = [0, 0x80000000, 0x7fc00001, 0xffc00200, 0x7f800000, 0xff800000]
                values = bits.view('<f4')
                binary = Path(folder) / f'{width}.bin'
                npy = Path(folder) / f'{width}.npy'
                values.tofile(binary)
                np.save(npy, values)
                indices = np.r_[4999, 0, np.arange(6), 31, 32, 33, generator.integers(0, 5000, 173)]
                for array in (np.memmap(binary, dtype='<f4', mode='r').reshape(-1, width), np.load(npy, mmap_mode='r')):
                    self.assertEqual(reader.read(array, indices).tobytes(), values[indices].tobytes())
                    self.assertEqual(reader.read(array[17:], indices[indices < 4983]).tobytes(), values[17:][indices[indices < 4983]].tobytes())
                    self.assertEqual(reader.read(array, []).shape, (0, width))
                    with self.assertRaises(AssertionError):
                        reader.read(array, [-1])
                    with self.assertRaises(AssertionError):
                        reader.read(array, [5000])

    def test_strings_and_ragged_indices(self):
        with tempfile.TemporaryDirectory(prefix='madcosmo-string-test-') as folder, BlockReader(2) as reader:
            path = Path(folder) / 'strings.npy'
            values = np.array([b'a\x00b', b'', b'abcdef', b'xyz'], dtype='S11')
            np.save(path, values)
            indices = [3, 1, 0, 3, 2]
            self.assertEqual(reader.read(np.load(path, mmap_mode='r'), indices).tobytes(), values[indices].tobytes())


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='madcosmo-checkpoint-test-')
        self.root = Path(self.temporary.name)
        self.staging = self.root / 'staging'
        self.staging.mkdir()
        np.save(self.staging / 'values.npy', np.arange(12, dtype=np.int32))
        self.store = RamCheckpoint(self.root / 'checkpoints', {'source': 'fixed', 'code': 'fixed'}, interval=0)
        self.progress = dict(report={'structures': 12}, completed_sources=[], source_positions={'7': 4})

    def tearDown(self):
        self.temporary.cleanup()

    def test_restore_exact_payload_and_partial_source_cursor(self):
        self.store(self.staging, self.progress)
        destination = self.root / 'restored'
        self.assertEqual(self.store.restore(destination), self.progress)
        self.assertEqual((destination / 'values.npy').read_bytes(), (self.staging / 'values.npy').read_bytes())

    def test_incomplete_generation_does_not_replace_committed_state(self):
        self.store(self.staging, self.progress)
        original = (self.store.root / 'CURRENT.json').read_bytes()
        with patch('ram_checkpoint.copy_checked', side_effect=OSError('interrupted copy')):
            with self.assertRaises(OSError):
                self.store(self.staging, self.progress)
        self.assertEqual((self.store.root / 'CURRENT.json').read_bytes(), original)
        self.assertEqual(self.store.restore(self.root / 'restored'), self.progress)

    def test_corrupt_payload_rejected(self):
        self.store(self.staging, self.progress)
        pointer = json.loads((self.store.root / 'CURRENT.json').read_text())['current']
        path = self.store.root / pointer['generation'] / 'values.npy'
        values = np.load(path, mmap_mode='r+')
        values[0] = 99
        values.flush()
        with self.assertRaises(AssertionError):
            self.store.restore(self.root / 'restored')

    def test_changed_binding_rejected(self):
        self.store(self.staging, self.progress)
        other = RamCheckpoint(self.store.root, {'source': 'different'})
        with self.assertRaises(AssertionError):
            other.restore(self.root / 'restored')

    def test_two_committed_generations_retained(self):
        for unused in range(3):
            self.store(self.staging, self.progress)
        self.assertEqual(len(list(self.store.root.glob('generation-*'))), 2)
        self.assertEqual(self.store.restore(self.root / 'restored'), self.progress)


if __name__ == '__main__':
    unittest.main()
