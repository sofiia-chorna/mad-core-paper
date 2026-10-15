import os
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np


class BlockReader:
    def __init__(self, workers=8):
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.executor.shutdown(wait=True)

    def read(self, array, indices):
        indices = np.asarray(indices, dtype=np.int64)
        assert indices.ndim == 1
        assert array.flags.c_contiguous and not array.dtype.hasobject
        assert np.all((indices >= 0) & (indices < len(array)))
        output = np.empty((len(indices),) + array.shape[1:], dtype=array.dtype)
        if not len(indices):
            return output
        ordered = np.all(indices[1:] >= indices[:-1])
        order = np.arange(len(indices)) if ordered else np.argsort(indices, kind='stable')
        selected = indices[order]
        row_bytes = array.dtype.itemsize * int(np.prod(array.shape[1:]))
        assert row_bytes > 0
        base = array
        while isinstance(base.base, np.ndarray):
            base = base.base
        offset = int(base.offset) + array.ctypes.data - base.ctypes.data
        positions = offset + selected * row_bytes
        buckets = positions // (1 << 20)
        cuts = np.r_[0, np.flatnonzero(np.diff(buckets)) + 1, len(selected)]
        spans = []
        for first, stop in zip(cuts[:-1], cuts[1:]):
            count = int(stop - first)
            size = int(positions[stop - 1] - positions[first] + row_bytes)
            if size <= max(65536, count * row_bytes * 16):
                spans.append((int(first), int(stop)))
            else:
                pages = positions[first:stop] // 65536
                small = np.r_[first, first + np.flatnonzero(np.diff(pages)) + 1, stop]
                spans.extend((int(a), int(b)) for a, b in zip(small[:-1], small[1:]))
        descriptor = os.open(array.filename, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_RANDOM)

            def copy_span(bounds):
                first, stop = bounds
                start = int(positions[first])
                remaining = int(positions[stop - 1] - start + row_bytes)
                chunks = []
                cursor = start
                while remaining:
                    block = os.pread(descriptor, remaining, cursor)
                    if not block:
                        raise EOFError((str(array.filename), cursor, remaining))
                    chunks.append(block)
                    cursor += len(block)
                    remaining -= len(block)
                values = np.frombuffer(b''.join(chunks), dtype=array.dtype)
                values = values.reshape((-1,) + array.shape[1:])
                output[order[first:stop]] = values[selected[first:stop] - selected[first]]

            futures = [self.executor.submit(copy_span, span) for span in spans]
            try:
                for future in futures:
                    future.result()
            finally:
                wait(futures)
        finally:
            os.close(descriptor)
        return output
