import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

from block_reader import BlockReader
from export import expanded_indices
from preflight import INDEX, ROOT


RUN = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1')
PLAN = RUN / 'production-v1/plan.json'
OUTPUT = RUN / 'thread-probe-v1'
SOURCE = 'collection/gdb13'
THREADS = (8, 16, 32, 64, 128)
ROWS_PER_TRIAL = 250000
BATCH_ROWS = 4096

PER_ROW_BINARY = (
    ('c', '<f4', (-1, 3, 3)),
    ('ecumetric_energy', '<f4', (-1,)),
    ('ecumetric_features', '<f4', (-1, 512)),
    ('voronoi_a', '<i4', (-1, 25)),
    ('voronoi_d', '<f4', (-1, 25)),
)
PER_ATOM_BINARY = (
    ('a', '<i4', (-1,)),
    ('x', '<f4', (-1, 3)),
    ('ecumetric_nc_forces', '<f4', (-1, 3)),
)
PER_ROW_NPY = (
    'pbc', 'zbl_energy_per_atom', 'zbl_max_atomic_energy', 'zbl_max_force',
    'dataset_id', 'configuration_id', 'filter_code', 'is_duplicate',
)


def source_code(name):
    names = np.load(INDEX / 'release_directory_names.npy')
    matches = [code for code in range(len(names)) if bytes(names[code]).decode() == name]
    assert len(matches) == 1, name
    return matches[0]


# pick the deepest part, away from the canary and ramp parts, that holds enough gdb13 rows
def choose_rows(plan, code, needed):
    excluded = set(plan['canary_indices'] + plan['ramp_indices'])
    directory = np.load(INDEX / 'release_directory.npy', mmap_mode='r')
    rows = np.load(INDEX / 'release_row_in_directory.npy', mmap_mode='r')
    for index in reversed(range(len(plan['parts']))):
        if index in excluded:
            continue
        part = plan['parts'][index]
        lo, hi = part['member_start'], part['member_stop']
        chosen = rows[lo:hi][directory[lo:hi] == code].astype(np.int64)
        if len(chosen) >= needed:
            return index, part['label'], np.sort(chosen)
    raise AssertionError('No part with enough rows from ' + SOURCE)


def open_source(directory):
    arrays = {'na': np.load(directory / 'na.npy', mmap_mode='r')}
    for name, dtype, shape in PER_ROW_BINARY + PER_ATOM_BINARY:
        arrays[name] = np.memmap(directory / (name + '.bin'), mode='r', dtype=dtype).reshape(shape)
    for name in PER_ROW_NPY:
        arrays[name] = np.load(directory / (name + '.npy'), mmap_mode='r')
    return arrays


# one batch reads the same arrays the exporter reads for a batch of selected rows
def read_batch(reader, arrays, rows):
    starts = reader.read(arrays['na'], rows)
    counts = reader.read(arrays['na'], rows + 1) - starts
    atoms = expanded_indices(starts, counts)

    payload = 0
    for name, _, _ in PER_ATOM_BINARY:
        payload += reader.read(arrays[name], atoms).nbytes
    for name, _, _ in PER_ROW_BINARY:
        payload += reader.read(arrays[name], rows).nbytes
    for name in PER_ROW_NPY:
        payload += reader.read(arrays[name], rows).nbytes

    return payload, int(counts.sum())


def check_bits(reader, arrays, rows):
    for name in ('ecumetric_features', 'voronoi_a', 'configuration_id'):
        fast = reader.read(arrays[name], rows)
        reference = np.asarray(arrays[name][rows])
        assert fast.dtype == reference.dtype and fast.tobytes() == reference.tobytes(), name


def run_trial(arrays, threads, rows):
    with BlockReader(threads) as reader:
        check_bits(reader, arrays, rows[:BATCH_ROWS])

        started = time.monotonic()
        payload = 0
        atoms = 0
        for begin in range(0, len(rows), BATCH_ROWS):
            batch_payload, batch_atoms = read_batch(reader, arrays, rows[begin:begin + BATCH_ROWS])
            payload += batch_payload
            atoms += batch_atoms
        seconds = time.monotonic() - started

    return dict(threads=threads, rows=len(rows), atoms=atoms, payload_bytes=payload, seconds=seconds,
                rows_per_second=len(rows) / seconds, payload_mib_per_second=payload / seconds / 2**20,
                first_row=int(rows[0]), last_row=int(rows[-1]))


def summarize(trials):
    summary = {}
    for threads in THREADS:
        rates = [trial['rows_per_second'] for trial in trials if trial['threads'] == threads]
        summary[str(threads)] = dict(trials=len(rates), mean_rows_per_second=float(np.mean(rates)),
                                     min_rows_per_second=float(np.min(rates)), max_rows_per_second=float(np.max(rates)))
    base = summary[str(THREADS[0])]['mean_rows_per_second']
    for entry in summary.values():
        entry['speedup_vs_8'] = entry['mean_rows_per_second'] / base
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / 'result.json'

    plan = json.loads(PLAN.read_text())
    code = source_code(SOURCE)

    # thread counts go up then down so a drifting cache or server load cancels between the two trials
    schedule = list(THREADS) + list(reversed(THREADS))
    index, label, chosen = choose_rows(plan, code, ROWS_PER_TRIAL * len(schedule))
    arrays = open_source(ROOT / SOURCE)

    result = dict(status='RUNNING', host=socket.gethostname(), source=SOURCE, part_index=index, part_label=label,
                  source_rows_available=len(chosen), rows_per_trial=ROWS_PER_TRIAL, batch_rows=BATCH_ROWS,
                  schedule=schedule, trials=[])
    print(json.dumps({key: value for key, value in result.items() if key != 'trials'}), flush=True)

    for position, threads in enumerate(schedule):
        rows = chosen[position * ROWS_PER_TRIAL:(position + 1) * ROWS_PER_TRIAL]
        trial = run_trial(arrays, threads, rows)
        result['trials'].append(trial)
        print(json.dumps(trial), flush=True)
        result_path.write_text(json.dumps(result, indent=2) + '\n')

    result['summary'] = summarize(result['trials'])
    result['status'] = 'PASS'
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
