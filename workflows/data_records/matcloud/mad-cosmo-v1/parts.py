import numpy as np

from export import report_label


def plan_interval(centers, cell_start, cell_stop, maximum_rows):
    assert 0 <= cell_start < cell_stop < len(centers)
    assert maximum_rows > 0
    first, last = map(int, centers[[cell_start, cell_stop]])
    assert first < last
    count = (last - first + maximum_rows - 1) // maximum_rows
    parts = []
    for number, lo in enumerate(range(first, last, maximum_rows), start=1):
        hi = min(lo + maximum_rows, last)
        start = cell_start if number == 1 else int(np.searchsorted(centers, lo, side='right')) - 1
        stop = cell_stop if number == count else int(np.searchsorted(centers, hi - 1, side='right'))
        part = dict(interval_cell_start=cell_start, interval_cell_stop=cell_stop,
                    cell_start=start, cell_stop=stop, member_start=lo, member_stop=hi,
                    structures=hi-lo, part_number=number, part_count=count)
        part['label'] = report_label(part)
        parts.append(part)
    return parts


def verify_parts(centers, parts):
    grouped = {}
    labels = set()
    for part in parts:
        assert part['label'] == report_label(part)
        assert part['label'] not in labels
        labels.add(part['label'])
        key = part['interval_cell_start'], part['interval_cell_stop']
        grouped.setdefault(key, []).append(part)
    for (start, stop), interval in grouped.items():
        ordered = sorted(interval, key=lambda part: part['part_number'])
        cursor = int(centers[start])
        for number, part in enumerate(ordered, start=1):
            assert part['part_number'] == number and part['part_count'] == len(ordered)
            assert part['member_start'] == cursor
            assert part['member_stop'] > cursor
            assert part['structures'] == part['member_stop'] - cursor
            assert start <= part['cell_start'] < part['cell_stop'] <= stop
            assert int(centers[part['cell_start']]) <= cursor
            assert int(centers[part['cell_stop']]) >= part['member_stop']
            cursor = part['member_stop']
        assert cursor == int(centers[stop])
    return grouped


def plan_remaining(centers, maximum_rows):
    intervals = [plan_interval(centers, 2**power, 2**(power+1), maximum_rows)
                 for power in (20, 21, 22, 23)]
    parts = []
    for number in range(max(map(len, intervals))):
        for interval in intervals:
            if number < len(interval):
                parts.append(interval[number])
    verify_parts(centers, parts)
    return parts
