import csv
import io
from collections import Counter
from pathlib import Path

import numpy as np


TARGET = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1/uploaded/files/dataset_metadata.tsv')
MAD_NEXT = Path('${STORE_ROOT}/mad-core/release/mad-next-coreset')
TOTAL_ROWS = 1_694_502_355
DROPPED_COLUMN = 'store_subfolder'

# the MAD-CORE style row that holds the citation of each MAD group, and the words of its title
COARSE = {'mad15': ('MAD-1.5', 'MAD-1.5 subset'),
          'mad_next': ('mad_next', 'MAD-NEXT subset'),
          'mad_sol': ('mad_sol', 'solvated-systems subset')}


def read(raw, terminator):
    rows = list(csv.reader(io.StringIO(raw.decode(), newline=''), delimiter='\t'))
    check = io.StringIO(newline='')
    csv.writer(check, delimiter='\t', lineterminator=terminator).writerows(rows)
    assert check.getvalue().encode() == raw
    return rows


# MAD-COSMO names the MAD structures by subset, so the table gets one row per subset
def main():
    raw = TARGET.read_bytes()
    terminator = '\r\n' if raw.split(b'\n', 1)[0].endswith(b'\r') else '\n'
    rows = read(raw, terminator)
    header, body = rows[0], rows[1:]
    column = {name: position for position, name in enumerate(header)}

    coarse = {row[column['dataset_id']]: row for row in body if row[column['dataset_id']] in {name for name, _ in COARSE.values()}}
    assert len(coarse) == len(COARSE)
    kept = [row for row in body if row[column['dataset_id']] not in coarse]

    names = np.load(MAD_NEXT / 'dataset_id.npy')
    groups = np.load(MAD_NEXT / 'dataset_group.npy')
    counts = Counter(zip((name.decode() for name in names), (group.decode() for group in groups)))
    assert sum(counts.values()) == len(names)

    added = []
    for (name, group), count in sorted(counts.items()):
        source_name, words = COARSE[group]
        assert name.startswith(group + ':')
        row = list(coarse[source_name])
        row[column['dataset_id']] = name
        row[column['dataset_group']] = group
        row[column['rows']] = str(count)
        row[column['title']] = f'Massive Atomic Diversity (MAD) dataset, {words}, {name.split(":", 1)[1]}'
        added.append(row)
    assert sum(int(row[column['rows']]) for row in added) == sum(int(row[column['rows']]) for row in coarse.values())

    position = column[DROPPED_COLUMN]
    table = [header] + kept + added
    table = [row[:position] + row[position + 1:] for row in table]
    assert sum(int(row[header.index('rows')]) for row in table[1:]) == TOTAL_ROWS

    out = io.StringIO(newline='')
    csv.writer(out, delimiter='\t', lineterminator=terminator).writerows(table)
    TARGET.write_bytes(out.getvalue().encode())
    print(f'{len(coarse)} MAD rows replaced by {len(added)} subset rows, {DROPPED_COLUMN} dropped, {len(table) - 1} datasets')


if __name__ == '__main__':
    main()
