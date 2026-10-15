import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols


STORE = Path('${STORE_ROOT}/mad-core')
FINAL = STORE / 'lda_pbe' / 'final'
REFERENCES = STORE / 'lda_pbe' / 'build' / 'references'
RELEASE = STORE / 'release'
CELL_FPS = RELEASE / 'mad-core' / 'cell_fps'
RECORD = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/xyz-2pow24-v2/files')
OUT = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/xyz-2pow24-v2/dft_internal')

ARMS = ('madcore_25k', 'random_25k', 'voronoi_25k', 'random_test_6k')
FUNCTIONALS = ('lda', 'pbe')
INTERVALS = ('0-2pow18', '2pow18-2pow19', '2pow19-2pow20', '2pow20-2pow21', '2pow21-2pow22', '2pow22-2pow23', '2pow23-2pow24')
INTERVAL_START = {name: (0 if index == 0 else 1 << (17 + index)) for index, name in enumerate(INTERVALS)}
KEY_VALUE = re.compile(r'(\w+)=("[^"]*"|\S+)')
SYMBOL = dict(enumerate(chemical_symbols))
POSITION_TOLERANCE = 1e-4
FORCE_CUT = 100.0
CHUNK = 200


def parse_header(line):
    return {key: value.strip('"') for key, value in KEY_VALUE.findall(line)}


# yield every frame of an extxyz file as header dict, species and the numeric columns
def read_frames(path):
    with open(path) as stream:
        while True:
            first = stream.readline()
            if not first:
                return
            count = int(first)
            header = parse_header(stream.readline())
            rows = [stream.readline().split() for _ in range(count)]
            species = [row[0] for row in rows]
            numbers = np.array([row[1:] for row in rows], dtype=np.float64).reshape(count, -1)
            yield header, species, numbers


def read_monomers(functional):
    energies = {}
    for header, species, _ in read_frames(REFERENCES / f'monomers_{functional}.xyz'):
        assert len(species) == 1
        energies[species[0]] = float(header['energy'])
    return energies


def interval_of(cell):
    return INTERVALS[max(18, int(cell).bit_length()) - 18]


def load_selected():
    selected = []
    for arm in ARMS:
        with open(FINAL / arm / 'manifest.tsv') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                entry = dict(arm=arm, configuration_id=row['configuration_id'], dataset_id=row['dataset_id'],
                             split_file=row['split_file'], split_row=int(row['split_row']), dft_bucket=row['dft_bucket'],
                             has_lda=row['has_lda'] == '1', has_pbe=row['has_pbe'] == '1',
                             e_lda=row['e_lda'], e_pbe=row['e_pbe'], fmax_lda=row['fmax_lda'], fmax_pbe=row['fmax_pbe'])
                if arm == 'madcore_25k':
                    entry['fps_order'] = int(row['fps_order'])
                elif arm == 'voronoi_25k':
                    entry['folder'] = row['release_folder']
                    entry['row'] = int(row['row_in_folder'])
                    entry['manifest_cell'] = int(row['cell'])
                    entry['member_rank'] = int(row['member_rank'])
                else:
                    group = row['dataset_group']
                    entry['folder'] = group if group == 'mad-next-coreset' else 'collection/' + group
                    entry['row'] = int(row['row_in_group'])
                selected.append(entry)
    return selected


# madcore picks point at their full record through the mad-core pointer columns
def resolve_madcore(selected, names):
    assert [name.decode() for name in np.load(RELEASE / 'mad-core' / 'release_directory_names.npy')] == names
    entries = [entry for entry in selected if entry['arm'] == 'madcore_25k']
    stop = max(entry['fps_order'] for entry in entries) + 1
    madcore_ids = np.load(RELEASE / 'mad-core' / 'configuration_id.npy', mmap_mode='r')[:stop]
    directory = np.load(RELEASE / 'mad-core' / 'release_directory.npy', mmap_mode='r')[:stop]
    row_in_directory = np.load(RELEASE / 'mad-core' / 'release_row_in_directory.npy', mmap_mode='r')[:stop]
    problems = []
    for entry in entries:
        order = entry['fps_order']
        if madcore_ids[order].decode() != entry['configuration_id']:
            problems.append(('madcore id', entry['configuration_id'], madcore_ids[order].decode()))
        entry['folder'] = names[int(directory[order])]
        entry['row'] = int(row_in_directory[order])
    return problems


def compare_geometry(header, species, positions, release_numbers, release_positions, release_cell, release_pbc):
    if len(species) != len(release_numbers) or any(SYMBOL[int(z)] != s for z, s in zip(release_numbers, species)):
        return 'species differ', None
    periodic = header['pbc'] == 'T T T'
    if periodic:
        lattice = np.array(header['Lattice'].split(), dtype=np.float64).reshape(3, 3)
        if np.abs(lattice - release_cell).max() > POSITION_TOLERANCE:
            return 'cell differs', None
        fractional = np.linalg.solve(lattice.T, (positions - release_positions).T).T
        shift = positions - release_positions - np.rint(fractional) @ lattice
    elif release_pbc.any():
        # isolated in the DFT but boxed in the release, atoms may be one image apart
        fractional = np.linalg.solve(release_cell.T, (positions - release_positions).T).T
        shift = positions - release_positions - np.rint(fractional) @ release_cell
    else:
        shift = positions - release_positions
    residual = float(np.abs(shift).max()) if len(shift) else 0.0
    return ('positions differ' if residual > POSITION_TOLERANCE else None), residual


# release lookups and geometry checks for a chunk of structures of one folder
def check_folder_chunk(folder, items):
    root = RELEASE / folder
    ids = np.load(root / 'configuration_id.npy', mmap_mode='r')
    offsets = np.load(root / 'na.npy', mmap_mode='r')
    numbers = np.memmap(root / 'a.bin', dtype=np.int32, mode='r')
    positions = np.memmap(root / 'x.bin', dtype=np.float32, mode='r').reshape(-1, 3)
    cells = np.memmap(root / 'c.bin', dtype=np.float32, mode='r').reshape(-1, 3, 3)
    pbc = np.load(root / 'pbc.npy', mmap_mode='r')
    filter_code = np.load(root / 'filter_code.npy', mmap_mode='r') if (root / 'filter_code.npy').exists() else None
    is_duplicate = np.load(root / 'is_duplicate.npy', mmap_mode='r') if (root / 'is_duplicate.npy').exists() else None
    assignment = np.memmap(root / 'voronoi_a.bin', dtype=np.int32, mode='r').reshape(-1, 25)

    results = []
    for row, arm, frames in sorted(items, key=lambda item: item[0]):
        first, last = int(offsets[row]), int(offsets[row + 1])
        release_numbers = np.asarray(numbers[first:last])
        release_positions = np.asarray(positions[first:last], dtype=np.float64)
        release_cell = np.asarray(cells[row], dtype=np.float64)
        release_pbc = np.asarray(pbc[row])
        result = dict(
            row=row, arm=arm, release_id=ids[row].decode(), cell=int(assignment[row, 24]),
            filter_code=int(filter_code[row]) if filter_code is not None else 0,
            is_duplicate=bool(is_duplicate[row]) if is_duplicate is not None else False,
            release_pbc=''.join('T' if flag else 'F' for flag in release_pbc), geometry={},
        )
        for functional, (header, species, frame_positions) in frames.items():
            result['geometry'][functional] = compare_geometry(header, species, frame_positions, release_numbers,
                                                              release_positions, release_cell, release_pbc)
        results.append(result)
    return folder, results


# member position in the cell ordering, its rank and whether it is the cell centre itself
def locate_cells(batch):
    center_offsets = np.load(CELL_FPS / 'center_offsets.npy', mmap_mode='r')
    directories = np.load(CELL_FPS / 'release_directory.npy', mmap_mode='r')
    rows = np.load(CELL_FPS / 'release_row_in_directory.npy', mmap_mode='r')
    ranks = np.load(CELL_FPS / 'fps_order_in_cell.npy', mmap_mode='r')
    madcore_directory = np.load(RELEASE / 'mad-core' / 'release_directory.npy', mmap_mode='r')
    madcore_row = np.load(RELEASE / 'mad-core' / 'release_row_in_directory.npy', mmap_mode='r')

    found = {}
    for cell, targets in batch:
        low, high = int(center_offsets[cell]), int(center_offsets[cell + 1])
        keys = (np.asarray(directories[low:high]).astype(np.int64) << 32) | np.asarray(rows[low:high]).astype(np.int64)
        hits = np.flatnonzero(np.isin(keys, np.array(targets, dtype=np.int64)))
        positions = defaultdict(list)
        for position in hits:
            positions[int(keys[position])].append(low + int(position))
        center_key = (int(madcore_directory[cell]) << 32) | int(madcore_row[cell])
        for target in targets:
            members = positions[target]
            found[(cell, target)] = dict(members=members, ranks=[int(ranks[member]) for member in members],
                                         is_center=target == center_key)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=64)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    names = [name.decode() for name in np.load(CELL_FPS / 'release_directory_names.npy')]
    index_of = {name: index for index, name in enumerate(names)}
    selected = load_selected()
    problems = resolve_madcore(selected, names)
    by_key = {(entry['arm'], entry['configuration_id']): entry for entry in selected}

    # every labelled frame, kept for the label checks and the geometry comparison
    frames = defaultdict(dict)
    duplicated_frames = []
    for arm in ARMS:
        for functional in FUNCTIONALS:
            for header, species, numbers in read_frames(FINAL / arm / f'{functional}.xyz'):
                key = (arm, header['configuration_id'])
                if key not in by_key:
                    problems.append(('frame not in manifest', arm, functional, header['configuration_id']))
                    continue
                if functional in frames[key]:
                    duplicated_frames.append(key)
                frames[key][functional] = (header, species, numbers)

    tasks = defaultdict(list)
    for entry in selected:
        key = (entry['arm'], entry['configuration_id'])
        positions = {functional: (header, species, numbers[:, :3]) for functional, (header, species, numbers) in frames.get(key, {}).items()}
        tasks[entry['folder']].append((entry['row'], entry['arm'], positions))
    jobs = [(folder, items[start:start + CHUNK]) for folder, items in tasks.items() for start in range(0, len(items), CHUNK)]

    release = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for folder, results in pool.map(check_folder_chunk, *zip(*jobs)):
            for result in results:
                release[(folder, result['row'], result['arm'])] = result

    for entry in selected:
        result = release[(entry['folder'], entry['row'], entry['arm'])]
        entry.update(cell=result['cell'], filter_code=result['filter_code'], is_duplicate=result['is_duplicate'],
                     release_pbc=result['release_pbc'])
        if result['release_id'] != entry['configuration_id']:
            problems.append(('release id', entry['arm'], entry['configuration_id'], result['release_id']))
        if entry['arm'] == 'voronoi_25k' and entry['cell'] != entry['manifest_cell']:
            problems.append(('voronoi cell', entry['configuration_id'], entry['cell'], entry['manifest_cell']))
        if entry['arm'] == 'madcore_25k' and entry['cell'] != entry['fps_order']:
            problems.append(('madcore cell', entry['configuration_id'], entry['cell'], entry['fps_order']))

    # cells are searched in batches, the largest cells alone
    population = np.diff(np.load(CELL_FPS / 'center_offsets.npy'))
    targets = defaultdict(set)
    for entry in selected:
        targets[entry['cell']].add((index_of[entry['folder']] << 32) | entry['row'])
    cells = sorted(targets, key=lambda cell: -population[cell])
    batches, current, current_size = [], [], 0
    for cell in cells:
        current.append((cell, sorted(targets[cell])))
        current_size += int(population[cell])
        if current_size > 2_000_000 or len(current) >= 500:
            batches.append(current)
            current, current_size = [], 0
    if current:
        batches.append(current)

    located = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for found in pool.map(locate_cells, batches):
            located.update(found)

    center_offsets = np.load(CELL_FPS / 'center_offsets.npy', mmap_mode='r')
    for entry in selected:
        found = located[(entry['cell'], (index_of[entry['folder']] << 32) | entry['row'])]
        if len(found['members']) != 1:
            problems.append(('not once in its cell', entry['arm'], entry['configuration_id'], entry['cell'], len(found['members'])))
            continue
        entry['member'] = found['members'][0]
        entry['rank_in_cell'] = found['ranks'][0]
        entry['interval'] = interval_of(entry['cell'])
        entry['madcosmo_row'] = entry['member'] - int(center_offsets[INTERVAL_START[entry['interval']]])
        entry['madcore_fps_order'] = entry['cell'] if found['is_center'] else -1
        if entry['arm'] == 'voronoi_25k' and entry['rank_in_cell'] != entry['member_rank']:
            problems.append(('voronoi rank', entry['configuration_id'], entry['rank_in_cell'], entry['member_rank']))
        if entry['arm'] == 'madcore_25k' and not found['is_center']:
            problems.append(('madcore pick is not its cell centre', entry['configuration_id']))

    monomers = {functional: read_monomers(functional) for functional in FUNCTIONALS}
    label_stats = defaultdict(Counter)
    worst = defaultdict(float)
    residuals = []
    geometry_flags = Counter()
    for entry in selected:
        key = (entry['arm'], entry['configuration_id'])
        result = release[(entry['folder'], entry['row'], entry['arm'])]
        for functional in FUNCTIONALS:
            labelled = functional in frames.get(key, {})
            if labelled != entry['has_' + functional]:
                problems.append(('has flag', entry['arm'], functional, entry['configuration_id']))
            if not labelled:
                continue
            header, species, numbers = frames[key][functional]
            stats = label_stats[(entry['arm'], functional)]
            stats['frames'] += 1

            energy = float(header['energy'])
            forces = numbers[:, 3:6]
            fmax = float(np.linalg.norm(forces, axis=1).max())
            corrected = energy - sum(monomers[functional][symbol] for symbol in species)
            worst['corrected_energy ' + functional] = max(worst['corrected_energy ' + functional], abs(corrected - float(header['corrected_energy'])))
            worst['manifest energy ' + functional] = max(worst['manifest energy ' + functional], abs(energy - float(entry['e_' + functional])))
            worst['manifest fmax ' + functional] = max(worst['manifest fmax ' + functional], abs(fmax - float(entry['fmax_' + functional])))

            if header['free_energy'] != header['energy']:
                stats['free_energy differs from energy'] += 1
            if header.get('atomization_energy') != header['corrected_energy']:
                stats['atomization_energy differs from corrected_energy'] += 1
            if not np.isfinite(energy) or not np.isfinite(numbers).all():
                stats['nonfinite'] += 1
            if header['functional'] != functional or header['subset'] != entry['arm']:
                stats['wrong functional or subset'] += 1
            if header.get('label_source', entry['arm']) != entry['arm']:
                stats['label taken from ' + header['label_source']] += 1
            periodic = header['pbc'] == 'T T T'
            stats['periodic' if periodic else 'isolated'] += 1
            if periodic != ('stress' in header):
                stats['stress presence wrong'] += 1
            if numbers.shape != (len(species), 6):
                stats['columns wrong'] += 1
            if fmax > FORCE_CUT:
                stats['above the 100 eV/A force cut'] += 1
            if not periodic and 'T' in entry['release_pbc']:
                stats['isolated here, boxed in the release'] += 1
            if periodic and entry['release_pbc'] != 'TTT':
                stats['periodic here, release pbc ' + entry['release_pbc']] += 1

            flag, residual = result['geometry'][functional]
            if flag:
                geometry_flags[f"{entry['arm']} {functional} {flag}"] += 1
                problems.append(('geometry', entry['arm'], functional, entry['configuration_id'], flag, residual))
            if residual is not None:
                residuals.append(residual)

    known = set()
    with open(RECORD / 'dataset_metadata.csv') as stream:
        for row in csv.DictReader(stream):
            known.add(row['dataset_id'])
    labelled_entries = [entry for entry in selected if entry['has_lda'] or entry['has_pbe']]
    missing_datasets = Counter(entry['dataset_id'] for entry in labelled_entries if entry['dataset_id'] not in known)

    counts = {}
    for arm in ARMS:
        entries = [entry for entry in selected if entry['arm'] == arm]
        labelled = [entry for entry in entries if entry['has_lda'] or entry['has_pbe']]
        counts[arm] = dict(
            selected=len(entries),
            lda=sum(entry['has_lda'] for entry in entries),
            pbe=sum(entry['has_pbe'] for entry in entries),
            both=sum(entry['has_lda'] and entry['has_pbe'] for entry in entries),
            neither=sum(not entry['has_lda'] and not entry['has_pbe'] for entry in entries),
            labelled_in_madcore=sum(entry.get('madcore_fps_order', -1) >= 0 for entry in labelled),
            labelled_filter_codes=dict(Counter(entry['filter_code'] for entry in labelled)),
            labelled_duplicates_in_release=sum(entry['is_duplicate'] for entry in labelled),
            labelled_by_interval=dict(Counter(entry.get('interval') for entry in labelled)),
        )
    lda_ids = {entry['configuration_id'] for entry in selected if entry['has_lda']}
    pbe_ids = {entry['configuration_id'] for entry in selected if entry['has_pbe']}
    counts['distinct structures over all arms'] = dict(lda=len(lda_ids), pbe=len(pbe_ids), both=len(lda_ids & pbe_ids),
                                                       either=len(lda_ids | pbe_ids))
    overlaps = Counter()
    arm_ids = {arm: {entry['configuration_id'] for entry in selected if entry['arm'] == arm} for arm in ARMS}
    for index, first in enumerate(ARMS):
        for second in ARMS[index + 1:]:
            overlaps[f'{first} and {second}'] = len(arm_ids[first] & arm_ids[second])

    with open(OUT / 'structures.tsv', 'w', newline='') as stream:
        columns = ['arm', 'configuration_id', 'dataset_id', 'folder', 'row', 'filter_code', 'is_duplicate', 'release_pbc',
                   'cell', 'rank_in_cell', 'madcore_fps_order', 'interval', 'madcosmo_row', 'has_lda', 'has_pbe',
                   'dft_bucket', 'split_file', 'split_row']
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(columns)
        for entry in selected:
            writer.writerow([int(entry[column]) if isinstance(entry.get(column), bool) else entry.get(column, '') for column in columns])

    report = dict(
        problem_count=len(problems), problem_kinds=dict(Counter(problem[0] for problem in problems)),
        problems=[list(map(str, problem)) for problem in problems[:300]],
        duplicated_frames=[list(key) for key in duplicated_frames],
        counts=counts, overlaps_between_arms=dict(overlaps),
        label_checks={f'{arm} {functional}': dict(stats) for (arm, functional), stats in label_stats.items()},
        worst_differences=dict(worst), geometry_flags=dict(geometry_flags),
        max_position_residual=max(residuals), labelled_dataset_ids_missing_from_record_csv=dict(missing_datasets),
    )
    (OUT / 'verify_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'problems'}, indent=1))
    print(json.dumps(report['problems'][:20], indent=1))


if __name__ == '__main__':
    main()
