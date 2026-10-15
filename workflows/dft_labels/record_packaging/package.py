import csv
import gzip
import hashlib
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ase.io import read


STORE = Path('${STORE_ROOT}/mad-core')
FINAL = STORE / 'lda_pbe' / 'final'
SETS = STORE / 'lda_pbe' / 'build' / 'sets'
REFERENCES = STORE / 'lda_pbe' / 'build' / 'references'
COMMON = STORE / 'lda_pbe' / 'campaign_history' / 'common'
RELEASE = STORE / 'release'
FILES = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/xyz-2pow24-v2/files')
INTERNAL = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/xyz-2pow24-v2/dft_internal')

ARMS = ('madcore_25k', 'random_25k', 'voronoi_25k', 'random_test_6k')
FUNCTIONALS = ('lda', 'pbe')
KINDS = ('isolated_fff', 'periodic_ttt', 'surfaces_ttf')
TTF_MIN_C = 80.0
GZIP_LEVEL = 6
KEY_VALUE = re.compile(r'(\w+)=("[^"]*"|\S+)')

# the mad-core tier names the MAD rows by these keys of dataset_metadata.csv
MAD_DATASET = {'mad15': 'MAD-1.5', 'mad_next': 'mad_next', 'mad_sol': 'mad_sol'}

STRING_KEYS = ('configuration_id', 'dataset_id', 'dataset_group', 'subset', 'functional', 'control')


def parse_header(line):
    return {key: value.strip('"') for key, value in KEY_VALUE.findall(line)}


# yield every frame as the raw count line, the parsed header and the raw atom lines
def raw_frames(path):
    with open(path) as stream:
        while True:
            first = stream.readline()
            if not first:
                return
            header = parse_header(stream.readline())
            body = ''.join(stream.readline() for _ in range(int(first)))
            yield first, header, body


def goes_to_ttf(atoms):
    return (not atoms.pbc[-1]) or atoms.get_cell()[2][2] > TTF_MIN_C


# the control template of every selected structure, recomputed with the routing rule of the campaign
def control_kinds(structures):
    kinds = {}
    mismatches = Counter()
    for arm in ARMS:
        surfaces = read(SETS / arm / 'surfaces.extxyz', ':')
        for entry in structures[arm].values():
            split = entry['split_file']
            if split == 'fff':
                kind = 'isolated_fff'
            elif split == 'ttt':
                kind = 'periodic_ttt'
            else:
                kind = 'surfaces_ttf' if goes_to_ttf(surfaces[int(entry['split_row'])]) else 'periodic_ttt'
            # madcore labels carry no bucket, the rule is the only record of their template
            if entry['dft_bucket'] not in ('', 'madcore_25k') and entry['dft_bucket'] != kind:
                mismatches[arm] += 1
            kinds[(arm, entry['configuration_id'])] = kind
    assert not mismatches, mismatches
    return kinds


def dataset_names(structures):
    madcore_id = np.load(RELEASE / 'mad-core' / 'dataset_id.npy', mmap_mode='r')
    madcore_group = np.load(RELEASE / 'mad-core' / 'dataset_group.npy', mmap_mode='r')
    names = {}
    disagreements = []
    for arm in ARMS:
        for entry in structures[arm].values():
            folder = entry['folder']
            manifest_id = entry['dataset_id']
            if folder == 'mad-next-coreset':
                dataset_id = MAD_DATASET[manifest_id.split(':')[0]] if ':' in manifest_id else manifest_id
                dataset_group = 'mad-next'
            else:
                dataset_id = manifest_id
                dataset_group = folder.removeprefix('collection/')
            order = int(entry['madcore_fps_order'])
            if order >= 0:
                release_names = (madcore_id[order].decode(), madcore_group[order].decode())
                if release_names != (dataset_id, dataset_group):
                    disagreements.append((arm, entry['configuration_id'], (dataset_id, dataset_group), release_names))
                dataset_id, dataset_group = release_names
            names[(arm, entry['configuration_id'])] = (dataset_id, dataset_group)
    return names, disagreements


def quoted(key, value):
    return f'{key}="{value}"' if key in STRING_KEYS or ' ' in value else f'{key}={value}'


def build_header(header, extra):
    values = dict(header)
    values.update(extra)
    order = ['Lattice', 'Properties', 'configuration_id', 'dataset_id', 'dataset_group', 'subset', 'functional',
             'fps_order', 'madcore_cell', 'control', 'energy', 'free_energy', 'corrected_energy', 'stress', 'pbc']
    return ' '.join(quoted(key, values[key]) for key in order if key in values) + '\n'


def write_functional(functional, structures, kinds, names):
    path = FILES / f'dft_{functional}.extxyz.gz'
    written = set()
    counts = Counter()
    skipped = []
    with open(path, 'wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=GZIP_LEVEL, mtime=0) as packed:
        stream = io.TextIOWrapper(packed, encoding='ascii', newline='')
        for arm in ARMS:
            for first, header, body in raw_frames(FINAL / arm / f'{functional}.xyz'):
                configuration_id = header['configuration_id']
                if configuration_id in written:
                    skipped.append((arm, configuration_id))
                    continue
                written.add(configuration_id)
                entry = structures[arm][configuration_id]
                dataset_id, dataset_group = names[(arm, configuration_id)]
                extra = dict(dataset_id=dataset_id, dataset_group=dataset_group, subset=arm, functional=functional,
                             fps_order=entry['madcore_fps_order'], madcore_cell=entry['cell'], control=kinds[(arm, configuration_id)])
                assert header['subset'] == arm and header['functional'] == functional
                assert header['free_energy'] == header['energy']
                stream.write(first + build_header(header, extra) + body)
                counts[arm] += 1
        stream.flush()
        stream.detach()
    return path, counts, skipped


def write_control():
    path = FILES / 'dft_control.tar.gz'
    members = [(COMMON / 'control' / kind / f'control_{functional}.in', f'{kind}/control_{functional}.in')
               for kind in KINDS for functional in ('pbe', 'pw-lda')]
    with open(path, 'wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=GZIP_LEVEL, mtime=0) as packed:
        with tarfile.open(fileobj=packed, mode='w', format=tarfile.PAX_FORMAT) as archive:
            for source, name in members:
                data = source.read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                info.mtime = 0
                archive.addfile(info, io.BytesIO(data))
    return path, len(members)


def checksums(path):
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            sha.update(block)
            md5.update(block)
    return dict(bytes=path.stat().st_size, sha256=sha.hexdigest(), md5=md5.hexdigest())


def main():
    structures = defaultdict(dict)
    with open(INTERNAL / 'structures.tsv') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            structures[row['arm']][row['configuration_id']] = row

    kinds = control_kinds(structures)
    names, disagreements = dataset_names(structures)

    report = dict(files={}, frames={}, skipped_as_already_written={}, name_disagreements=[list(map(str, item)) for item in disagreements])
    for functional in FUNCTIONALS:
        path, counts, skipped = write_functional(functional, structures, kinds, names)
        report['frames'][functional] = dict(counts, total=sum(counts.values()))
        report['skipped_as_already_written'][functional] = [list(item) for item in skipped]
        report['files'][path.name] = checksums(path)

    for functional in FUNCTIONALS:
        target = FILES / f'dft_isolated_atoms_{functional}.xyz'
        target.write_bytes((REFERENCES / f'monomers_{functional}.xyz').read_bytes())
        report['files'][target.name] = checksums(target)

    path, member_count = write_control()
    report['files'][path.name] = checksums(path)
    report['control_members'] = member_count

    (INTERNAL / 'package_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(frames=report['frames'], skipped={key: len(value) for key, value in report['skipped_as_already_written'].items()},
                          name_disagreements=len(disagreements), files=report['files']), indent=1))


if __name__ == '__main__':
    main()
