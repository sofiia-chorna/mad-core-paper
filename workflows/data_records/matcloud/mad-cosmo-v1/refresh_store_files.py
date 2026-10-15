import hashlib
import shutil
from pathlib import Path


UPLOADED = Path('${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1/uploaded/files')
STORE = Path('${STORE_ROOT}/mad-core/matcloud/mad-cosmo-v1')
SYNCED = ('README.md', 'LICENSES.md', 'dataset_metadata.tsv', 'example.py',
          'madcosmo_cells_0-2pow18_data.h5', 'madcosmo_cells_0-2pow18_features.h5')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 26), b''):
            digest.update(block)
    return digest.hexdigest()


# these files follow the uploaded folder, the merged data files of the later intervals and their checksums stay as they are
def main():
    sums = {}
    for line in (STORE / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ')
        sums[name] = digest

    for name in SYNCED:
        source, target = UPLOADED / name, STORE / 'files' / name
        digest = sha256(source)
        if not target.exists() or sums.get(f'files/{name}') != digest or sha256(target) != digest:
            shutil.copyfile(source, target)
            print('copied', name, flush=True)
        assert sha256(target) == digest
        sums[f'files/{name}'] = digest

    (STORE / 'SHA256SUMS').write_text(''.join(f'{sums[name]}  {name}\n' for name in sorted(sums)))
    listed = {name.removeprefix('files/') for name in sums}
    present = {path.name for path in (STORE / 'files').iterdir()}
    print(f'{len(listed)} files listed in SHA256SUMS, {len(present)} in the store folder, same set: {listed == present}')


if __name__ == '__main__':
    main()
