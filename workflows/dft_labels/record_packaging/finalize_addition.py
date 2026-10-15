import hashlib
import json
from datetime import datetime
from pathlib import Path

RECORD = Path("${DEV_ROOT}/phace-atomic/matcloud-upload/xyz-2pow24-v2")
ADDITION = RECORD / "r2scan_addition"
RECORD_BEFORE = Path(__file__).resolve().parent / "out" / "record_before_addition"

# the files of the addition, other files in the staging folder are not part of the record
NEW = ("dft_r2scan.extxyz.gz", "dft_isolated_atoms_r2scan.xyz")
REPLACED = ("README.md", "LICENSES.md", "dft_control.tar.gz")
NEW_FAMILY = "dft"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    sums = {}
    for line in (RECORD_BEFORE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        sums[name] = digest

    index = json.loads((RECORD_BEFORE / "index.json").read_text())
    assert all(f"files/{name}" in sums for name in REPLACED) and not any(f"files/{name}" in sums for name in NEW)

    # the unchanged record files must still match their published checksums
    unchanged = [name for name in sums if name.split("/", 1)[1] not in REPLACED]
    for name in unchanged:
        assert sha256(RECORD / name) == sums[name], name

    for name in NEW + REPLACED:
        path = ADDITION / "files" / name
        digest = sha256(path)
        sums[f"files/{name}"] = digest

        entry = index["files"].get(name, {"family": NEW_FAMILY})
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = digest
        index["files"][name] = entry

    index["files"] = dict(sorted(index["files"].items()))
    index["record"]["r2scan_addition_built_at"] = datetime.now().isoformat(timespec="seconds")

    (ADDITION / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())))
    (ADDITION / "index.json").write_text(json.dumps(index, indent=1) + "\n")

    print(json.dumps({"files_in_record": len(sums), "unchanged_verified": len(unchanged), "new": NEW, "replaced": REPLACED}, indent=1))


if __name__ == "__main__":
    main()
