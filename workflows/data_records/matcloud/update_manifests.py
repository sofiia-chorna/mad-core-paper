import argparse
import hashlib
import json
import sys
from pathlib import Path

RECORD = Path("${DEV_ROOT}/phace-atomic/matcloud-upload/xyz-2pow24-v2")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", help="index.json family of files that are new in files/")
    args = parser.parse_args()

    sums = {}
    for line in (RECORD / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        sums[name[len("files/"):]] = digest

    index = json.loads((RECORD / "index.json").read_text())
    present = sorted(path.name for path in (RECORD / "files").iterdir())

    # a new file needs a family for index.json
    added = [name for name in present if name not in index["files"]]
    if added and not args.family:
        sys.exit(f"new files {added}, run again with --family")

    removed = sorted(set(index["files"]) - set(present))
    for name in removed:
        del index["files"][name]
        sums.pop(name, None)

    changed = []
    for name in present:
        path = RECORD / "files" / name
        digest = sha256(path)

        if sums.get(name) != digest:
            changed.append(name)

        sums[name] = digest
        entry = index["files"].get(name, {"family": args.family})
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = digest
        index["files"][name] = entry

    index["files"] = dict(sorted(index["files"].items()))

    (RECORD / "SHA256SUMS").write_text("".join(f"{sums[name]}  files/{name}\n" for name in sorted(sums)))
    (RECORD / "index.json").write_text(json.dumps(index, indent=1) + "\n")

    print("files", len(present))
    print("changed or new", changed)
    print("removed", removed)


if __name__ == "__main__":
    main()
