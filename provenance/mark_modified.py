import csv
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

rows = list(csv.reader(open(os.path.join(HERE, "extracted_files.csv"))))[1:]
out = csv.writer(open(os.path.join(HERE, "modified_files.csv"), "w", newline=""))
out.writerow(["release_path", "original_sha256", "release_sha256"])

modified = 0

for release_path, original_path, source, revision, original_sha in rows:
    path = os.path.join(ROOT, release_path)

    if not os.path.exists(path):
        print("missing in release", release_path, file=sys.stderr)
        continue

    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()

    if digest != original_sha:
        out.writerow([release_path, original_sha, digest])
        modified += 1

print(modified, "of", len(rows), "copied files differ from their original")
