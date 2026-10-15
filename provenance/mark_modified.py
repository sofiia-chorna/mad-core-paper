# lists the copied files whose content differs from the original recorded in extracted_files.tsv
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
rows = [line.rstrip("\n").split("\t") for line in open(os.path.join(HERE, "extracted_files.tsv"))][1:]
out = open(os.path.join(HERE, "modified_files.tsv"), "w")
out.write("release_path\toriginal_sha256\trelease_sha256\n")
modified = 0
for release_path, original_path, source, revision, original_sha in rows:
    path = os.path.join(ROOT, release_path)
    if not os.path.exists(path):
        print("missing in release:", release_path, file=sys.stderr)
        continue
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != original_sha:
        out.write(f"{release_path}\t{original_sha}\t{digest}\n")
        modified += 1
print(f"{modified} of {len(rows)} copied files differ from their original")
