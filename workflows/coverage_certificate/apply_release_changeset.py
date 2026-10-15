import hashlib
import json
import os
import shutil
import sys
from datetime import date

RELEASE_ROOT = "${STORE_ROOT}/mad-core/release"
TIER = f"{RELEASE_ROOT}/mad-core"
STAGING = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert/out/release_changeset"
BACKUP = f"{TIER}/_backup_epsilon_sq_2026-09-28"
ROOT_BACKUP = f"{RELEASE_ROOT}/_backup_2026-09-28"
ROOT_FILES = ["FORMAT.md", "dataset_metadata.tsv"]
EPSILON_ROW = ("| epsilon_sq.npy         | (ns,) | float64 | mad-next-coreset: squared separation distance at pick time. "
               "mad-core: squared distance from row i to its nearest earlier row (minimum over j < i) in the scaled ecumetric metric, "
               "recomputed in float64 from the released features, +inf at row 0. The smallest epsilon_sq among the first n rows is the "
               "squared separation of the first n picks. Inside the accrete-anchor block (rows < 856,012) the sequence is not monotone, "
               "each accretion level opens with anchor points admitted farthest-first, then plain FPS. Not an interatomic distance. "
               "The previous mad-core column (per-level proxy, float32 FPS values beyond row 856,012) is kept in mad-core/_backup_epsilon_sq_2026-09-28 |")
LEVELS_ROW = ("| levels.json            |       |         | mad-core only. Per-level table for the sizes 2^1..2^24 and the 856,012-row anchor: size, anchor members, "
              "exact separation, coverage over the 184,180,658-row accrete pool (run-time float32 value from the construction log, its reconstruction "
              "from the published margin, the exact lower bound with its witness row, exact upper bound pending), status of the separation-coverage check. "
              "The condition sep(S_k) >= cov_X(S_k) refers to the accrete pool X, not to the full collection |")
FILES = ["epsilon_sq.npy", "levels.json"]
OPTIONAL = ["nearest_earlier_pick.npy"]
APPLY = "--apply" in sys.argv
WITH_OPTIONAL = "--with-nearest" in sys.argv


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 26), b""):
            h.update(block)
    return h.hexdigest()


def edited_format(text):
    lines = text.split("\n")
    hits = [i for i, line in enumerate(lines) if line.startswith("| epsilon_sq.npy ") and "float64" in line]
    assert len(hits) == 1 and not any(line.startswith("| levels.json") for line in lines)
    lines[hits[0]] = EPSILON_ROW
    lines.insert(hits[0] + 1, LEVELS_ROW)
    return "\n".join(lines)


def main():
    files = FILES + (OPTIONAL if WITH_OPTIONAL else [])
    entries = json.load(open(f"{STAGING}/manifest_entries.json"))
    tier_manifest = json.load(open(f"{TIER}/MANIFEST.json"))
    root_manifest = json.load(open(f"{RELEASE_ROOT}/MANIFEST.json"))
    print(f"mode: {'APPLY' if APPLY else 'DRY RUN'}; files: {files}")
    print(f"backup dir: {BACKUP} (old epsilon_sq.npy + old MANIFEST.json)")
    for name in files:
        src = f"{STAGING}/{name}"
        print(f"  {name}: {os.path.getsize(src):,} B sha256 {entries[name]['sha256'][:16]}… -> {TIER}/{name}"
              + ("  (replaces existing)" if os.path.exists(f"{TIER}/{name}") else "  (new file)"))
    print(f"  {TIER}/MANIFEST.json: update entries, format_version {tier_manifest.get('format_version')} -> 3, updated_at {date.today().isoformat()}")
    print(f"  {RELEASE_ROOT}/MANIFEST.json: directories['mad-core'] sha256 refreshed")
    format_text = open(f"{RELEASE_ROOT}/FORMAT.md").read()
    new_format = edited_format(format_text)
    print(f"  {RELEASE_ROOT}/FORMAT.md: epsilon_sq row rewritten for both tiers, levels.json row added (backup in {ROOT_BACKUP})")
    print(f"  {RELEASE_ROOT}/MANIFEST.json: files entries refreshed for {ROOT_FILES}")
    if not APPLY:
        print(EPSILON_ROW)
        print(LEVELS_ROW)
        return

    os.makedirs(ROOT_BACKUP, exist_ok=True)
    for name in ROOT_FILES + ["MANIFEST.json"]:
        if not os.path.exists(f"{ROOT_BACKUP}/{name}"):
            shutil.copy2(f"{RELEASE_ROOT}/{name}", f"{ROOT_BACKUP}/{name}")
    with open(f"{RELEASE_ROOT}/.FORMAT.md.tmp", "w") as f:
        f.write(new_format)
    os.replace(f"{RELEASE_ROOT}/.FORMAT.md.tmp", f"{RELEASE_ROOT}/FORMAT.md")

    os.makedirs(BACKUP, exist_ok=True)
    for name in ("epsilon_sq.npy", "MANIFEST.json"):
        if not os.path.exists(f"{BACKUP}/{name}"):
            shutil.copy2(f"{TIER}/{name}", f"{BACKUP}/{name}")
    for name in files:
        tmp = f"{TIER}/.{name}.tmp"
        shutil.copyfile(f"{STAGING}/{name}", tmp)
        assert sha256(tmp) == entries[name]["sha256"], name
        os.replace(tmp, f"{TIER}/{name}")
        tier_manifest["files"][name] = entries[name]
    tier_manifest["format_version"] = 3
    tier_manifest["updated_at"] = date.today().isoformat()
    tier_manifest["epsilon_sq"] = ("float64 squared distance to the nearest earlier pick, recomputed 2026-09-28 from the released features; "
                                   "+inf at row 0; previous column (per-level proxy for rows < 856,012, float32 FPS values beyond) kept in _backup_epsilon_sq_2026-09-28")
    with open(f"{TIER}/MANIFEST.json", "w") as f:
        json.dump(tier_manifest, f, indent=1)
    root_manifest["directories"]["mad-core"] = sha256(f"{TIER}/MANIFEST.json")
    for name in ROOT_FILES:
        root_manifest["files"][name] = {"bytes": os.path.getsize(f"{RELEASE_ROOT}/{name}"), "sha256": sha256(f"{RELEASE_ROOT}/{name}")}
    root_manifest["updated_at"] = date.today().isoformat()
    with open(f"{RELEASE_ROOT}/MANIFEST.json", "w") as f:
        json.dump(root_manifest, f, indent=1)
    print("applied")


if __name__ == "__main__":
    main()
