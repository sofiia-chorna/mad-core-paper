import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


upload_dir = Path("${DEV_ROOT}/phace-atomic/huggingface/upload")
release_manifest = Path("${STORE_ROOT}/mad-core/release/MANIFEST.json")

repository = "lab-cosmo/ecumetric"
version_tag = "v1.0.0"

model_files = [
    "ecumetric_model.pt",
    "ecumetric_explorer.pt",
]
uploaded_files = ["README.md"] + model_files


def sha256_of(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(64 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


# the folder must hold exactly the uploaded files
found_files = sorted(path.name for path in upload_dir.iterdir())
if found_files != sorted(uploaded_files):
    raise SystemExit(f"{upload_dir} holds {found_files}, expected {sorted(uploaded_files)}")

# the model files must be the ones the release manifest pins
pinned_files = json.loads(release_manifest.read_text())["files"]

for name in model_files:
    file_hash = sha256_of(upload_dir / name)

    if name not in pinned_files:
        raise SystemExit(f"{name} is not in {release_manifest}")

    if file_hash != pinned_files[name]["sha256"]:
        raise SystemExit(f"{name} is not the file pinned in {release_manifest}")

    print(f"{name} {file_hash}, matches the release")

print(f"README.md {sha256_of(upload_dir / 'README.md')}")


api = HfApi()

# the repository already exists, its visibility is set on the Hugging Face page
print(f"{repository} private {api.repo_info(repository).private}")

commit = api.upload_folder(
    repo_id=repository,
    folder_path=upload_dir,
    allow_patterns=uploaded_files,
    commit_message=f"ecumetric {version_tag}",
)

# the version tag points at this commit, a tag of the same name from an earlier upload is moved
existing_tags = {tag.name: tag.target_commit for tag in api.list_repo_refs(repository).tags}

if version_tag in existing_tags:
    print(f"moving {version_tag} from {existing_tags[version_tag]} to {commit.oid}")
    api.delete_tag(repository, tag=version_tag)

api.create_tag(
    repository,
    tag=version_tag,
    revision=commit.oid,
)

print(f"uploaded to {commit.commit_url}, tagged {version_tag}")
