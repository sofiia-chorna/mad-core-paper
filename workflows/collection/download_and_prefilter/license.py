from huggingface_hub import HfApi

api = HfApi()

# Option 1: from tags (lightweight, no full=True needed)
datasets = list(api.list_datasets(
    author="colabfit",
    expand=["tags", "description", "downloads"],
))
for ds in datasets:
    license_tags = [t.split(":")[-1] for t in (ds.tags or []) if t.startswith("license:")]
    license_str = license_tags[0] if license_tags else "not specified"
    print(f"{ds.id:50s}  {license_str:20s}  {ds.downloads} downloads")

licenses = {}
# Option 2: from card_data (richer, needs full=True)
datasets = list(api.list_datasets(author="colabfit", full=True))
for ds in datasets:
    license_str = "not specified"
    if ds.card_data:
        license_str = ds.card_data.get("license", "not specified")
    licenses[ds.id] = license_str    
