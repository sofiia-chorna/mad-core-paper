project_dir="${DEV_ROOT}/phace-atomic"

feat_model="${STORE_ROOT}/mad-core/models/phace_atomic_ep92.pt"
feat_store_root="${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92"
feat_sidecar_root="$project_dir/featurize_zbl/sidecars"
feat_manifest_default="$project_dir/featurize_zbl/manifests/feat_manifest.txt"
feat_keep_cols="configuration_id,structure_hash,dataset_id,names,atomic_numbers,positions,cell,pbc,nsites,zbl_max_force,zbl_energy_per_atom,zbl_max_atomic_energy"

run_persistent_worker() {
    python "$project_dir/inference/persistent_worker.py" \
        --manifest    "${1:-${MANIFEST:-$feat_manifest_default}}" \
        --model       "$feat_model" \
        --store-root  "$feat_store_root" \
        --sidecar-root "$feat_sidecar_root" \
        --keep-cols   "$feat_keep_cols" --keep-cols-missing-ok \
        --meanstd --max-nsites 1000 --nl-backend metatomic
}
