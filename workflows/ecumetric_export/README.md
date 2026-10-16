How the model files of the records and of Hugging Face were made. The epoch 92
checkpoint named its architecture experimental.phace, which only exists in a private fork, so
on 2026-10-08 it was re-exported under the public name experimental.space with every weight
tensor unchanged (space_swap, the checks are in apply.sh). export_model.sh writes
ecumetric_model.pt, build_explorer.py packs the model, the scaler and the sketch-map
projector into ecumetric_explorer.pt for chemiscope, check.py compares both with the release
files, upload.py sends them to huggingface.co/lab-cosmo/ecumetric.
