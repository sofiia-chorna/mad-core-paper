# Replaced ecumetric model files, 2026-10-08

This folder keeps the `ecumetric_model.pt` and `ecumetric_model.ckpt` that the release
shipped until 2026-10-08, and the root `MANIFEST.json` as it was before they were replaced.
They are kept because earlier records pin their sha256 (for example the mad-cosmo-v1
`madcosmo_source_manifests.json`).

The old checkpoint names its architecture `experimental.phace`, which only exists in a
private metatrain fork, and the old exported model only loads with metatomic-torch 0.1.11.
The new checkpoint holds the same weights under the public metatrain architecture
`experimental.space`. The new exported model was made from it with
`mtt export` (metatrain 2026.4, metatomic-torch 0.1.17, torch 2.9.1).

Checks done before the swap

- all 530 weight tensors of the old checkpoint are bit-identical in the new one, for the
  current, best and EMA weights. The new checkpoint adds 14 output scaler buffers, the newer
  metatrain storage of the same scaling
- on 256 mad-core rows (the first 64 and 192 random) the energy is bit-identical, and the
  features, non-conservative forces and the five heads agree to float32 rounding
  (largest relative difference 4.4e-7)
- both models are equally close to `mad-core/ecumetric_features.bin`, at most 3.1e-5 in the
  scaled ecumetric metric, against a level 24 separation of 0.218

The old `ecumetric_model.pt` is byte-identical to `mad-core/models/phace_atomic_ep92.pt`,
the model all release features were computed with.

| file | old sha256 (this folder) | new sha256 (release root) |
| --- | --- | --- |
| ecumetric_model.pt | f2cf2608dc16260995399e363c07de2f2e111f7619e577a8b1e2ca0fb5b4b9cb | 34df930261ee78a19bae485e6b8cc159c179dc0eb03007906f8c72000132f5d6 |
| ecumetric_model.ckpt | 8a916e91e8cf86970ec7b77680cb26aec1ab12654ea56ed99c3dc4ce94af32b2 | de2a622436dd030961595fa622a8efbc1c257fd438dd4d2bf85fb65f50800d9d |

Build and verification scripts are in the repository under
`release/build/ecumetric_space_swap/`.
