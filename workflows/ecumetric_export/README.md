---
license: bsd-3-clause
language:
  - en
tags:
  - atomistic-machine-learning
  - materials-science
---

# ecumetric

This repository contains the ecumetric model and sketch-map projector which were used in
MAD-CORE.

The ecumetric model provides descriptors used to construct the
MAD-CORE hierarchy. The model uses the SPACE architecture of
[metatrain](https://github.com/metatensor/metatrain). Its per-atom `feature` output is
pooled into one 512-dimensional vector per structure, with concatenation of the mean and
the standard deviation over atoms. Distances between these vectors, after scaling,
define the ecumetric metric.

The ecumetric explorer model maps the ecumetric descriptors to eight-dimensional
sketch-map coordinates for exploration and visualization.

## Exploring a dataset with chemiscope

```python
import ase.io
import chemiscope

structures = ase.io.read("structures.xyz", ":")
chemiscope.explore(structures, featurizer="ecumetric-1.0")
```

## Citation

- MAD-CORE paper (in preparation)
- SPACE architecture: https://arxiv.org/abs/2610.10830
- sketch-map: https://doi.org/10.1073/pnas.1108486108
