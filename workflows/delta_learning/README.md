The r2SCAN minus LDA and r2SCAN minus PBE corrections and the direct r2SCAN fine-tune.
The 20,881 MAD-CORE structures with all three labels are split once with RandomState(42)
into 18,000, 1,000 and 1,881 (scripts/build_shared_split.py, manifests/). Each model starts
from PET-OMat-S with a fresh head, a least squares composition baseline, learning rate 1e-4,
150 epochs, batch 64, and the checkpoint with the lowest validation energy times force MAE
(epochs 102, 101 and 113). A corrected prediction is the DFT value plus the learned
difference. reports/main/metrics.json of both experiments are the sources of the SI table.
The three checkpoints are in the models archive.
