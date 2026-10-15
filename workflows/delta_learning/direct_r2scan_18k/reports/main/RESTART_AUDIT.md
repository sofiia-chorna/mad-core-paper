# direct_r2scan_18k, restart audit

Checks from the checkpoints and training log of each run: allocations, optimizer and scheduler step counts, normalization, best checkpoint tracking.

## direct_r2scan--omat-s--madcore_shared18000--s42

- 1 allocation(s); epochs logged per allocation [(0, 149)]; steps per epoch 281
- model.ckpt: epoch 149, best_epoch 113, best_metric 5.692388e-02, adam steps [42150], scheduler last_epoch 42150, expected 42150
- model.ckpt (final): epoch 149, best_epoch 113, best_metric 5.692388e-02, adam steps [42150], scheduler last_epoch 42150, expected 42150
- model_0.ckpt: epoch 0, best_epoch 0, best_metric 2.430058e+00, adam steps [281], scheduler last_epoch 281, expected 281
- model_100.ckpt: epoch 100, best_epoch 100, best_metric 5.877081e-02, adam steps [28381], scheduler last_epoch 28381, expected 28381
- model_125.ckpt: epoch 125, best_epoch 113, best_metric 5.692388e-02, adam steps [35406], scheduler last_epoch 35406, expected 35406
- model_25.ckpt: epoch 25, best_epoch 23, best_metric 1.198043e-01, adam steps [7306], scheduler last_epoch 7306, expected 7306
- model_50.ckpt: epoch 50, best_epoch 45, best_metric 8.238727e-02, adam steps [14331], scheduler last_epoch 14331, expected 14331
- model_75.ckpt: epoch 75, best_epoch 74, best_metric 6.765170e-02, adam steps [21356], scheduler last_epoch 21356, expected 21356
- composition weights and scale identical across all checkpoints and best states, largest gaps 0.0e+00 and 0.0e+00 eV
- lineage mae_prod minimum from the log at epoch 113, final best_epoch 113, checkpoint best_metric over the log product 9.999e-07 (a unit factor)
- best weights differ from the last weights by at most 3.710e-03
- not restored by metatrain on a restart: random number generator and sampler state, so redone epochs saw another batch order

- PASS lineage covers epochs 0 to 149 without gaps
- PASS model.ckpt: optimizer and scheduler step counts equal (epoch + 1) x 281 = 42150
- PASS model.ckpt (final): optimizer and scheduler step counts equal (epoch + 1) x 281 = 42150
- PASS model_0.ckpt: optimizer and scheduler step counts equal (epoch + 1) x 281 = 281
- PASS model_100.ckpt: optimizer and scheduler step counts equal (epoch + 1) x 281 = 28381
- PASS model_125.ckpt: optimizer and scheduler step counts equal (epoch + 1) x 281 = 35406
- PASS model_25.ckpt: optimizer and scheduler step counts equal (epoch + 1) x 281 = 7306
- PASS model_50.ckpt: optimizer and scheduler step counts equal (epoch + 1) x 281 = 14331
- PASS model_75.ckpt: optimizer and scheduler step counts equal (epoch + 1) x 281 = 21356
- PASS normalization unchanged across all checkpoints
- PASS final best epoch (113) is the lineage minimum of validation mae_prod (113)
- PASS model_0.ckpt: recorded best epoch (0) is the lineage minimum up to its epoch (0)
- PASS model_100.ckpt: recorded best epoch (100) is the lineage minimum up to its epoch (100)
- PASS model_125.ckpt: recorded best epoch (113) is the lineage minimum up to its epoch (113)
- PASS model_25.ckpt: recorded best epoch (23) is the lineage minimum up to its epoch (23)
- PASS model_50.ckpt: recorded best epoch (45) is the lineage minimum up to its epoch (45)
- PASS model_75.ckpt: recorded best epoch (74) is the lineage minimum up to its epoch (74)
- PASS best_metric never increases from one checkpoint to the next
