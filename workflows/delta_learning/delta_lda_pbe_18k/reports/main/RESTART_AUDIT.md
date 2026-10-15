# delta_lda_pbe_18k, restart audit

Checks from the checkpoints and training log of each run: allocations, optimizer and scheduler step counts, normalization, best checkpoint tracking.

## delta_lda--omat-s--madcore_shared18000--s42

- 2 allocation(s); epochs logged per allocation [(0, 105), (101, 149)]; steps per epoch 281
- allocation 2 resumed from epoch 100 checkpoint, first logged epoch 101
- epochs 101 to 105 redone, learning rate gap 0.0e+00, training loss ratio at epoch 101 0.964
- model.ckpt: epoch 149, best_epoch 102, best_metric 1.777449e-03, adam steps [42150], scheduler last_epoch 42150, expected 42150
- model.ckpt (final): epoch 149, best_epoch 102, best_metric 1.777449e-03, adam steps [42150], scheduler last_epoch 42150, expected 42150
- model_0.ckpt: epoch 0, best_epoch 0, best_metric 1.399493e-02, adam steps [281], scheduler last_epoch 281, expected 281
- model_100.ckpt: epoch 100, best_epoch 94, best_metric 1.777819e-03, adam steps [28381], scheduler last_epoch 28381, expected 28381
- model_125.ckpt: epoch 125, best_epoch 102, best_metric 1.777449e-03, adam steps [35406], scheduler last_epoch 35406, expected 35406
- model_25.ckpt: epoch 25, best_epoch 25, best_metric 2.234147e-03, adam steps [7306], scheduler last_epoch 7306, expected 7306
- model_50.ckpt: epoch 50, best_epoch 45, best_metric 1.949582e-03, adam steps [14331], scheduler last_epoch 14331, expected 14331
- model_75.ckpt: epoch 75, best_epoch 70, best_metric 1.817194e-03, adam steps [21356], scheduler last_epoch 21356, expected 21356
- composition weights and scale identical across all checkpoints and best states, largest gaps 0.0e+00 and 0.0e+00 eV
- lineage mae_prod minimum from the log at epoch 94, final best_epoch 102, checkpoint best_metric over the log product 1.000e-06 (a unit factor)
- best weights differ from the last weights by at most 6.081e-03
- not restored by metatrain on a restart: random number generator and sampler state, so redone epochs saw another batch order

- PASS allocation 2 starts one epoch after its checkpoint (100 then 101)
- PASS allocation 2: scheduler restored, learning rates of the redone epochs identical
- PASS allocation 2: training loss continues (ratio within 0.8 to 1.25)
- PASS allocation 2: no warmup after the restart, learning rate non increasing
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
- PASS final best epoch (102) is the lineage minimum of validation mae_prod (94, recorded epoch within 9.3e-06 relative of the log minimum, below the log rounding)
- PASS model_0.ckpt: recorded best epoch (0) is the lineage minimum up to its epoch (0)
- PASS model_100.ckpt: recorded best epoch (94) is the lineage minimum up to its epoch (94)
- PASS model_125.ckpt: recorded best epoch (102) is the lineage minimum up to its epoch (94, within 9.3e-06 relative, below the log rounding)
- PASS model_25.ckpt: recorded best epoch (25) is the lineage minimum up to its epoch (25)
- PASS model_50.ckpt: recorded best epoch (45) is the lineage minimum up to its epoch (45)
- PASS model_75.ckpt: recorded best epoch (70) is the lineage minimum up to its epoch (70)
- PASS best_metric never increases from one checkpoint to the next

## delta_pbe--omat-s--madcore_shared18000--s42

- 2 allocation(s); epochs logged per allocation [(0, 99), (76, 149)]; steps per epoch 281
- allocation 2 resumed from epoch 75 checkpoint, first logged epoch 76
- epochs 76 to 99 redone, learning rate gap 0.0e+00, training loss ratio at epoch 76 0.972
- model.ckpt: epoch 149, best_epoch 101, best_metric 1.135759e-03, adam steps [42150], scheduler last_epoch 42150, expected 42150
- model.ckpt (final): epoch 149, best_epoch 101, best_metric 1.135759e-03, adam steps [42150], scheduler last_epoch 42150, expected 42150
- model_0.ckpt: epoch 0, best_epoch 0, best_metric 4.236334e-03, adam steps [281], scheduler last_epoch 281, expected 281
- model_100.ckpt: epoch 100, best_epoch 98, best_metric 1.138238e-03, adam steps [28381], scheduler last_epoch 28381, expected 28381
- model_125.ckpt: epoch 125, best_epoch 101, best_metric 1.135759e-03, adam steps [35406], scheduler last_epoch 35406, expected 35406
- model_25.ckpt: epoch 25, best_epoch 20, best_metric 1.467491e-03, adam steps [7306], scheduler last_epoch 7306, expected 7306
- model_50.ckpt: epoch 50, best_epoch 50, best_metric 1.271130e-03, adam steps [14331], scheduler last_epoch 14331, expected 14331
- model_75.ckpt: epoch 75, best_epoch 67, best_metric 1.175145e-03, adam steps [21356], scheduler last_epoch 21356, expected 21356
- composition weights and scale identical across all checkpoints and best states, largest gaps 0.0e+00 and 0.0e+00 eV
- lineage mae_prod minimum from the log at epoch 101, final best_epoch 101, checkpoint best_metric over the log product 1.000e-06 (a unit factor)
- best weights differ from the last weights by at most 6.348e-03
- not restored by metatrain on a restart: random number generator and sampler state, so redone epochs saw another batch order

- PASS allocation 2 starts one epoch after its checkpoint (75 then 76)
- PASS allocation 2: scheduler restored, learning rates of the redone epochs identical
- PASS allocation 2: training loss continues (ratio within 0.8 to 1.25)
- PASS allocation 2: no warmup after the restart, learning rate non increasing
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
- PASS final best epoch (101) is the lineage minimum of validation mae_prod (101)
- PASS model_0.ckpt: recorded best epoch (0) is the lineage minimum up to its epoch (0)
- PASS model_100.ckpt: recorded best epoch (98) is the lineage minimum up to its epoch (98)
- PASS model_125.ckpt: recorded best epoch (101) is the lineage minimum up to its epoch (101)
- PASS model_25.ckpt: recorded best epoch (20) is the lineage minimum up to its epoch (20)
- PASS model_50.ckpt: recorded best epoch (50) is the lineage minimum up to its epoch (50)
- PASS model_75.ckpt: recorded best epoch (67) is the lineage minimum up to its epoch (67)
- PASS best_metric never increases from one checkpoint to the next
