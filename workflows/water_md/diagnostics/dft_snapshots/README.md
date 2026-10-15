# dft_snapshots

Bounded single-point check of the three water potentials against LDA, PBE and r2SCAN DFT (2026-10-13). Read-only on the
production trajectories and on `analysis/` and `../proton_transfer/`. No MD, no training, no AIMD.

## Why

The approximate LDA and PBE potentials are PET-MAD-1.6-M minus a learned r2SCAN minus LDA (or PBE) correction. Their job is to
reproduce the functional, not experiment, so a comparison with the literature can only be qualitative. Whether they reproduce
their functional on liquid-water geometries was never tested, and no r2SCAN label exists for any liquid-water configuration
(checked in the shared manifests, the LDA/PBE campaign product and the MAD-1.6 r2SCAN sets).

## Steps and files

1. `select_snapshots.py` (rules fixed before any comparison, recorded in `tables/event.json`):
   - late snapshots, seed 1 at step 140,000 (35.0 ps) and seed 3 at step 200,000 (50.0 ps) of each potential, six frames;
   - the event, the lasting proton transfer of an approximate LDA run after 20 ps with one crossing and the largest settled stay
     on both sides (hydrogen 41, oxygen 39 to oxygen 54, run lda_corrected_seed3, 28.05 ps); during is the frame of smallest
     donor H minus acceptor H distance difference within 0.1 ps of the crossing, before and after are 0.5 ps away;
   - `tables/event_classes.tsv` classifies every approximate LDA episode (transient returns within 0.1 ps, longer returns,
     lasting transfers settled 0.5 ps on both sides, open) and the coordination-defect stretches by duration, using the
     definitions of `../proton_transfer/inspect_events.py` (nearest oxygen owns a hydrogen, periodic minimum image);
     `tables/episodes_with_settled_stays.tsv`, `tables/event_series.tsv` (donor H, acceptor H and O O distances around the event);
   - `snapshots/*.extxyz`, nine frames with the MD forces as `md_forces`, `tables/snapshots.tsv`.
2. `build_aims.py`: `aims/<snapshot>/<functional>/{control.in,geometry.in}`. pw-lda and pbe use the campaign periodic_ttt
   control files verbatim (scratch copy asserted byte-identical to the aa002 copy), r2scan the Lyra bundle periodic template
   verbatim; tight defaults_2020 basis for H and O (asserted identical between campaign and bundle). The two protocols differ in
   smearing, mixing and sc_init_iter as recorded in DFT_CAMPAIGN.md section 9a; that is what the labels were made with.
   Geometries are re-read and asserted equal to the snapshots to 1e-9 A. k_grid_density 8 gives a 6x6x6 grid for this
   9.86 A cell, as it did for every periodic label. Cold SCF starts.
3. `submit_probe.sh` (debug, one r2scan single point, shows the libxc functional works in the Clariden build) and `submit.sh`
   (normal, one node, four slots of 72 ranks, r2scan then pw-lda then pbe, finished pairs skipped on resubmission), both through
   `run_aims_frames.sh`. Job ids in `logs/JOBIDS.txt`. The outputs in `aims/` (3.5 GB, matrices included) are diagnostics and may be deleted.
4. `extract_labelled_water.py`: the H and O only frames with at least 24 atoms of the LDA/PBE campaign random arms
   (`existing_labels/water_{lda,pbe}.xyz`, `frames.json` with density, arm and membership in the shared sets).
5. `evaluate_models.py`: the float64 copies of PET-MAD-1.6-M, delta_LDA and delta_PBE on the nine snapshots and the labelled
   frames (`tables/model_predictions.{json,npz}`), with the check that baseline minus correction equals the stored MD forces.
6. `plot_force_parity.py` writes `figures/force_parity_snapshots.{png,pdf}` (model against DFT force components of the nine snapshots, both panels on identical axes, one colour for all nine).
7. `literature_table.py` writes `REPORT_LITERATURE.md`, `compare_labelled.py` writes `REPORT_LABELLED.md`, `compare.py` writes
   `REPORT_SNAPSHOTS.md` once the aims outputs exist. `REPORT.md` is the summary.
