# Reassessment of the water MD with LDA, PBE and r2SCAN single points

Summary of `REPORT_LITERATURE.md`, `../proton_transfer/` plus `tables/event_classes.tsv`, `REPORT_LABELLED.md` and
`REPORT_SNAPSHOTS.md` (2026-10-13). Jobs 3651137 (probe, debug, 14 min) and 3651188 (normal, 2 h 06 min); all 27 single
points converged in 19 to 27 SCF cycles. Nothing in `analysis/`, `../proton_transfer/` or the manuscript was changed.

## 1. Literature

Qualitative consistency, not validation (`REPORT_LITERATURE.md`). The approximate LDA potential at 300 K gives an O-O first
peak at 2.57 A of height 4.81 ± 0.14, first minimum 0.19 at 3.09 A, FWHM 0.28 A; the LDA liquid of Todorova et al. at 350 K
(32 molecules, 0.975 g/cm3, deuterium masses, Car-Parrinello, Pade LDA) has 2.59 A, 5.07, 0.23 at 3.12 A, FWHM 0.21 A. The
approximate PBE potential gives 2.73 A, 3.48 ± 0.14, minimum 0.39 at 3.28 A, second peak 1.51, n_OO(3.28 A) 4.18; DiStasio
et al. at 300 K (64 molecules, deuterium masses, Car-Parrinello) report 2.69 A, 3.28, 0.37 at 3.28 A, 1.44, 4.03, and
Todorova at 350 K 2.70 A, 2.99, 0.47. Both approximate potentials deviate from experiment (2.76 to 2.80 A, 2.57 to 2.62,
minimum 0.82 to 0.84) in the direction and size class their functionals do in the literature. The differences in temperature,
density, nuclear mass, dynamics and basis rule out a quantitative match at the 0.1 to 0.3 level in g. Neither paper reports
proton transfer or ionic defects in LDA water, so that feature needed the direct check below.

## 2. Coordination events in the approximate LDA runs

`tables/event_classes.tsv` (20 to 50 ps window for the fractions, all frames for the stretches), definitions of
`../proton_transfer/inspect_events.py`: a hydrogen belongs to its nearest oxygen under periodic minimum image (checked
against 27 explicit images), an episode is the span between two settled stays of at least 0.5 ps on one oxygen.

| per run, seeds 0 to 4 | transient switches, returned within 0.1 ps | longer returns | lasting transfers, settled 0.5 ps on both sides | open at an end | defect stretches, of which over 0.5 ps | longest defect | hydrogens on another oxygen at 50 ps |
|---|---|---|---|---|---|---|---|
| seed 0 | 151 | 81 | 77 | 1 | 350, 9 | 4.1 ps | 32 |
| seed 1 | 90 | 12 | 21 | 1 | 109, 1 | 2.2 ps | 5 |
| seed 2 | 78 | 23 | 11 | 6 | 172, 0 | 0.2 ps | 9 |
| seed 3 | 98 | 26 | 51 | 1 | 180, 0 | 0.3 ps | 18 |
| seed 4 | 124 | 50 | 40 | 3 | 277, 1 | 0.7 ps | 24 |

Three classes: most crossings are transient switches of a shared proton that return within a few frames; a minority are
lasting proton exchanges, after which the hydrogen stays on the new oxygen for the rest of the run; and the ionic defects
they create (an oxygen with one or three hydrogens) usually live for one to a few frames, with persistent defects only in
seed 0 (a 4.1 ps stretch at 29 to 33 ps with up to six defect oxygens, 55 % of its frames with a defect) and one 2.2 ps
stretch in seed 1. Minimum O-O distance at a transfer 2.28 to 2.33 A. The baseline and approximate PBE runs have no crossing
at all. The representative event used below: hydrogen 41 moving from oxygen 39 to oxygen 54 in seed 3 at 28.05 ps, one
crossing, settled 28 ps before and 22 ps after, O-O 2.40 A and donor H 1.195 A, acceptor H 1.207 A at the during frame.

## 3. Labels used

No r2SCAN label exists for any liquid-water configuration (shared manifests, LDA/PBE product and MAD-1.6 r2SCAN sets
scanned), so the bounded single points were run: the nine snapshots of `tables/snapshots.tsv` (two late frames per
potential, seed 1 at 35 ps and seed 3 at 50 ps, and before/during/after of the event) with the campaign pw-lda and pbe
control files and the Lyra r2scan template, verbatim, tight basis, 6x6x6 k-grid from k_grid_density 8, cells and
geometries fixed. The LDA/PBE campaign also holds labelled water-like frames outside the correction training set (an
80-molecule liquid frame at 1.249 g/cm3, water clusters of 8 to 70 molecules); they test the combined potentials against
their functional without r2SCAN (`REPORT_LABELLED.md`).

## 4. Forces and energies

Pooled over the nine snapshots, MAE in meV/A per component, H / O (`REPORT_SNAPSHOTS.md`):

| comparison | H | O | size of the reference difference, RMS |
|---|---|---|---|
| PET-MAD-1.6-M against r2SCAN | 10 | 14 | (forces 712 / 991) |
| delta_LDA against F_r2SCAN minus F_LDA | 22 | 25 | 438 / 478 |
| delta_PBE against F_r2SCAN minus F_PBE | 29 | 22 | 226 / 272 |
| approx. LDA against LDA | 25 | 31 | baseline alone would be 377 / 408 |
| approx. PBE against PBE | 33 | 28 | baseline alone would be 195 / 233 |

The errors are the same on every snapshot, whichever potential generated it (per-snapshot RMSE 26 to 46 meV/A), including
the three event frames with a shared proton. The decomposition approx minus target = (baseline minus r2SCAN) + (r2SCAN minus
target minus delta) shows the correction term (23 to 38 meV/A) dominating the baseline term (12 to 18 meV/A); both are small
against the 386 meV/A (LDA) and 206 meV/A (PBE) that the corrections must supply. On the labelled frames the same holds:
25 meV/A on the 80-molecule liquid frame for both potentials, 36 to 52 meV/A on the clusters.

Relative energies over the nine H64O32 snapshots (no reference atoms needed, free energies for aims): PET-MAD-1.6-M matches
r2SCAN to 0.4 meV/atom, approx. LDA matches LDA to 1.4 meV/atom over a 52 meV/atom spread, approx. PBE matches PBE to
2.0 meV/atom over a 22 meV/atom spread. LDA DFT itself places the LDA-trajectory frames 45 meV/atom below the
baseline-trajectory frames of the same seeds; r2SCAN places them 6 and 16 meV/atom above. On the labelled clusters the relative energies agree
to 0.1 to 1.4 meV/atom.

The event: the before, during and after frames are whole-cell geometries 0.5 ps apart, so their absolute energy differences
(LDA -0.54 and -0.60 eV, PBE -0.10 and -0.03 eV, r2SCAN +0.21 and +0.38 eV relative to before) contain the thermal motion of
the other 93 atoms and are not separable from that background on their own. What is robust is the comparison between
functionals at identical geometries, where the background cancels: relative to r2SCAN, LDA stabilises the shared-proton frame
by 0.75 eV and the ion-pair frame (H3O+ OH-) by 0.98 eV, and relative to PBE by 0.44 and 0.58 eV. Each approximate potential
reproduces its own functional's ranking of the three frames to 0.02 to 0.13 eV per cell. The forces on the transferring proton along the donor to
acceptor axis agree between approx. LDA and DFT LDA to 0.05 eV/A at all three frames (before -0.96 / -1.01, during -0.73 /
-0.75, after -0.99 / -0.97 eV/A), and likewise for the two oxygens. The correction is not extrapolating on these geometries;
it reproduces what LDA does.

## 5. Updated interpretation

- The MD results are a property of the functionals as the corrections represent them, not an artefact of the correction.
  On liquid-water geometries sampled by all three potentials the baseline reproduces r2SCAN forces to 11 meV/A and each
  combined potential reproduces its target functional to about 30 meV/A and 1 to 2 meV/atom, against functional
  differences of 200 to 400 meV/A and 20 to 50 meV/atom.
- The overstructured, short-hydrogen-bond LDA liquid with its stretched O-H bonds (1.02 A), wider and softer bend and
  frequent shared protons is supported by LDA single points: LDA prefers those geometries by tens of meV/atom and, at
  identical geometries, stabilises the shared-proton and ion-pair frames of the representative transfer relative to r2SCAN by
  0.75 and 1.0 eV. This is consistent with the
  overbinding reported for LDA water and clusters in the literature, which does not itself report the transfers; nothing
  here contradicts it, and the number and rate of transfers remain unvalidated (see the next point).
- What the check cannot say: nine single points diagnose local force and relative-energy errors; they do not establish
  converged radial distribution functions, transfer rates, or the free-energy balance that decides how often an ion pair
  forms. The LDA transfer counts differ by a factor of seven between seeds and would need much longer or more trajectories
  and, for a rate, direct LDA dynamics, which was not run.
- Wording stays "approximate LDA" and "approximate PBE" models. The one correction to the earlier interpretation is that the
  possible proton-transfer events are no longer only possible: they are what the approximate LDA potential does, and LDA
  DFT reproduces its forces and relative energies on those geometries; they remain unobserved in actual LDA dynamics.
- No retraining is indicated by this check.
