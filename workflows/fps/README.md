The metric and the first orderings. fit_scaler_union.py fits one mean vector and one
global scale on the 856,012 MAD-NEXT features (data/ecumetric_scaler.npz). run_fps.py is the
multi GPU farthest point sampling of the coresets package. runs/mad_union_meanstd is the
MAD-NEXT ordering of Figs. 4 and 5, seeded at the hydrogen atom. submit_fps_family.sh
selects at most 10 M structures from each large dataset, with the ZBL thresholds 200 eV/A,
50 eV/atom and 100 eV/atom, using the earlier scaler fitted on MAD-1.5 only, which does not
change the picks.
