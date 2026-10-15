# Existing LDA and PBE labels on water-like structures

Frames of the LDA/PBE campaign whose atoms are only H and O with H to O ratio 2, outside the correction training set (random arms). They carry LDA and PBE labels but no r2SCAN label, so only the combined potentials can be tested on them. Forces as MAE / RMSE / max in meV/A per Cartesian component.

## periodic liquid water frame, 80 molecules, 1.249 g/cm3 (AENET), 1 frame(s), 240 atoms

| comparison | H | O | all | RMS of the reference, H / O |
|---|---|---|---|---|
| C1 approx. LDA against LDA | 23 / 29 / 99 | 28 / 37 / 120 | 25 / 32 / 120 | 1412 / 1932 |
| C2 approx. PBE against PBE | 24 / 30 / 93 | 28 / 35 / 104 | 25 / 31 / 104 | 1436 / 1934 |
| D1 context, baseline alone against LDA | 384 / 447 / 894 | 419 / 504 / 1268 | 396 / 467 / 1268 | 1412 / 1932 |
| D2 context, baseline alone against PBE | 196 / 227 / 438 | 234 / 278 / 617 | 209 / 245 / 617 | 1436 / 1934 |
| D5 context, delta_LDA against F_PBE minus F_LDA (not its target) | 25 / 33 / 126 | 31 / 40 / 126 | 27 / 35 / 126 | 1412 / 1932 |

## periodic ice frame, 64 molecules in a 23.5 A box (water_ice), 1 frame(s), 192 atoms

| comparison | H | O | all | RMS of the reference, H / O |
|---|---|---|---|---|
| C1 approx. LDA against LDA | 69 / 87 / 290 | 57 / 70 / 244 | 65 / 82 / 290 | 2296 / 2556 |
| C2 approx. PBE against PBE | 44 / 58 / 209 | 64 / 84 / 229 | 51 / 67 / 229 | 2223 / 2472 |
| D1 context, baseline alone against LDA | 94 / 123 / 382 | 101 / 125 / 381 | 96 / 124 / 382 | 2296 / 2556 |
| D2 context, baseline alone against PBE | 151 / 187 / 509 | 154 / 187 / 565 | 152 / 187 / 565 | 2223 / 2472 |
| D5 context, delta_LDA against F_PBE minus F_LDA (not its target) | 61 / 72 / 189 | 40 / 49 / 146 | 54 / 66 / 189 | 2296 / 2556 |

## isolated clusters, 70 molecules (OMol25), 5 frame(s), 1050 atoms

| comparison | H | O | all | RMS of the reference, H / O |
|---|---|---|---|---|
| C1 approx. LDA against LDA | 33 / 43 / 179 | 43 / 55 / 241 | 36 / 48 / 241 | 896 / 1195 |
| C2 approx. PBE against PBE | 33 / 43 / 188 | 43 / 54 / 187 | 37 / 47 / 188 | 801 / 1117 |
| D1 context, baseline alone against LDA | 370 / 433 / 942 | 393 / 462 / 1032 | 377 / 443 / 1032 | 896 / 1195 |
| D2 context, baseline alone against PBE | 221 / 257 / 561 | 264 / 307 / 639 | 235 / 275 / 639 | 801 / 1117 |
| D5 context, delta_LDA against F_PBE minus F_LDA (not its target) | 26 / 34 / 135 | 31 / 40 / 156 | 28 / 36 / 156 | 896 / 1195 |

Relative energies among these 5 frames of identical composition, each source minus its own mean, meV/atom:

| frame | DFT LDA | approx. LDA | DFT PBE | approx. PBE | PET-MAD-1.6-M |
|---|---|---|---|---|---|
| OMol25:62daf1ffc9474347ab98972e92bfdc39 | -0.4 | +0.4 | -2.2 | -1.2 | +1.2 |
| OMol25:a9c8083ded8740fbaf6307e7121b869e | -4.0 | -3.6 | +1.0 | +1.2 | -1.7 |
| OMol25:21ccf27cbb644425b88b709c717ed3b5 | +6.3 | +5.7 | +3.5 | +2.6 | +3.8 |
| OMol25:9bf220babd814a79b2267090740f94f4 | +2.8 | +2.4 | +1.3 | +0.4 | -1.7 |
| OMol25:f115a580430847ef9e672968cfbee13d | -4.8 | -4.8 | -3.6 | -3.0 | -1.6 |

MAE of the relative energies: approx. LDA vs DFT LDA 0.5, approx. PBE vs DFT PBE 0.7 meV/atom.

## isolated clusters, 40 molecules (OMol25), 2 frame(s), 240 atoms

| comparison | H | O | all | RMS of the reference, H / O |
|---|---|---|---|---|
| C1 approx. LDA against LDA | 34 / 44 / 213 | 46 / 62 / 259 | 38 / 51 / 259 | 1745 / 2404 |
| C2 approx. PBE against PBE | 30 / 39 / 168 | 43 / 55 / 173 | 34 / 45 / 173 | 1700 / 2367 |
| D1 context, baseline alone against LDA | 338 / 398 / 913 | 371 / 438 / 1021 | 349 / 412 / 1021 | 1745 / 2404 |
| D2 context, baseline alone against PBE | 212 / 246 / 518 | 263 / 308 / 609 | 229 / 268 / 609 | 1700 / 2367 |
| D5 context, delta_LDA against F_PBE minus F_LDA (not its target) | 27 / 34 / 120 | 36 / 45 / 129 | 30 / 38 / 129 | 1745 / 2404 |

Relative energies among these 2 frames of identical composition, each source minus its own mean, meV/atom:

| frame | DFT LDA | approx. LDA | DFT PBE | approx. PBE | PET-MAD-1.6-M |
|---|---|---|---|---|---|
| OMol25:10bf637b33894615b972c541823c881e | +35.1 | +35.2 | +26.0 | +27.3 | +30.2 |
| OMol25:985c048ee1f54ff5834798cc38f7ecf8 | -35.1 | -35.2 | -26.0 | -27.3 | -30.2 |

MAE of the relative energies: approx. LDA vs DFT LDA 0.1, approx. PBE vs DFT PBE 1.4 meV/atom.

## isolated clusters, 8 to 25 molecules, 31 frame(s), 1128 atoms

| comparison | H | O | all | RMS of the reference, H / O |
|---|---|---|---|---|
| C1 approx. LDA against LDA | 45 / 59 / 385 | 65 / 106 / 1402 | 52 / 78 / 1402 | 718 / 1417 |
| C2 approx. PBE against PBE | 36 / 46 / 281 | 57 / 97 / 1388 | 43 / 67 / 1388 | 669 / 1454 |
| D1 context, baseline alone against LDA | 319 / 373 / 985 | 355 / 422 / 1170 | 331 / 390 / 1170 | 718 / 1417 |
| D2 context, baseline alone against PBE | 225 / 260 / 618 | 277 / 324 / 1078 | 242 / 283 / 1078 | 669 / 1454 |
| D5 context, delta_LDA against F_PBE minus F_LDA (not its target) | 34 / 43 / 189 | 44 / 55 / 215 | 37 / 47 / 215 | 718 / 1417 |

