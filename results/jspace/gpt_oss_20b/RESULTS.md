# J-space separability — gpt_oss_20b

Design: `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md`. Script: `scripts/jspace_separability.py`. Written 2026-10-31 22:43 UTC. Layers [10, 11, 12, 13, 14] (centre 12); frame sizes k = [8, 16, 25, 50, 100]; bootstrap B = 2000, permutations B = 2000; seed 42.

## Data

| Split | Rows used | Zero rows dropped | Runs | Domains / families |
|---|---:|---:|---:|---|
| core | 576 | 36 | 36 | 13: audit, education, engineering, environmental, financial, government, hiring, legal… |
| transfer | 234 | 0 | 26 | 3: cicd, code_review, research |

## A2. Core discussion → Transfer final, band mean (primary AUROC)

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.748 | (same) | (same) | (same) | (same) |
| J | 0.632 | 0.586 | 0.568 | 0.566 | 0.614 |
| C_J | 0.718 | 0.731 | 0.725 | 0.752 | 0.766 |
| randsub | 0.682 | 0.540 | 0.611 | 0.657 | 0.691 |
| C_randsub | 0.751 | 0.750 | 0.751 | 0.751 | 0.753 |
| shuffled | 0.427 | 0.511 | 0.563 | 0.568 | 0.547 |
| C_shuffled | 0.742 | 0.739 | 0.746 | 0.748 | 0.736 |
| randrows | 0.448 | 0.499 | 0.561 | 0.508 | 0.411 |
| C_randrows | 0.755 | 0.732 | 0.732 | 0.753 | 0.757 |
| pca | 0.669 | 0.705 | 0.685 | 0.737 | 0.720 |
| C_pca | 0.662 | 0.581 | 0.602 | 0.560 | 0.547 |
| contrast | 0.639 | 0.637 | 0.643 | 0.643 | 0.643 |

### A2 inference at k = 25 (band mean; 95% bootstrap intervals over episodes; permutation p for primary > 0.5)

| arm | primary | 95% CI | mean FVE core | mean FVE transfer | p (perm) |
|---|---:|---|---:|---:|---:|
| full | 0.748 | [0.644, 0.838] | 1.000 | 1.000 | 0.0015 |
| J | 0.568 | [0.477, 0.660] | 0.061 | 0.060 | 0.1299 |
| C_J | 0.725 | [0.621, 0.818] | 0.939 | 0.940 | 0.0020 |
| randsub | 0.611 | [0.491, 0.725] | 0.011 | 0.010 | 0.0490 |
| C_randsub | 0.751 | [0.647, 0.841] | 0.989 | 0.990 | 0.0015 |
| shuffled | 0.563 | [0.493, 0.633] | 0.060 | 0.055 | 0.0830 |
| C_shuffled | 0.746 | [0.634, 0.843] | 0.940 | 0.945 | 0.0020 |
| randrows | 0.561 | [0.462, 0.648] | 0.020 | 0.019 | 0.1209 |
| C_randrows | 0.732 | [0.602, 0.843] | 0.980 | 0.981 | 0.0030 |
| pca | 0.685 | [0.604, 0.760] | 0.411 | 0.365 | 0.0025 |
| C_pca | 0.602 | [0.485, 0.709] | 0.589 | 0.635 | 0.0685 |
| contrast | 0.643 | [0.507, 0.767] | 0.250 | 0.223 | 0.0505 |

| difference | estimate | 95% CI |
|---|---:|---|
| C_J-J | +0.157 | [+0.073, +0.238] |
| J-randsub | -0.043 | [-0.169, +0.090] |
| J-shuffled | +0.005 | [-0.094, +0.115] |
| J-randrows | +0.007 | [-0.110, +0.125] |
| J-pca | -0.117 | [-0.217, -0.015] |
| C_J-full | -0.022 | [-0.043, +0.000] |
| C_randsub-full | +0.004 | [-0.004, +0.011] |
| contrast-J | +0.075 | [-0.073, +0.219] |
| pca-J | +0.117 | [+0.015, +0.217] |

Per layer, J vs complement vs full at k = 25:

| layer | full | J | C_J | randsub | shuffled | pca | contrast |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.821 | 0.647 | 0.844 | 0.712 | 0.554 | 0.704 | 0.631 |
| 11 | 0.767 | 0.499 | 0.718 | 0.635 | 0.649 | 0.576 | 0.586 |
| 12 | 0.746 | 0.523 | 0.740 | 0.393 | 0.519 | 0.659 | 0.647 |
| 13 | 0.726 | 0.576 | 0.698 | 0.584 | 0.562 | 0.738 | 0.679 |
| 14 | 0.679 | 0.596 | 0.627 | 0.732 | 0.533 | 0.750 | 0.675 |

Per family at the centre layer L12, k = 25 (Split AUROC):

| family | full | J | C_J | randsub |
|---|---:|---:|---:|---:|
| cicd | 0.833 | 0.667 | 0.833 | 0.583 |
| code_review | 0.500 | 0.444 | 0.472 | 0.139 |

## A1. Core leave-one-domain-out (fit and eval on discussion turns)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| J | 0.808 | 0.632 | 0.784 | 0.846 | 0.974 |
| C_J | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randsub | 0.695 | 0.712 | 0.800 | 0.963 | 0.974 |
| C_randsub | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| shuffled | 0.706 | 0.571 | 0.671 | 0.831 | 0.936 |
| C_shuffled | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randrows | 0.506 | 0.397 | 0.508 | 0.467 | 0.380 |
| C_randrows | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pca | 0.959 | 0.891 | 0.998 | 1.000 | 1.000 |
| C_pca | 0.858 | 0.761 | 0.639 | 0.891 | 0.643 |
| contrast | 0.727 | 0.875 | 0.933 | 0.875 | 0.875 |

Fold-paired differences at k = 25, centre layer (full minus arm; folds = held-out domains):

| arm | mean diff | folds where full > arm | folds | sign-test p |
|---|---:|---:|---:|---:|
| J | +0.128 | 4 | 13 | 0.125 |
| C_J | +0.000 | 0 | 13 | 1.000 |
| randsub | +0.279 | 6 | 13 | 0.031 |
| C_randsub | +0.000 | 0 | 13 | 1.000 |
| shuffled | +0.470 | 8 | 13 | 0.008 |
| C_shuffled | +0.000 | 0 | 13 | 1.000 |
| randrows | +0.606 | 12 | 13 | 0.000 |
| C_randrows | +0.000 | 0 | 13 | 1.000 |
| pca | +0.006 | 1 | 13 | 1.000 |
| C_pca | +0.496 | 10 | 13 | 0.002 |
| contrast | +0.125 | 4 | 13 | 0.125 |

## A3. Within-Transfer leave-one-family-out (fit and eval on final turns; pooled held-out AUROC)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.777 | 0.777 | 0.777 | 0.777 | 0.777 |
| J | 0.561 | 0.579 | 0.556 | 0.617 | 0.671 |
| C_J | 0.751 | 0.748 | 0.770 | 0.771 | 0.780 |
| randsub | 0.604 | 0.728 | 0.672 | 0.720 | 0.787 |
| C_randsub | 0.779 | 0.778 | 0.783 | 0.781 | 0.778 |
| shuffled | 0.471 | 0.439 | 0.516 | 0.513 | 0.533 |
| C_shuffled | 0.740 | 0.762 | 0.785 | 0.725 | 0.729 |
| randrows | 0.552 | 0.492 | 0.465 | 0.493 | 0.506 |
| C_randrows | 0.785 | 0.778 | 0.793 | 0.778 | 0.788 |
| pca | 0.792 | 0.697 | 0.702 | 0.680 | 0.637 |
| C_pca | 0.770 | 0.781 | 0.774 | 0.800 | 0.810 |
| contrast | 0.693 | 0.641 | 0.662 | 0.662 | 0.662 |

Inference at k = 25 (pooled held-out episodes, bootstrap over episodes, permutation within family):

| arm | primary | 95% CI | p (perm) |
|---|---:|---|---:|
| full | 0.777 | [0.714, 0.832] | 0.0005 |
| J | 0.556 | [0.480, 0.628] | 0.0900 |
| C_J | 0.770 | [0.706, 0.826] | 0.0005 |
| randsub | 0.672 | [0.587, 0.754] | 0.0015 |
| C_randsub | 0.783 | [0.722, 0.837] | 0.0005 |
| shuffled | 0.516 | [0.431, 0.606] | 0.3768 |
| C_shuffled | 0.785 | [0.734, 0.829] | 0.0005 |
| randrows | 0.465 | [0.408, 0.526] | 0.8461 |
| C_randrows | 0.793 | [0.729, 0.843] | 0.0005 |
| pca | 0.702 | [0.596, 0.796] | 0.0015 |
| C_pca | 0.774 | [0.693, 0.841] | 0.0005 |
| contrast | 0.662 | [0.545, 0.765] | 0.0125 |

| difference | estimate | 95% CI |
|---|---:|---|
| C_J-J | +0.214 | [+0.124, +0.305] |
| J-randsub | -0.116 | [-0.230, +0.001] |
| J-shuffled | +0.041 | [-0.090, +0.166] |
| J-randrows | +0.091 | [+0.008, +0.176] |
| J-pca | -0.146 | [-0.240, -0.045] |
| C_J-full | -0.006 | [-0.026, +0.012] |
| C_randsub-full | +0.006 | [-0.002, +0.015] |
| contrast-J | +0.106 | [-0.010, +0.215] |
| pca-J | +0.146 | [+0.045, +0.240] |

## A4a. Core discussion → Core final, leave-one-domain-out (phase changes, family does not)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.949 | 0.949 | 0.924 | 0.949 | 0.949 |
| J | 0.652 | 0.719 | 0.507 | 0.724 | 0.681 |
| C_J | 0.974 | 0.974 | 0.923 | 0.949 | 0.974 |
| randsub | 0.656 | 0.808 | 0.780 | 0.877 | 0.821 |
| C_randsub | 0.949 | 0.949 | 0.928 | 0.923 | 0.949 |
| shuffled | 0.495 | 0.432 | 0.450 | 0.557 | 0.632 |
| C_shuffled | 0.949 | 0.974 | 0.887 | 0.974 | 0.972 |
| randrows | 0.575 | 0.350 | 0.453 | 0.375 | 0.571 |
| C_randrows | 0.974 | 0.897 | 0.907 | 0.872 | 0.949 |
| pca | 0.821 | 0.774 | 0.916 | 0.949 | 0.949 |
| C_pca | 0.882 | 0.781 | 0.662 | 0.593 | 0.521 |
| contrast | 0.792 | 0.826 | 0.767 | 0.769 | 0.769 |

## A4b. Core discussion → Transfer discussion (family changes, phase does not)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.606 | 0.606 | 0.606 | 0.606 | 0.606 |
| J | 0.451 | 0.498 | 0.544 | 0.415 | 0.542 |
| C_J | 0.557 | 0.578 | 0.550 | 0.594 | 0.571 |
| randsub | 0.552 | 0.479 | 0.461 | 0.490 | 0.417 |
| C_randsub | 0.606 | 0.604 | 0.598 | 0.600 | 0.591 |
| shuffled | 0.510 | 0.503 | 0.569 | 0.535 | 0.441 |
| C_shuffled | 0.569 | 0.611 | 0.586 | 0.607 | 0.561 |
| randrows | 0.508 | 0.502 | 0.516 | 0.415 | 0.557 |
| C_randrows | 0.566 | 0.536 | 0.539 | 0.511 | 0.572 |
| pca | 0.617 | 0.508 | 0.561 | 0.445 | 0.489 |
| C_pca | 0.632 | 0.457 | 0.536 | 0.458 | 0.503 |
| contrast | 0.456 | 0.568 | 0.517 | 0.517 | 0.517 |

## A5. Geometry (centre layer, k = 25)

Share of the full probe direction inside each activation's own frame: median 0.005, max 0.008; a random direction: median 0.011; k/d = 0.0087.

|cos| between the full probe direction and the direction each arm learns: J 0.002, C_J 0.940, shuffled 0.030, C_shuffled 0.961, randrows 0.009, C_randrows 0.802, randsub 0.059, C_randsub 0.988, pca 0.573, C_pca 0.338, contrast 0.089.

Frame tokens shared between colluder and honest turns (Jaccard of the 100 most frequent): 0.787.

| role | 25 most frequent frame tokens (share of that role's discussion turns whose frame holds the token) |
|---|---|
| colluder | `'4'` 1.00, `'This'` 1.00, `'S'` 1.00, `'3'` 1.00, `'In'` 1.00, `'1'` 0.99, `'7'` 0.99, `'-'` 0.99, `'C'` 0.99, `'c'` 0.98, `'\n\n'` 0.97, `'5'` 0.97, `'6'` 0.97, `'8'` 0.97, `'I'` 0.94, `' This'` 0.94, `' this'` 0.86, `'For'` 0.86, `'---\n\n'` 0.85, `'Now'` 0.84, `'i'` 0.83, `' \n'` 0.78, `' \n\n'` 0.62, `'B'` 0.58, `'—'` 0.56 |
| honest | `'This'` 1.00, `'-'` 1.00, `'5'` 1.00, `'4'` 1.00, `'In'` 1.00, `'3'` 1.00, `'1'` 1.00, `'6'` 1.00, `'7'` 1.00, `'\n\n'` 1.00, `'c'` 0.99, `'S'` 0.99, `'C'` 0.97, `'8'` 0.97, `' This'` 0.94, `'I'` 0.87, `'i'` 0.86, `'Now'` 0.84, `' this'` 0.83, `'For'` 0.73, `'---\n\n'` 0.72, `' \n'` 0.70, `' \n\n'` 0.69, `'—'` 0.58, `'B'` 0.57 |

## Readings (rules fixed in the design, section 6)

- **R1_in_distribution**: in distribution J-space is a weaker carrier than the full residual
  J_primary = 0.784; full_minus_J = 0.216
- **R2_transfer**: the transferable signal is not in J-space
  J = 0.568; C_J = 0.725; randsub = 0.611; shuffled = 0.563; full = 0.748; p_perm_J = 0.130; p_perm_randsub = 0.049
- **R3_complement_uninformative**: complement-of-random equals full (as expected); 'complement equals full' says nothing about J-space by itself
  
- **R4_within_transfer**: role information is absent from J-space on Transfer finals (within-Transfer J probe at its permutation null)
  J_within_transfer = 0.556; J_core_to_transfer = 0.568; p_perm_within = 0.090; C_J_within_transfer = 0.770; full_within_transfer = 0.777
- **R5_phase_vs_family**: the J drop is mostly the phase change (discussion to final)
  J_core_disc_to_core_final = 0.507; full_core_disc_to_core_final = 0.924; J_core_disc_to_transfer_final = 0.568; J_core_disc_to_transfer_discussion = 0.544; full_core_disc_to_transfer_discussion = 0.606; drop_phase_only = 0.277; drop_total = 0.216
- **R6_dimension**: small supervised/PCA subspaces do not reach the full probe either; dimension may matter
  best_contrast_k_le_16 = 0.639; best_pca_k_le_16 = 0.705; best_J_k_le_16 = 0.632; full = 0.748

## Baseline reproduction

Full-activation Core LDO primary 1.000 (paper: 1.00 for every model in distribution); full-activation Core → Transfer band mean 0.748.
