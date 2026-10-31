# J-space separability — gpt_oss_20b_gamma

Design: `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md`. Script: `scripts/jspace_separability.py`. Written 2026-10-31 23:06 UTC. Layers [10, 11, 12, 13, 14] (centre 12); frame sizes k = [8, 16, 25, 50, 100]; bootstrap B = 2000, permutations B = 2000; seed 42.

## Data

| Split | Rows used | Zero rows dropped | Runs | Domains / families |
|---|---:|---:|---:|---|
| core | 576 | 36 | 36 | 13: audit, education, engineering, environmental, financial, government, hiring, legal… |
| transfer | 234 | 0 | 26 | 3: cicd, code_review, research |

## A2. Core discussion → Transfer final, band mean (primary AUROC)

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.748 | (same) | (same) | (same) | (same) |
| J | 0.527 | 0.572 | 0.615 | 0.641 | 0.568 |
| C_J | 0.758 | 0.730 | 0.715 | 0.738 | 0.773 |
| randsub | 0.682 | 0.540 | 0.611 | 0.657 | 0.691 |
| C_randsub | 0.751 | 0.750 | 0.751 | 0.751 | 0.753 |
| shuffled | 0.503 | 0.548 | 0.517 | 0.548 | 0.485 |
| C_shuffled | 0.746 | 0.714 | 0.749 | 0.732 | 0.757 |
| randrows | 0.502 | 0.464 | 0.501 | 0.475 | 0.393 |
| C_randrows | 0.756 | 0.738 | 0.723 | 0.753 | 0.756 |
| pca | 0.669 | 0.705 | 0.685 | 0.737 | 0.720 |
| C_pca | 0.662 | 0.581 | 0.602 | 0.560 | 0.547 |
| contrast | 0.639 | 0.637 | 0.643 | 0.643 | 0.643 |

### A2 inference at k = 25 (band mean; 95% bootstrap intervals over episodes; permutation p for primary > 0.5)

| arm | primary | 95% CI | mean FVE core | mean FVE transfer | p (perm) |
|---|---:|---|---:|---:|---:|
| full | 0.748 | [0.644, 0.838] | 1.000 | 1.000 | 0.0015 |
| J | 0.615 | [0.501, 0.735] | 0.041 | 0.048 | 0.0505 |
| C_J | 0.715 | [0.592, 0.819] | 0.959 | 0.952 | 0.0035 |
| randsub | 0.611 | [0.491, 0.725] | 0.011 | 0.010 | 0.0490 |
| C_randsub | 0.751 | [0.647, 0.841] | 0.989 | 0.990 | 0.0015 |
| shuffled | 0.517 | [0.447, 0.588] | 0.039 | 0.035 | 0.3428 |
| C_shuffled | 0.749 | [0.640, 0.843] | 0.961 | 0.965 | 0.0010 |
| randrows | 0.501 | [0.423, 0.577] | 0.010 | 0.010 | 0.4828 |
| C_randrows | 0.723 | [0.596, 0.829] | 0.990 | 0.990 | 0.0025 |
| pca | 0.685 | [0.604, 0.760] | 0.411 | 0.365 | 0.0025 |
| C_pca | 0.602 | [0.485, 0.709] | 0.589 | 0.635 | 0.0685 |
| contrast | 0.643 | [0.507, 0.767] | 0.250 | 0.223 | 0.0505 |

| difference | estimate | 95% CI |
|---|---:|---|
| C_J-J | +0.099 | [-0.030, +0.230] |
| J-randsub | +0.004 | [-0.125, +0.134] |
| J-shuffled | +0.099 | [-0.043, +0.247] |
| J-randrows | +0.114 | [-0.036, +0.266] |
| J-pca | -0.070 | [-0.188, +0.045] |
| C_J-full | -0.033 | [-0.083, +0.011] |
| C_randsub-full | +0.004 | [-0.004, +0.011] |
| contrast-J | +0.028 | [-0.087, +0.141] |
| pca-J | +0.070 | [-0.045, +0.188] |

Per layer, J vs complement vs full at k = 25:

| layer | full | J | C_J | randsub | shuffled | pca | contrast |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.821 | 0.714 | 0.751 | 0.712 | 0.467 | 0.704 | 0.631 |
| 11 | 0.767 | 0.738 | 0.655 | 0.635 | 0.487 | 0.576 | 0.586 |
| 12 | 0.746 | 0.570 | 0.775 | 0.393 | 0.574 | 0.659 | 0.647 |
| 13 | 0.726 | 0.471 | 0.738 | 0.584 | 0.598 | 0.738 | 0.679 |
| 14 | 0.679 | 0.584 | 0.655 | 0.732 | 0.458 | 0.750 | 0.675 |

Per family at the centre layer L12, k = 25 (Split AUROC):

| family | full | J | C_J | randsub |
|---|---:|---:|---:|---:|
| cicd | 0.833 | 0.889 | 0.861 | 0.583 |
| code_review | 0.500 | 0.417 | 0.500 | 0.139 |

## A1. Core leave-one-domain-out (fit and eval on discussion turns)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| J | 0.636 | 0.662 | 0.802 | 0.912 | 0.937 |
| C_J | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randsub | 0.695 | 0.712 | 0.800 | 0.963 | 0.974 |
| C_randsub | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| shuffled | 0.506 | 0.663 | 0.621 | 0.795 | 0.542 |
| C_shuffled | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randrows | 0.447 | 0.419 | 0.457 | 0.556 | 0.563 |
| C_randrows | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pca | 0.959 | 0.891 | 0.998 | 1.000 | 1.000 |
| C_pca | 0.858 | 0.761 | 0.639 | 0.891 | 0.643 |
| contrast | 0.727 | 0.875 | 0.933 | 0.875 | 0.875 |

Fold-paired differences at k = 25, centre layer (full minus arm; folds = held-out domains):

| arm | mean diff | folds where full > arm | folds | sign-test p |
|---|---:|---:|---:|---:|
| J | +0.317 | 8 | 13 | 0.008 |
| C_J | +0.000 | 0 | 13 | 1.000 |
| randsub | +0.279 | 6 | 13 | 0.031 |
| C_randsub | +0.000 | 0 | 13 | 1.000 |
| shuffled | +0.241 | 5 | 13 | 0.062 |
| C_shuffled | +0.000 | 0 | 13 | 1.000 |
| randrows | +0.576 | 12 | 13 | 0.000 |
| C_randrows | +0.000 | 0 | 13 | 1.000 |
| pca | +0.006 | 1 | 13 | 1.000 |
| C_pca | +0.496 | 10 | 13 | 0.002 |
| contrast | +0.125 | 4 | 13 | 0.125 |

## A3. Within-Transfer leave-one-family-out (fit and eval on final turns; pooled held-out AUROC)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.777 | 0.777 | 0.777 | 0.777 | 0.777 |
| J | 0.581 | 0.588 | 0.614 | 0.536 | 0.609 |
| C_J | 0.767 | 0.705 | 0.757 | 0.784 | 0.735 |
| randsub | 0.604 | 0.728 | 0.672 | 0.720 | 0.787 |
| C_randsub | 0.779 | 0.778 | 0.783 | 0.781 | 0.778 |
| shuffled | 0.472 | 0.424 | 0.554 | 0.521 | 0.591 |
| C_shuffled | 0.717 | 0.696 | 0.732 | 0.695 | 0.692 |
| randrows | 0.582 | 0.518 | 0.467 | 0.529 | 0.553 |
| C_randrows | 0.781 | 0.785 | 0.792 | 0.787 | 0.788 |
| pca | 0.792 | 0.697 | 0.702 | 0.680 | 0.637 |
| C_pca | 0.770 | 0.781 | 0.774 | 0.800 | 0.810 |
| contrast | 0.693 | 0.641 | 0.662 | 0.662 | 0.662 |

Inference at k = 25 (pooled held-out episodes, bootstrap over episodes, permutation within family):

| arm | primary | 95% CI | p (perm) |
|---|---:|---|---:|
| full | 0.777 | [0.714, 0.832] | 0.0005 |
| J | 0.614 | [0.542, 0.685] | 0.0095 |
| C_J | 0.757 | [0.702, 0.806] | 0.0005 |
| randsub | 0.672 | [0.587, 0.754] | 0.0015 |
| C_randsub | 0.783 | [0.722, 0.837] | 0.0005 |
| shuffled | 0.554 | [0.471, 0.648] | 0.1489 |
| C_shuffled | 0.732 | [0.675, 0.784] | 0.0005 |
| randrows | 0.467 | [0.396, 0.537] | 0.7756 |
| C_randrows | 0.792 | [0.730, 0.845] | 0.0005 |
| pca | 0.702 | [0.596, 0.796] | 0.0015 |
| C_pca | 0.774 | [0.693, 0.841] | 0.0005 |
| contrast | 0.662 | [0.545, 0.765] | 0.0125 |

| difference | estimate | 95% CI |
|---|---:|---|
| C_J-J | +0.142 | [+0.071, +0.217] |
| J-randsub | -0.058 | [-0.148, +0.033] |
| J-shuffled | +0.060 | [-0.038, +0.153] |
| J-randrows | +0.147 | [+0.040, +0.263] |
| J-pca | -0.088 | [-0.179, +0.008] |
| C_J-full | -0.020 | [-0.054, +0.012] |
| C_randsub-full | +0.006 | [-0.002, +0.015] |
| contrast-J | +0.048 | [-0.052, +0.140] |
| pca-J | +0.088 | [-0.008, +0.179] |

## A4a. Core discussion → Core final, leave-one-domain-out (phase changes, family does not)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.949 | 0.949 | 0.924 | 0.949 | 0.949 |
| J | 0.401 | 0.675 | 0.694 | 0.684 | 0.809 |
| C_J | 0.949 | 0.949 | 0.909 | 0.991 | 0.897 |
| randsub | 0.656 | 0.808 | 0.780 | 0.877 | 0.821 |
| C_randsub | 0.949 | 0.949 | 0.928 | 0.923 | 0.949 |
| shuffled | 0.486 | 0.541 | 0.536 | 0.574 | 0.593 |
| C_shuffled | 0.974 | 0.974 | 0.889 | 0.974 | 0.940 |
| randrows | 0.584 | 0.424 | 0.441 | 0.476 | 0.629 |
| C_randrows | 0.974 | 0.949 | 0.917 | 0.872 | 0.897 |
| pca | 0.821 | 0.774 | 0.916 | 0.949 | 0.949 |
| C_pca | 0.882 | 0.781 | 0.662 | 0.593 | 0.521 |
| contrast | 0.792 | 0.826 | 0.767 | 0.769 | 0.769 |

## A4b. Core discussion → Transfer discussion (family changes, phase does not)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.606 | 0.606 | 0.606 | 0.606 | 0.606 |
| J | 0.497 | 0.515 | 0.519 | 0.440 | 0.528 |
| C_J | 0.570 | 0.577 | 0.575 | 0.589 | 0.605 |
| randsub | 0.552 | 0.479 | 0.461 | 0.490 | 0.417 |
| C_randsub | 0.606 | 0.604 | 0.598 | 0.600 | 0.591 |
| shuffled | 0.606 | 0.572 | 0.493 | 0.499 | 0.488 |
| C_shuffled | 0.608 | 0.596 | 0.591 | 0.544 | 0.585 |
| randrows | 0.504 | 0.507 | 0.520 | 0.399 | 0.510 |
| C_randrows | 0.591 | 0.558 | 0.557 | 0.518 | 0.575 |
| pca | 0.617 | 0.508 | 0.561 | 0.445 | 0.489 |
| C_pca | 0.632 | 0.457 | 0.536 | 0.458 | 0.503 |
| contrast | 0.456 | 0.568 | 0.517 | 0.517 | 0.517 |

## A5. Geometry (centre layer, k = 25)

Share of the full probe direction inside each activation's own frame: median 0.007, max 0.009; a random direction: median 0.009; k/d = 0.0087.

|cos| between the full probe direction and the direction each arm learns: J 0.018, C_J 0.950, shuffled 0.007, C_shuffled 0.951, randrows 0.013, C_randrows 0.842, randsub 0.059, C_randsub 0.988, pca 0.573, C_pca 0.338, contrast 0.089.

Frame tokens shared between colluder and honest turns (Jaccard of the 100 most frequent): 0.775.

| role | 25 most frequent frame tokens (share of that role's discussion turns whose frame holds the token) |
|---|---|
| colluder | `'---\n\n'` 1.00, `'Now'` 1.00, `'This'` 1.00, `''` 1.00, `'---'` 0.98, `'7'` 0.97, `'Conclusion'` 0.97, `'4'` 0.97, `'5'` 0.97, `'Furthermore'` 0.97, `'6'` 0.96, `'Additionally'` 0.96, `'Note'` 0.95, `'21'` 0.94, `'For'` 0.89, `'###'` 0.89, `'In'` 0.88, `''` 0.85, `'From'` 0.73, `'1'` 0.70, `'##'` 0.64, `'3'` 0.62, `'Regarding'` 0.61, `'---\n'` 0.50, `' This'` 0.49 |
| honest | `'Now'` 1.00, `'---\n\n'` 1.00, `'This'` 1.00, `'5'` 1.00, `'4'` 1.00, `''` 1.00, `'7'` 1.00, `'---'` 0.98, `'Additionally'` 0.97, `'Note'` 0.97, `'6'` 0.97, `'Furthermore'` 0.94, `'Conclusion'` 0.94, `''` 0.90, `'21'` 0.87, `'###'` 0.83, `'For'` 0.79, `'We'` 0.76, `'From'` 0.75, `'In'` 0.70, `'1'` 0.64, `'##'` 0.62, `'Regarding'` 0.56, `'202'` 0.56, `'3'` 0.55 |

## Readings (rules fixed in the design, section 6)

- **R1_in_distribution**: in distribution J-space is a weaker carrier than the full residual
  J_primary = 0.802; full_minus_J = 0.198
- **R2_transfer**: J-space and its complement are not clearly separated on Transfer (condition i fails)
  J = 0.615; C_J = 0.715; randsub = 0.611; shuffled = 0.517; full = 0.748; p_perm_J = 0.050; p_perm_randsub = 0.049
- **R3_complement_uninformative**: complement-of-random equals full (as expected); 'complement equals full' says nothing about J-space by itself
  
- **R4_within_transfer**: intermediate: within-Transfer J probe above chance but below 0.80, or Core-fit J above 0.65
  J_within_transfer = 0.614; J_core_to_transfer = 0.615; p_perm_within = 0.009; C_J_within_transfer = 0.757; full_within_transfer = 0.777
- **R5_phase_vs_family**: the J drop is shared between phase and family
  J_core_disc_to_core_final = 0.694; full_core_disc_to_core_final = 0.924; J_core_disc_to_transfer_final = 0.615; J_core_disc_to_transfer_discussion = 0.519; full_core_disc_to_transfer_discussion = 0.606; drop_phase_only = 0.108; drop_total = 0.187
- **R6_dimension**: small supervised/PCA subspaces do not reach the full probe either; dimension may matter
  best_contrast_k_le_16 = 0.639; best_pca_k_le_16 = 0.705; best_J_k_le_16 = 0.572; full = 0.748

## Baseline reproduction

Full-activation Core LDO primary 1.000 (paper: 1.00 for every model in distribution); full-activation Core → Transfer band mean 0.748.
