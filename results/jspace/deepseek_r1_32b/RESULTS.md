# J-space separability — deepseek_r1_32b

Design: `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md`. Script: `scripts/jspace_separability.py`. Written 2026-10-31 22:43 UTC. Layers [26, 27, 28, 29, 30] (centre 28); frame sizes k = [8, 16, 25, 50, 100]; bootstrap B = 2000, permutations B = 2000; seed 42.

## Data

| Split | Rows used | Zero rows dropped | Runs | Domains / families |
|---|---:|---:|---:|---|
| core | 896 | 56 | 56 | 14: education, engineering, environmental, financial, food_safety, government, hiring, insurance… |
| transfer | 928 | 8 | 72 | 6: cicd, code_review, news, pump_dump, rag, research |

## A2. Core discussion → Transfer final, band mean (primary AUROC)

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.733 | (same) | (same) | (same) | (same) |
| J | 0.523 | 0.546 | 0.500 | 0.550 | 0.577 |
| C_J | 0.736 | 0.728 | 0.729 | 0.726 | 0.725 |
| randsub | 0.512 | 0.520 | 0.550 | 0.538 | 0.631 |
| C_randsub | 0.734 | 0.734 | 0.734 | 0.735 | 0.738 |
| shuffled | 0.452 | 0.572 | 0.513 | 0.495 | 0.549 |
| C_shuffled | 0.726 | 0.732 | 0.727 | 0.727 | 0.729 |
| randrows | 0.506 | 0.533 | 0.448 | 0.491 | 0.494 |
| C_randrows | 0.724 | 0.714 | 0.715 | 0.721 | 0.713 |
| pca | 0.596 | 0.669 | 0.680 | 0.701 | 0.736 |
| C_pca | 0.631 | 0.575 | 0.537 | 0.512 | 0.515 |
| contrast | 0.626 | 0.584 | 0.630 | 0.626 | 0.626 |

### A2 inference at k = 25 (band mean; 95% bootstrap intervals over episodes; permutation p for primary > 0.5)

| arm | primary | 95% CI | mean FVE core | mean FVE transfer | p (perm) |
|---|---:|---|---:|---:|---:|
| full | 0.733 | [0.669, 0.794] | 1.000 | 1.000 | 0.0005 |
| J | 0.500 | [0.435, 0.566] | 0.035 | 0.033 | 0.5242 |
| C_J | 0.729 | [0.666, 0.792] | 0.965 | 0.967 | 0.0005 |
| randsub | 0.550 | [0.492, 0.604] | 0.005 | 0.005 | 0.0585 |
| C_randsub | 0.734 | [0.669, 0.795] | 0.995 | 0.995 | 0.0005 |
| shuffled | 0.513 | [0.465, 0.562] | 0.035 | 0.032 | 0.3113 |
| C_shuffled | 0.727 | [0.663, 0.789] | 0.965 | 0.968 | 0.0005 |
| randrows | 0.448 | [0.395, 0.501] | 0.017 | 0.016 | 0.9730 |
| C_randrows | 0.715 | [0.653, 0.776] | 0.983 | 0.984 | 0.0005 |
| pca | 0.680 | [0.617, 0.741] | 0.246 | 0.191 | 0.0005 |
| C_pca | 0.537 | [0.460, 0.617] | 0.754 | 0.809 | 0.1894 |
| contrast | 0.630 | [0.557, 0.695] | 0.188 | 0.170 | 0.0005 |

| difference | estimate | 95% CI |
|---|---:|---|
| C_J-J | +0.229 | [+0.161, +0.292] |
| J-randsub | -0.050 | [-0.120, +0.013] |
| J-shuffled | -0.013 | [-0.083, +0.053] |
| J-randrows | +0.053 | [-0.045, +0.147] |
| J-pca | -0.180 | [-0.269, -0.093] |
| C_J-full | -0.004 | [-0.017, +0.009] |
| C_randsub-full | +0.000 | [-0.002, +0.003] |
| contrast-J | +0.130 | [+0.047, +0.210] |
| pca-J | +0.180 | [+0.093, +0.269] |

Per layer, J vs complement vs full at k = 25:

| layer | full | J | C_J | randsub | shuffled | pca | contrast |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 26 | 0.743 | 0.565 | 0.738 | 0.588 | 0.522 | 0.648 | 0.698 |
| 27 | 0.719 | 0.465 | 0.710 | 0.526 | 0.486 | 0.647 | 0.650 |
| 28 | 0.735 | 0.429 | 0.728 | 0.640 | 0.525 | 0.665 | 0.627 |
| 29 | 0.738 | 0.544 | 0.723 | 0.519 | 0.426 | 0.709 | 0.618 |
| 30 | 0.732 | 0.500 | 0.747 | 0.479 | 0.607 | 0.733 | 0.557 |

Per family at the centre layer L28, k = 25 (Split AUROC):

| family | full | J | C_J | randsub |
|---|---:|---:|---:|---:|
| cicd | 0.694 | 0.361 | 0.750 | 0.472 |
| code_review | 0.900 | 0.700 | 0.933 | 0.700 |
| news | 0.583 | 0.500 | 0.611 | 0.583 |
| pump_dump | 0.778 | 0.222 | 0.806 | 0.639 |
| rag | 0.667 | 0.633 | 0.667 | 0.833 |
| research | 0.472 | 0.417 | 0.389 | 0.583 |

## A1. Core leave-one-domain-out (fit and eval on discussion turns)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| J | 0.823 | 0.822 | 0.906 | 1.000 | 0.999 |
| C_J | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randsub | 0.836 | 0.979 | 0.974 | 0.976 | 0.976 |
| C_randsub | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| shuffled | 0.679 | 0.834 | 0.779 | 0.976 | 0.994 |
| C_shuffled | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randrows | 0.455 | 0.433 | 0.458 | 0.361 | 0.702 |
| C_randrows | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pca | 0.975 | 0.976 | 0.990 | 1.000 | 1.000 |
| C_pca | 0.698 | 0.613 | 0.659 | 0.885 | 0.704 |
| contrast | 0.948 | 0.918 | 0.950 | 0.974 | 0.974 |

Fold-paired differences at k = 25, centre layer (full minus arm; folds = held-out domains):

| arm | mean diff | folds where full > arm | folds | sign-test p |
|---|---:|---:|---:|---:|
| J | +0.090 | 5 | 14 | 0.062 |
| C_J | +0.000 | 0 | 14 | 1.000 |
| randsub | +0.001 | 1 | 14 | 1.000 |
| C_randsub | +0.000 | 0 | 14 | 1.000 |
| shuffled | +0.146 | 7 | 14 | 0.016 |
| C_shuffled | +0.000 | 0 | 14 | 1.000 |
| randrows | +0.503 | 12 | 14 | 0.000 |
| C_randrows | +0.000 | 0 | 14 | 1.000 |
| pca | +0.000 | 0 | 14 | 1.000 |
| C_pca | +0.357 | 14 | 14 | 0.000 |
| contrast | +0.022 | 3 | 14 | 0.250 |

## A3. Within-Transfer leave-one-family-out (fit and eval on final turns; pooled held-out AUROC)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.827 | 0.827 | 0.827 | 0.827 | 0.827 |
| J | 0.522 | 0.503 | 0.539 | 0.577 | 0.524 |
| C_J | 0.818 | 0.807 | 0.800 | 0.808 | 0.800 |
| randsub | 0.501 | 0.557 | 0.556 | 0.575 | 0.634 |
| C_randsub | 0.827 | 0.827 | 0.827 | 0.827 | 0.826 |
| shuffled | 0.509 | 0.523 | 0.513 | 0.505 | 0.561 |
| C_shuffled | 0.807 | 0.794 | 0.810 | 0.783 | 0.775 |
| randrows | 0.509 | 0.511 | 0.470 | 0.467 | 0.501 |
| C_randrows | 0.796 | 0.776 | 0.755 | 0.715 | 0.698 |
| pca | 0.499 | 0.640 | 0.737 | 0.709 | 0.760 |
| C_pca | 0.816 | 0.809 | 0.827 | 0.838 | 0.833 |
| contrast | 0.525 | 0.578 | 0.626 | 0.607 | 0.607 |

Inference at k = 25 (pooled held-out episodes, bootstrap over episodes, permutation within family):

| arm | primary | 95% CI | p (perm) |
|---|---:|---|---:|
| full | 0.827 | [0.770, 0.876] | 0.0005 |
| J | 0.539 | [0.497, 0.581] | 0.0390 |
| C_J | 0.800 | [0.738, 0.854] | 0.0005 |
| randsub | 0.556 | [0.499, 0.612] | 0.0395 |
| C_randsub | 0.827 | [0.770, 0.876] | 0.0005 |
| shuffled | 0.513 | [0.491, 0.535] | 0.1039 |
| C_shuffled | 0.810 | [0.758, 0.855] | 0.0005 |
| randrows | 0.470 | [0.418, 0.518] | 0.8576 |
| C_randrows | 0.755 | [0.688, 0.815] | 0.0005 |
| pca | 0.737 | [0.677, 0.790] | 0.0005 |
| C_pca | 0.827 | [0.766, 0.876] | 0.0005 |
| contrast | 0.626 | [0.553, 0.696] | 0.0030 |

| difference | estimate | 95% CI |
|---|---:|---|
| C_J-J | +0.261 | [+0.192, +0.328] |
| J-randsub | -0.017 | [-0.079, +0.040] |
| J-shuffled | +0.026 | [-0.021, +0.071] |
| J-randrows | +0.069 | [+0.007, +0.139] |
| J-pca | -0.197 | [-0.263, -0.125] |
| C_J-full | -0.027 | [-0.041, -0.013] |
| C_randsub-full | +0.000 | [-0.002, +0.003] |
| contrast-J | +0.087 | [+0.011, +0.164] |
| pca-J | +0.197 | [+0.125, +0.263] |

## A4a. Core discussion → Core final, leave-one-domain-out (phase changes, family does not)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 |
| J | 0.738 | 0.672 | 0.579 | 0.702 | 0.703 |
| C_J | 1.000 | 1.000 | 0.997 | 1.000 | 1.000 |
| randsub | 0.570 | 0.762 | 0.641 | 0.850 | 0.861 |
| C_randsub | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 |
| shuffled | 0.611 | 0.340 | 0.471 | 0.362 | 0.525 |
| C_shuffled | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 |
| randrows | 0.419 | 0.302 | 0.496 | 0.508 | 0.442 |
| C_randrows | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pca | 0.832 | 0.929 | 0.971 | 1.000 | 1.000 |
| C_pca | 0.681 | 0.552 | 0.554 | 0.703 | 0.600 |
| contrast | 0.564 | 0.911 | 0.878 | 0.871 | 0.871 |

## A4b. Core discussion → Transfer discussion (family changes, phase does not)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.942 | 0.942 | 0.942 | 0.942 | 0.942 |
| J | 0.523 | 0.556 | 0.560 | 0.566 | 0.627 |
| C_J | 0.937 | 0.941 | 0.939 | 0.941 | 0.942 |
| randsub | 0.547 | 0.575 | 0.608 | 0.601 | 0.697 |
| C_randsub | 0.942 | 0.941 | 0.941 | 0.942 | 0.942 |
| shuffled | 0.495 | 0.490 | 0.529 | 0.555 | 0.586 |
| C_shuffled | 0.941 | 0.938 | 0.939 | 0.935 | 0.941 |
| randrows | 0.481 | 0.492 | 0.478 | 0.498 | 0.485 |
| C_randrows | 0.947 | 0.947 | 0.947 | 0.950 | 0.940 |
| pca | 0.625 | 0.858 | 0.864 | 0.917 | 0.929 |
| C_pca | 0.754 | 0.530 | 0.538 | 0.536 | 0.517 |
| contrast | 0.655 | 0.721 | 0.740 | 0.748 | 0.748 |

## A5. Geometry (centre layer, k = 25)

Share of the full probe direction inside each activation's own frame: median 0.005, max 0.008; a random direction: median 0.007; k/d = 0.0049.

|cos| between the full probe direction and the direction each arm learns: J 0.022, C_J 0.972, shuffled 0.019, C_shuffled 0.986, randrows 0.011, C_randrows 0.950, randsub 0.044, C_randsub 0.996, pca 0.795, C_pca 0.200, contrast 0.255.

Frame tokens shared between colluder and honest turns (Jaccard of the 100 most frequent): 0.818.

| role | 25 most frequent frame tokens (share of that role's discussion turns whose frame holds the token) |
|---|---|
| colluder | `'딮'` 1.00, `''` 1.00, `'뼐'` 1.00, `'먮'` 1.00, `'ᐕ'` 1.00, `'텦'` 1.00, `'㉲'` 0.99, `'겇'` 0.99, `'ᩈ'` 0.97, `'�'` 0.96, `'ᄢ'` 0.95, `'ﯲ'` 0.95, `'�'` 0.89, `'ﭑ'` 0.85, `'괙'` 0.84, `'ﭲ'` 0.82, `'겥'` 0.76, `'�'` 0.74, `'ﯣ'` 0.73, `'ﯶ'` 0.71, `'읅'` 0.66, `'�'` 0.61, `'퓝'` 0.58, `'ꍲ'` 0.45, `'쵥'` 0.43 |
| honest | `'딮'` 1.00, `'뼐'` 1.00, `'텦'` 0.99, `'ᐕ'` 0.99, `''` 0.99, `'먮'` 0.98, `'�'` 0.98, `'ᩈ'` 0.96, `'㉲'` 0.90, `'�'` 0.86, `'�'` 0.83, `'ﭲ'` 0.81, `'ﭑ'` 0.79, `'ﯲ'` 0.78, `'ᄢ'` 0.74, `'겇'` 0.72, `'괙'` 0.71, `'�'` 0.71, `'�'` 0.67, `'ꍲ'` 0.64, `'퓝'` 0.64, `'ﯶ'` 0.59, `'쵥'` 0.58, `'ﯣ'` 0.53, `'ꡔ'` 0.52 |

## Readings (rules fixed in the design, section 6)

- **R1_in_distribution**: in distribution J-space is a weaker carrier than the full residual
  J_primary = 0.906; full_minus_J = 0.094
- **R2_transfer**: the transferable signal is not in J-space
  J = 0.500; C_J = 0.729; randsub = 0.550; shuffled = 0.513; full = 0.733; p_perm_J = 0.524; p_perm_randsub = 0.058
- **R3_complement_uninformative**: complement-of-random equals full (as expected); 'complement equals full' says nothing about J-space by itself
  
- **R4_within_transfer**: intermediate: within-Transfer J probe above chance but below 0.80, or Core-fit J above 0.65
  J_within_transfer = 0.539; J_core_to_transfer = 0.500; p_perm_within = 0.039; C_J_within_transfer = 0.800; full_within_transfer = 0.827
- **R5_phase_vs_family**: the J drop is mostly the phase change (discussion to final)
  J_core_disc_to_core_final = 0.579; full_core_disc_to_core_final = 0.999; J_core_disc_to_transfer_final = 0.500; J_core_disc_to_transfer_discussion = 0.560; full_core_disc_to_transfer_discussion = 0.942; drop_phase_only = 0.327; drop_total = 0.405
- **R6_dimension**: small supervised/PCA subspaces do not reach the full probe either; dimension may matter
  best_contrast_k_le_16 = 0.626; best_pca_k_le_16 = 0.669; best_J_k_le_16 = 0.546; full = 0.733

## Baseline reproduction

Full-activation Core LDO primary 1.000 (paper: 1.00 for every model in distribution); full-activation Core → Transfer band mean 0.733.
