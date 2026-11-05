# J-space separability — gemma2_9b_jpaper_nnomp_k25

Design: `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md`. Script: `scripts/jspace_separability.py`. Written 2026-11-05 08:05 UTC. Layers [19, 20, 21, 22, 23] (centre 21); frame sizes k = [8, 16, 25, 50, 100]; bootstrap B = 2000, permutations B = 2000; seed 42.

## Data

| Split | Rows used | Zero rows dropped | Runs | Domains / families |
|---|---:|---:|---:|---|
| core | 1700 | 0 | 100 | 17: audit, cybersecurity, education, engineering, environmental, financial, food_safety, government… |
| transfer | 936 | 0 | 72 | 6: cicd, code_review, news, pump_dump, rag, research |

## A2. Core discussion → Transfer final, band mean (primary AUROC)

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.902 | (same) | (same) | (same) | (same) |
| J | 0.664 | 0.583 | 0.602 | 0.604 | 0.651 |
| C_J | 0.899 | 0.895 | 0.898 | 0.890 | 0.891 |
| randsub | 0.572 | 0.600 | 0.647 | 0.638 | 0.698 |
| C_randsub | 0.902 | 0.903 | 0.900 | 0.901 | 0.899 |
| shuffled | 0.595 | 0.552 | 0.549 | 0.675 | 0.694 |
| C_shuffled | 0.899 | 0.895 | 0.895 | 0.894 | 0.892 |
| randrows | 0.492 | 0.502 | 0.466 | 0.488 | 0.490 |
| C_randrows | 0.897 | 0.895 | 0.894 | 0.892 | 0.888 |
| pca | 0.849 | 0.855 | 0.856 | 0.886 | 0.886 |
| C_pca | 0.598 | 0.636 | 0.617 | 0.519 | 0.509 |
| contrast | 0.831 | 0.753 | 0.690 | 0.798 | 0.798 |
| Jpaper | — | — | 0.755 | — | — |
| C_Jpaper | — | — | 0.891 | — | — |
| shuffled_paper | — | — | 0.559 | — | — |
| C_shuffled_paper | — | — | 0.895 | — | — |

### A2 inference at k = 25 (band mean; 95% bootstrap intervals over episodes; permutation p for primary > 0.5)

| arm | primary | 95% CI | mean FVE core | mean FVE transfer | p (perm) |
|---|---:|---|---:|---:|---:|
| full | 0.902 | [0.863, 0.935] | 1.000 | 1.000 | 0.0005 |
| J | 0.602 | [0.550, 0.658] | 0.057 | 0.053 | 0.0020 |
| C_J | 0.898 | [0.859, 0.931] | 0.943 | 0.947 | 0.0005 |
| randsub | 0.647 | [0.602, 0.693] | 0.007 | 0.007 | 0.0005 |
| C_randsub | 0.900 | [0.861, 0.933] | 0.993 | 0.993 | 0.0005 |
| shuffled | 0.549 | [0.503, 0.597] | 0.056 | 0.052 | 0.0395 |
| C_shuffled | 0.895 | [0.854, 0.930] | 0.944 | 0.948 | 0.0005 |
| randrows | 0.466 | [0.413, 0.521] | 0.028 | 0.028 | 0.8836 |
| C_randrows | 0.894 | [0.852, 0.931] | 0.972 | 0.972 | 0.0005 |
| pca | 0.856 | [0.807, 0.898] | 0.643 | 0.381 | 0.0005 |
| C_pca | 0.617 | [0.559, 0.676] | 0.357 | 0.619 | 0.0025 |
| contrast | 0.690 | [0.634, 0.742] | 0.401 | 0.246 | 0.0005 |
| Jpaper | 0.755 | [0.709, 0.796] | 0.011 | 0.013 | 0.0005 |
| C_Jpaper | 0.891 | [0.850, 0.927] | 0.989 | 0.987 | 0.0005 |
| shuffled_paper | 0.559 | [0.511, 0.611] | 0.011 | 0.013 | 0.0260 |
| C_shuffled_paper | 0.895 | [0.854, 0.930] | 0.989 | 0.987 | 0.0005 |

| difference | estimate | 95% CI |
|---|---:|---|
| C_J-J | +0.295 | [+0.235, +0.352] |
| J-randsub | -0.045 | [-0.091, +0.003] |
| J-shuffled | +0.053 | [+0.004, +0.103] |
| J-randrows | +0.136 | [+0.067, +0.205] |
| J-pca | -0.254 | [-0.312, -0.191] |
| C_J-full | -0.005 | [-0.009, -0.001] |
| C_randsub-full | -0.002 | [-0.004, +0.000] |
| contrast-J | +0.088 | [+0.025, +0.149] |
| pca-J | +0.254 | [+0.191, +0.312] |
| Jpaper-J | +0.153 | [+0.094, +0.208] |
| Jpaper-randsub | +0.108 | [+0.052, +0.160] |
| Jpaper-shuffled | +0.206 | [+0.148, +0.258] |
| Jpaper-pca | -0.101 | [-0.157, -0.043] |
| C_Jpaper-Jpaper | +0.136 | [+0.083, +0.190] |
| C_Jpaper-full | -0.011 | [-0.019, -0.006] |
| C_Jpaper-C_J | -0.007 | [-0.012, -0.002] |
| shuffled_paper-J | -0.043 | [-0.114, +0.030] |
| shuffled_paper-randsub | -0.088 | [-0.155, -0.018] |
| shuffled_paper-shuffled | +0.010 | [-0.061, +0.084] |
| shuffled_paper-pca | -0.297 | [-0.365, -0.221] |
| C_shuffled_paper-shuffled_paper | +0.336 | [+0.264, +0.401] |
| C_shuffled_paper-full | -0.007 | [-0.013, -0.002] |
| C_shuffled_paper-C_J | -0.003 | [-0.008, +0.003] |

Per layer, J vs complement vs full at k = 25:

| layer | full | J | C_J | randsub | shuffled | pca | contrast |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 0.884 | 0.585 | 0.885 | 0.538 | 0.570 | 0.823 | 0.677 |
| 20 | 0.885 | 0.570 | 0.869 | 0.696 | 0.493 | 0.823 | 0.651 |
| 21 | 0.941 | 0.756 | 0.935 | 0.750 | 0.626 | 0.905 | 0.659 |
| 22 | 0.930 | 0.522 | 0.928 | 0.753 | 0.580 | 0.921 | 0.792 |
| 23 | 0.873 | 0.577 | 0.873 | 0.497 | 0.478 | 0.807 | 0.672 |

Per family at the centre layer L21, k = 25 (Split AUROC):

| family | full | J | C_J | randsub |
|---|---:|---:|---:|---:|
| cicd | 1.000 | 0.917 | 1.000 | 0.806 |
| code_review | 1.000 | 0.861 | 1.000 | 0.917 |
| news | 0.917 | 0.611 | 0.889 | 0.694 |
| pump_dump | 1.000 | 0.750 | 0.972 | 0.278 |
| rag | 1.000 | 0.333 | 1.000 | 1.000 |
| research | 1.000 | 0.389 | 1.000 | 0.750 |

## A1. Core leave-one-domain-out (fit and eval on discussion turns)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| J | 0.908 | 0.912 | 0.989 | 1.000 | 1.000 |
| C_J | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randsub | 0.942 | 0.998 | 0.981 | 1.000 | 1.000 |
| C_randsub | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| shuffled | 0.868 | 0.966 | 0.977 | 1.000 | 1.000 |
| C_shuffled | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randrows | 0.502 | 0.558 | 0.544 | 0.615 | 0.875 |
| C_randrows | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pca | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| C_pca | 0.700 | 0.766 | 0.860 | 0.918 | 0.668 |
| contrast | 0.969 | 0.976 | 0.977 | 0.971 | 0.971 |
| Jpaper | — | — | 0.995 | — | — |
| C_Jpaper | — | — | 1.000 | — | — |
| shuffled_paper | — | — | 0.965 | — | — |
| C_shuffled_paper | — | — | 1.000 | — | — |

Fold-paired differences at k = 25, centre layer (full minus arm; folds = held-out domains):

| arm | mean diff | folds where full > arm | folds | sign-test p |
|---|---:|---:|---:|---:|
| J | +0.006 | 2 | 17 | 0.500 |
| C_J | +0.000 | 0 | 17 | 1.000 |
| randsub | +0.005 | 1 | 17 | 1.000 |
| C_randsub | +0.000 | 0 | 17 | 1.000 |
| shuffled | +0.005 | 1 | 17 | 1.000 |
| C_shuffled | +0.000 | 0 | 17 | 1.000 |
| randrows | +0.513 | 17 | 17 | 0.000 |
| C_randrows | +0.000 | 0 | 17 | 1.000 |
| pca | +0.000 | 0 | 17 | 1.000 |
| C_pca | +0.120 | 15 | 17 | 0.000 |
| contrast | +0.029 | 2 | 17 | 0.500 |
| Jpaper | +0.001 | 1 | 17 | 1.000 |
| C_Jpaper | +0.000 | 0 | 17 | 1.000 |
| shuffled_paper | +0.007 | 2 | 17 | 0.500 |
| C_shuffled_paper | +0.000 | 0 | 17 | 1.000 |

## A3. Within-Transfer leave-one-family-out (fit and eval on final turns; pooled held-out AUROC)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.977 | 0.977 | 0.977 | 0.977 | 0.977 |
| J | 0.650 | 0.656 | 0.744 | 0.808 | 0.863 |
| C_J | 0.976 | 0.975 | 0.975 | 0.977 | 0.974 |
| randsub | 0.653 | 0.737 | 0.742 | 0.822 | 0.880 |
| C_randsub | 0.976 | 0.977 | 0.976 | 0.977 | 0.977 |
| shuffled | 0.567 | 0.637 | 0.660 | 0.759 | 0.825 |
| C_shuffled | 0.976 | 0.976 | 0.977 | 0.975 | 0.972 |
| randrows | 0.545 | 0.532 | 0.583 | 0.532 | 0.574 |
| C_randrows | 0.976 | 0.977 | 0.976 | 0.974 | 0.972 |
| pca | 0.917 | 0.903 | 0.902 | 0.914 | 0.931 |
| C_pca | 0.972 | 0.975 | 0.976 | 0.976 | 0.978 |
| contrast | 0.784 | 0.890 | 0.915 | 0.918 | 0.918 |
| Jpaper | — | — | 0.854 | — | — |
| C_Jpaper | — | — | 0.978 | — | — |
| shuffled_paper | — | — | 0.608 | — | — |
| C_shuffled_paper | — | — | 0.977 | — | — |

Inference at k = 25 (pooled held-out episodes, bootstrap over episodes, permutation within family):

| arm | primary | 95% CI | p (perm) |
|---|---:|---|---:|
| full | 0.977 | [0.963, 0.988] | 0.0005 |
| J | 0.744 | [0.702, 0.784] | 0.0005 |
| C_J | 0.975 | [0.961, 0.988] | 0.0005 |
| randsub | 0.742 | [0.694, 0.786] | 0.0005 |
| C_randsub | 0.976 | [0.962, 0.988] | 0.0005 |
| shuffled | 0.660 | [0.613, 0.707] | 0.0005 |
| C_shuffled | 0.977 | [0.963, 0.989] | 0.0005 |
| randrows | 0.583 | [0.533, 0.629] | 0.0030 |
| C_randrows | 0.976 | [0.963, 0.988] | 0.0005 |
| pca | 0.902 | [0.873, 0.931] | 0.0005 |
| C_pca | 0.976 | [0.961, 0.989] | 0.0005 |
| contrast | 0.915 | [0.884, 0.944] | 0.0005 |
| Jpaper | 0.854 | [0.820, 0.887] | 0.0005 |
| C_Jpaper | 0.978 | [0.965, 0.989] | 0.0005 |
| shuffled_paper | 0.608 | [0.562, 0.653] | 0.0005 |
| C_shuffled_paper | 0.977 | [0.963, 0.989] | 0.0005 |

| difference | estimate | 95% CI |
|---|---:|---|
| C_J-J | +0.231 | [+0.190, +0.272] |
| J-randsub | +0.003 | [-0.050, +0.055] |
| J-shuffled | +0.084 | [+0.035, +0.132] |
| J-randrows | +0.161 | [+0.097, +0.221] |
| J-pca | -0.158 | [-0.208, -0.111] |
| C_J-full | -0.001 | [-0.003, +0.000] |
| C_randsub-full | -0.000 | [-0.001, +0.001] |
| contrast-J | +0.170 | [+0.124, +0.217] |
| pca-J | +0.158 | [+0.111, +0.208] |
| Jpaper-J | +0.109 | [+0.057, +0.163] |
| Jpaper-randsub | +0.112 | [+0.060, +0.165] |
| Jpaper-shuffled | +0.193 | [+0.144, +0.244] |
| Jpaper-pca | -0.048 | [-0.084, -0.017] |
| C_Jpaper-Jpaper | +0.124 | [+0.092, +0.158] |
| C_Jpaper-full | +0.001 | [-0.000, +0.004] |
| C_Jpaper-C_J | +0.003 | [+0.001, +0.005] |
| shuffled_paper-J | -0.137 | [-0.201, -0.068] |
| shuffled_paper-randsub | -0.134 | [-0.190, -0.075] |
| shuffled_paper-shuffled | -0.053 | [-0.119, +0.018] |
| shuffled_paper-pca | -0.294 | [-0.345, -0.242] |
| C_shuffled_paper-shuffled_paper | +0.369 | [+0.322, +0.416] |
| C_shuffled_paper-full | +0.001 | [-0.001, +0.003] |
| C_shuffled_paper-C_J | +0.002 | [-0.000, +0.004] |

## A4a. Core discussion → Core final, leave-one-domain-out (phase changes, family does not)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 |
| J | 0.820 | 0.885 | 0.872 | 0.929 | 0.947 |
| C_J | 0.998 | 0.998 | 0.997 | 0.998 | 0.996 |
| randsub | 0.954 | 0.950 | 0.912 | 0.935 | 0.965 |
| C_randsub | 0.998 | 0.998 | 0.998 | 1.000 | 1.000 |
| shuffled | 0.861 | 0.916 | 0.789 | 0.882 | 0.940 |
| C_shuffled | 0.998 | 0.998 | 0.997 | 0.998 | 0.998 |
| randrows | 0.534 | 0.544 | 0.522 | 0.506 | 0.642 |
| C_randrows | 0.998 | 0.998 | 0.997 | 0.998 | 0.998 |
| pca | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 |
| C_pca | 0.645 | 0.737 | 0.780 | 0.806 | 0.639 |
| contrast | 0.954 | 0.961 | 0.949 | 0.958 | 0.958 |
| Jpaper | — | — | 0.930 | — | — |
| C_Jpaper | — | — | 0.998 | — | — |
| shuffled_paper | — | — | 0.761 | — | — |
| C_shuffled_paper | — | — | 0.997 | — | — |

## A4b. Core discussion → Transfer discussion (family changes, phase does not)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.959 | 0.959 | 0.959 | 0.959 | 0.959 |
| J | 0.478 | 0.498 | 0.504 | 0.552 | 0.520 |
| C_J | 0.954 | 0.954 | 0.956 | 0.953 | 0.951 |
| randsub | 0.550 | 0.613 | 0.602 | 0.552 | 0.608 |
| C_randsub | 0.958 | 0.958 | 0.958 | 0.956 | 0.956 |
| shuffled | 0.468 | 0.483 | 0.506 | 0.484 | 0.497 |
| C_shuffled | 0.955 | 0.953 | 0.953 | 0.950 | 0.950 |
| randrows | 0.511 | 0.507 | 0.529 | 0.530 | 0.534 |
| C_randrows | 0.952 | 0.952 | 0.950 | 0.950 | 0.948 |
| pca | 0.847 | 0.852 | 0.894 | 0.944 | 0.945 |
| C_pca | 0.513 | 0.510 | 0.492 | 0.548 | 0.497 |
| contrast | 0.860 | 0.850 | 0.839 | 0.787 | 0.787 |
| Jpaper | — | — | 0.613 | — | — |
| C_Jpaper | — | — | 0.956 | — | — |
| shuffled_paper | — | — | 0.589 | — | — |
| C_shuffled_paper | — | — | 0.955 | — | — |

## A5. Geometry (centre layer, k = 25)

Share of the full probe direction inside each activation's own frame: median 0.020, max 0.024; a random direction: median 0.010; k/d = 0.0070.

|cos| between the full probe direction and the direction each arm learns: J 0.042, C_J 0.885, shuffled 0.025, C_shuffled 0.885, randrows 0.117, C_randrows 0.858, randsub 0.044, C_randsub 0.961, pca 0.904, C_pca 0.024, contrast 0.363, Jpaper 0.105, C_Jpaper 0.865, shuffled_paper 0.081, C_shuffled_paper 0.869.

Frame tokens shared between colluder and honest turns (Jaccard of the 100 most frequent): 0.920.

| role | 25 most frequent frame tokens (share of that role's discussion turns whose frame holds the token) |
|---|---|
| colluder | `'The'` 1.00, `'1'` 1.00, `'2'` 1.00, `'This'` 1.00, `'\n\n'` 1.00, `'I'` 1.00, `'<strong>'` 1.00, `' '` 1.00, `'('` 1.00, `'\n'` 1.00, `'A'` 0.99, `'0'` 0.98, `'In'` 0.97, `'It'` 0.97, `'5'` 0.97, `' and'` 0.96, `'-'` 0.93, `'3'` 0.89, `'.'` 0.75, `'_'` 0.71, `' the'` 0.71, `'4'` 0.69, `''` 0.63, `'['` 0.63, `'<em>'` 0.42 |
| honest | `'The'` 1.00, `'1'` 1.00, `'2'` 1.00, `'<strong>'` 1.00, `'\n\n'` 1.00, `' '` 1.00, `'This'` 1.00, `'\n'` 1.00, `'I'` 1.00, `'('` 1.00, `'3'` 0.99, `' and'` 0.99, `'A'` 0.97, `'0'` 0.96, `'In'` 0.96, `'5'` 0.95, `'.'` 0.91, `' the'` 0.89, `'It'` 0.89, `'-'` 0.88, `'4'` 0.82, `'6'` 0.80, `'\\'` 0.59, `'If'` 0.58, `''` 0.48 |

## Readings (rules fixed in the design, section 6)

- **R1_in_distribution**: J-space separates colluders in distribution
  J_primary = 0.989; full_minus_J = 0.011
- **R2_transfer**: mixed: C minus J clear, J vs random unclear, complement vs full differs
  J = 0.602; C_J = 0.898; randsub = 0.647; shuffled = 0.549; full = 0.902; p_perm_J = 0.002; p_perm_randsub = 0.000
- **R3_complement_uninformative**: complement-of-random equals full (as expected); 'complement equals full' says nothing about J-space by itself
  
- **R4_within_transfer**: intermediate: within-Transfer J probe above chance but below 0.80, or Core-fit J above 0.65
  J_within_transfer = 0.744; J_core_to_transfer = 0.602; p_perm_within = 0.000; C_J_within_transfer = 0.975; full_within_transfer = 0.977
- **R5_phase_vs_family**: the J drop is mostly the family change
  J_core_disc_to_core_final = 0.872; full_core_disc_to_core_final = 0.998; J_core_disc_to_transfer_final = 0.602; J_core_disc_to_transfer_discussion = 0.504; full_core_disc_to_transfer_discussion = 0.959; drop_phase_only = 0.117; drop_total = 0.387
- **R6_dimension**: small supervised/PCA subspaces do not reach the full probe either; dimension may matter
  best_contrast_k_le_16 = 0.831; best_pca_k_le_16 = 0.855; best_J_k_le_16 = 0.664; full = 0.902

## Baseline reproduction

Full-activation Core LDO primary 1.000 (paper: 1.00 for every model in distribution); full-activation Core → Transfer band mean 0.902.
