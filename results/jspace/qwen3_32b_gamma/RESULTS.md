# J-space separability — qwen3_32b_gamma

Design: `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md`. Script: `scripts/jspace_separability.py`. Written 2026-10-31 23:06 UTC. Layers [26, 27, 28, 29, 30] (centre 28); frame sizes k = [8, 16, 25, 50, 100]; bootstrap B = 2000, permutations B = 2000; seed 42.

## Data

| Split | Rows used | Zero rows dropped | Runs | Domains / families |
|---|---:|---:|---:|---|
| core | 1054 | 0 | 62 | 14: audit, cyber, eng, env, fin, food, gov, hire… |
| transfer | 288 | 0 | 72 | 6: cicd, code_review, news, pump_dump, rag, research |

## A2. Core discussion → Transfer final, band mean (primary AUROC)

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.773 | (same) | (same) | (same) | (same) |
| J | 0.548 | 0.583 | 0.542 | 0.546 | 0.583 |
| C_J | 0.772 | 0.756 | 0.749 | 0.760 | 0.782 |
| randsub | 0.528 | 0.506 | 0.508 | 0.519 | 0.546 |
| C_randsub | 0.771 | 0.779 | 0.779 | 0.775 | 0.783 |
| shuffled | 0.541 | 0.511 | 0.542 | 0.498 | 0.484 |
| C_shuffled | 0.780 | 0.779 | 0.787 | 0.772 | 0.771 |
| randrows | 0.455 | 0.476 | 0.477 | 0.519 | 0.492 |
| C_randrows | 0.776 | 0.776 | 0.775 | 0.778 | 0.785 |
| pca | 0.637 | 0.798 | 0.784 | 0.767 | 0.758 |
| C_pca | 0.552 | 0.468 | 0.478 | 0.478 | 0.512 |
| contrast | 0.450 | 0.574 | 0.500 | 0.460 | 0.460 |

### A2 inference at k = 25 (band mean; 95% bootstrap intervals over episodes; permutation p for primary > 0.5)

| arm | primary | 95% CI | mean FVE core | mean FVE transfer | p (perm) |
|---|---:|---|---:|---:|---:|
| full | 0.773 | [0.692, 0.847] | 1.000 | 1.000 | 0.0005 |
| J | 0.542 | [0.495, 0.593] | 0.019 | 0.015 | 0.0690 |
| C_J | 0.749 | [0.667, 0.824] | 0.981 | 0.985 | 0.0005 |
| randsub | 0.508 | [0.446, 0.565] | 0.005 | 0.005 | 0.3988 |
| C_randsub | 0.779 | [0.700, 0.853] | 0.995 | 0.995 | 0.0005 |
| shuffled | 0.542 | [0.492, 0.589] | 0.015 | 0.011 | 0.0615 |
| C_shuffled | 0.787 | [0.709, 0.858] | 0.985 | 0.989 | 0.0005 |
| randrows | 0.477 | [0.426, 0.529] | 0.003 | 0.002 | 0.7996 |
| C_randrows | 0.775 | [0.694, 0.850] | 0.997 | 0.998 | 0.0005 |
| pca | 0.784 | [0.704, 0.857] | 0.326 | 0.264 | 0.0005 |
| C_pca | 0.478 | [0.385, 0.565] | 0.674 | 0.736 | 0.6437 |
| contrast | 0.500 | [0.411, 0.590] | 0.141 | 0.130 | 0.4943 |

| difference | estimate | 95% CI |
|---|---:|---|
| C_J-J | +0.206 | [+0.109, +0.293] |
| J-randsub | +0.034 | [-0.033, +0.108] |
| J-shuffled | +0.001 | [-0.067, +0.075] |
| J-randrows | +0.066 | [-0.000, +0.137] |
| J-pca | -0.242 | [-0.327, -0.146] |
| C_J-full | -0.024 | [-0.039, -0.009] |
| C_randsub-full | +0.006 | [+0.001, +0.011] |
| contrast-J | -0.042 | [-0.146, +0.059] |
| pca-J | +0.242 | [+0.146, +0.327] |

Per layer, J vs complement vs full at k = 25:

| layer | full | J | C_J | randsub | shuffled | pca | contrast |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 26 | 0.767 | 0.566 | 0.760 | 0.507 | 0.577 | 0.700 | 0.511 |
| 27 | 0.836 | 0.582 | 0.796 | 0.546 | 0.413 | 0.794 | 0.468 |
| 28 | 0.730 | 0.468 | 0.697 | 0.555 | 0.445 | 0.771 | 0.478 |
| 29 | 0.744 | 0.618 | 0.710 | 0.425 | 0.652 | 0.812 | 0.540 |
| 30 | 0.788 | 0.478 | 0.781 | 0.507 | 0.621 | 0.845 | 0.505 |

Per family at the centre layer L28, k = 25 (Split AUROC):

| family | full | J | C_J | randsub |
|---|---:|---:|---:|---:|
| cicd | 0.667 | 0.222 | 0.778 | 0.833 |
| code_review | 0.694 | 0.306 | 0.667 | 0.306 |
| news | 0.639 | 0.722 | 0.611 | 0.639 |
| pump_dump | 0.333 | 0.472 | 0.306 | 0.278 |
| rag | 0.722 | 0.222 | 0.500 | 0.611 |
| research | 0.917 | 0.611 | 0.861 | 0.361 |

## A1. Core leave-one-domain-out (fit and eval on discussion turns)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| J | 0.940 | 0.960 | 0.996 | 1.000 | 1.000 |
| C_J | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randsub | 0.944 | 0.915 | 0.988 | 1.000 | 1.000 |
| C_randsub | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| shuffled | 0.368 | 0.478 | 0.562 | 0.636 | 0.845 |
| C_shuffled | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randrows | 0.386 | 0.265 | 0.477 | 0.446 | 0.581 |
| C_randrows | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pca | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| C_pca | 0.588 | 0.669 | 0.692 | 0.829 | 0.661 |
| contrast | 0.863 | 0.946 | 0.948 | 0.927 | 0.927 |

Fold-paired differences at k = 25, centre layer (full minus arm; folds = held-out domains):

| arm | mean diff | folds where full > arm | folds | sign-test p |
|---|---:|---:|---:|---:|
| J | +0.003 | 1 | 14 | 1.000 |
| C_J | +0.000 | 0 | 14 | 1.000 |
| randsub | +0.003 | 1 | 14 | 1.000 |
| C_randsub | +0.000 | 0 | 14 | 1.000 |
| shuffled | +0.495 | 13 | 14 | 0.000 |
| C_shuffled | +0.000 | 0 | 14 | 1.000 |
| randrows | +0.624 | 12 | 14 | 0.000 |
| C_randrows | +0.000 | 0 | 14 | 1.000 |
| pca | +0.000 | 0 | 14 | 1.000 |
| C_pca | +0.294 | 13 | 14 | 0.000 |
| contrast | +0.045 | 3 | 14 | 0.250 |

## A3. Within-Transfer leave-one-family-out (fit and eval on final turns; pooled held-out AUROC)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.924 | 0.924 | 0.924 | 0.924 | 0.924 |
| J | 0.626 | 0.662 | 0.633 | 0.620 | 0.662 |
| C_J | 0.921 | 0.918 | 0.915 | 0.912 | 0.907 |
| randsub | 0.506 | 0.542 | 0.555 | 0.660 | 0.716 |
| C_randsub | 0.925 | 0.924 | 0.924 | 0.922 | 0.922 |
| shuffled | 0.511 | 0.515 | 0.503 | 0.535 | 0.543 |
| C_shuffled | 0.919 | 0.918 | 0.910 | 0.909 | 0.908 |
| randrows | 0.454 | 0.467 | 0.471 | 0.519 | 0.486 |
| C_randrows | 0.921 | 0.921 | 0.919 | 0.916 | 0.907 |
| pca | 0.690 | 0.775 | 0.672 | 0.805 | 0.886 |
| C_pca | 0.893 | 0.894 | 0.891 | 0.885 | 0.882 |
| contrast | 0.598 | 0.711 | 0.717 | 0.719 | 0.719 |

Inference at k = 25 (pooled held-out episodes, bootstrap over episodes, permutation within family):

| arm | primary | 95% CI | p (perm) |
|---|---:|---|---:|
| full | 0.924 | [0.889, 0.955] | 0.0005 |
| J | 0.633 | [0.573, 0.685] | 0.0005 |
| C_J | 0.915 | [0.876, 0.950] | 0.0005 |
| randsub | 0.555 | [0.506, 0.601] | 0.0245 |
| C_randsub | 0.924 | [0.888, 0.954] | 0.0005 |
| shuffled | 0.503 | [0.454, 0.553] | 0.4728 |
| C_shuffled | 0.910 | [0.873, 0.944] | 0.0005 |
| randrows | 0.471 | [0.418, 0.521] | 0.8401 |
| C_randrows | 0.919 | [0.882, 0.951] | 0.0005 |
| pca | 0.672 | [0.605, 0.735] | 0.0010 |
| C_pca | 0.891 | [0.852, 0.927] | 0.0005 |
| contrast | 0.717 | [0.640, 0.789] | 0.0005 |

| difference | estimate | 95% CI |
|---|---:|---|
| C_J-J | +0.282 | [+0.229, +0.339] |
| J-randsub | +0.078 | [+0.011, +0.144] |
| J-shuffled | +0.130 | [+0.061, +0.197] |
| J-randrows | +0.162 | [+0.088, +0.233] |
| J-pca | -0.039 | [-0.122, +0.047] |
| C_J-full | -0.009 | [-0.017, -0.002] |
| C_randsub-full | -0.000 | [-0.002, +0.001] |
| contrast-J | +0.084 | [-0.001, +0.169] |
| pca-J | +0.039 | [-0.047, +0.122] |

## A4a. Core discussion → Core final, leave-one-domain-out (phase changes, family does not)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| J | 0.770 | 0.744 | 0.663 | 0.712 | 0.841 |
| C_J | 1.000 | 0.976 | 1.000 | 1.000 | 1.000 |
| randsub | 0.481 | 0.435 | 0.700 | 0.560 | 0.772 |
| C_randsub | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| shuffled | 0.495 | 0.652 | 0.455 | 0.586 | 0.560 |
| C_shuffled | 1.000 | 1.000 | 1.000 | 0.994 | 1.000 |
| randrows | 0.535 | 0.606 | 0.492 | 0.516 | 0.363 |
| C_randrows | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pca | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| C_pca | 0.714 | 0.573 | 0.613 | 0.773 | 0.574 |
| contrast | 0.602 | 0.726 | 0.733 | 0.672 | 0.672 |

## A4b. Core discussion → Transfer discussion (family changes, phase does not)

not available for this model (see `data` above).

## A5. Geometry (centre layer, k = 25)

Share of the full probe direction inside each activation's own frame: median 0.005, max 0.011; a random direction: median 0.005; k/d = 0.0049.

|cos| between the full probe direction and the direction each arm learns: J 0.011, C_J 0.952, shuffled 0.003, C_shuffled 0.990, randrows 0.050, C_randrows 0.996, randsub 0.045, C_randsub 0.997, pca 0.931, C_pca 0.111, contrast 0.299.

Frame tokens shared between colluder and honest turns (Jaccard of the 100 most frequent): 0.550.

| role | 25 most frequent frame tokens (share of that role's discussion turns whose frame holds the token) |
|---|---|
| colluder | `"='\\"` 1.00, `' vücud'` 0.98, `'("\\'` 0.95, `'输卵'` 0.74, `'okemon'` 0.73, `'ettel'` 0.71, `'prehensive'` 0.70, `'coder'` 0.68, `'颎'` 0.65, `';)'` 0.54, `'или'` 0.54, `' steadfast'` 0.51, `'lays'` 0.50, `'.\n\n'` 0.47, `' vüc'` 0.41, `'こう'` 0.41, `'.\n'` 0.40, `'raisal'` 0.39, `' mijn'` 0.38, `'opian'` 0.38, `'浙江省'` 0.34, `'ofi'` 0.34, `'окумент'` 0.34, `'ategorical'` 0.33, `' köz'` 0.33 |
| honest | `"='\\"` 1.00, `'ettel'` 0.95, `'("\\'` 0.94, `' vücud'` 0.91, `'浙江省'` 0.82, `'okemon'` 0.79, `'.\n\n'` 0.76, `'prehensive'` 0.75, `'или'` 0.63, `'输卵'` 0.60, `';)'` 0.56, `' vüc'` 0.55, `'coder'` 0.54, `'颎'` 0.53, `'opian'` 0.49, `'﹩'` 0.49, `'.\n'` 0.43, `'执法人员'` 0.38, `'окумент'` 0.36, `'.constant'` 0.35, `' mijn'` 0.35, `'فاع'` 0.32, `'lays'` 0.31, `'.atan'` 0.31, `'ofi'` 0.30 |

## Readings (rules fixed in the design, section 6)

- **R1_in_distribution**: J-space separates colluders in distribution
  J_primary = 0.996; full_minus_J = 0.004
- **R2_transfer**: mixed: C minus J clear, J vs random unclear, complement vs full differs
  J = 0.542; C_J = 0.749; randsub = 0.508; shuffled = 0.542; full = 0.773; p_perm_J = 0.069; p_perm_randsub = 0.399
- **R3_complement_uninformative**: complement-of-random equals full (as expected); 'complement equals full' says nothing about J-space by itself
  
- **R4_within_transfer**: intermediate: within-Transfer J probe above chance but below 0.80, or Core-fit J above 0.65
  J_within_transfer = 0.633; J_core_to_transfer = 0.542; p_perm_within = 0.000; C_J_within_transfer = 0.915; full_within_transfer = 0.924
- **R5_phase_vs_family**: the J drop is mostly the phase change (discussion to final)
  J_core_disc_to_core_final = 0.663; full_core_disc_to_core_final = 1.000; J_core_disc_to_transfer_final = 0.542; J_core_disc_to_transfer_discussion = None; full_core_disc_to_transfer_discussion = None; drop_phase_only = 0.333; drop_total = 0.454
- **R6_dimension**: dimension is not the explanation: a supervised or PCA subspace of at most 16 dimensions reaches the full transfer AUROC while J does not
  best_contrast_k_le_16 = 0.574; best_pca_k_le_16 = 0.798; best_J_k_le_16 = 0.583; full = 0.773

## Baseline reproduction

Full-activation Core LDO primary 1.000 (paper: 1.00 for every model in distribution); full-activation Core → Transfer band mean 0.773.
