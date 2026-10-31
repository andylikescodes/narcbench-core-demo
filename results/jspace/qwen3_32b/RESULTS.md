# J-space separability — qwen3_32b

Design: `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md`. Script: `scripts/jspace_separability.py`. Written 2026-10-31 22:43 UTC. Layers [26, 27, 28, 29, 30] (centre 28); frame sizes k = [8, 16, 25, 50, 100]; bootstrap B = 2000, permutations B = 2000; seed 42.

## Data

| Split | Rows used | Zero rows dropped | Runs | Domains / families |
|---|---:|---:|---:|---|
| core | 1054 | 0 | 62 | 14: audit, cyber, eng, env, fin, food, gov, hire… |
| transfer | 288 | 0 | 72 | 6: cicd, code_review, news, pump_dump, rag, research |

## A2. Core discussion → Transfer final, band mean (primary AUROC)

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.773 | (same) | (same) | (same) | (same) |
| J | 0.594 | 0.515 | 0.513 | 0.564 | 0.519 |
| C_J | 0.776 | 0.769 | 0.775 | 0.766 | 0.770 |
| randsub | 0.528 | 0.506 | 0.508 | 0.519 | 0.546 |
| C_randsub | 0.771 | 0.779 | 0.779 | 0.775 | 0.783 |
| shuffled | 0.508 | 0.538 | 0.506 | 0.523 | 0.517 |
| C_shuffled | 0.792 | 0.799 | 0.806 | 0.790 | 0.795 |
| randrows | 0.502 | 0.500 | 0.536 | 0.501 | 0.502 |
| C_randrows | 0.776 | 0.778 | 0.778 | 0.785 | 0.784 |
| pca | 0.637 | 0.798 | 0.784 | 0.767 | 0.758 |
| C_pca | 0.552 | 0.468 | 0.478 | 0.478 | 0.512 |
| contrast | 0.450 | 0.574 | 0.500 | 0.460 | 0.460 |

### A2 inference at k = 25 (band mean; 95% bootstrap intervals over episodes; permutation p for primary > 0.5)

| arm | primary | 95% CI | mean FVE core | mean FVE transfer | p (perm) |
|---|---:|---|---:|---:|---:|
| full | 0.773 | [0.692, 0.847] | 1.000 | 1.000 | 0.0005 |
| J | 0.513 | [0.459, 0.564] | 0.032 | 0.024 | 0.3118 |
| C_J | 0.775 | [0.696, 0.846] | 0.968 | 0.976 | 0.0005 |
| randsub | 0.508 | [0.446, 0.565] | 0.005 | 0.005 | 0.3988 |
| C_randsub | 0.779 | [0.700, 0.853] | 0.995 | 0.995 | 0.0005 |
| shuffled | 0.506 | [0.461, 0.550] | 0.025 | 0.017 | 0.4013 |
| C_shuffled | 0.806 | [0.731, 0.874] | 0.975 | 0.983 | 0.0005 |
| randrows | 0.536 | [0.491, 0.581] | 0.003 | 0.003 | 0.0855 |
| C_randrows | 0.778 | [0.698, 0.852] | 0.997 | 0.997 | 0.0005 |
| pca | 0.784 | [0.704, 0.857] | 0.326 | 0.264 | 0.0005 |
| C_pca | 0.478 | [0.385, 0.565] | 0.674 | 0.736 | 0.6437 |
| contrast | 0.500 | [0.411, 0.590] | 0.141 | 0.130 | 0.4943 |

| difference | estimate | 95% CI |
|---|---:|---|
| C_J-J | +0.262 | [+0.172, +0.345] |
| J-randsub | +0.005 | [-0.053, +0.066] |
| J-shuffled | +0.007 | [-0.067, +0.077] |
| J-randrows | -0.023 | [-0.093, +0.043] |
| J-pca | -0.271 | [-0.362, -0.172] |
| C_J-full | +0.002 | [-0.019, +0.022] |
| C_randsub-full | +0.006 | [+0.001, +0.011] |
| contrast-J | -0.013 | [-0.106, +0.082] |
| pca-J | +0.271 | [+0.172, +0.362] |

Per layer, J vs complement vs full at k = 25:

| layer | full | J | C_J | randsub | shuffled | pca | contrast |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 26 | 0.767 | 0.512 | 0.770 | 0.507 | 0.521 | 0.700 | 0.511 |
| 27 | 0.836 | 0.516 | 0.792 | 0.546 | 0.487 | 0.794 | 0.468 |
| 28 | 0.730 | 0.541 | 0.765 | 0.555 | 0.499 | 0.771 | 0.478 |
| 29 | 0.744 | 0.467 | 0.758 | 0.425 | 0.513 | 0.812 | 0.540 |
| 30 | 0.788 | 0.530 | 0.789 | 0.507 | 0.510 | 0.845 | 0.505 |

Per family at the centre layer L28, k = 25 (Split AUROC):

| family | full | J | C_J | randsub |
|---|---:|---:|---:|---:|
| cicd | 0.667 | 0.472 | 0.722 | 0.833 |
| code_review | 0.694 | 0.306 | 0.722 | 0.306 |
| news | 0.639 | 0.639 | 0.694 | 0.639 |
| pump_dump | 0.333 | 0.417 | 0.500 | 0.278 |
| rag | 0.722 | 0.417 | 0.750 | 0.611 |
| research | 0.917 | 0.583 | 0.972 | 0.361 |

## A1. Core leave-one-domain-out (fit and eval on discussion turns)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| J | 0.860 | 0.997 | 0.991 | 1.000 | 1.000 |
| C_J | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randsub | 0.944 | 0.915 | 0.988 | 1.000 | 1.000 |
| C_randsub | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| shuffled | 0.437 | 0.655 | 0.580 | 0.399 | 0.743 |
| C_shuffled | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randrows | 0.501 | 0.372 | 0.502 | 0.557 | 0.639 |
| C_randrows | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pca | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| C_pca | 0.588 | 0.669 | 0.692 | 0.829 | 0.661 |
| contrast | 0.863 | 0.946 | 0.948 | 0.927 | 0.927 |

Fold-paired differences at k = 25, centre layer (full minus arm; folds = held-out domains):

| arm | mean diff | folds where full > arm | folds | sign-test p |
|---|---:|---:|---:|---:|
| J | +0.024 | 5 | 14 | 0.062 |
| C_J | +0.000 | 0 | 14 | 1.000 |
| randsub | +0.003 | 1 | 14 | 1.000 |
| C_randsub | +0.000 | 0 | 14 | 1.000 |
| shuffled | +0.444 | 12 | 14 | 0.000 |
| C_shuffled | +0.000 | 0 | 14 | 1.000 |
| randrows | +0.647 | 13 | 14 | 0.000 |
| C_randrows | +0.000 | 0 | 14 | 1.000 |
| pca | +0.000 | 0 | 14 | 1.000 |
| C_pca | +0.294 | 13 | 14 | 0.000 |
| contrast | +0.045 | 3 | 14 | 0.250 |

## A3. Within-Transfer leave-one-family-out (fit and eval on final turns; pooled held-out AUROC)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 0.924 | 0.924 | 0.924 | 0.924 | 0.924 |
| J | 0.576 | 0.567 | 0.600 | 0.660 | 0.679 |
| C_J | 0.917 | 0.916 | 0.904 | 0.905 | 0.897 |
| randsub | 0.506 | 0.542 | 0.555 | 0.660 | 0.716 |
| C_randsub | 0.925 | 0.924 | 0.924 | 0.922 | 0.922 |
| shuffled | 0.506 | 0.493 | 0.504 | 0.518 | 0.508 |
| C_shuffled | 0.918 | 0.912 | 0.908 | 0.901 | 0.889 |
| randrows | 0.465 | 0.480 | 0.469 | 0.529 | 0.472 |
| C_randrows | 0.921 | 0.921 | 0.920 | 0.914 | 0.902 |
| pca | 0.690 | 0.775 | 0.672 | 0.805 | 0.886 |
| C_pca | 0.893 | 0.894 | 0.891 | 0.885 | 0.882 |
| contrast | 0.598 | 0.711 | 0.717 | 0.719 | 0.719 |

Inference at k = 25 (pooled held-out episodes, bootstrap over episodes, permutation within family):

| arm | primary | 95% CI | p (perm) |
|---|---:|---|---:|
| full | 0.924 | [0.889, 0.955] | 0.0005 |
| J | 0.600 | [0.545, 0.656] | 0.0010 |
| C_J | 0.904 | [0.863, 0.941] | 0.0005 |
| randsub | 0.555 | [0.506, 0.601] | 0.0245 |
| C_randsub | 0.924 | [0.888, 0.954] | 0.0005 |
| shuffled | 0.504 | [0.452, 0.556] | 0.4573 |
| C_shuffled | 0.908 | [0.868, 0.943] | 0.0005 |
| randrows | 0.469 | [0.425, 0.513] | 0.8881 |
| C_randrows | 0.920 | [0.884, 0.951] | 0.0005 |
| pca | 0.672 | [0.605, 0.735] | 0.0010 |
| C_pca | 0.891 | [0.852, 0.927] | 0.0005 |
| contrast | 0.717 | [0.640, 0.789] | 0.0005 |

| difference | estimate | 95% CI |
|---|---:|---|
| C_J-J | +0.304 | [+0.252, +0.357] |
| J-randsub | +0.046 | [-0.012, +0.105] |
| J-shuffled | +0.096 | [+0.022, +0.172] |
| J-randrows | +0.131 | [+0.061, +0.204] |
| J-pca | -0.072 | [-0.155, +0.013] |
| C_J-full | -0.020 | [-0.031, -0.010] |
| C_randsub-full | -0.000 | [-0.002, +0.001] |
| contrast-J | +0.117 | [+0.038, +0.198] |
| pca-J | +0.072 | [-0.013, +0.155] |

## A4a. Core discussion → Core final, leave-one-domain-out (phase changes, family does not)

Band mean over layers that were run (per layer in `summary.json`); primary AUROC.

| arm | k=8 | k=16 | k=25 | k=50 | k=100 |
|---|---:|---:|---:|---:|---:|
| full | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| J | 0.683 | 0.544 | 0.700 | 0.647 | 0.845 |
| C_J | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| randsub | 0.481 | 0.435 | 0.700 | 0.560 | 0.772 |
| C_randsub | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| shuffled | 0.568 | 0.577 | 0.537 | 0.466 | 0.485 |
| C_shuffled | 1.000 | 0.994 | 0.994 | 1.000 | 1.000 |
| randrows | 0.611 | 0.546 | 0.502 | 0.596 | 0.345 |
| C_randrows | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pca | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| C_pca | 0.714 | 0.573 | 0.613 | 0.773 | 0.574 |
| contrast | 0.602 | 0.726 | 0.733 | 0.672 | 0.672 |

## A4b. Core discussion → Transfer discussion (family changes, phase does not)

not available for this model (see `data` above).

## A5. Geometry (centre layer, k = 25)

Share of the full probe direction inside each activation's own frame: median 0.006, max 0.011; a random direction: median 0.005; k/d = 0.0049.

|cos| between the full probe direction and the direction each arm learns: J 0.121, C_J 0.932, shuffled 0.008, C_shuffled 0.976, randrows 0.108, C_randrows 0.996, randsub 0.045, C_randsub 0.997, pca 0.931, C_pca 0.111, contrast 0.299.

Frame tokens shared between colluder and honest turns (Jaccard of the 100 most frequent): 0.562.

| role | 25 most frequent frame tokens (share of that role's discussion turns whose frame holds the token) |
|---|---|
| colluder | `"='\\"` 1.00, `' silently'` 0.97, `' vücud'` 0.96, `'prehensive'` 0.96, `'("\\'` 0.84, `'.\n'` 0.83, `'coder'` 0.80, `'lays'` 0.77, `' steadfast'` 0.59, `'浙江省'` 0.57, `'otal'` 0.57, `'raisal'` 0.54, `' vüc'` 0.53, `'<>('` 0.49, `' mijn'` 0.48, `' slightly'` 0.48, `'Fade'` 0.48, `'赪'` 0.47, `'фессиона'` 0.44, `'ofi'` 0.43, `'erts'` 0.43, `'или'` 0.42, `'circ'` 0.36, `'itest'` 0.35, `'(high'` 0.35 |
| honest | `"='\\"` 1.00, `'prehensive'` 0.97, `'otal'` 0.97, `'("\\'` 0.89, `'浙江省'` 0.89, `'.\n'` 0.88, `' silently'` 0.88, `'coder'` 0.82, `' vücud'` 0.80, `' vüc'` 0.70, `'lays'` 0.68, `'<>('` 0.61, `'Fade'` 0.56, `' slightly'` 0.56, `'или'` 0.56, `'.constant'` 0.54, `'.\n\n'` 0.53, `' mijn'` 0.49, `'circ'` 0.42, `'ofi'` 0.41, `'(high'` 0.41, `'ettel'` 0.38, `' �'` 0.36, `' steadfast'` 0.36, `'�'` 0.35 |

## Readings (rules fixed in the design, section 6)

- **R1_in_distribution**: J-space separates colluders in distribution
  J_primary = 0.991; full_minus_J = 0.009
- **R2_transfer**: the transferable signal is not in J-space
  J = 0.513; C_J = 0.775; randsub = 0.508; shuffled = 0.506; full = 0.773; p_perm_J = 0.312; p_perm_randsub = 0.399
- **R3_complement_uninformative**: complement-of-random equals full (as expected); 'complement equals full' says nothing about J-space by itself
  
- **R4_within_transfer**: intermediate: within-Transfer J probe above chance but below 0.80, or Core-fit J above 0.65
  J_within_transfer = 0.600; J_core_to_transfer = 0.513; p_perm_within = 0.001; C_J_within_transfer = 0.904; full_within_transfer = 0.924
- **R5_phase_vs_family**: the J drop is mostly the phase change (discussion to final)
  J_core_disc_to_core_final = 0.700; full_core_disc_to_core_final = 1.000; J_core_disc_to_transfer_final = 0.513; J_core_disc_to_transfer_discussion = None; full_core_disc_to_transfer_discussion = None; drop_phase_only = 0.291; drop_total = 0.478
- **R6_dimension**: dimension is not the explanation: a supervised or PCA subspace of at most 16 dimensions reaches the full transfer AUROC while J does not
  best_contrast_k_le_16 = 0.574; best_pca_k_le_16 = 0.798; best_J_k_le_16 = 0.594; full = 0.773

## Baseline reproduction

Full-activation Core LDO primary 1.000 (paper: 1.00 for every model in distribution); full-activation Core → Transfer band mean 0.773.
