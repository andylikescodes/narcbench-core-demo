# Frozen Gemma-2-9B J-space numbers with uncertainty

Source: `results/transfer/FULL_core_to_transfer_*/metrics.json` (HARD Transfer, 36 collusion + 36 control episodes; Core discussion → Transfer final). No per-episode scores were kept, so standard errors are Hanley–McNeil approximations; the primary's SE is the mean of the three metric SEs (treats them as perfectly correlated, which is conservative). Contrasts treat arms as independent. Method: `scripts/jspace_offline_stats.py`.

## Band means (L19–23, or L19/21/23 for the Jacobian arms)

| arm | primary | SE | 95% CI | z vs 0.5 | p | Peak [CI] | Split [CI] | Asym [CI] |
|---|---:|---:|---|---:|---:|---|---|---|
| full | 0.902 | 0.037 | [0.829, 0.975] | 10.80 | 3.33e-27 | 0.874 [0.79, 0.96] | 0.904 [0.83, 0.98] | 0.929 [0.87, 0.99] |
| W_U k25 J-only | 0.602 | 0.067 | [0.471, 0.733] | 1.53 | 0.126 | 0.625 [0.50, 0.75] | 0.565 [0.43, 0.70] | 0.616 [0.49, 0.75] |
| W_U k25 complement | 0.898 | 0.038 | [0.823, 0.972] | 10.45 | 1.5e-25 | 0.869 [0.78, 0.95] | 0.897 [0.82, 0.97] | 0.928 [0.86, 0.99] |
| jac k25 J-only | 0.549 | 0.068 | [0.416, 0.683] | 0.73 | 0.467 | 0.586 [0.45, 0.72] | 0.523 [0.39, 0.66] | 0.540 [0.41, 0.67] |
| jac k25 complement | 0.891 | 0.039 | [0.814, 0.968] | 9.92 | 3.56e-23 | 0.867 [0.78, 0.95] | 0.888 [0.81, 0.97] | 0.917 [0.85, 0.98] |

## Contrasts

| contrast | difference | SE | 95% CI | z | p |
|---|---:|---:|---|---:|---:|
| W_U k25 complement minus W_U k25 J-only | +0.296 | 0.077 | [+0.145, +0.446] | 3.85 | 0.000116 |
| jac k25 complement minus jac k25 J-only | +0.341 | 0.079 | [+0.187, +0.495] | 4.34 | 1.4e-05 |
| full minus W_U k25 complement | +0.005 | 0.053 | [-0.100, +0.109] | 0.09 | 0.931 |
| full minus jac k25 complement | +0.012 | 0.054 | [-0.095, +0.118] | 0.21 | 0.832 |

## Per layer, primary with 95% CI

| layer | full | W_U k25 J-only | W_U k25 complement | jac k25 J-only | jac k25 complement |
|---:|---:|---:|---:|---:|---:|
| 19 | 0.884 [0.80, 0.96] | 0.585 [0.45, 0.72] | 0.885 [0.81, 0.96] | 0.480 [0.35, 0.61] | 0.864 [0.78, 0.95] |
| 20 | 0.885 [0.81, 0.96] | 0.570 [0.44, 0.70] | 0.869 [0.78, 0.95] | — | — |
| 21 | 0.941 [0.89, 1.00] | 0.756 [0.65, 0.87] | 0.935 [0.88, 0.99] | 0.570 [0.44, 0.70] | 0.935 [0.88, 0.99] |
| 22 | 0.930 [0.87, 0.99] | 0.522 [0.39, 0.66] | 0.928 [0.87, 0.99] | — | — |
| 23 | 0.873 [0.79, 0.96] | 0.577 [0.44, 0.71] | 0.873 [0.79, 0.96] | 0.599 [0.47, 0.73] | 0.873 [0.79, 0.96] |

## Per family (12 episodes each; Split AUROC) for the W_U k25 arms, by layer

| layer | family | full | J-only | complement |
|---:|---|---:|---:|---:|
| 19 | cicd | 1.000 | 0.917 | 1.000 |
| 19 | code_review | 1.000 | 0.667 | 1.000 |
| 19 | news | 0.417 | 0.556 | 0.333 |
| 19 | pump_dump | 0.583 | 0.333 | 0.556 |
| 19 | rag | 1.000 | 0.306 | 1.000 |
| 19 | research | 0.972 | 0.389 | 0.972 |
| 20 | cicd | 1.000 | 0.361 | 0.972 |
| 20 | code_review | 1.000 | 0.778 | 1.000 |
| 20 | news | 0.472 | 0.611 | 0.472 |
| 20 | pump_dump | 0.667 | 0.556 | 0.500 |
| 20 | rag | 0.972 | 0.333 | 0.972 |
| 20 | research | 0.944 | 0.806 | 0.889 |
| 21 | cicd | 1.000 | 0.917 | 1.000 |
| 21 | code_review | 1.000 | 0.861 | 1.000 |
| 21 | news | 0.917 | 0.611 | 0.889 |
| 21 | pump_dump | 1.000 | 0.750 | 0.972 |
| 21 | rag | 1.000 | 0.333 | 1.000 |
| 21 | research | 1.000 | 0.389 | 1.000 |
| 22 | cicd | 1.000 | 0.278 | 1.000 |
| 22 | code_review | 1.000 | 0.583 | 1.000 |
| 22 | news | 0.889 | 0.194 | 0.889 |
| 22 | pump_dump | 0.944 | 0.750 | 0.944 |
| 22 | rag | 0.944 | 0.528 | 0.972 |
| 22 | research | 1.000 | 0.639 | 1.000 |
| 23 | cicd | 1.000 | 0.889 | 1.000 |
| 23 | code_review | 1.000 | 0.889 | 1.000 |
| 23 | news | 0.667 | 0.417 | 0.639 |
| 23 | pump_dump | 0.778 | 0.444 | 0.667 |
| 23 | rag | 0.750 | 0.306 | 0.694 |
| 23 | research | 1.000 | 0.556 | 1.000 |

## Core leave-one-domain-out battery (50 + 50 episodes; `docs/NEXT-SCIENCE.md`)

Fold-mean AUROCs; the CI below treats each as a single AUROC over 100 episodes, which ignores the fold structure and is only indicative.

| variant | J-only | 95% CI (indicative) | complement |
|---|---:|---|---:|
| wu_topk k8 | 0.894 | [0.829, 0.959] | 1.000 |
| wu_topk k16 | 0.950 | [0.905, 0.995] | 1.000 |
| wu_topk k25 | 0.989 | [0.968, 1.000] | 1.000 |
| wu_topk k50 | 1.000 | [1.000, 1.000] | 1.000 |
| wu_fixed k25 | 0.990 | [0.970, 1.000] | 1.000 |
| jac_topk k8 | 0.900 | [0.837, 0.963] | 1.000 |
| jac_topk k16 | 0.919 | [0.862, 0.976] | 1.000 |
| jac_topk k25 | 0.961 | [0.922, 1.000] | 1.000 |
| jac_topk k50 | 0.997 | [0.986, 1.000] | 1.000 |

## What these numbers can and cannot say

- Every J-only Transfer arm is compatible with chance at this sample size (p above 0.05 against 0.5), and every complement arm is far above chance.
- The complement-minus-J-only contrast is large and significant even under the conservative independence approximation.
- What the frozen numbers lack: a dimension-matched control (a random 25-dimensional subspace and its complement), the shuffled-frame control, a within-Transfer J probe, and the phase-versus-family split. `scripts/jspace_separability.py` adds all of these; it needs the activation caches, which are not in git.
