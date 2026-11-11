# jspace-interp results summary

Inputs: 16 sweep file(s), 4 steering file(s). Planned cells 80, ran 80.

Primary J-space arm (the columns headed 'J fill' below): the non-negative combination of the vectors the pursuit chose (`J{k}_recon` in the sweep files), the form of the workspace paper's Methods. The column headed 'J pursuit' is the span of the same vectors without the fill; the span of exactly k vectors is `J_span` in SUMMARY.json.

## original (k=25)

- cells: 35; J fill clears both nulls in 1, J pursuit in 2; J fill beats PCA (paired CI) in 0, PCA beats J fill in 6; full beats J fill in 12
- best J fill: last L22 = 0.929 [0.89, 0.96] (full 0.967, PCA 0.937, rand sub p95 0.851, rand atoms p95 0.969, clears both: False, J-PCA CI [-0.06, 0.04], recovery 0.92)

| position | full | J fill | J pursuit | PCA | rand sub p95 | rand atoms p95 | clears |
|---|---|---|---|---|---|---|---|
| last | 0.945 | 0.819 | 0.815 | 0.879 | 0.808 | 0.875 | 0/5 |
| eot | 0.780 | 0.590 | 0.587 | 0.695 | 0.719 | 0.763 | 0/5 |
| prevote | 0.517 | 0.538 | 0.541 | 0.493 | 0.600 | 0.588 | 0/5 |
| reply_mean | 0.706 | 0.582 | 0.583 | 0.679 | 0.691 | 0.658 | 0/5 |
| frac25 | 0.522 | 0.535 | 0.531 | 0.507 | 0.595 | 0.580 | 0/5 |
| frac50 | 0.527 | 0.540 | 0.539 | 0.538 | 0.598 | 0.596 | 1/5 |
| frac75 | 0.571 | 0.471 | 0.470 | 0.546 | 0.619 | 0.602 | 0/5 |

## swap (k=25)

- cells: 35; J fill clears both nulls in 3, J pursuit in 3; J fill beats PCA (paired CI) in 0, PCA beats J fill in 1; full beats J fill in 2
- best J fill: prevote L22 = 0.612 [0.51, 0.72] (full 0.504, PCA 0.501, rand sub p95 0.594, rand atoms p95 0.606, clears both: True, J-PCA CI [-0.01, 0.23], recovery n/a (full residual at chance))

| position | full | J fill | J pursuit | PCA | rand sub p95 | rand atoms p95 | clears |
|---|---|---|---|---|---|---|---|
| last | 0.479 | 0.465 | 0.464 | 0.465 | 0.609 | 0.595 | 0/5 |
| eot | 0.491 | 0.477 | 0.485 | 0.491 | 0.611 | 0.594 | 0/5 |
| prevote | 0.469 | 0.527 | 0.544 | 0.480 | 0.587 | 0.572 | 2/5 |
| reply_mean | 0.554 | 0.562 | 0.563 | 0.541 | 0.649 | 0.627 | 0/5 |
| frac25 | 0.488 | 0.513 | 0.512 | 0.643 | 0.612 | 0.596 | 0/5 |
| frac50 | 0.519 | 0.566 | 0.567 | 0.610 | 0.608 | 0.602 | 1/5 |
| frac75 | 0.573 | 0.465 | 0.467 | 0.514 | 0.601 | 0.592 | 0/5 |

## nosys (k=25)

- cells: 5; J fill clears both nulls in 0, J pursuit in 0; J fill beats PCA (paired CI) in 1, PCA beats J fill in 0; full beats J fill in 0
- best J fill: last L22 = 0.576 [0.49, 0.66] (full 0.490, PCA 0.441, rand sub p95 0.588, rand atoms p95 0.587, clears both: False, J-PCA CI [0.03, 0.24], recovery n/a (full residual at chance))

| position | full | J fill | J pursuit | PCA | rand sub p95 | rand atoms p95 | clears |
|---|---|---|---|---|---|---|---|
| last | 0.488 | 0.525 | 0.496 | 0.466 | 0.608 | 0.586 | 0/5 |

## prompt_only (k=25)

- cells: 5; J fill clears both nulls in 0, J pursuit in 0; J fill beats PCA (paired CI) in 0, PCA beats J fill in 4; full beats J fill in 4
- best J fill: prompt_end L20 = 1.000 [1.00, 1.00] (full 1.000, PCA 1.000, rand sub p95 1.000, rand atoms p95 0.994, clears both: False, J-PCA CI [0.00, 0.00], recovery 1.00)

| position | full | J fill | J pursuit | PCA | rand sub p95 | rand atoms p95 | clears |
|---|---|---|---|---|---|---|---|
| prompt_end | 1.000 | 0.976 | 0.980 | 1.000 | 1.000 | 0.997 | 0/5 |

## prompt_only probe (prompt_end)

| layer | full | J fill | PCA | rand p95 |
|---|---|---|---|---|
| 19 | 1.000 | 0.989 | 1.000 | 1.000 |
| 20 | 1.000 | 1.000 | 1.000 | 1.000 |
| 21 | 1.000 | 0.980 | 1.000 | 1.000 |
| 22 | 1.000 | 0.966 | 1.000 | 1.000 |
| 23 | 1.000 | 0.947 | 1.000 | 0.999 |

## matched vocabulary (k=25)

| cell | arm | colluder vs warned | warned vs unwarned (p) | warned vs control | echo | ep warned vs control | ep collusion vs warned |
|---|---|---|---|---|---|---|---|
| original/eot/L22 | full | 1.000 | 0.765 (0.00) | 0.714 | 0.08 | 0.720 | 1.000 |
| original/eot/L22 | J_fill | 0.980 | 0.547 (0.12) | 0.508 | 0.00 | 0.481 | 0.987 |
| original/eot/L22 | pca | 1.000 | 0.701 (0.00) | 0.671 | 0.06 | 0.638 | 1.000 |
| original/eot/L21 | full | 1.000 | 0.759 (0.00) | 0.731 | 0.09 | 0.721 | 1.000 |
| original/eot/L21 | J_fill | 0.959 | 0.556 (0.02) | 0.576 | 0.09 | 0.571 | 0.964 |
| original/eot/L21 | pca | 1.000 | 0.673 (0.00) | 0.653 | 0.06 | 0.637 | 1.000 |
| original/eot/L23 | full | 1.000 | 0.828 (0.00) | 0.798 | 0.12 | 0.811 | 1.000 |
| original/eot/L23 | J_fill | 0.964 | 0.556 (0.06) | 0.554 | 0.05 | 0.589 | 0.983 |
| original/eot/L23 | pca | 1.000 | 0.812 (0.00) | 0.776 | 0.11 | 0.753 | 1.000 |
| original/eot/L20 | full | 1.000 | 0.746 (0.00) | 0.732 | 0.09 | 0.678 | 1.000 |
| original/eot/L20 | J_fill | 0.967 | 0.523 (0.18) | 0.508 | 0.02 | 0.461 | 0.983 |
| original/eot/L20 | pca | 1.000 | 0.577 (0.01) | 0.570 | 0.03 | 0.542 | 1.000 |
| original/eot/L19 | full | 0.999 | 0.706 (0.00) | 0.712 | 0.10 | 0.630 | 1.000 |
| original/eot/L19 | J_fill | 0.931 | 0.568 (0.00) | 0.530 | 0.07 | 0.467 | 0.949 |
| original/eot/L19 | pca | 1.000 | 0.621 (0.00) | 0.630 | 0.07 | 0.583 | 1.000 |
| original/frac25/L22 | full | 0.974 | 0.538 (0.16) | 0.491 | -0.01 | 0.497 | 0.991 |
| original/frac25/L22 | J_fill | 0.718 | 0.478 (0.38) | 0.457 | -0.27 | 0.423 | 0.771 |
| original/frac25/L22 | pca | 0.862 | 0.552 (0.05) | 0.522 | 0.04 | 0.402 | 0.918 |
| original/frac25/L21 | full | 0.966 | 0.530 (0.28) | 0.484 | -0.02 | 0.492 | 0.979 |
| original/frac25/L21 | J_fill | 0.671 | 0.497 (0.77) | 0.485 | -0.19 | 0.474 | 0.645 |
| original/frac25/L21 | pca | 0.835 | 0.531 (0.22) | 0.524 | 0.04 | 0.456 | 0.883 |
| original/frac25/L23 | full | 0.963 | 0.546 (0.13) | 0.506 | 0.01 | 0.509 | 0.979 |
| original/frac25/L23 | J_fill | 0.728 | 0.505 (0.82) | 0.487 | -0.13 | 0.464 | 0.758 |
| original/frac25/L23 | pca | 0.870 | 0.559 (0.03) | 0.537 | 0.06 | 0.435 | 0.897 |
| original/frac25/L20 | full | 0.965 | 0.531 (0.34) | 0.478 | -0.04 | 0.486 | 0.976 |
| original/frac25/L20 | J_fill | 0.616 | 0.528 (0.47) | 0.499 | -0.07 | 0.463 | 0.586 |
| original/frac25/L20 | pca | 0.774 | 0.535 (0.15) | 0.508 | 0.03 | 0.440 | 0.795 |
| original/frac25/L19 | full | 0.959 | 0.530 (0.31) | 0.465 | -0.05 | 0.467 | 0.963 |
| original/frac25/L19 | J_fill | 0.601 | 0.483 (0.88) | 0.472 | -0.34 | 0.500 | 0.573 |
| original/frac25/L19 | pca | 0.758 | 0.508 (0.66) | 0.493 | -0.01 | 0.453 | 0.795 |
| original/frac50/L22 | full | 0.973 | 0.522 (0.14) | 0.490 | -0.01 | 0.510 | 0.988 |
| original/frac50/L22 | J_fill | 0.654 | 0.499 (0.88) | 0.481 | -0.09 | 0.599 | 0.599 |
| original/frac50/L22 | pca | 0.872 | 0.538 (0.16) | 0.571 | 0.11 | 0.505 | 0.899 |
| original/frac50/L21 | full | 0.957 | 0.533 (0.07) | 0.507 | 0.01 | 0.492 | 0.968 |
| original/frac50/L21 | J_fill | 0.635 | 0.507 (0.73) | 0.486 | 0.02 | 0.555 | 0.514 |
| original/frac50/L21 | pca | 0.849 | 0.558 (0.04) | 0.551 | 0.10 | 0.471 | 0.887 |
| original/frac50/L23 | full | 0.955 | 0.522 (0.14) | 0.516 | 0.02 | 0.508 | 0.968 |
| original/frac50/L23 | J_fill | 0.733 | 0.502 (0.98) | 0.459 | -0.21 | 0.525 | 0.634 |
| original/frac50/L23 | pca | 0.888 | 0.526 (0.28) | 0.586 | 0.13 | 0.511 | 0.896 |
| original/frac50/L20 | full | 0.946 | 0.546 (0.04) | 0.513 | 0.00 | 0.523 | 0.961 |
| original/frac50/L20 | J_fill | 0.568 | 0.562 (0.25) | 0.510 | 0.12 | 0.521 | 0.479 |
| original/frac50/L20 | pca | 0.788 | 0.553 (0.06) | 0.543 | 0.11 | 0.515 | 0.840 |
| original/frac50/L19 | full | 0.935 | 0.519 (0.33) | 0.496 | -0.02 | 0.458 | 0.945 |
| original/frac50/L19 | J_fill | 0.571 | 0.539 (0.61) | 0.507 | 0.02 | 0.509 | 0.449 |
| original/frac50/L19 | pca | 0.682 | 0.541 (0.14) | 0.524 | 0.10 | 0.517 | 0.740 |
| original/frac75/L22 | full | 0.982 | 0.537 (0.10) | 0.551 | 0.04 | 0.512 | 0.993 |
| original/frac75/L22 | J_fill | 0.660 | 0.530 (0.39) | 0.566 | 0.27 | 0.541 | 0.742 |
| original/frac75/L22 | pca | 0.876 | 0.559 (0.02) | 0.619 | 0.23 | 0.589 | 0.875 |
| original/frac75/L21 | full | 0.975 | 0.526 (0.25) | 0.574 | 0.06 | 0.533 | 0.984 |
| original/frac75/L21 | J_fill | 0.528 | 0.508 (0.94) | 0.579 | 0.59 | 0.606 | 0.557 |
| original/frac75/L21 | pca | 0.858 | 0.562 (0.01) | 0.603 | 0.23 | 0.562 | 0.865 |
| original/frac75/L23 | full | 0.973 | 0.533 (0.13) | 0.562 | 0.07 | 0.529 | 0.981 |
| original/frac75/L23 | J_fill | 0.713 | 0.546 (0.32) | 0.542 | 0.15 | 0.556 | 0.672 |
| original/frac75/L23 | pca | 0.882 | 0.548 (0.05) | 0.603 | 0.21 | 0.577 | 0.870 |
| original/frac75/L20 | full | 0.967 | 0.545 (0.18) | 0.563 | 0.06 | 0.591 | 0.972 |
| original/frac75/L20 | J_fill | 0.570 | 0.497 (0.92) | 0.581 | 0.53 | 0.555 | 0.570 |
| original/frac75/L20 | pca | 0.818 | 0.588 (0.00) | 0.617 | 0.29 | 0.568 | 0.817 |
| original/frac75/L19 | full | 0.948 | 0.542 (0.18) | 0.559 | 0.07 | 0.538 | 0.942 |
| original/frac75/L19 | J_fill | 0.520 | 0.547 (0.27) | 0.600 | 0.85 | 0.486 | 0.593 |
| original/frac75/L19 | pca | 0.739 | 0.588 (0.01) | 0.606 | 0.33 | 0.547 | 0.721 |
| original/last/L22 | full | 1.000 | 0.694 (0.00) | 0.612 | 0.05 | 0.632 | 1.000 |
| original/last/L22 | J_fill | 0.989 | 0.803 (0.00) | 0.812 | 0.24 | 0.844 | 0.996 |
| original/last/L22 | pca | 0.999 | 0.572 (0.00) | 0.525 | 0.02 | 0.527 | 1.000 |
| original/last/L21 | full | 1.000 | 0.658 (0.00) | 0.601 | 0.05 | 0.621 | 1.000 |
| original/last/L21 | J_fill | 0.992 | 0.630 (0.00) | 0.648 | 0.10 | 0.652 | 0.996 |
| original/last/L21 | pca | 0.999 | 0.561 (0.00) | 0.524 | 0.01 | 0.540 | 1.000 |
| original/last/L23 | full | 1.000 | 0.699 (0.00) | 0.628 | 0.06 | 0.648 | 1.000 |
| original/last/L23 | J_fill | 0.998 | 0.628 (0.00) | 0.659 | 0.09 | 0.619 | 0.999 |
| original/last/L23 | pca | 0.996 | 0.617 (0.00) | 0.580 | 0.05 | 0.571 | 1.000 |
| original/last/L20 | full | 1.000 | 0.634 (0.00) | 0.587 | 0.04 | 0.622 | 1.000 |
| original/last/L20 | J_fill | 0.973 | 0.575 (0.00) | 0.597 | 0.11 | 0.546 | 0.987 |
| original/last/L20 | pca | 1.000 | 0.556 (0.00) | 0.525 | 0.01 | 0.534 | 1.000 |
| original/last/L19 | full | 1.000 | 0.618 (0.00) | 0.577 | 0.04 | 0.586 | 1.000 |
| original/last/L19 | J_fill | 0.923 | 0.510 (0.43) | 0.546 | 0.06 | 0.539 | 0.949 |
| original/last/L19 | pca | 1.000 | 0.545 (0.02) | 0.513 | 0.01 | 0.503 | 1.000 |
| original/prevote/L22 | full | 1.000 | 0.736 (0.00) | 0.741 | 0.13 | 0.738 | 1.000 |
| original/prevote/L22 | J_fill | 0.923 | 0.557 (0.03) | 0.592 | 0.14 | 0.546 | 0.950 |
| original/prevote/L22 | pca | 0.998 | 0.718 (0.00) | 0.745 | 0.17 | 0.700 | 0.999 |
| original/prevote/L21 | full | 1.000 | 0.702 (0.00) | 0.735 | 0.14 | 0.703 | 1.000 |
| original/prevote/L21 | J_fill | 0.888 | 0.613 (0.00) | 0.596 | 0.17 | 0.539 | 0.886 |
| original/prevote/L21 | pca | 0.994 | 0.714 (0.00) | 0.778 | 0.20 | 0.711 | 0.998 |
| original/prevote/L23 | full | 1.000 | 0.741 (0.00) | 0.728 | 0.13 | 0.712 | 1.000 |
| original/prevote/L23 | J_fill | 0.907 | 0.578 (0.00) | 0.602 | 0.15 | 0.581 | 0.952 |
| original/prevote/L23 | pca | 0.997 | 0.733 (0.00) | 0.752 | 0.16 | 0.690 | 1.000 |
| original/prevote/L20 | full | 0.999 | 0.691 (0.00) | 0.723 | 0.16 | 0.668 | 1.000 |
| original/prevote/L20 | J_fill | 0.799 | 0.607 (0.00) | 0.659 | 0.30 | 0.602 | 0.814 |
| original/prevote/L20 | pca | 0.996 | 0.727 (0.00) | 0.796 | 0.21 | 0.758 | 0.997 |
| original/prevote/L19 | full | 0.993 | 0.652 (0.00) | 0.679 | 0.14 | 0.617 | 0.995 |
| original/prevote/L19 | J_fill | 0.759 | 0.573 (0.08) | 0.589 | 0.25 | 0.538 | 0.753 |
| original/prevote/L19 | pca | 0.974 | 0.682 (0.00) | 0.728 | 0.18 | 0.652 | 0.986 |
| original/reply_mean/L22 | full | 1.000 | 0.626 (0.00) | 0.559 | 0.03 | 0.551 | 1.000 |
| original/reply_mean/L22 | J_fill | 0.852 | 0.519 (0.55) | 0.501 | -0.00 | 0.511 | 0.894 |
| original/reply_mean/L22 | pca | 0.984 | 0.626 (0.00) | 0.610 | 0.08 | 0.554 | 0.995 |
| original/reply_mean/L21 | full | 1.000 | 0.630 (0.00) | 0.559 | 0.04 | 0.539 | 1.000 |
| original/reply_mean/L21 | J_fill | 0.790 | 0.523 (0.16) | 0.496 | -0.00 | 0.471 | 0.840 |
| original/reply_mean/L21 | pca | 0.983 | 0.622 (0.00) | 0.601 | 0.08 | 0.556 | 0.995 |
| original/reply_mean/L23 | full | 1.000 | 0.614 (0.00) | 0.551 | 0.04 | 0.533 | 1.000 |
| original/reply_mean/L23 | J_fill | 0.904 | 0.538 (0.13) | 0.483 | -0.06 | 0.461 | 0.933 |
| original/reply_mean/L23 | pca | 0.992 | 0.626 (0.00) | 0.608 | 0.08 | 0.561 | 0.998 |
| original/reply_mean/L20 | full | 1.000 | 0.625 (0.00) | 0.571 | 0.04 | 0.530 | 1.000 |
| original/reply_mean/L20 | J_fill | 0.793 | 0.557 (0.00) | 0.540 | 0.16 | 0.567 | 0.834 |
| original/reply_mean/L20 | pca | 0.960 | 0.614 (0.00) | 0.603 | 0.09 | 0.574 | 0.988 |
| original/reply_mean/L19 | full | 1.000 | 0.620 (0.00) | 0.556 | 0.03 | 0.521 | 1.000 |
| original/reply_mean/L19 | J_fill | 0.804 | 0.561 (0.00) | 0.546 | 0.12 | 0.547 | 0.851 |
| original/reply_mean/L19 | pca | 0.969 | 0.618 (0.00) | 0.607 | 0.10 | 0.588 | 0.989 |
| prompt_only/prompt_end/L22 | full | 1.000 | 0.925 (0.00) | 0.912 | 0.24 | 0.901 | 1.000 |
| prompt_only/prompt_end/L22 | J_fill | 0.973 | 0.886 (0.00) | 0.874 | 0.40 | 0.861 | 0.941 |
| prompt_only/prompt_end/L22 | pca | 1.000 | 0.874 (0.00) | 0.870 | 0.21 | 0.864 | 1.000 |
| prompt_only/prompt_end/L21 | full | 1.000 | 0.876 (0.00) | 0.865 | 0.20 | 0.849 | 1.000 |
| prompt_only/prompt_end/L21 | J_fill | 0.989 | 0.478 (0.87) | 0.501 | -0.01 | 0.608 | 0.954 |
| prompt_only/prompt_end/L21 | pca | 1.000 | 0.832 (0.00) | 0.826 | 0.17 | 0.825 | 1.000 |
| prompt_only/prompt_end/L23 | full | 1.000 | 0.884 (0.00) | 0.871 | 0.23 | 0.853 | 1.000 |
| prompt_only/prompt_end/L23 | J_fill | 0.986 | 0.830 (0.00) | 0.819 | 0.28 | 0.823 | 0.972 |
| prompt_only/prompt_end/L23 | pca | 1.000 | 0.833 (0.00) | 0.828 | 0.19 | 0.831 | 1.000 |
| prompt_only/prompt_end/L20 | full | 1.000 | 0.854 (0.00) | 0.847 | 0.18 | 0.833 | 1.000 |
| prompt_only/prompt_end/L20 | J_fill | 0.951 | 0.880 (0.00) | 0.858 | 0.40 | 0.841 | 0.931 |
| prompt_only/prompt_end/L20 | pca | 1.000 | 0.763 (0.00) | 0.755 | 0.14 | 0.784 | 1.000 |
| prompt_only/prompt_end/L19 | full | 1.000 | 0.801 (0.00) | 0.807 | 0.15 | 0.792 | 1.000 |
| prompt_only/prompt_end/L19 | J_fill | 0.946 | 0.802 (0.00) | 0.811 | 0.34 | 0.785 | 0.929 |
| prompt_only/prompt_end/L19 | pca | 1.000 | 0.776 (0.00) | 0.767 | 0.15 | 0.788 | 1.000 |

## steering

- original L22: scale 33.30 (given (median ||h_J||)), signs role, prefix cache True (check {"rows": 3, "max_abs_diff_y0": 0.0625009536743164}), 400 prefixes, 20 random directions
  baseline: {"colluder": {"n": 100, "y0_mean": 3.54286301612854, "frac_target": 0.73}, "honest": {"n": 300, "y0_mean": -8.631031370162964, "frac_target": 0.04666666666666667}}; random at alpha 4: {"colluder|4.0": {"mean_delta_min": -0.9495815753936767, "mean_delta_max": 0.38129464149475095, "abs_p95": 0.860556393146515}, "honest|4.0": {"mean_delta_min": -0.1119172477722168, "mean_delta_max": 1.9425266710917155, "abs_p95": 1.3321047981580103}}
  - concept:deception_secrecy / colluder (n=100): dy at alpha 0.5/1/2/4 = 0.037/-0.007/-0.119/-0.468; flips at alpha 2/4 = 0.03/0.06; sign ok False, beats random all False, BH any False, dose monotone False, passes all bars False
  - concept:deception_secrecy / honest (n=300): dy at alpha 0.5/1/2/4 = 0.389/0.830/1.868/3.205; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - concept:coordination / colluder (n=100): dy at alpha 0.5/1/2/4 = 0.075/0.115/0.171/0.195; flips at alpha 2/4 = 0.01/0.03; sign ok False, beats random all False, BH any False, dose monotone True, passes all bars False
  - concept:coordination / honest (n=300): dy at alpha 0.5/1/2/4 = 0.345/0.741/1.763/2.495; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - concept:steering_persuasion / colluder (n=100): dy at alpha 0.5/1/2/4 = 0.072/0.099/0.129/0.050; flips at alpha 2/4 = 0.01/0.04; sign ok False, beats random all False, BH any False, dose monotone False, passes all bars False
  - concept:steering_persuasion / honest (n=300): dy at alpha 0.5/1/2/4 = 0.374/0.786/1.817/2.681; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - concept:honesty_standards / colluder (n=100): dy at alpha 0.5/1/2/4 = 0.106/0.187/0.293/0.377; flips at alpha 2/4 = 0.01/0.03; sign ok False, beats random all False, BH any False, dose monotone True, passes all bars False
  - concept:honesty_standards / honest (n=300): dy at alpha 0.5/1/2/4 = 0.345/0.733/1.727/2.413; flips at alpha 2/4 = 0.00/0.01; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - jspace_role_probe / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.066/-0.137/-0.310/-0.665; flips at alpha 2/4 = 0.00/0.01; sign ok True, beats random all False, BH any False, dose monotone True, passes all bars False
  - jspace_role_probe / honest (n=300): dy at alpha 0.5/1/2/4 = 0.053/0.170/0.554/1.902; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all False, BH any False, dose monotone True, passes all bars False
- swap L22: scale 33.88 (given (median ||h_J||)), signs role, prefix cache True (check {"rows": 3, "max_abs_diff_y0": 0.125}), 400 prefixes, 20 random directions
  baseline: {"colluder": {"n": 100, "y0_mean": -3.84608154296875, "frac_target": 0.24}, "honest": {"n": 300, "y0_mean": -8.631031370162964, "frac_target": 0.04666666666666667}}; random at alpha 4: {"colluder|4.0": {"mean_delta_min": -0.6278294944763183, "mean_delta_max": 0.6873270225524902, "abs_p95": 0.630804370880127}, "honest|4.0": {"mean_delta_min": -0.10765312194824218, "mean_delta_max": 1.988994820912679, "abs_p95": 1.3719005265235906}}
  - concept:deception_secrecy / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.173/-0.348/-0.582/-0.879; flips at alpha 2/4 = 0.01/0.03; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - concept:deception_secrecy / honest (n=300): dy at alpha 0.5/1/2/4 = 0.393/0.843/1.911/3.250; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - concept:coordination / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.092/-0.184/-0.287/-0.335; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all False, BH any False, dose monotone True, passes all bars False
  - concept:coordination / honest (n=300): dy at alpha 0.5/1/2/4 = 0.354/0.758/1.793/2.526; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - concept:steering_persuasion / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.122/-0.266/-0.430/-0.613; flips at alpha 2/4 = 0.00/0.02; sign ok True, beats random all False, BH any False, dose monotone True, passes all bars False
  - concept:steering_persuasion / honest (n=300): dy at alpha 0.5/1/2/4 = 0.379/0.805/1.851/2.706; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - concept:honesty_standards / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.092/-0.188/-0.281/-0.256; flips at alpha 2/4 = 0.00/0.02; sign ok True, beats random all False, BH any False, dose monotone False, passes all bars False
  - concept:honesty_standards / honest (n=300): dy at alpha 0.5/1/2/4 = 0.349/0.743/1.747/2.446; flips at alpha 2/4 = 0.00/0.01; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - jspace_role_probe / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.202/-0.414/-0.824/-1.540; flips at alpha 2/4 = 0.03/0.10; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - jspace_role_probe / honest (n=300): dy at alpha 0.5/1/2/4 = 0.238/0.512/1.108/2.379; flips at alpha 2/4 = 0.01/0.01; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
- swap L23: scale 39.48 (given (median ||h_J||)), signs role, prefix cache True (check {"rows": 3, "max_abs_diff_y0": 0.125}), 400 prefixes, 20 random directions
  baseline: {"colluder": {"n": 100, "y0_mean": -3.84608154296875, "frac_target": 0.24}, "honest": {"n": 300, "y0_mean": -8.631031370162964, "frac_target": 0.04666666666666667}}; random at alpha 4: {"colluder|4.0": {"mean_delta_min": -0.41941452980041505, "mean_delta_max": 0.6440356159210205, "abs_p95": 0.4306455841064455}, "honest|4.0": {"mean_delta_min": -0.29547542889912926, "mean_delta_max": 1.8823740323384603, "abs_p95": 1.4158366368611657}}
  - concept:deception_secrecy / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.254/-0.522/-0.982/-1.600; flips at alpha 2/4 = 0.01/0.03; sign ok True, beats random all True, BH any True, dose monotone True, passes all bars True
  - concept:deception_secrecy / honest (n=300): dy at alpha 0.5/1/2/4 = 0.503/1.059/2.330/3.611; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all True, BH any True, dose monotone True, passes all bars True
  - concept:coordination / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.185/-0.365/-0.636/-0.916; flips at alpha 2/4 = 0.00/0.03; sign ok True, beats random all True, BH any True, dose monotone True, passes all bars True
  - concept:coordination / honest (n=300): dy at alpha 0.5/1/2/4 = 0.429/0.919/2.089/2.962; flips at alpha 2/4 = 0.01/0.01; sign ok True, beats random all True, BH any True, dose monotone True, passes all bars True
  - concept:steering_persuasion / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.212/-0.446/-0.818/-1.265; flips at alpha 2/4 = 0.00/0.03; sign ok True, beats random all True, BH any True, dose monotone True, passes all bars True
  - concept:steering_persuasion / honest (n=300): dy at alpha 0.5/1/2/4 = 0.483/1.001/2.196/3.133; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all True, BH any True, dose monotone True, passes all bars True
  - concept:honesty_standards / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.141/-0.245/-0.473/-0.613; flips at alpha 2/4 = 0.00/0.02; sign ok True, beats random all True, BH any True, dose monotone True, passes all bars True
  - concept:honesty_standards / honest (n=300): dy at alpha 0.5/1/2/4 = 0.395/0.840/1.931/2.662; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all True, BH any True, dose monotone True, passes all bars True
  - jspace_role_probe / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.298/-0.628/-1.327/-2.045; flips at alpha 2/4 = 0.06/0.10; sign ok True, beats random all True, BH any True, dose monotone True, passes all bars True
  - jspace_role_probe / honest (n=300): dy at alpha 0.5/1/2/4 = 0.314/0.628/1.147/1.853; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all True, BH any True, dose monotone True, passes all bars True
- swap L23 (lens null): scale 39.48 (given (median ||h_J||)), signs role, prefix cache True (check {"rows": 3, "max_abs_diff_y0": 0.125}), 400 prefixes, 20 random directions
  baseline: {"colluder": {"n": 100, "y0_mean": -3.84608154296875, "frac_target": 0.24}, "honest": {"n": 300, "y0_mean": -8.631031370162964, "frac_target": 0.04666666666666667}}; random at alpha 4: {"colluder|4.0": {"mean_delta_min": -1.2295494079589844, "mean_delta_max": -0.8044942378997803, "abs_p95": 1.1829898352622985}, "honest|4.0": {"mean_delta_min": 2.180951566696167, "mean_delta_max": 2.9485335032145183, "abs_p95": 2.870173971494039}}
  - concept:deception_secrecy / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.254/-0.522/-0.982/-1.600; flips at alpha 2/4 = 0.01/0.03; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - concept:deception_secrecy / honest (n=300): dy at alpha 0.5/1/2/4 = 0.503/1.059/2.330/3.611; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - concept:coordination / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.185/-0.365/-0.636/-0.916; flips at alpha 2/4 = 0.00/0.03; sign ok True, beats random all False, BH any False, dose monotone True, passes all bars False
  - concept:coordination / honest (n=300): dy at alpha 0.5/1/2/4 = 0.429/0.919/2.089/2.962; flips at alpha 2/4 = 0.01/0.01; sign ok True, beats random all False, BH any False, dose monotone True, passes all bars False
  - concept:steering_persuasion / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.212/-0.446/-0.818/-1.265; flips at alpha 2/4 = 0.00/0.03; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - concept:steering_persuasion / honest (n=300): dy at alpha 0.5/1/2/4 = 0.483/1.001/2.196/3.133; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all True, BH any False, dose monotone True, passes all bars False
  - concept:honesty_standards / colluder (n=100): dy at alpha 0.5/1/2/4 = -0.141/-0.245/-0.473/-0.613; flips at alpha 2/4 = 0.00/0.02; sign ok True, beats random all False, BH any False, dose monotone True, passes all bars False
  - concept:honesty_standards / honest (n=300): dy at alpha 0.5/1/2/4 = 0.395/0.840/1.931/2.662; flips at alpha 2/4 = 0.00/0.00; sign ok True, beats random all False, BH any False, dose monotone True, passes all bars False

## not run

- condition/position pairs with missing cells: none
- steering runs missing: original:L23
