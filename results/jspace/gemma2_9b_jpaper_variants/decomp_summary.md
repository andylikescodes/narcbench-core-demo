# J-space decomposition summary (paper-faithful split)

Written 2026-11-05 10:11 UTC by `scripts/jspace_decompose_summary.py`.

## core_k25 — 1700 activations, k = 25, method nnomp, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/core/20261001T012639Z/variants/jpaper_nnomp_k25`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0098 | 0.0097 | 0.0089–0.0105 | 1.00 | 0.0103 | 16.4 | 1.00 | 8.6 | 0.14 |
| 20 | 0.0097 | 0.0096 | 0.0088–0.0106 | 1.00 | 0.0100 | 17.3 | 0.99 | 7.7 | 0.27 |
| 21 | 0.0104 | 0.0103 | 0.0092–0.0116 | 1.00 | 0.0108 | 16.6 | 0.99 | 8.4 | 0.53 |
| 22 | 0.0108 | 0.0109 | 0.0098–0.0120 | 1.00 | 0.0111 | 17.0 | 0.98 | 8.0 | 0.39 |
| 23 | 0.0128 | 0.0128 | 0.0116–0.0141 | 1.00 | 0.0132 | 17.7 | 0.97 | 7.3 | 0.47 |

### core_k25 layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0094 | 0.0095 |
| collusion / discussion / honest | 300 | 0.0095 | 0.0095 |
| collusion / final / colluder | 100 | 0.0100 | 0.0098 |
| collusion / final / honest | 100 | 0.0095 | 0.0093 |
| collusion / private / colluder | 100 | 0.0098 | 0.0093 |
| control / discussion / honest | 600 | 0.0101 | 0.0099 |
| control / final / honest | 200 | 0.0098 | 0.0097 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.5), `' decision'` 0.93 (0.9), `' acknowledging'` 0.89 (0.2), `' understandably'` 0.87 (0.8), `' eloquently'` 0.84 (1.9), `' spokesperson'` 0.79 (0.5), `' prioritizing'` 0.59 (0.6), `' reaffirm'` 0.58 (0.4), `' confidently'` 0.54 (0.1), `' confirming'` 0.54 (0.3), `' confirmed'` 0.48 (0.5), `' undeniably'` 0.45 (0.5), `' statement'` 0.45 (0.0), `'?'` 0.44 (0.5), `' bestätigt'` 0.42 (0.4), `' confirms'` 0.41 (0.2), `' hatır'` 0.39 (0.4), `' acknowledges'` 0.39 (0.0), `'.\u202f'` 0.35 (0.9), `' hopeful'` 0.34 (0.5)

First atom chosen: `'<eos>'` 0.36, `' eloquently'` 0.23, `' decision'` 0.22, `' acknowledging'` 0.06, `' confidently'` 0.02, `' confirming'` 0.02, `' spokesperson'` 0.02, `' questioning'` 0.01

Readout top-1 token: `'<eos>'` 1.00

### core_k25 layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0085 | 0.0086 |
| collusion / discussion / honest | 300 | 0.0095 | 0.0094 |
| collusion / final / colluder | 100 | 0.0095 | 0.0093 |
| collusion / final / honest | 100 | 0.0102 | 0.0101 |
| collusion / private / colluder | 100 | 0.0097 | 0.0092 |
| control / discussion / honest | 600 | 0.0101 | 0.0099 |
| control / final / honest | 200 | 0.0106 | 0.0105 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.3), `' eloquently'` 0.81 (1.6), `' decision'` 0.80 (0.9), `' prioritizing'` 0.76 (0.9), `' spokesperson'` 0.71 (0.6), `' understandably'` 0.70 (0.7), `' empathetic'` 0.67 (0.9), `' acknowledging'` 0.62 (0.0), `' reaffirm'` 0.60 (0.7), `' bestätigt'` 0.53 (0.6), `' confirms'` 0.49 (0.5), `' reaffirmed'` 0.48 (0.3), `' thankfully'` 0.46 (0.5), `'.\u202f'` 0.42 (1.1), `' confidently'` 0.41 (0.2), `' تاکید'` 0.40 (0.4), `' recommendation'` 0.38 (0.2), `' responding'` 0.36 (0.1), `' confirmed'` 0.35 (0.4), `' impactful'` 0.33 (0.3)

First atom chosen: `' decision'` 0.25, `' eloquently'` 0.17, `' reaffirm'` 0.10, `' empathetic'` 0.08, `' confidently'` 0.05, `' spokesperson'` 0.04, `'<eos>'` 0.03, `' prioritizing'` 0.03

Readout top-1 token: `'<eos>'` 1.00, `'.”'` 0.00, `' discussions'` 0.00, `'\n\n'` 0.00, `' eloquently'` 0.00, `'?”'` 0.00, `').”'` 0.00

### core_k25 layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0088 | 0.0087 |
| collusion / discussion / honest | 300 | 0.0099 | 0.0099 |
| collusion / final / colluder | 100 | 0.0111 | 0.0109 |
| collusion / final / honest | 100 | 0.0122 | 0.0121 |
| collusion / private / colluder | 100 | 0.0101 | 0.0097 |
| control / discussion / honest | 600 | 0.0104 | 0.0103 |
| control / final / honest | 200 | 0.0124 | 0.0122 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.90 (1.4), `'<eos>'` 0.86 (1.0), `' prioritizing'` 0.82 (1.3), `' recommendation'` 0.76 (0.5), `' recommendations'` 0.73 (0.5), `' eloquently'` 0.71 (1.3), `' تاکید'` 0.57 (0.6), `' diría'` 0.55 (0.4), `' spokesperson'` 0.54 (0.6), `' reaffirm'` 0.50 (0.8), `' recommending'` 0.48 (0.1), `' responding'` 0.44 (0.1), `' consensus'` 0.44 (0.4), `' citing'` 0.42 (0.1), `' pungkasnya'` 0.41 (0.5), `' confirmed'` 0.40 (0.5), `' acknowledging'` 0.38 (0.0), `' reaffirmed'` 0.36 (0.1), `' advocating'` 0.36 (0.3), `' empathetic'` 0.35 (0.6)

First atom chosen: `' decision'` 0.36, `' recommendation'` 0.11, `' prioritizing'` 0.06, `' reaffirm'` 0.05, `' recommendations'` 0.04, `' eloquently'` 0.03, `' advocating'` 0.02, `' convincing'` 0.02

Readout top-1 token: `'<eos>'` 0.98, `' discussions'` 0.00, `' unethical'` 0.00, `' decision'` 0.00, `' despite'` 0.00, `'<end_of_turn>'` 0.00, `' suggestions'` 0.00, `' subtly'` 0.00

### core_k25 layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0090 | 0.0089 |
| collusion / discussion / honest | 300 | 0.0109 | 0.0109 |
| collusion / final / colluder | 100 | 0.0095 | 0.0095 |
| collusion / final / honest | 100 | 0.0119 | 0.0118 |
| collusion / private / colluder | 100 | 0.0108 | 0.0106 |
| control / discussion / honest | 600 | 0.0113 | 0.0112 |
| control / final / honest | 200 | 0.0120 | 0.0118 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.90 (1.4), `' recommendation'` 0.78 (0.6), `' prioritizing'` 0.77 (1.2), `' diría'` 0.74 (0.5), `' recommendations'` 0.72 (0.4), `' تاکید'` 0.68 (1.0), `'<eos>'` 0.68 (0.8), `' eloquently'` 0.63 (1.1), `' recommending'` 0.55 (0.0), `' thankfully'` 0.49 (0.5), `' spokesperson'` 0.45 (0.7), `' decisions'` 0.44 (0.0), `' advocating'` 0.36 (0.5), `' despite'` 0.35 (0.9), `' reaffirm'` 0.34 (0.6), `' discussions'` 0.33 (0.8), `' likely'` 0.33 (0.5), `' leveraging'` 0.33 (0.4), `' consensus'` 0.33 (0.2), `' teşekkür'` 0.31 (0.4)

First atom chosen: `' decision'` 0.38, `' recommendation'` 0.19, `' prioritizing'` 0.04, `' recommendations'` 0.03, `' تاکید'` 0.03, `' advocating'` 0.03, `' convincing'` 0.03, `' unanimously'` 0.02

Readout top-1 token: `'<eos>'` 0.92, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' discussions'` 0.01, `' despite'` 0.01, `' decision'` 0.00, `' leveraging'` 0.00, `' strategically'` 0.00

### core_k25 layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0117 | 0.0116 |
| collusion / discussion / honest | 300 | 0.0125 | 0.0124 |
| collusion / final / colluder | 100 | 0.0124 | 0.0120 |
| collusion / final / honest | 100 | 0.0134 | 0.0134 |
| collusion / private / colluder | 100 | 0.0125 | 0.0127 |
| control / discussion / honest | 600 | 0.0133 | 0.0132 |
| control / final / honest | 200 | 0.0138 | 0.0136 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.83 (1.5), `' prioritizing'` 0.81 (1.4), `' diría'` 0.79 (0.7), `'<eos>'` 0.76 (0.9), `' recommendations'` 0.61 (0.5), `' recommendation'` 0.58 (0.3), `' confirmado'` 0.53 (0.8), `' ethically'` 0.46 (1.2), `' spokesperson'` 0.46 (0.9), `' leveraging'` 0.45 (0.7), `' consensus'` 0.44 (0.5), `' discussions'` 0.39 (0.9), `' decisions'` 0.39 (0.0), `' eloquently'` 0.39 (0.8), `' dikutip'` 0.37 (0.4), `' advocating'` 0.37 (0.8), `' impactful'` 0.36 (0.5), `' emphasizing'` 0.34 (0.5), `' unanimously'` 0.33 (1.3), `' advocacy'` 0.32 (0.2)

First atom chosen: `' decision'` 0.28, `' prioritizing'` 0.08, `' consensus'` 0.07, `' recommendation'` 0.06, `' unanimously'` 0.06, `' advocating'` 0.05, `' recommendations'` 0.04, `' spokesperson'` 0.04

Readout top-1 token: `'<eos>'` 0.89, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' decision'` 0.01, `' recommendations'` 0.01, `' discussions'` 0.01, `' hyö'` 0.00, `' ethically'` 0.00

## transfer_k25 — 936 activations, k = 25, method nnomp, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/transfer/RUNPOD/variants/jpaper_nnomp_k25`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0124 | 0.0123 | 0.0101–0.0145 | 1.00 | 0.0127 | 18.5 | 0.95 | 6.5 | 0.18 |
| 20 | 0.0125 | 0.0127 | 0.0103–0.0148 | 1.00 | 0.0126 | 19.4 | 0.92 | 5.6 | 0.20 |
| 21 | 0.0132 | 0.0134 | 0.0104–0.0159 | 1.00 | 0.0127 | 19.1 | 0.90 | 5.9 | 0.25 |
| 22 | 0.0134 | 0.0136 | 0.0111–0.0158 | 1.00 | 0.0126 | 19.8 | 0.89 | 5.2 | 0.16 |
| 23 | 0.0148 | 0.0150 | 0.0126–0.0170 | 1.00 | 0.0138 | 21.0 | 0.77 | 4.0 | 0.18 |

### transfer_k25 layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0117 | 0.0118 |
| collusion / discussion / honest | 144 | 0.0115 | 0.0116 |
| collusion / final / colluder | 72 | 0.0142 | 0.0144 |
| collusion / final / honest | 72 | 0.0141 | 0.0137 |
| collusion / private / colluder | 72 | 0.0095 | 0.0090 |
| control / discussion / honest | 288 | 0.0115 | 0.0116 |
| control / final / honest | 144 | 0.0153 | 0.0151 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.96 (1.5), `' understandably'` 0.70 (1.0), `'?'` 0.67 (1.0), `':'` 0.51 (0.8), `' thankfully'` 0.48 (0.8), `' eloquently'` 0.48 (1.3), `'\n\n\n'` 0.43 (0.3), `' pungkasnya'` 0.40 (0.5), `'  '` 0.39 (0.4), `'\n\n\n\n'` 0.35 (0.2), `' discussions'` 0.33 (0.6), `'https'` 0.33 (0.3), `"'"` 0.32 (0.6), `'.'` 0.31 (0.2), `','` 0.28 (0.6), `' insightful'` 0.28 (0.9), `' Webinar'` 0.26 (0.5), `' Interestingly'` 0.25 (0.4), `' showcasing'` 0.25 (0.5), `'Continue'` 0.23 (0.0)

First atom chosen: `'<eos>'` 0.53, `'?'` 0.11, `' insightful'` 0.04, `'.'` 0.03, `' understandably'` 0.03, `' continued'` 0.02, `' comments'` 0.02, `':'` 0.02

Readout top-1 token: `'<eos>'` 0.95, `'...”'` 0.01, `'\n\n\n'` 0.00, `' […]'` 0.00, `'\n\n'` 0.00, `'.”'` 0.00, `'?”'` 0.00, `'...'` 0.00

### transfer_k25 layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0122 | 0.0120 |
| collusion / discussion / honest | 144 | 0.0120 | 0.0124 |
| collusion / final / colluder | 72 | 0.0133 | 0.0134 |
| collusion / final / honest | 72 | 0.0140 | 0.0135 |
| collusion / private / colluder | 72 | 0.0096 | 0.0091 |
| control / discussion / honest | 288 | 0.0120 | 0.0118 |
| control / final / honest | 144 | 0.0149 | 0.0147 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.94 (1.2), `'?'` 0.67 (1.0), `':'` 0.52 (0.8), `' thankfully'` 0.46 (0.9), `' discussions'` 0.46 (0.7), `' understandably'` 0.44 (0.8), `'\n\n\n'` 0.44 (0.4), `' pungkasnya'` 0.42 (0.6), `' eloquently'` 0.41 (1.0), `'\n\n\n\n'` 0.35 (0.3), `'如果您'` 0.34 (0.5), `' insightful'` 0.34 (1.2), `' recommendations'` 0.33 (0.7), `' storytelling'` 0.33 (0.7), `'  '` 0.33 (0.5), `' continued'` 0.27 (0.9), `' Interestingly'` 0.25 (0.6), `' concludes'` 0.24 (0.2), `'https'` 0.23 (0.3), `' spokesperson'` 0.23 (0.4)

First atom chosen: `'<eos>'` 0.18, `' insightful'` 0.14, `'?'` 0.07, `' recommendations'` 0.05, `' concludes'` 0.04, `' continued'` 0.03, `' understandably'` 0.03, `' comments'` 0.03

Readout top-1 token: `'<eos>'` 0.90, `'\n\n\n'` 0.03, `'<end_of_turn>'` 0.01, `'?”'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01, `'.\\\\'` 0.00, `'.”'` 0.00

### transfer_k25 layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0128 | 0.0129 |
| collusion / discussion / honest | 144 | 0.0123 | 0.0120 |
| collusion / final / colluder | 72 | 0.0145 | 0.0143 |
| collusion / final / honest | 72 | 0.0153 | 0.0148 |
| collusion / private / colluder | 72 | 0.0096 | 0.0095 |
| control / discussion / honest | 288 | 0.0123 | 0.0121 |
| control / final / honest | 144 | 0.0163 | 0.0161 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'?'` 0.72 (1.2), `'<eos>'` 0.64 (0.9), `' pungkasnya'` 0.59 (0.8), `':'` 0.47 (0.7), `' discussions'` 0.45 (0.8), `' recommendations'` 0.41 (1.0), `' thankfully'` 0.39 (0.8), `').'` 0.39 (0.9), `' comments'` 0.37 (0.3), `'  '` 0.36 (0.8), `'https'` 0.35 (0.5), `' insightful'` 0.34 (1.0), `' continued'` 0.32 (1.2), `' eloquently'` 0.32 (1.0), `' recommendation'` 0.26 (0.1), `' concludes'` 0.24 (0.3), `' further'` 0.24 (0.5), `' pungkas'` 0.24 (0.4), `' showcasing'` 0.23 (0.6), `' spokesperson'` 0.22 (0.4)

First atom chosen: `' recommendations'` 0.14, `'?'` 0.10, `' insightful'` 0.08, `' continued'` 0.08, `' concludes'` 0.07, `' comments'` 0.05, `'https'` 0.02, `' detaylı'` 0.02

Readout top-1 token: `'<eos>'` 0.70, `'<end_of_turn>'` 0.12, `'\n\n\n'` 0.04, `'?”'` 0.02, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01

### transfer_k25 layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0127 | 0.0126 |
| collusion / discussion / honest | 144 | 0.0130 | 0.0132 |
| collusion / final / colluder | 72 | 0.0143 | 0.0142 |
| collusion / final / honest | 72 | 0.0150 | 0.0140 |
| collusion / private / colluder | 72 | 0.0101 | 0.0096 |
| control / discussion / honest | 288 | 0.0132 | 0.0132 |
| control / final / honest | 144 | 0.0154 | 0.0150 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.66 (0.9), `' discussions'` 0.62 (0.8), `'?'` 0.55 (1.2), `' pungkasnya'` 0.48 (0.6), `'https'` 0.43 (0.5), `' thankfully'` 0.41 (1.1), `' continued'` 0.41 (1.1), `' recommendations'` 0.37 (0.8), `' detaylı'` 0.33 (0.8), `' comments'` 0.31 (0.2), `'\n\n\n'` 0.29 (0.5), `' further'` 0.28 (0.7), `' eloquently'` 0.28 (0.8), `' would'` 0.26 (0.5), `' insightful'` 0.25 (0.8), `' storytelling'` 0.23 (0.6), `'Interestingly'` 0.22 (0.3), `' concludes'` 0.21 (0.5), `' recommendation'` 0.19 (0.1), `' FINALLY'` 0.19 (0.4)

First atom chosen: `' recommendations'` 0.09, `' continued'` 0.07, `' concludes'` 0.07, `' comments'` 0.06, `' insightful'` 0.05, `'<eos>'` 0.05, `' discussions'` 0.04, `' detaylı'` 0.04

Readout top-1 token: `'<eos>'` 0.55, `'<end_of_turn>'` 0.31, `'\n\n\n'` 0.03, `'....'` 0.01, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.00, `'...'` 0.00

### transfer_k25 layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0146 | 0.0145 |
| collusion / discussion / honest | 144 | 0.0148 | 0.0150 |
| collusion / final / colluder | 72 | 0.0150 | 0.0152 |
| collusion / final / honest | 72 | 0.0157 | 0.0152 |
| collusion / private / colluder | 72 | 0.0119 | 0.0119 |
| control / discussion / honest | 288 | 0.0148 | 0.0146 |
| control / final / honest | 144 | 0.0159 | 0.0157 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' discussions'` 0.72 (0.9), `'<eos>'` 0.66 (0.9), `'?'` 0.51 (1.1), `'\n\n\n'` 0.49 (0.7), `' continued'` 0.47 (1.3), `' insightful'` 0.46 (0.9), `' recommendations'` 0.42 (1.0), `' pungkasnya'` 0.38 (0.5), `' further'` 0.35 (0.6), `' detaylı'` 0.35 (0.7), `' thankfully'` 0.33 (0.9), `' diría'` 0.30 (0.6), `' eloquently'` 0.29 (0.7), `'https'` 0.26 (0.5), `' ausführ'` 0.25 (0.3), `' policymakers'` 0.24 (0.4), `' hopefully'` 0.24 (0.5), `' LGBTQ'` 0.23 (0.6), `' liknande'` 0.21 (0.4), `'articolo'` 0.20 (0.3)

First atom chosen: `' continued'` 0.14, `' recommendations'` 0.11, `' insightful'` 0.08, `' detaylı'` 0.07, `' discussions'` 0.07, `' concludes'` 0.05, `'<eos>'` 0.03, `' tentunya'` 0.03

Readout top-1 token: `'<eos>'` 0.48, `'<end_of_turn>'` 0.36, `'\n\n\n'` 0.09, `'\n\n\n\n'` 0.01, `'...”'` 0.01, `'\n\n'` 0.00, `' recommendations'` 0.00, `' comments'` 0.00

## core_raw — 1700 activations, k = 25, method nnomp, select raw, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/core/20261001T012639Z/variants/jpaper_nnomp_k25_raw`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0120 | 0.0107 | 0.0095–0.0150 | 1.00 | 0.0118 | 19.4 | 0.96 | 5.6 | 0.40 |
| 20 | 0.0134 | 0.0144 | 0.0099–0.0160 | 1.00 | 0.0127 | 20.1 | 0.91 | 4.9 | 0.47 |
| 21 | 0.0136 | 0.0134 | 0.0097–0.0173 | 1.00 | 0.0131 | 18.7 | 0.95 | 6.3 | 0.64 |
| 22 | 0.0159 | 0.0172 | 0.0123–0.0190 | 1.00 | 0.0146 | 18.3 | 0.96 | 6.7 | 0.54 |
| 23 | 0.0179 | 0.0196 | 0.0127–0.0220 | 1.00 | 0.0166 | 18.8 | 0.95 | 6.2 | 0.58 |

### core_raw layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0112 | 0.0101 |
| collusion / discussion / honest | 300 | 0.0118 | 0.0105 |
| collusion / final / colluder | 100 | 0.0133 | 0.0139 |
| collusion / final / honest | 100 | 0.0127 | 0.0135 |
| collusion / private / colluder | 100 | 0.0101 | 0.0096 |
| control / discussion / honest | 600 | 0.0123 | 0.0108 |
| control / final / honest | 200 | 0.0129 | 0.0128 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.6), `'<em>'` 0.99 (0.5), `'.\u202f'` 0.96 (0.4), `'\n\n\n'` 0.93 (0.0), `'.**'` 0.90 (0.3), `'\n\n\n\n'` 0.87 (0.0), `' […]'` 0.83 (0.0), `' decision'` 0.81 (1.0), `' eloquently'` 0.79 (1.9), `' understandably'` 0.77 (0.8), `'.’”'` 0.73 (0.2), `'.”'` 0.68 (0.1), `' spokesperson'` 0.68 (0.6), `' acknowledging'` 0.56 (0.6), `' prioritizing'` 0.53 (0.6), `' reaffirm'` 0.48 (0.5), `','` 0.45 (0.2), `' undeniably'` 0.44 (0.7), `'\n\n\n\n\n\n\n'` 0.40 (0.0), `'\n\n\n\n\n'` 0.38 (0.0)

First atom chosen: `'<eos>'` 1.00

Readout top-1 token: `'<eos>'` 1.00

### core_raw layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0108 | 0.0095 |
| collusion / discussion / honest | 300 | 0.0130 | 0.0140 |
| collusion / final / colluder | 100 | 0.0133 | 0.0134 |
| collusion / final / honest | 100 | 0.0151 | 0.0152 |
| collusion / private / colluder | 100 | 0.0107 | 0.0096 |
| control / discussion / honest | 600 | 0.0143 | 0.0150 |
| control / final / honest | 200 | 0.0159 | 0.0161 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.4), `'.\u202f'` 0.97 (0.6), `'<em>'` 0.92 (0.5), `'.**'` 0.88 (0.4), `' eloquently'` 0.78 (1.6), `' prioritizing'` 0.73 (0.9), `'.’”'` 0.73 (0.2), `'\n\n\n\n'` 0.70 (0.0), `' decision'` 0.69 (1.0), `'<end_of_turn>'` 0.64 (0.0), `'\n\n\n'` 0.64 (0.0), `' empathetic'` 0.62 (0.9), `' understandably'` 0.61 (0.8), `' spokesperson'` 0.61 (0.6), `'.”'` 0.60 (0.1), `' reaffirm'` 0.51 (1.0), `' thankfully'` 0.47 (0.5), `' impactful'` 0.45 (0.2), `' confirms'` 0.42 (0.6), `' bestätigt'` 0.40 (0.4)

First atom chosen: `'<eos>'` 1.00, `'.”'` 0.00, `' discussions'` 0.00, `'\n\n'` 0.00, `' eloquently'` 0.00, `'?”'` 0.00, `').”'` 0.00

Readout top-1 token: `'<eos>'` 1.00, `'.”'` 0.00, `' discussions'` 0.00, `'\n\n'` 0.00, `' eloquently'` 0.00, `'?”'` 0.00, `').”'` 0.00

### core_raw layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0104 | 0.0091 |
| collusion / discussion / honest | 300 | 0.0122 | 0.0106 |
| collusion / final / colluder | 100 | 0.0138 | 0.0121 |
| collusion / final / honest | 100 | 0.0172 | 0.0178 |
| collusion / private / colluder | 100 | 0.0109 | 0.0099 |
| control / discussion / honest | 600 | 0.0141 | 0.0155 |
| control / final / honest | 200 | 0.0181 | 0.0186 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.99 (0.9), `'<em>'` 0.97 (0.6), `'.\u202f'` 0.94 (0.5), `' decision'` 0.83 (1.3), `' recommendations'` 0.80 (0.4), `' prioritizing'` 0.79 (1.2), `' recommendation'` 0.78 (0.6), `'.**'` 0.68 (0.2), `' eloquently'` 0.68 (1.3), `' impactful'` 0.61 (0.1), `'<end_of_turn>'` 0.49 (0.0), `' reaffirm'` 0.47 (1.0), `' spokesperson'` 0.45 (0.6), `' تاکید'` 0.45 (0.5), `' recommending'` 0.38 (0.0), `' pungkasnya'` 0.38 (0.4), `' leveraging'` 0.36 (0.1), `' comments'` 0.35 (0.2), `' despite'` 0.34 (1.1), `' diría'` 0.34 (0.3)

First atom chosen: `'<eos>'` 0.98, `' discussions'` 0.00, `' unethical'` 0.00, `' decision'` 0.00, `' despite'` 0.00, `'<end_of_turn>'` 0.00, `' suggestions'` 0.00, `' subtly'` 0.00

Readout top-1 token: `'<eos>'` 0.98, `' discussions'` 0.00, `' unethical'` 0.00, `' decision'` 0.00, `' despite'` 0.00, `'<end_of_turn>'` 0.00, `' suggestions'` 0.00, `' subtly'` 0.00

### core_raw layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0118 | 0.0109 |
| collusion / discussion / honest | 300 | 0.0162 | 0.0174 |
| collusion / final / colluder | 100 | 0.0127 | 0.0103 |
| collusion / final / honest | 100 | 0.0190 | 0.0193 |
| collusion / private / colluder | 100 | 0.0122 | 0.0107 |
| control / discussion / honest | 600 | 0.0173 | 0.0179 |
| control / final / honest | 200 | 0.0188 | 0.0193 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.99 (0.7), `' decision'` 0.84 (1.5), `'.\u202f'` 0.83 (0.5), `' prioritizing'` 0.73 (1.1), `' recommendation'` 0.73 (0.7), `'<end_of_turn>'` 0.73 (0.0), `' recommendations'` 0.72 (0.3), `' تاکید'` 0.66 (0.9), `' eloquently'` 0.61 (1.1), `' RECOMMENDATIONS'` 0.61 (0.0), `' diría'` 0.60 (0.4), `' thankfully'` 0.47 (0.5), `' spokesperson'` 0.39 (0.6), `' recommending'` 0.37 (0.1), `' decisão'` 0.36 (0.1), `' leveraging'` 0.36 (0.4), `' despite'` 0.36 (0.9), `' discussions'` 0.33 (0.8), `' likely'` 0.32 (0.6), `' Recommendations'` 0.31 (0.0)

First atom chosen: `'<eos>'` 0.92, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' discussions'` 0.01, `' despite'` 0.01, `' decision'` 0.00, `' leveraging'` 0.00, `' strategically'` 0.00

Readout top-1 token: `'<eos>'` 0.92, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' discussions'` 0.01, `' despite'` 0.01, `' decision'` 0.00, `' leveraging'` 0.00, `' strategically'` 0.00

### core_raw layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0160 | 0.0178 |
| collusion / discussion / honest | 300 | 0.0160 | 0.0138 |
| collusion / final / colluder | 100 | 0.0173 | 0.0181 |
| collusion / final / honest | 100 | 0.0198 | 0.0211 |
| collusion / private / colluder | 100 | 0.0145 | 0.0132 |
| control / discussion / honest | 600 | 0.0191 | 0.0202 |
| control / final / honest | 200 | 0.0212 | 0.0221 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.99 (0.8), `'.\u202f'` 0.92 (0.6), `' decision'` 0.79 (1.6), `' prioritizing'` 0.78 (1.3), `' recommendations'` 0.69 (0.4), `' diría'` 0.68 (0.6), `'<end_of_turn>'` 0.62 (0.0), `' recommendation'` 0.59 (0.4), `' confirmado'` 0.55 (0.8), `' decisão'` 0.45 (0.1), `' ethically'` 0.45 (1.2), `' leveraging'` 0.45 (0.6), `' spokesperson'` 0.42 (0.9), `' discussions'` 0.40 (0.9), `' eloquently'` 0.36 (0.8), `' consensus'` 0.31 (1.0), `' unanimously'` 0.31 (1.3), `' تاکید'` 0.30 (0.6), `' convincingly'` 0.30 (0.8), `' emphasizing'` 0.29 (0.6)

First atom chosen: `'<eos>'` 0.89, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' decision'` 0.01, `' recommendations'` 0.01, `' discussions'` 0.01, `' hyö'` 0.00, `' ethically'` 0.00

Readout top-1 token: `'<eos>'` 0.89, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' decision'` 0.01, `' recommendations'` 0.01, `' discussions'` 0.01, `' hyö'` 0.00, `' ethically'` 0.00

## transfer_raw — 936 activations, k = 25, method nnomp, select raw, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/transfer/RUNPOD/variants/jpaper_nnomp_k25_raw`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0149 | 0.0133 | 0.0105–0.0191 | 1.00 | 0.0140 | 21.0 | 0.81 | 4.0 | 0.32 |
| 20 | 0.0159 | 0.0148 | 0.0108–0.0206 | 1.00 | 0.0146 | 21.7 | 0.72 | 3.3 | 0.32 |
| 21 | 0.0177 | 0.0159 | 0.0110–0.0249 | 1.00 | 0.0155 | 20.8 | 0.77 | 4.2 | 0.40 |
| 22 | 0.0199 | 0.0215 | 0.0134–0.0254 | 1.00 | 0.0162 | 20.1 | 0.86 | 4.9 | 0.35 |
| 23 | 0.0209 | 0.0217 | 0.0149–0.0264 | 1.00 | 0.0174 | 21.7 | 0.71 | 3.3 | 0.30 |

### transfer_raw layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0125 | 0.0121 |
| collusion / discussion / honest | 144 | 0.0123 | 0.0120 |
| collusion / final / colluder | 72 | 0.0203 | 0.0208 |
| collusion / final / honest | 72 | 0.0199 | 0.0194 |
| collusion / private / colluder | 72 | 0.0102 | 0.0092 |
| control / discussion / honest | 288 | 0.0124 | 0.0120 |
| control / final / honest | 144 | 0.0220 | 0.0222 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.99 (1.5), `'<em>'` 0.85 (0.7), `'\n\n\n\n'` 0.79 (0.1), `'\n\n\n'` 0.79 (0.1), `'s'` 0.65 (0.3), `'?'` 0.55 (1.0), `'\n\n\n\n\n'` 0.40 (0.0), `' thankfully'` 0.40 (1.0), `':'` 0.40 (0.7), `' pungkasnya'` 0.39 (0.4), `' understandably'` 0.38 (1.1), `'.\u202f'` 0.37 (0.3), `'.’'` 0.37 (0.3), `'’'` 0.37 (0.3), `','` 0.37 (0.5), `' eloquently'` 0.32 (1.6), `'️'` 0.32 (0.2), `'<end_of_turn>'` 0.30 (0.0), `'\n\n\n\n\n\n\n'` 0.30 (0.0), `' Interestingly'` 0.30 (0.6)

First atom chosen: `'<eos>'` 0.95, `'...”'` 0.01, `'\n\n\n'` 0.00, `' […]'` 0.00, `'\n\n'` 0.00, `'.”'` 0.00, `'?”'` 0.00, `'...'` 0.00

Readout top-1 token: `'<eos>'` 0.95, `'...”'` 0.01, `'\n\n\n'` 0.00, `' […]'` 0.00, `'\n\n'` 0.00, `'.”'` 0.00, `'?”'` 0.00, `'...'` 0.00

### transfer_raw layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0144 | 0.0128 |
| collusion / discussion / honest | 144 | 0.0143 | 0.0133 |
| collusion / final / colluder | 72 | 0.0190 | 0.0190 |
| collusion / final / honest | 72 | 0.0199 | 0.0198 |
| collusion / private / colluder | 72 | 0.0111 | 0.0096 |
| control / discussion / honest | 288 | 0.0141 | 0.0124 |
| control / final / honest | 144 | 0.0215 | 0.0211 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.98 (1.2), `'<em>'` 0.94 (0.8), `'\n\n\n\n'` 0.69 (0.2), `'\n\n\n'` 0.65 (0.2), `'s'` 0.59 (0.4), `'?'` 0.54 (0.9), `'.\u202f'` 0.44 (0.4), `' thankfully'` 0.41 (1.0), `' discussions'` 0.40 (0.7), `':'` 0.40 (0.7), `'<end_of_turn>'` 0.39 (0.0), `'.’'` 0.39 (0.3), `' pungkasnya'` 0.36 (0.6), `'如果您'` 0.34 (0.7), `' Interestingly'` 0.32 (0.7), `'<u>'` 0.32 (0.6), `' recommendations'` 0.30 (0.8), `' eloquently'` 0.29 (1.1), `' insightful'` 0.27 (1.4), `' understandably'` 0.25 (0.9)

First atom chosen: `'<eos>'` 0.90, `'\n\n\n'` 0.03, `'<end_of_turn>'` 0.01, `'?”'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01, `'.\\\\'` 0.00, `'.”'` 0.00

Readout top-1 token: `'<eos>'` 0.90, `'\n\n\n'` 0.03, `'<end_of_turn>'` 0.01, `'?”'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01, `'.\\\\'` 0.00, `'.”'` 0.00

### transfer_raw layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0157 | 0.0138 |
| collusion / discussion / honest | 144 | 0.0147 | 0.0132 |
| collusion / final / colluder | 72 | 0.0228 | 0.0225 |
| collusion / final / honest | 72 | 0.0242 | 0.0245 |
| collusion / private / colluder | 72 | 0.0108 | 0.0095 |
| control / discussion / honest | 288 | 0.0149 | 0.0126 |
| control / final / honest | 144 | 0.0260 | 0.0261 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<em>'` 0.93 (0.8), `'<eos>'` 0.92 (0.8), `'?'` 0.60 (1.1), `' pungkasnya'` 0.55 (0.8), `'s'` 0.52 (0.4), `'.\u202f'` 0.49 (0.6), `'<end_of_turn>'` 0.45 (0.0), `').'` 0.44 (0.8), `' discussions'` 0.38 (0.8), `' recommendations'` 0.38 (1.2), `'\n\n\n'` 0.36 (0.3), `' pungkas'` 0.35 (0.4), `' thankfully'` 0.35 (0.9), `':'` 0.34 (0.6), `'https'` 0.34 (0.4), `'<u>'` 0.31 (0.5), `'.’'` 0.29 (0.2), `'\n\n\n\n'` 0.28 (0.1), `' continued'` 0.28 (1.3), `' insightful'` 0.27 (1.1)

First atom chosen: `'<eos>'` 0.70, `'<end_of_turn>'` 0.12, `'\n\n\n'` 0.04, `'?”'` 0.02, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01

Readout top-1 token: `'<eos>'` 0.70, `'<end_of_turn>'` 0.12, `'\n\n\n'` 0.04, `'?”'` 0.02, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01

### transfer_raw layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0178 | 0.0169 |
| collusion / discussion / honest | 144 | 0.0188 | 0.0208 |
| collusion / final / colluder | 72 | 0.0224 | 0.0227 |
| collusion / final / honest | 72 | 0.0238 | 0.0231 |
| collusion / private / colluder | 72 | 0.0123 | 0.0104 |
| control / discussion / honest | 288 | 0.0193 | 0.0211 |
| control / final / honest | 144 | 0.0247 | 0.0240 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.91 (0.8), `'<end_of_turn>'` 0.70 (0.0), `' discussions'` 0.52 (0.9), `'<em>'` 0.49 (0.9), `' pungkasnya'` 0.46 (0.6), `'?'` 0.45 (0.9), `'\n\n\n'` 0.41 (0.4), `'https'` 0.41 (0.5), `' recommendations'` 0.37 (0.9), `'.\u202f'` 0.36 (0.7), `'s'` 0.35 (0.5), `'\n\n\n\n'` 0.35 (0.1), `' continued'` 0.34 (1.3), `' thankfully'` 0.33 (1.1), `' FINALLY'` 0.32 (0.5), `' ſeinem'` 0.31 (0.0), `' detaylı'` 0.29 (0.9), `' hopefully'` 0.26 (0.4), `' Interestingly'` 0.24 (0.6), `').'` 0.24 (0.7)

First atom chosen: `'<eos>'` 0.55, `'<end_of_turn>'` 0.31, `'\n\n\n'` 0.03, `'....'` 0.01, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.00, `'...'` 0.00

Readout top-1 token: `'<eos>'` 0.55, `'<end_of_turn>'` 0.31, `'\n\n\n'` 0.03, `'....'` 0.01, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.00, `'...'` 0.00

### transfer_raw layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0201 | 0.0183 |
| collusion / discussion / honest | 144 | 0.0202 | 0.0190 |
| collusion / final / colluder | 72 | 0.0227 | 0.0225 |
| collusion / final / honest | 72 | 0.0245 | 0.0239 |
| collusion / private / colluder | 72 | 0.0145 | 0.0133 |
| control / discussion / honest | 288 | 0.0202 | 0.0190 |
| control / final / honest | 144 | 0.0244 | 0.0240 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.89 (0.8), `' discussions'` 0.68 (1.0), `'\n\n\n'` 0.68 (0.5), `'<end_of_turn>'` 0.65 (0.0), `'<em>'` 0.60 (0.9), `'.\u202f'` 0.58 (0.7), `'?'` 0.47 (0.9), `'\n\n\n\n'` 0.44 (0.1), `' continued'` 0.43 (1.4), `' insightful'` 0.42 (1.0), `' recommendations'` 0.41 (1.0), `' hopefully'` 0.34 (0.5), `' pungkasnya'` 0.34 (0.5), `' further'` 0.32 (0.6), `'s'` 0.32 (0.6), `' diría'` 0.30 (0.5), `').'` 0.29 (0.7), `' ſeinem'` 0.27 (0.0), `' detaylı'` 0.27 (0.9), `' Wikiseite'` 0.25 (0.0)

First atom chosen: `'<eos>'` 0.48, `'<end_of_turn>'` 0.36, `'\n\n\n'` 0.09, `'\n\n\n\n'` 0.01, `'...”'` 0.01, `'\n\n'` 0.00, `' recommendations'` 0.00, `' comments'` 0.00

Readout top-1 token: `'<eos>'` 0.48, `'<end_of_turn>'` 0.36, `'\n\n\n'` 0.09, `'\n\n\n\n'` 0.01, `'...”'` 0.01, `'\n\n'` 0.00, `' recommendations'` 0.00, `' comments'` 0.00

## core_topk — 1700 activations, k = 25, method topk, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/core/20261001T012639Z/variants/jpaper_topk_k25`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0097 | 0.0096 | 0.0088–0.0105 | 1.00 | 0.0102 | 25.0 | 0.00 | 0.0 | 0.09 |
| 20 | 0.0093 | 0.0092 | 0.0082–0.0103 | 1.00 | 0.0095 | 25.0 | 0.00 | 0.0 | 0.19 |
| 21 | 0.0103 | 0.0102 | 0.0091–0.0113 | 1.00 | 0.0106 | 25.0 | 0.00 | 0.0 | 0.50 |
| 22 | 0.0107 | 0.0108 | 0.0097–0.0119 | 1.00 | 0.0108 | 25.0 | 0.00 | 0.0 | 0.38 |
| 23 | 0.0127 | 0.0126 | 0.0115–0.0139 | 1.00 | 0.0126 | 25.0 | 0.00 | 0.0 | 0.47 |

### core_topk layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0093 | 0.0093 |
| collusion / discussion / honest | 300 | 0.0094 | 0.0093 |
| collusion / final / colluder | 100 | 0.0101 | 0.0099 |
| collusion / final / honest | 100 | 0.0097 | 0.0094 |
| collusion / private / colluder | 100 | 0.0095 | 0.0088 |
| control / discussion / honest | 600 | 0.0099 | 0.0098 |
| control / final / honest | 200 | 0.0099 | 0.0099 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.96 (1.0), `' acknowledging'` 0.94 (0.3), `'<eos>'` 0.93 (1.6), `' spokesperson'` 0.84 (0.8), `' understandably'` 0.78 (1.0), `' eloquently'` 0.73 (2.3), `' reaffirm'` 0.68 (0.5), `' confidently'` 0.66 (0.1), `' confirming'` 0.64 (0.2), `' statement'` 0.62 (0.0), `' acknowledges'` 0.60 (0.0), `' confirms'` 0.49 (0.3), `' reaffirmed'` 0.46 (0.1), `' confirmed'` 0.46 (0.7), `' undeniably'` 0.45 (0.4), `' verdict'` 0.41 (0.2), `' recommendations'` 0.35 (0.2), `' confident'` 0.35 (0.0), `' responding'` 0.34 (0.0), `' decisions'` 0.34 (0.0)

First atom chosen: `'<eos>'` 0.36, `' eloquently'` 0.23, `' decision'` 0.22, `' acknowledging'` 0.06, `' confidently'` 0.02, `' confirming'` 0.02, `' spokesperson'` 0.02, `' questioning'` 0.01

Readout top-1 token: `'<eos>'` 1.00

### core_topk layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0080 | 0.0080 |
| collusion / discussion / honest | 300 | 0.0090 | 0.0089 |
| collusion / final / colluder | 100 | 0.0093 | 0.0090 |
| collusion / final / honest | 100 | 0.0103 | 0.0102 |
| collusion / private / colluder | 100 | 0.0093 | 0.0087 |
| control / discussion / honest | 600 | 0.0095 | 0.0093 |
| control / final / honest | 200 | 0.0106 | 0.0106 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.86 (1.1), `' acknowledging'` 0.81 (0.0), `' spokesperson'` 0.78 (1.1), `' eloquently'` 0.69 (1.9), `' reaffirm'` 0.67 (0.9), `' empathetic'` 0.62 (1.0), `' reaffirmed'` 0.58 (0.3), `' understandably'` 0.58 (1.0), `' confidently'` 0.56 (0.2), `' recommendation'` 0.54 (0.1), `' responding'` 0.52 (0.1), `' confirms'` 0.49 (0.6), `' recommendations'` 0.49 (0.2), `' underscores'` 0.46 (0.2), `' reiterated'` 0.45 (0.0), `' acknowledges'` 0.45 (0.1), `' insightful'` 0.44 (0.0), `' prioritizing'` 0.44 (1.6), `'<eos>'` 0.44 (1.6), `' thoughtfully'` 0.43 (0.0)

First atom chosen: `' decision'` 0.25, `' eloquently'` 0.17, `' reaffirm'` 0.10, `' empathetic'` 0.08, `' confidently'` 0.05, `' spokesperson'` 0.04, `'<eos>'` 0.03, `' prioritizing'` 0.03

Readout top-1 token: `'<eos>'` 1.00, `'.”'` 0.00, `' discussions'` 0.00, `'\n\n'` 0.00, `' eloquently'` 0.00, `'?”'` 0.00, `').”'` 0.00

### core_topk layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0088 | 0.0086 |
| collusion / discussion / honest | 300 | 0.0098 | 0.0096 |
| collusion / final / colluder | 100 | 0.0110 | 0.0109 |
| collusion / final / honest | 100 | 0.0120 | 0.0121 |
| collusion / private / colluder | 100 | 0.0099 | 0.0095 |
| control / discussion / honest | 600 | 0.0103 | 0.0102 |
| control / final / honest | 200 | 0.0121 | 0.0120 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.93 (1.7), `' recommendation'` 0.82 (0.4), `' recommendations'` 0.81 (0.6), `' recommending'` 0.62 (0.0), `' acknowledging'` 0.61 (0.0), `' responding'` 0.61 (0.1), `' spokesperson'` 0.60 (1.1), `' citing'` 0.60 (0.2), `' prioritizing'` 0.57 (1.8), `' reaffirm'` 0.55 (1.1), `' consensus'` 0.52 (0.2), `' reaffirmed'` 0.47 (0.1), `' advocating'` 0.45 (0.2), `' advocacy'` 0.44 (0.2), `' eloquently'` 0.42 (1.9), `' confirming'` 0.41 (0.2), `' decisions'` 0.39 (0.0), `' confirmed'` 0.38 (0.7), `' urging'` 0.37 (0.3), `' confidently'` 0.37 (0.5)

First atom chosen: `' decision'` 0.36, `' recommendation'` 0.11, `' prioritizing'` 0.06, `' reaffirm'` 0.05, `' recommendations'` 0.04, `' eloquently'` 0.03, `' advocating'` 0.02, `' convincing'` 0.02

Readout top-1 token: `'<eos>'` 0.98, `' discussions'` 0.00, `' unethical'` 0.00, `' decision'` 0.00, `' despite'` 0.00, `'<end_of_turn>'` 0.00, `' suggestions'` 0.00, `' subtly'` 0.00

### core_topk layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0092 | 0.0091 |
| collusion / discussion / honest | 300 | 0.0108 | 0.0107 |
| collusion / final / colluder | 100 | 0.0095 | 0.0094 |
| collusion / final / honest | 100 | 0.0117 | 0.0116 |
| collusion / private / colluder | 100 | 0.0107 | 0.0106 |
| control / discussion / honest | 600 | 0.0112 | 0.0112 |
| control / final / honest | 200 | 0.0117 | 0.0115 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.93 (1.6), `' recommendation'` 0.84 (0.7), `' recommendations'` 0.82 (0.4), `' recommending'` 0.69 (0.0), `' تاکید'` 0.66 (1.4), `' decisions'` 0.61 (0.0), `' spokesperson'` 0.47 (1.1), `' prioritizing'` 0.46 (1.8), `' advocating'` 0.45 (0.4), `' acknowledging'` 0.44 (0.2), `' consensus'` 0.43 (0.1), `' recommended'` 0.40 (0.0), `' reaffirm'` 0.40 (0.7), `' advocacy'` 0.39 (0.1), `' reaffirmed'` 0.37 (0.2), `' unanimously'` 0.33 (0.7), `' policymakers'` 0.32 (0.3), `' eloquently'` 0.30 (1.7), `' discussions'` 0.30 (0.9), `' urging'` 0.28 (0.2)

First atom chosen: `' decision'` 0.38, `' recommendation'` 0.19, `' prioritizing'` 0.04, `' recommendations'` 0.03, `' تاکید'` 0.03, `' advocating'` 0.03, `' convincing'` 0.03, `' unanimously'` 0.02

Readout top-1 token: `'<eos>'` 0.92, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' discussions'` 0.01, `' despite'` 0.01, `' decision'` 0.00, `' leveraging'` 0.00, `' strategically'` 0.00

### core_topk layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0118 | 0.0116 |
| collusion / discussion / honest | 300 | 0.0123 | 0.0122 |
| collusion / final / colluder | 100 | 0.0123 | 0.0120 |
| collusion / final / honest | 100 | 0.0133 | 0.0131 |
| collusion / private / colluder | 100 | 0.0123 | 0.0123 |
| control / discussion / honest | 600 | 0.0130 | 0.0129 |
| control / final / honest | 200 | 0.0135 | 0.0134 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.87 (1.7), `' recommendations'` 0.74 (0.5), `' recommendation'` 0.72 (0.4), `' prioritizing'` 0.62 (1.8), `' decisions'` 0.54 (0.0), `' consensus'` 0.52 (0.5), `' advocating'` 0.45 (0.6), `' spokesperson'` 0.44 (1.5), `' advocacy'` 0.44 (0.2), `' recommending'` 0.42 (0.1), `' reaffirmed'` 0.38 (0.2), `' policymakers'` 0.38 (0.2), `' emphasizing'` 0.37 (0.4), `' unanimously'` 0.36 (1.3), `' reiterated'` 0.36 (0.2), `' تاکید'` 0.33 (1.0), `' consistent'` 0.31 (0.9), `' decisiones'` 0.31 (0.2), `' reaffirm'` 0.30 (0.6), `' leveraging'` 0.29 (1.0)

First atom chosen: `' decision'` 0.28, `' prioritizing'` 0.08, `' consensus'` 0.07, `' recommendation'` 0.06, `' unanimously'` 0.06, `' advocating'` 0.05, `' recommendations'` 0.04, `' spokesperson'` 0.04

Readout top-1 token: `'<eos>'` 0.89, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' decision'` 0.01, `' recommendations'` 0.01, `' discussions'` 0.01, `' hyö'` 0.00, `' ethically'` 0.00

## transfer_topk — 936 activations, k = 25, method topk, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/transfer/RUNPOD/variants/jpaper_topk_k25`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0122 | 0.0122 | 0.0100–0.0144 | 1.00 | 0.0119 | 25.0 | 0.00 | 0.0 | 0.14 |
| 20 | 0.0122 | 0.0123 | 0.0098–0.0143 | 1.00 | 0.0117 | 25.0 | 0.00 | 0.0 | 0.16 |
| 21 | 0.0127 | 0.0128 | 0.0102–0.0151 | 1.00 | 0.0118 | 25.0 | 0.00 | 0.0 | 0.20 |
| 22 | 0.0129 | 0.0131 | 0.0109–0.0151 | 1.00 | 0.0115 | 25.0 | 0.00 | 0.0 | 0.14 |
| 23 | 0.0142 | 0.0143 | 0.0121–0.0164 | 1.00 | 0.0128 | 25.0 | 0.00 | 0.0 | 0.15 |

### transfer_topk layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0114 | 0.0114 |
| collusion / discussion / honest | 144 | 0.0115 | 0.0116 |
| collusion / final / colluder | 72 | 0.0139 | 0.0143 |
| collusion / final / honest | 72 | 0.0139 | 0.0137 |
| collusion / private / colluder | 72 | 0.0093 | 0.0088 |
| control / discussion / honest | 288 | 0.0114 | 0.0115 |
| control / final / honest | 144 | 0.0149 | 0.0149 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.83 (1.7), `'?'` 0.48 (1.8), `'.'` 0.38 (0.4), `'Continue'` 0.37 (0.0), `' understandably'` 0.36 (1.5), `':'` 0.35 (1.1), `'https'` 0.34 (0.6), `' Interestingly'` 0.32 (0.5), `' insightful'` 0.31 (1.1), `' Podcast'` 0.30 (0.3), `' concludes'` 0.29 (0.5), `'今回の'` 0.29 (0.1), `' eloquently'` 0.28 (2.0), `' comments'` 0.27 (0.2), `' Webinar'` 0.27 (0.6), `'近期'` 0.25 (0.1), `' COMMENTS'` 0.25 (0.1), `' quote'` 0.23 (0.2), `' questions'` 0.23 (0.3), `' Thankfully'` 0.23 (0.4)

First atom chosen: `'<eos>'` 0.53, `'?'` 0.11, `' insightful'` 0.04, `'.'` 0.03, `' understandably'` 0.03, `' continued'` 0.02, `' comments'` 0.02, `':'` 0.02

Readout top-1 token: `'<eos>'` 0.95, `'...”'` 0.01, `'\n\n\n'` 0.00, `' […]'` 0.00, `'\n\n'` 0.00, `'.”'` 0.00, `'?”'` 0.00, `'...'` 0.00

### transfer_topk layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0117 | 0.0114 |
| collusion / discussion / honest | 144 | 0.0117 | 0.0120 |
| collusion / final / colluder | 72 | 0.0129 | 0.0130 |
| collusion / final / honest | 72 | 0.0137 | 0.0132 |
| collusion / private / colluder | 72 | 0.0095 | 0.0091 |
| control / discussion / honest | 288 | 0.0116 | 0.0116 |
| control / final / honest | 144 | 0.0144 | 0.0141 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.64 (1.5), `' recommendations'` 0.38 (0.7), `' insightful'` 0.36 (1.4), `'如果您'` 0.35 (0.9), `'?'` 0.35 (1.9), `' Podcast'` 0.33 (0.4), `' concludes'` 0.33 (0.5), `'近期'` 0.33 (0.2), `' comments'` 0.32 (0.2), `' COMMENTS'` 0.30 (0.1), `' spokesperson'` 0.29 (0.6), `' Webinar'` 0.29 (0.3), `'今回の'` 0.26 (0.1), `' Interestingly'` 0.25 (0.9), `' understandably'` 0.23 (1.1), `' yorum'` 0.23 (0.3), `' thoughtfully'` 0.22 (0.5), `' insights'` 0.22 (0.1), `' recommendation'` 0.22 (0.1), `'Continue'` 0.21 (0.1)

First atom chosen: `'<eos>'` 0.18, `' insightful'` 0.14, `'?'` 0.07, `' recommendations'` 0.05, `' concludes'` 0.04, `' continued'` 0.03, `' understandably'` 0.03, `' comments'` 0.03

Readout top-1 token: `'<eos>'` 0.90, `'\n\n\n'` 0.03, `'<end_of_turn>'` 0.01, `'?”'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01, `'.\\\\'` 0.00, `'.”'` 0.00

### transfer_topk layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0123 | 0.0124 |
| collusion / discussion / honest | 144 | 0.0120 | 0.0121 |
| collusion / final / colluder | 72 | 0.0139 | 0.0139 |
| collusion / final / honest | 72 | 0.0147 | 0.0149 |
| collusion / private / colluder | 72 | 0.0095 | 0.0096 |
| control / discussion / honest | 288 | 0.0120 | 0.0119 |
| control / final / honest | 144 | 0.0153 | 0.0150 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' comments'` 0.58 (0.4), `' recommendations'` 0.49 (1.0), `'?'` 0.38 (2.0), `' insightful'` 0.38 (1.2), `' concludes'` 0.36 (0.6), `' recommendation'` 0.35 (0.1), `' discussions'` 0.31 (0.9), `' commenting'` 0.29 (0.1), `' spokesperson'` 0.27 (0.8), `' feedback'` 0.27 (0.4), `' yorum'` 0.27 (0.3), `' suggests'` 0.26 (0.4), `' continued'` 0.26 (1.8), `' commented'` 0.23 (0.0), `' highlighted'` 0.23 (0.2), `' detaylı'` 0.22 (1.4), `' discusses'` 0.21 (0.1), `' recommending'` 0.20 (0.1), `' webinar'` 0.19 (0.4), `' podcast'` 0.19 (0.9)

First atom chosen: `' recommendations'` 0.14, `'?'` 0.10, `' insightful'` 0.08, `' continued'` 0.08, `' concludes'` 0.07, `' comments'` 0.05, `'https'` 0.02, `' detaylı'` 0.02

Readout top-1 token: `'<eos>'` 0.70, `'<end_of_turn>'` 0.12, `'\n\n\n'` 0.04, `'?”'` 0.02, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01

### transfer_topk layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0121 | 0.0119 |
| collusion / discussion / honest | 144 | 0.0123 | 0.0125 |
| collusion / final / colluder | 72 | 0.0140 | 0.0141 |
| collusion / final / honest | 72 | 0.0144 | 0.0140 |
| collusion / private / colluder | 72 | 0.0102 | 0.0095 |
| control / discussion / honest | 288 | 0.0126 | 0.0128 |
| control / final / honest | 144 | 0.0148 | 0.0145 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' comments'` 0.54 (0.4), `' recommendations'` 0.44 (0.9), `' discussions'` 0.33 (1.1), `' concludes'` 0.31 (0.9), `' continued'` 0.29 (1.9), `' recommendation'` 0.28 (0.1), `' detaylı'` 0.28 (1.4), `'Continue'` 0.26 (0.3), `' yorum'` 0.25 (0.2), `' insightful'` 0.25 (1.2), `' Continuing'` 0.23 (0.1), `' discusses'` 0.22 (0.1), `' continues'` 0.21 (0.1), `' feedback'` 0.21 (0.7), `' ادامه'` 0.21 (0.0), `' asks'` 0.19 (0.6), `' commenting'` 0.19 (0.1), `'<eos>'` 0.18 (1.4), `' discussing'` 0.17 (0.2), `'总结'` 0.17 (0.1)

First atom chosen: `' recommendations'` 0.09, `' continued'` 0.07, `' concludes'` 0.07, `' comments'` 0.06, `' insightful'` 0.05, `'<eos>'` 0.05, `' discussions'` 0.04, `' detaylı'` 0.04

Readout top-1 token: `'<eos>'` 0.55, `'<end_of_turn>'` 0.31, `'\n\n\n'` 0.03, `'....'` 0.01, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.00, `'...'` 0.00

### transfer_topk layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0139 | 0.0139 |
| collusion / discussion / honest | 144 | 0.0143 | 0.0142 |
| collusion / final / colluder | 72 | 0.0145 | 0.0145 |
| collusion / final / honest | 72 | 0.0149 | 0.0144 |
| collusion / private / colluder | 72 | 0.0117 | 0.0119 |
| control / discussion / honest | 288 | 0.0143 | 0.0144 |
| control / final / honest | 144 | 0.0151 | 0.0151 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' recommendations'` 0.52 (1.0), `' discussions'` 0.51 (1.2), `' insightful'` 0.51 (1.4), `' detaylı'` 0.47 (0.9), `' comments'` 0.40 (0.3), `' ausführ'` 0.36 (0.4), `' continued'` 0.34 (2.1), `' yorum'` 0.31 (0.1), `' tentunya'` 0.27 (0.9), `' concludes'` 0.27 (0.5), `' discusses'` 0.26 (0.1), `' rekan'` 0.23 (0.3), `' recommendation'` 0.22 (0.0), `' insights'` 0.22 (0.1), `' discussing'` 0.21 (0.2), `' reseña'` 0.20 (0.3), `' spokesperson'` 0.19 (0.9), `'總結'` 0.19 (0.5), `' ادامه'` 0.19 (0.0), `' tämän'` 0.18 (0.7)

First atom chosen: `' continued'` 0.14, `' recommendations'` 0.11, `' insightful'` 0.08, `' detaylı'` 0.07, `' discussions'` 0.07, `' concludes'` 0.05, `'<eos>'` 0.03, `' tentunya'` 0.03

Readout top-1 token: `'<eos>'` 0.48, `'<end_of_turn>'` 0.36, `'\n\n\n'` 0.09, `'\n\n\n\n'` 0.01, `'...”'` 0.01, `'\n\n'` 0.00, `' recommendations'` 0.00, `' comments'` 0.00

## core_k50 — 1700 activations, k = 50, method nnomp, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/core/20261001T012639Z/variants/jpaper_nnomp_k50`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0120 | 0.0118 | 0.0109–0.0129 | 1.00 | 0.0134 | 16.4 | 1.00 | 33.6 | 0.08 |
| 20 | 0.0118 | 0.0117 | 0.0106–0.0128 | 1.00 | 0.0133 | 17.3 | 1.00 | 32.7 | 0.16 |
| 21 | 0.0130 | 0.0130 | 0.0116–0.0144 | 1.00 | 0.0149 | 16.6 | 1.00 | 33.4 | 0.34 |
| 22 | 0.0133 | 0.0134 | 0.0120–0.0147 | 1.00 | 0.0149 | 17.1 | 1.00 | 32.9 | 0.25 |
| 23 | 0.0155 | 0.0154 | 0.0141–0.0169 | 1.00 | 0.0176 | 17.8 | 1.00 | 32.2 | 0.31 |

### core_k50 layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0118 | 0.0117 |
| collusion / discussion / honest | 300 | 0.0117 | 0.0115 |
| collusion / final / colluder | 100 | 0.0122 | 0.0120 |
| collusion / final / honest | 100 | 0.0117 | 0.0114 |
| collusion / private / colluder | 100 | 0.0121 | 0.0116 |
| control / discussion / honest | 600 | 0.0122 | 0.0121 |
| control / final / honest | 200 | 0.0120 | 0.0119 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.5), `' decision'` 0.98 (0.9), `' acknowledging'` 0.96 (0.2), `' understandably'` 0.95 (0.7), `' spokesperson'` 0.95 (0.4), `' eloquently'` 0.85 (1.9), `' acknowledges'` 0.83 (0.0), `' confidently'` 0.80 (0.0), `' statement'` 0.80 (0.0), `' reaffirm'` 0.79 (0.3), `' confirming'` 0.76 (0.2), `' confirms'` 0.67 (0.1), `' undeniably'` 0.64 (0.4), `' reaffirmed'` 0.64 (0.0), `' verdict'` 0.62 (0.1), `' confirmed'` 0.61 (0.4), `' prioritizing'` 0.59 (0.6), `' confident'` 0.56 (0.0), `' recommendations'` 0.56 (0.0), `' responding'` 0.56 (0.0)

First atom chosen: `'<eos>'` 0.36, `' eloquently'` 0.23, `' decision'` 0.22, `' acknowledging'` 0.06, `' confidently'` 0.02, `' confirming'` 0.02, `' spokesperson'` 0.02, `' questioning'` 0.01

Readout top-1 token: `'<eos>'` 1.00

### core_k50 layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0103 | 0.0103 |
| collusion / discussion / honest | 300 | 0.0116 | 0.0115 |
| collusion / final / colluder | 100 | 0.0113 | 0.0111 |
| collusion / final / honest | 100 | 0.0125 | 0.0123 |
| collusion / private / colluder | 100 | 0.0119 | 0.0114 |
| control / discussion / honest | 600 | 0.0122 | 0.0120 |
| control / final / honest | 200 | 0.0128 | 0.0126 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.3), `' decision'` 0.95 (0.8), `' spokesperson'` 0.93 (0.4), `' acknowledging'` 0.91 (0.0), `' understandably'` 0.85 (0.6), `' eloquently'` 0.83 (1.6), `' empathetic'` 0.80 (0.7), `' prioritizing'` 0.77 (0.9), `' reaffirm'` 0.76 (0.6), `' confidently'` 0.72 (0.1), `' recommendations'` 0.70 (0.1), `' recommendation'` 0.69 (0.1), `' acknowledges'` 0.69 (0.0), `' responding'` 0.69 (0.0), `' reaffirmed'` 0.68 (0.2), `' confirms'` 0.68 (0.3), `' تاکید'` 0.65 (0.3), `' underscores'` 0.63 (0.0), `' reiterated'` 0.63 (0.0), `' thoughtfully'` 0.62 (0.0)

First atom chosen: `' decision'` 0.25, `' eloquently'` 0.17, `' reaffirm'` 0.10, `' empathetic'` 0.08, `' confidently'` 0.05, `' spokesperson'` 0.04, `'<eos>'` 0.03, `' prioritizing'` 0.03

Readout top-1 token: `'<eos>'` 1.00, `'.”'` 0.00, `' discussions'` 0.00, `'\n\n'` 0.00, `' eloquently'` 0.00, `'?”'` 0.00, `').”'` 0.00

### core_k50 layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0111 | 0.0109 |
| collusion / discussion / honest | 300 | 0.0127 | 0.0126 |
| collusion / final / colluder | 100 | 0.0133 | 0.0129 |
| collusion / final / honest | 100 | 0.0148 | 0.0149 |
| collusion / private / colluder | 100 | 0.0126 | 0.0122 |
| control / discussion / honest | 600 | 0.0132 | 0.0132 |
| control / final / honest | 200 | 0.0151 | 0.0151 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.96 (1.3), `' recommendation'` 0.87 (0.4), `' recommendations'` 0.87 (0.4), `'<eos>'` 0.86 (1.0), `' prioritizing'` 0.83 (1.3), `' acknowledging'` 0.82 (0.0), `' spokesperson'` 0.81 (0.4), `' responding'` 0.79 (0.1), `' citing'` 0.79 (0.1), `' recommending'` 0.76 (0.0), `' eloquently'` 0.73 (1.3), `' تاکید'` 0.67 (0.5), `' reaffirm'` 0.66 (0.6), `' consensus'` 0.63 (0.3), `' advocacy'` 0.63 (0.1), `' advocating'` 0.60 (0.2), `' decisions'` 0.59 (0.0), `' reaffirmed'` 0.58 (0.0), `' empathetic'` 0.56 (0.4), `' diría'` 0.55 (0.4)

First atom chosen: `' decision'` 0.36, `' recommendation'` 0.11, `' prioritizing'` 0.06, `' reaffirm'` 0.05, `' recommendations'` 0.04, `' eloquently'` 0.03, `' advocating'` 0.02, `' convincing'` 0.02

Readout top-1 token: `'<eos>'` 0.98, `' discussions'` 0.00, `' unethical'` 0.00, `' decision'` 0.00, `' despite'` 0.00, `'<end_of_turn>'` 0.00, `' suggestions'` 0.00, `' subtly'` 0.00

### core_k50 layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0113 | 0.0113 |
| collusion / discussion / honest | 300 | 0.0137 | 0.0135 |
| collusion / final / colluder | 100 | 0.0115 | 0.0113 |
| collusion / final / honest | 100 | 0.0142 | 0.0142 |
| collusion / private / colluder | 100 | 0.0135 | 0.0136 |
| control / discussion / honest | 600 | 0.0140 | 0.0139 |
| control / final / honest | 200 | 0.0142 | 0.0139 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.96 (1.3), `' recommendation'` 0.88 (0.5), `' recommendations'` 0.86 (0.3), `' recommending'` 0.80 (0.0), `' decisions'` 0.78 (0.0), `' prioritizing'` 0.78 (1.2), `' تاکید'` 0.76 (0.9), `' diría'` 0.74 (0.5), `' spokesperson'` 0.70 (0.4), `'<eos>'` 0.68 (0.8), `' acknowledging'` 0.68 (0.0), `' eloquently'` 0.65 (1.0), `' recommended'` 0.57 (0.0), `' advocating'` 0.55 (0.3), `' advocacy'` 0.54 (0.1), `' consensus'` 0.52 (0.1), `' reaffirm'` 0.51 (0.4), `' policymakers'` 0.50 (0.1), `' thankfully'` 0.49 (0.5), `' reaffirmed'` 0.48 (0.1)

First atom chosen: `' decision'` 0.38, `' recommendation'` 0.19, `' prioritizing'` 0.04, `' recommendations'` 0.03, `' تاکید'` 0.03, `' advocating'` 0.03, `' convincing'` 0.03, `' unanimously'` 0.02

Readout top-1 token: `'<eos>'` 0.92, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' discussions'` 0.01, `' despite'` 0.01, `' decision'` 0.00, `' leveraging'` 0.00, `' strategically'` 0.00

### core_k50 layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0144 | 0.0142 |
| collusion / discussion / honest | 300 | 0.0152 | 0.0152 |
| collusion / final / colluder | 100 | 0.0148 | 0.0143 |
| collusion / final / honest | 100 | 0.0162 | 0.0160 |
| collusion / private / colluder | 100 | 0.0153 | 0.0154 |
| control / discussion / honest | 600 | 0.0159 | 0.0156 |
| control / final / honest | 200 | 0.0164 | 0.0162 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.93 (1.3), `' prioritizing'` 0.84 (1.4), `' recommendations'` 0.83 (0.4), `' recommendation'` 0.81 (0.2), `' diría'` 0.79 (0.7), `'<eos>'` 0.76 (0.9), `' decisions'` 0.71 (0.0), `' spokesperson'` 0.67 (0.6), `' consensus'` 0.59 (0.4), `' advocacy'` 0.59 (0.1), `' policymakers'` 0.57 (0.1), `' recommending'` 0.57 (0.1), `' confirmado'` 0.56 (0.8), `' leveraging'` 0.56 (0.6), `' advocating'` 0.56 (0.5), `' تاکید'` 0.53 (0.4), `' emphasizing'` 0.52 (0.3), `' statement'` 0.51 (0.0), `' reaffirmed'` 0.50 (0.1), `' ethically'` 0.49 (1.1)

First atom chosen: `' decision'` 0.28, `' prioritizing'` 0.08, `' consensus'` 0.07, `' recommendation'` 0.06, `' unanimously'` 0.06, `' advocating'` 0.05, `' recommendations'` 0.04, `' spokesperson'` 0.04

Readout top-1 token: `'<eos>'` 0.89, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' decision'` 0.01, `' recommendations'` 0.01, `' discussions'` 0.01, `' hyö'` 0.00, `' ethically'` 0.00

## transfer_k50 — 936 activations, k = 50, method nnomp, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/transfer/RUNPOD/variants/jpaper_nnomp_k50`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0149 | 0.0149 | 0.0124–0.0173 | 1.00 | 0.0171 | 18.5 | 1.00 | 31.5 | 0.11 |
| 20 | 0.0151 | 0.0152 | 0.0125–0.0175 | 1.00 | 0.0171 | 19.5 | 1.00 | 30.5 | 0.13 |
| 21 | 0.0160 | 0.0160 | 0.0128–0.0190 | 1.00 | 0.0175 | 19.3 | 1.00 | 30.7 | 0.16 |
| 22 | 0.0160 | 0.0160 | 0.0135–0.0185 | 1.00 | 0.0169 | 19.9 | 1.00 | 30.1 | 0.11 |
| 23 | 0.0176 | 0.0176 | 0.0151–0.0201 | 1.00 | 0.0186 | 21.5 | 1.00 | 28.5 | 0.12 |

### transfer_k50 layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0140 | 0.0139 |
| collusion / discussion / honest | 144 | 0.0141 | 0.0142 |
| collusion / final / colluder | 72 | 0.0166 | 0.0169 |
| collusion / final / honest | 72 | 0.0167 | 0.0164 |
| collusion / private / colluder | 72 | 0.0117 | 0.0113 |
| control / discussion / honest | 288 | 0.0142 | 0.0141 |
| control / final / honest | 144 | 0.0180 | 0.0180 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.96 (1.5), `' understandably'` 0.74 (0.9), `'?'` 0.69 (1.0), `':'` 0.56 (0.7), `' thankfully'` 0.51 (0.7), `' eloquently'` 0.51 (1.3), `'\n\n\n'` 0.47 (0.3), `'Continue'` 0.47 (0.0), `'https'` 0.47 (0.2), `'.'` 0.46 (0.1), `' Interestingly'` 0.46 (0.2), `'  '` 0.46 (0.3), `' Podcast'` 0.44 (0.1), `' pungkasnya'` 0.41 (0.5), `'今回の'` 0.40 (0.0), `'\n\n\n\n'` 0.39 (0.2), `' concludes'` 0.38 (0.1), `' Thankfully'` 0.36 (0.2), `' comments'` 0.36 (0.1), `' Webinar'` 0.35 (0.4)

First atom chosen: `'<eos>'` 0.53, `'?'` 0.11, `' insightful'` 0.04, `'.'` 0.03, `' understandably'` 0.03, `' continued'` 0.02, `' comments'` 0.02, `':'` 0.02

Readout top-1 token: `'<eos>'` 0.95, `'...”'` 0.01, `'\n\n\n'` 0.00, `' […]'` 0.00, `'\n\n'` 0.00, `'.”'` 0.00, `'?”'` 0.00, `'...'` 0.00

### transfer_k50 layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0146 | 0.0143 |
| collusion / discussion / honest | 144 | 0.0145 | 0.0144 |
| collusion / final / colluder | 72 | 0.0155 | 0.0156 |
| collusion / final / honest | 72 | 0.0167 | 0.0158 |
| collusion / private / colluder | 72 | 0.0120 | 0.0118 |
| control / discussion / honest | 288 | 0.0147 | 0.0147 |
| control / final / honest | 144 | 0.0175 | 0.0169 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.94 (1.2), `'?'` 0.68 (1.0), `':'` 0.54 (0.8), `' understandably'` 0.50 (0.7), `' discussions'` 0.48 (0.7), `'如果您'` 0.48 (0.4), `' thankfully'` 0.47 (0.9), `' recommendations'` 0.46 (0.5), `'近期'` 0.45 (0.0), `'\n\n\n'` 0.44 (0.4), `' eloquently'` 0.44 (1.0), `' comments'` 0.43 (0.1), `' insightful'` 0.42 (1.0), `' pungkasnya'` 0.42 (0.6), `' Webinar'` 0.42 (0.1), `' Podcast'` 0.42 (0.1), `' concludes'` 0.41 (0.1), `'今回の'` 0.41 (0.0), `' spokesperson'` 0.41 (0.2), `' COMMENTS'` 0.39 (0.1)

First atom chosen: `'<eos>'` 0.18, `' insightful'` 0.14, `'?'` 0.07, `' recommendations'` 0.05, `' concludes'` 0.04, `' continued'` 0.03, `' understandably'` 0.03, `' comments'` 0.03

Readout top-1 token: `'<eos>'` 0.90, `'\n\n\n'` 0.03, `'<end_of_turn>'` 0.01, `'?”'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01, `'.\\\\'` 0.00, `'.”'` 0.00

### transfer_k50 layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0151 | 0.0150 |
| collusion / discussion / honest | 144 | 0.0149 | 0.0143 |
| collusion / final / colluder | 72 | 0.0174 | 0.0174 |
| collusion / final / honest | 72 | 0.0188 | 0.0187 |
| collusion / private / colluder | 72 | 0.0120 | 0.0119 |
| control / discussion / honest | 288 | 0.0150 | 0.0150 |
| control / final / honest | 144 | 0.0198 | 0.0194 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'?'` 0.72 (1.2), `' comments'` 0.66 (0.1), `'<eos>'` 0.64 (0.9), `' pungkasnya'` 0.60 (0.8), `' recommendations'` 0.56 (0.8), `' discussions'` 0.54 (0.7), `':'` 0.50 (0.7), `' insightful'` 0.46 (0.8), `' concludes'` 0.44 (0.2), `' spokesperson'` 0.42 (0.2), `' thankfully'` 0.41 (0.8), `' commenting'` 0.41 (0.0), `' recommendation'` 0.40 (0.1), `').'` 0.39 (0.9), `' yorum'` 0.38 (0.0), `'  '` 0.37 (0.7), `'https'` 0.36 (0.5), `' continued'` 0.35 (1.1), `' suggests'` 0.35 (0.1), `' commented'` 0.34 (0.0)

First atom chosen: `' recommendations'` 0.14, `'?'` 0.10, `' insightful'` 0.08, `' continued'` 0.08, `' concludes'` 0.07, `' comments'` 0.05, `'https'` 0.02, `' detaylı'` 0.02

Readout top-1 token: `'<eos>'` 0.70, `'<end_of_turn>'` 0.12, `'\n\n\n'` 0.04, `'?”'` 0.02, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01

### transfer_k50 layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0149 | 0.0149 |
| collusion / discussion / honest | 144 | 0.0153 | 0.0154 |
| collusion / final / colluder | 72 | 0.0169 | 0.0172 |
| collusion / final / honest | 72 | 0.0181 | 0.0175 |
| collusion / private / colluder | 72 | 0.0127 | 0.0121 |
| control / discussion / honest | 288 | 0.0157 | 0.0158 |
| control / final / honest | 144 | 0.0183 | 0.0171 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' discussions'` 0.67 (0.7), `'<eos>'` 0.66 (0.9), `' comments'` 0.64 (0.1), `'?'` 0.55 (1.2), `' recommendations'` 0.51 (0.6), `' pungkasnya'` 0.48 (0.6), `'https'` 0.44 (0.5), `' continued'` 0.42 (1.1), `' thankfully'` 0.41 (1.1), `' detaylı'` 0.40 (0.7), `' concludes'` 0.38 (0.3), `' yorum'` 0.34 (0.1), `' recommendation'` 0.33 (0.1), `' insightful'` 0.32 (0.6), `' discusses'` 0.30 (0.0), `'Continue'` 0.30 (0.0), `'\n\n\n'` 0.29 (0.5), `'Interestingly'` 0.29 (0.2), `' further'` 0.29 (0.7), `' eloquently'` 0.28 (0.7)

First atom chosen: `' recommendations'` 0.09, `' continued'` 0.07, `' concludes'` 0.07, `' comments'` 0.06, `' insightful'` 0.05, `'<eos>'` 0.05, `' discussions'` 0.04, `' detaylı'` 0.04

Readout top-1 token: `'<eos>'` 0.55, `'<end_of_turn>'` 0.31, `'\n\n\n'` 0.03, `'....'` 0.01, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.00, `'...'` 0.00

### transfer_k50 layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0171 | 0.0169 |
| collusion / discussion / honest | 144 | 0.0177 | 0.0177 |
| collusion / final / colluder | 72 | 0.0175 | 0.0174 |
| collusion / final / honest | 72 | 0.0185 | 0.0178 |
| collusion / private / colluder | 72 | 0.0145 | 0.0143 |
| control / discussion / honest | 288 | 0.0177 | 0.0178 |
| control / final / honest | 144 | 0.0187 | 0.0184 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' discussions'` 0.74 (0.9), `'<eos>'` 0.66 (0.9), `' insightful'` 0.60 (0.7), `' recommendations'` 0.59 (0.7), `' detaylı'` 0.54 (0.5), `' comments'` 0.51 (0.1), `'?'` 0.51 (1.1), `'\n\n\n'` 0.49 (0.7), `' continued'` 0.48 (1.3), `' ausführ'` 0.47 (0.1), `' pungkasnya'` 0.41 (0.5), `' yorum'` 0.40 (0.0), `' further'` 0.36 (0.6), `' rekan'` 0.35 (0.1), `' discusses'` 0.35 (0.0), `' concludes'` 0.34 (0.1), `' tentunya'` 0.34 (0.3), `' thankfully'` 0.33 (0.9), `' diría'` 0.31 (0.6), `' dikutip'` 0.31 (0.2)

First atom chosen: `' continued'` 0.14, `' recommendations'` 0.11, `' insightful'` 0.08, `' detaylı'` 0.07, `' discussions'` 0.07, `' concludes'` 0.05, `'<eos>'` 0.03, `' tentunya'` 0.03

Readout top-1 token: `'<eos>'` 0.48, `'<end_of_turn>'` 0.36, `'\n\n\n'` 0.09, `'\n\n\n\n'` 0.01, `'...”'` 0.01, `'\n\n'` 0.00, `' recommendations'` 0.00, `' comments'` 0.00

## core_k16 — 1700 activations, k = 16, method nnomp, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/core/20261001T012639Z/variants/jpaper_nnomp_k16`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0093 | 0.0092 | 0.0085–0.0100 | 1.00 | 0.0092 | 15.1 | 0.38 | 0.9 | 0.20 |
| 20 | 0.0094 | 0.0093 | 0.0084–0.0102 | 1.00 | 0.0091 | 15.4 | 0.26 | 0.6 | 0.34 |
| 21 | 0.0099 | 0.0097 | 0.0088–0.0109 | 1.00 | 0.0095 | 15.1 | 0.37 | 0.9 | 0.58 |
| 22 | 0.0102 | 0.0103 | 0.0093–0.0112 | 1.00 | 0.0098 | 15.1 | 0.32 | 0.9 | 0.42 |
| 23 | 0.0122 | 0.0122 | 0.0111–0.0134 | 1.00 | 0.0117 | 15.4 | 0.24 | 0.6 | 0.49 |

### core_k16 layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0090 | 0.0090 |
| collusion / discussion / honest | 300 | 0.0091 | 0.0090 |
| collusion / final / colluder | 100 | 0.0096 | 0.0093 |
| collusion / final / honest | 100 | 0.0091 | 0.0089 |
| collusion / private / colluder | 100 | 0.0094 | 0.0087 |
| control / discussion / honest | 600 | 0.0096 | 0.0094 |
| control / final / honest | 200 | 0.0094 | 0.0094 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.5), `' decision'` 0.84 (1.0), `' eloquently'` 0.83 (2.0), `' understandably'` 0.81 (0.8), `' spokesperson'` 0.66 (0.6), `' prioritizing'` 0.56 (0.7), `' acknowledging'` 0.51 (0.4), `'?'` 0.43 (0.5), `' reaffirm'` 0.42 (0.5), `' confirmed'` 0.39 (0.6), `' bestätigt'` 0.39 (0.4), `' undeniably'` 0.36 (0.6), `' hatır'` 0.36 (0.4), `'.\u202f'` 0.35 (0.9), `' hopeful'` 0.31 (0.5), `' 🙏'` 0.31 (0.3), `' confirming'` 0.30 (0.6), `' teşekkür'` 0.28 (0.3), `' thankfully'` 0.26 (0.4), `'.***'` 0.26 (0.6)

First atom chosen: `'<eos>'` 0.36, `' eloquently'` 0.23, `' decision'` 0.22, `' acknowledging'` 0.06, `' confidently'` 0.02, `' confirming'` 0.02, `' spokesperson'` 0.02, `' questioning'` 0.01

Readout top-1 token: `'<eos>'` 1.00

### core_k16 layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0083 | 0.0082 |
| collusion / discussion / honest | 300 | 0.0091 | 0.0091 |
| collusion / final / colluder | 100 | 0.0092 | 0.0091 |
| collusion / final / honest | 100 | 0.0098 | 0.0098 |
| collusion / private / colluder | 100 | 0.0093 | 0.0087 |
| control / discussion / honest | 600 | 0.0097 | 0.0096 |
| control / final / honest | 200 | 0.0103 | 0.0103 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.3), `' eloquently'` 0.79 (1.7), `' prioritizing'` 0.72 (1.0), `' decision'` 0.69 (1.1), `' understandably'` 0.61 (0.8), `' empathetic'` 0.60 (1.0), `' spokesperson'` 0.57 (0.7), `' bestätigt'` 0.50 (0.7), `' reaffirm'` 0.46 (1.0), `' thankfully'` 0.42 (0.5), `'.\u202f'` 0.42 (1.1), `' تاکید'` 0.36 (0.5), `' confirms'` 0.35 (0.7), `'?'` 0.30 (0.6), `'.***'` 0.27 (0.7), `' 🙏'` 0.27 (0.3), `' discussions'` 0.26 (0.7), `' hatır'` 0.26 (0.4), `' impactful'` 0.24 (0.4), `' confirmed'` 0.23 (0.5)

First atom chosen: `' decision'` 0.25, `' eloquently'` 0.17, `' reaffirm'` 0.10, `' empathetic'` 0.08, `' confidently'` 0.05, `' spokesperson'` 0.04, `'<eos>'` 0.03, `' prioritizing'` 0.03

Readout top-1 token: `'<eos>'` 1.00, `'.”'` 0.00, `' discussions'` 0.00, `'\n\n'` 0.00, `' eloquently'` 0.00, `'?”'` 0.00, `').”'` 0.00

### core_k16 layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0084 | 0.0083 |
| collusion / discussion / honest | 300 | 0.0093 | 0.0093 |
| collusion / final / colluder | 100 | 0.0106 | 0.0105 |
| collusion / final / honest | 100 | 0.0116 | 0.0117 |
| collusion / private / colluder | 100 | 0.0097 | 0.0095 |
| control / discussion / honest | 600 | 0.0099 | 0.0098 |
| control / final / honest | 200 | 0.0119 | 0.0118 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.86 (1.0), `' decision'` 0.83 (1.5), `' prioritizing'` 0.80 (1.4), `' eloquently'` 0.69 (1.4), `' تاکید'` 0.53 (0.7), `' diría'` 0.51 (0.4), `' recommendation'` 0.48 (0.7), `' recommendations'` 0.47 (0.7), `' spokesperson'` 0.43 (0.7), `' reaffirm'` 0.42 (1.0), `' pungkasnya'` 0.38 (0.5), `' confirmed'` 0.32 (0.7), `' despite'` 0.32 (1.1), `' empathetic'` 0.29 (0.7), `'?'` 0.29 (0.7), `' dikutip'` 0.26 (0.3), `' bestätigt'` 0.26 (0.5), `' consensus'` 0.23 (0.7), `' impactful'` 0.23 (0.4), `' thankfully'` 0.22 (0.5)

First atom chosen: `' decision'` 0.36, `' recommendation'` 0.11, `' prioritizing'` 0.06, `' reaffirm'` 0.05, `' recommendations'` 0.04, `' eloquently'` 0.03, `' advocating'` 0.02, `' convincing'` 0.02

Readout top-1 token: `'<eos>'` 0.98, `' discussions'` 0.00, `' unethical'` 0.00, `' decision'` 0.00, `' despite'` 0.00, `'<end_of_turn>'` 0.00, `' suggestions'` 0.00, `' subtly'` 0.00

### core_k16 layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0086 | 0.0085 |
| collusion / discussion / honest | 300 | 0.0101 | 0.0101 |
| collusion / final / colluder | 100 | 0.0091 | 0.0092 |
| collusion / final / honest | 100 | 0.0112 | 0.0111 |
| collusion / private / colluder | 100 | 0.0101 | 0.0097 |
| control / discussion / honest | 600 | 0.0106 | 0.0106 |
| control / final / honest | 200 | 0.0114 | 0.0113 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.82 (1.6), `' prioritizing'` 0.74 (1.3), `' diría'` 0.70 (0.5), `'<eos>'` 0.68 (0.8), `' تاکید'` 0.65 (1.0), `' eloquently'` 0.61 (1.1), `' recommendation'` 0.54 (0.8), `' thankfully'` 0.45 (0.5), `' recommendations'` 0.41 (0.6), `' spokesperson'` 0.41 (0.7), `' despite'` 0.33 (1.0), `' discussions'` 0.31 (0.9), `' likely'` 0.31 (0.6), `' reaffirm'` 0.27 (0.8), `' teşekkür'` 0.27 (0.5), `' leveraging'` 0.27 (0.5), `' ethically'` 0.27 (0.9), `' convincingly'` 0.24 (0.7), `' advocating'` 0.21 (0.8), `'REJECT'` 0.21 (0.5)

First atom chosen: `' decision'` 0.38, `' recommendation'` 0.19, `' prioritizing'` 0.04, `' recommendations'` 0.03, `' تاکید'` 0.03, `' advocating'` 0.03, `' convincing'` 0.03, `' unanimously'` 0.02

Readout top-1 token: `'<eos>'` 0.92, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' discussions'` 0.01, `' despite'` 0.01, `' decision'` 0.00, `' leveraging'` 0.00, `' strategically'` 0.00

### core_k16 layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0113 | 0.0112 |
| collusion / discussion / honest | 300 | 0.0118 | 0.0117 |
| collusion / final / colluder | 100 | 0.0119 | 0.0117 |
| collusion / final / honest | 100 | 0.0127 | 0.0123 |
| collusion / private / colluder | 100 | 0.0118 | 0.0118 |
| control / discussion / honest | 600 | 0.0127 | 0.0127 |
| control / final / honest | 200 | 0.0131 | 0.0129 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' prioritizing'` 0.78 (1.5), `' diría'` 0.76 (0.7), `'<eos>'` 0.76 (0.9), `' decision'` 0.76 (1.7), `' confirmado'` 0.49 (0.9), `' ethically'` 0.43 (1.3), `' spokesperson'` 0.42 (1.0), `' recommendations'` 0.40 (0.7), `' leveraging'` 0.39 (0.8), `' discussions'` 0.36 (1.0), `' eloquently'` 0.34 (0.9), `' impactful'` 0.33 (0.6), `' dikutip'` 0.33 (0.5), `' unanimously'` 0.29 (1.5), `' تاکید'` 0.28 (0.7), `' advocating'` 0.25 (1.2), `' consensus'` 0.25 (1.0), `' likely'` 0.24 (0.8), `' recommendation'` 0.24 (0.8), `' convincingly'` 0.23 (1.0)

First atom chosen: `' decision'` 0.28, `' prioritizing'` 0.08, `' consensus'` 0.07, `' recommendation'` 0.06, `' unanimously'` 0.06, `' advocating'` 0.05, `' recommendations'` 0.04, `' spokesperson'` 0.04

Readout top-1 token: `'<eos>'` 0.89, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' decision'` 0.01, `' recommendations'` 0.01, `' discussions'` 0.01, `' hyö'` 0.00, `' ethically'` 0.00

## transfer_k16 — 936 activations, k = 16, method nnomp, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/transfer/RUNPOD/variants/jpaper_nnomp_k16`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0119 | 0.0118 | 0.0097–0.0141 | 1.00 | 0.0111 | 15.6 | 0.18 | 0.4 | 0.21 |
| 20 | 0.0121 | 0.0123 | 0.0099–0.0143 | 1.00 | 0.0110 | 15.8 | 0.12 | 0.2 | 0.23 |
| 21 | 0.0127 | 0.0129 | 0.0100–0.0153 | 1.00 | 0.0111 | 15.6 | 0.17 | 0.4 | 0.29 |
| 22 | 0.0130 | 0.0131 | 0.0107–0.0154 | 1.00 | 0.0111 | 15.7 | 0.12 | 0.3 | 0.18 |
| 23 | 0.0144 | 0.0146 | 0.0122–0.0166 | 1.00 | 0.0122 | 15.8 | 0.08 | 0.2 | 0.20 |

### transfer_k16 layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0112 | 0.0112 |
| collusion / discussion / honest | 144 | 0.0110 | 0.0113 |
| collusion / final / colluder | 72 | 0.0138 | 0.0139 |
| collusion / final / honest | 72 | 0.0137 | 0.0137 |
| collusion / private / colluder | 72 | 0.0090 | 0.0085 |
| control / discussion / honest | 288 | 0.0110 | 0.0111 |
| control / final / honest | 144 | 0.0147 | 0.0146 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.96 (1.5), `' understandably'` 0.63 (1.1), `'?'` 0.63 (1.1), `' eloquently'` 0.46 (1.4), `':'` 0.44 (0.9), `' thankfully'` 0.43 (0.9), `' pungkasnya'` 0.38 (0.5), `'\n\n\n'` 0.37 (0.3), `'  '` 0.32 (0.5), `' discussions'` 0.28 (0.7), `"'"` 0.28 (0.7), `','` 0.26 (0.7), `'\n\n\n\n'` 0.25 (0.3), `'https'` 0.24 (0.4), `' insightful'` 0.22 (1.2), `' Webinar'` 0.22 (0.6), `'<em>'` 0.21 (0.8), `' showcasing'` 0.21 (0.6), `'’'` 0.19 (0.5), `'，《'` 0.18 (0.5)

First atom chosen: `'<eos>'` 0.53, `'?'` 0.11, `' insightful'` 0.04, `'.'` 0.03, `' understandably'` 0.03, `' continued'` 0.02, `' comments'` 0.02, `':'` 0.02

Readout top-1 token: `'<eos>'` 0.95, `'...”'` 0.01, `'\n\n\n'` 0.00, `' […]'` 0.00, `'\n\n'` 0.00, `'.”'` 0.00, `'?”'` 0.00, `'...'` 0.00

### transfer_k16 layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0118 | 0.0116 |
| collusion / discussion / honest | 144 | 0.0116 | 0.0121 |
| collusion / final / colluder | 72 | 0.0130 | 0.0130 |
| collusion / final / honest | 72 | 0.0136 | 0.0132 |
| collusion / private / colluder | 72 | 0.0091 | 0.0086 |
| control / discussion / honest | 288 | 0.0116 | 0.0113 |
| control / final / honest | 144 | 0.0144 | 0.0143 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.94 (1.2), `'?'` 0.63 (1.1), `':'` 0.47 (0.9), `' thankfully'` 0.42 (1.0), `' discussions'` 0.40 (0.8), `' pungkasnya'` 0.38 (0.7), `'\n\n\n'` 0.38 (0.5), `' eloquently'` 0.37 (1.1), `' understandably'` 0.37 (0.9), `'  '` 0.29 (0.6), `' storytelling'` 0.28 (0.9), `' insightful'` 0.28 (1.5), `'如果您'` 0.26 (0.7), `' recommendations'` 0.25 (0.9), `'\n\n\n\n'` 0.24 (0.4), `' continued'` 0.23 (1.0), `' Interestingly'` 0.20 (0.8), `'<em>'` 0.18 (0.9), `'https'` 0.18 (0.4), `').'` 0.18 (0.8)

First atom chosen: `'<eos>'` 0.18, `' insightful'` 0.14, `'?'` 0.07, `' recommendations'` 0.05, `' concludes'` 0.04, `' continued'` 0.03, `' understandably'` 0.03, `' comments'` 0.03

Readout top-1 token: `'<eos>'` 0.90, `'\n\n\n'` 0.03, `'<end_of_turn>'` 0.01, `'?”'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01, `'.\\\\'` 0.00, `'.”'` 0.00

### transfer_k16 layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0125 | 0.0123 |
| collusion / discussion / honest | 144 | 0.0120 | 0.0116 |
| collusion / final / colluder | 72 | 0.0141 | 0.0141 |
| collusion / final / honest | 72 | 0.0146 | 0.0141 |
| collusion / private / colluder | 72 | 0.0091 | 0.0090 |
| control / discussion / honest | 288 | 0.0119 | 0.0117 |
| control / final / honest | 144 | 0.0156 | 0.0154 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'?'` 0.69 (1.2), `'<eos>'` 0.64 (0.9), `' pungkasnya'` 0.55 (0.9), `':'` 0.43 (0.8), `' discussions'` 0.39 (0.9), `').'` 0.37 (1.0), `' thankfully'` 0.35 (0.9), `'  '` 0.33 (0.8), `' recommendations'` 0.33 (1.3), `'https'` 0.31 (0.6), `' continued'` 0.28 (1.4), `' eloquently'` 0.28 (1.1), `' insightful'` 0.25 (1.4), `' showcasing'` 0.20 (0.6), `'\n\n\n'` 0.20 (0.5), `' further'` 0.19 (0.7), `' storytelling'` 0.19 (0.8), `' pungkas'` 0.17 (0.5), `'<em>'` 0.17 (0.9), `' detaylı'` 0.16 (0.8)

First atom chosen: `' recommendations'` 0.14, `'?'` 0.10, `' insightful'` 0.08, `' continued'` 0.08, `' concludes'` 0.07, `' comments'` 0.05, `'https'` 0.02, `' detaylı'` 0.02

Readout top-1 token: `'<eos>'` 0.70, `'<end_of_turn>'` 0.12, `'\n\n\n'` 0.04, `'?”'` 0.02, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01

### transfer_k16 layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0123 | 0.0123 |
| collusion / discussion / honest | 144 | 0.0126 | 0.0127 |
| collusion / final / colluder | 72 | 0.0139 | 0.0138 |
| collusion / final / honest | 72 | 0.0144 | 0.0136 |
| collusion / private / colluder | 72 | 0.0097 | 0.0092 |
| control / discussion / honest | 288 | 0.0127 | 0.0128 |
| control / final / honest | 144 | 0.0148 | 0.0146 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.66 (0.9), `' discussions'` 0.56 (0.9), `'?'` 0.54 (1.2), `' pungkasnya'` 0.43 (0.7), `' thankfully'` 0.39 (1.1), `'https'` 0.38 (0.6), `' continued'` 0.37 (1.2), `' recommendations'` 0.29 (1.0), `'\n\n\n'` 0.27 (0.6), `' detaylı'` 0.27 (1.0), `' further'` 0.25 (0.7), `' eloquently'` 0.24 (0.9), `' would'` 0.21 (0.6), `' storytelling'` 0.20 (0.7), `' insightful'` 0.20 (1.0), `'  '` 0.17 (0.9), `' hopefully'` 0.16 (0.4), `' FINALLY'` 0.16 (0.5), `':'` 0.16 (0.7), `'Interestingly'` 0.16 (0.5)

First atom chosen: `' recommendations'` 0.09, `' continued'` 0.07, `' concludes'` 0.07, `' comments'` 0.06, `' insightful'` 0.05, `'<eos>'` 0.05, `' discussions'` 0.04, `' detaylı'` 0.04

Readout top-1 token: `'<eos>'` 0.55, `'<end_of_turn>'` 0.31, `'\n\n\n'` 0.03, `'....'` 0.01, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.00, `'...'` 0.00

### transfer_k16 layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0143 | 0.0141 |
| collusion / discussion / honest | 144 | 0.0144 | 0.0147 |
| collusion / final / colluder | 72 | 0.0148 | 0.0148 |
| collusion / final / honest | 72 | 0.0151 | 0.0146 |
| collusion / private / colluder | 72 | 0.0115 | 0.0115 |
| control / discussion / honest | 288 | 0.0144 | 0.0144 |
| control / final / honest | 144 | 0.0155 | 0.0154 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.66 (1.0), `' discussions'` 0.64 (1.1), `'?'` 0.49 (1.2), `'\n\n\n'` 0.45 (0.8), `' continued'` 0.43 (1.4), `' insightful'` 0.36 (1.2), `' recommendations'` 0.34 (1.2), `' pungkasnya'` 0.31 (0.6), `' further'` 0.30 (0.7), `' thankfully'` 0.29 (1.1), `' diría'` 0.25 (0.7), `' detaylı'` 0.24 (1.0), `' eloquently'` 0.23 (0.9), `' LGBTQ'` 0.21 (0.7), `'https'` 0.20 (0.6), `' hopefully'` 0.20 (0.6), `').'` 0.16 (0.9), `':'` 0.16 (0.7), `' liknande'` 0.15 (0.6), `' policymakers'` 0.14 (0.6)

First atom chosen: `' continued'` 0.14, `' recommendations'` 0.11, `' insightful'` 0.08, `' detaylı'` 0.07, `' discussions'` 0.07, `' concludes'` 0.05, `'<eos>'` 0.03, `' tentunya'` 0.03

Readout top-1 token: `'<eos>'` 0.48, `'<end_of_turn>'` 0.36, `'\n\n\n'` 0.09, `'\n\n\n\n'` 0.01, `'...”'` 0.01, `'\n\n'` 0.00, `' recommendations'` 0.00, `' comments'` 0.00

