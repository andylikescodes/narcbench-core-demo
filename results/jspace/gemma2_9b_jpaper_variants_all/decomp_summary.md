# J-space decomposition summary (paper-faithful split)

Written 2026-11-05 11:24 UTC by `scripts/jspace_decompose_summary.py`.

## core_gp — 1700 activations, k = 25, method gp, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/core/20261001T012639Z/variants/jpaper_gp_k25`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0098 | 0.0097 | 0.0089–0.0105 | 1.00 | 0.0103 | 15.6 | 1.00 | 9.4 | 0.14 |
| 20 | 0.0097 | 0.0097 | 0.0088–0.0107 | 1.00 | 0.0100 | 16.5 | 0.99 | 8.5 | 0.26 |
| 21 | 0.0104 | 0.0104 | 0.0092–0.0116 | 1.00 | 0.0108 | 16.0 | 0.99 | 9.0 | 0.53 |
| 22 | 0.0108 | 0.0109 | 0.0098–0.0120 | 1.00 | 0.0111 | 16.6 | 0.98 | 8.4 | 0.39 |
| 23 | 0.0129 | 0.0128 | 0.0116–0.0141 | 1.00 | 0.0132 | 17.3 | 0.97 | 7.7 | 0.48 |

### core_gp layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0094 | 0.0095 |
| collusion / discussion / honest | 300 | 0.0096 | 0.0095 |
| collusion / final / colluder | 100 | 0.0101 | 0.0098 |
| collusion / final / honest | 100 | 0.0095 | 0.0093 |
| collusion / private / colluder | 100 | 0.0098 | 0.0094 |
| control / discussion / honest | 600 | 0.0101 | 0.0099 |
| control / final / honest | 200 | 0.0098 | 0.0098 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.5), `' decision'` 0.93 (0.9), `' acknowledging'` 0.90 (0.2), `' understandably'` 0.85 (0.8), `' eloquently'` 0.83 (2.0), `' spokesperson'` 0.79 (0.5), `' reaffirm'` 0.58 (0.4), `' prioritizing'` 0.58 (0.6), `' confidently'` 0.55 (0.1), `' confirming'` 0.55 (0.3), `' confirmed'` 0.51 (0.5), `' statement'` 0.46 (0.0), `'?'` 0.43 (0.5), `' undeniably'` 0.43 (0.5), `' confirms'` 0.41 (0.2), `' acknowledges'` 0.41 (0.0), `' bestätigt'` 0.40 (0.4), `' hatır'` 0.39 (0.4), `'.\u202f'` 0.35 (0.9), `' hopeful'` 0.34 (0.5)

First atom chosen: `'<eos>'` 0.36, `' eloquently'` 0.23, `' decision'` 0.22, `' acknowledging'` 0.06, `' confidently'` 0.02, `' confirming'` 0.02, `' spokesperson'` 0.02, `' questioning'` 0.01

Readout top-1 token: `'<eos>'` 1.00

### core_gp layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0085 | 0.0086 |
| collusion / discussion / honest | 300 | 0.0095 | 0.0094 |
| collusion / final / colluder | 100 | 0.0095 | 0.0094 |
| collusion / final / honest | 100 | 0.0102 | 0.0101 |
| collusion / private / colluder | 100 | 0.0097 | 0.0093 |
| control / discussion / honest | 600 | 0.0101 | 0.0100 |
| control / final / honest | 200 | 0.0106 | 0.0106 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.3), `' eloquently'` 0.80 (1.7), `' decision'` 0.79 (0.9), `' prioritizing'` 0.74 (0.9), `' spokesperson'` 0.71 (0.6), `' understandably'` 0.68 (0.7), `' empathetic'` 0.66 (0.9), `' acknowledging'` 0.64 (0.0), `' reaffirm'` 0.60 (0.7), `' bestätigt'` 0.51 (0.7), `' reaffirmed'` 0.49 (0.3), `' confirms'` 0.47 (0.5), `' thankfully'` 0.46 (0.5), `' confidently'` 0.42 (0.2), `'.\u202f'` 0.42 (1.1), `' recommendation'` 0.40 (0.2), `' تاکید'` 0.39 (0.4), `' responding'` 0.38 (0.1), `' confirmed'` 0.36 (0.3), `' recommendations'` 0.34 (0.1)

First atom chosen: `' decision'` 0.25, `' eloquently'` 0.17, `' reaffirm'` 0.10, `' empathetic'` 0.08, `' confidently'` 0.05, `' spokesperson'` 0.04, `'<eos>'` 0.03, `' prioritizing'` 0.03

Readout top-1 token: `'<eos>'` 1.00, `'.”'` 0.00, `' discussions'` 0.00, `'\n\n'` 0.00, `' eloquently'` 0.00, `'?”'` 0.00, `').”'` 0.00

### core_gp layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0088 | 0.0087 |
| collusion / discussion / honest | 300 | 0.0100 | 0.0099 |
| collusion / final / colluder | 100 | 0.0111 | 0.0109 |
| collusion / final / honest | 100 | 0.0122 | 0.0122 |
| collusion / private / colluder | 100 | 0.0101 | 0.0098 |
| control / discussion / honest | 600 | 0.0104 | 0.0103 |
| control / final / honest | 200 | 0.0125 | 0.0122 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.90 (1.4), `'<eos>'` 0.86 (1.0), `' prioritizing'` 0.81 (1.4), `' recommendation'` 0.76 (0.5), `' recommendations'` 0.73 (0.5), `' eloquently'` 0.70 (1.4), `' تاکید'` 0.54 (0.6), `' spokesperson'` 0.54 (0.6), `' diría'` 0.51 (0.4), `' reaffirm'` 0.50 (0.8), `' recommending'` 0.48 (0.1), `' responding'` 0.46 (0.1), `' consensus'` 0.45 (0.4), `' citing'` 0.44 (0.1), `' pungkasnya'` 0.40 (0.5), `' acknowledging'` 0.40 (0.0), `' confirmed'` 0.39 (0.6), `' advocating'` 0.37 (0.3), `' reaffirmed'` 0.37 (0.1), `' empathetic'` 0.35 (0.6)

First atom chosen: `' decision'` 0.36, `' recommendation'` 0.11, `' prioritizing'` 0.06, `' reaffirm'` 0.05, `' recommendations'` 0.04, `' eloquently'` 0.03, `' advocating'` 0.02, `' convincing'` 0.02

Readout top-1 token: `'<eos>'` 0.98, `' discussions'` 0.00, `' unethical'` 0.00, `' decision'` 0.00, `' despite'` 0.00, `'<end_of_turn>'` 0.00, `' suggestions'` 0.00, `' subtly'` 0.00

### core_gp layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0090 | 0.0089 |
| collusion / discussion / honest | 300 | 0.0109 | 0.0109 |
| collusion / final / colluder | 100 | 0.0095 | 0.0095 |
| collusion / final / honest | 100 | 0.0119 | 0.0118 |
| collusion / private / colluder | 100 | 0.0108 | 0.0106 |
| control / discussion / honest | 600 | 0.0113 | 0.0112 |
| control / final / honest | 200 | 0.0120 | 0.0119 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.90 (1.4), `' recommendation'` 0.78 (0.6), `' prioritizing'` 0.76 (1.2), `' recommendations'` 0.73 (0.4), `' diría'` 0.72 (0.5), `' تاکید'` 0.68 (1.0), `'<eos>'` 0.68 (0.8), `' eloquently'` 0.62 (1.1), `' recommending'` 0.56 (0.0), `' thankfully'` 0.46 (0.5), `' spokesperson'` 0.45 (0.7), `' decisions'` 0.44 (0.0), `' advocating'` 0.36 (0.5), `' reaffirm'` 0.35 (0.6), `' despite'` 0.34 (0.9), `' discussions'` 0.34 (0.8), `' consensus'` 0.33 (0.2), `' likely'` 0.33 (0.6), `' leveraging'` 0.31 (0.5), `' teşekkür'` 0.31 (0.4)

First atom chosen: `' decision'` 0.38, `' recommendation'` 0.19, `' prioritizing'` 0.04, `' recommendations'` 0.03, `' تاکید'` 0.03, `' advocating'` 0.03, `' convincing'` 0.03, `' unanimously'` 0.02

Readout top-1 token: `'<eos>'` 0.92, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' discussions'` 0.01, `' despite'` 0.01, `' decision'` 0.00, `' leveraging'` 0.00, `' strategically'` 0.00

### core_gp layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0118 | 0.0117 |
| collusion / discussion / honest | 300 | 0.0125 | 0.0124 |
| collusion / final / colluder | 100 | 0.0124 | 0.0120 |
| collusion / final / honest | 100 | 0.0135 | 0.0134 |
| collusion / private / colluder | 100 | 0.0126 | 0.0127 |
| control / discussion / honest | 600 | 0.0133 | 0.0132 |
| control / final / honest | 200 | 0.0138 | 0.0137 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.83 (1.5), `' prioritizing'` 0.80 (1.4), `' diría'` 0.77 (0.7), `'<eos>'` 0.76 (0.9), `' recommendations'` 0.61 (0.5), `' recommendation'` 0.59 (0.3), `' confirmado'` 0.52 (0.8), `' ethically'` 0.46 (1.2), `' leveraging'` 0.45 (0.7), `' spokesperson'` 0.45 (0.9), `' consensus'` 0.44 (0.5), `' decisions'` 0.39 (0.0), `' discussions'` 0.39 (0.9), `' eloquently'` 0.39 (0.8), `' advocating'` 0.37 (0.8), `' dikutip'` 0.36 (0.4), `' impactful'` 0.35 (0.5), `' emphasizing'` 0.35 (0.5), `' advocacy'` 0.33 (0.2), `' unanimously'` 0.32 (1.3)

First atom chosen: `' decision'` 0.28, `' prioritizing'` 0.08, `' consensus'` 0.07, `' recommendation'` 0.06, `' unanimously'` 0.06, `' advocating'` 0.05, `' recommendations'` 0.04, `' spokesperson'` 0.04

Readout top-1 token: `'<eos>'` 0.89, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' decision'` 0.01, `' recommendations'` 0.01, `' discussions'` 0.01, `' hyö'` 0.00, `' ethically'` 0.00

## transfer_gp — 936 activations, k = 25, method gp, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/transfer/RUNPOD/variants/jpaper_gp_k25`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0124 | 0.0124 | 0.0101–0.0146 | 1.00 | 0.0127 | 17.9 | 0.97 | 7.1 | 0.18 |
| 20 | 0.0126 | 0.0127 | 0.0103–0.0148 | 1.00 | 0.0126 | 18.8 | 0.94 | 6.2 | 0.20 |
| 21 | 0.0132 | 0.0134 | 0.0104–0.0159 | 1.00 | 0.0128 | 18.5 | 0.91 | 6.5 | 0.24 |
| 22 | 0.0135 | 0.0136 | 0.0112–0.0158 | 1.00 | 0.0126 | 19.2 | 0.89 | 5.8 | 0.16 |
| 23 | 0.0148 | 0.0150 | 0.0126–0.0171 | 1.00 | 0.0139 | 20.5 | 0.80 | 4.5 | 0.18 |

### transfer_gp layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0118 | 0.0118 |
| collusion / discussion / honest | 144 | 0.0115 | 0.0116 |
| collusion / final / colluder | 72 | 0.0142 | 0.0142 |
| collusion / final / honest | 72 | 0.0142 | 0.0138 |
| collusion / private / colluder | 72 | 0.0095 | 0.0091 |
| control / discussion / honest | 288 | 0.0116 | 0.0118 |
| control / final / honest | 144 | 0.0153 | 0.0152 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.96 (1.5), `' understandably'` 0.68 (1.0), `'?'` 0.66 (1.0), `':'` 0.51 (0.8), `' thankfully'` 0.47 (0.8), `' eloquently'` 0.47 (1.4), `'\n\n\n'` 0.45 (0.3), `' pungkasnya'` 0.41 (0.5), `'  '` 0.38 (0.4), `'\n\n\n\n'` 0.37 (0.2), `'https'` 0.33 (0.3), `' discussions'` 0.32 (0.6), `'.'` 0.31 (0.2), `"'"` 0.30 (0.6), `' insightful'` 0.29 (1.0), `','` 0.28 (0.6), `' Webinar'` 0.25 (0.5), `' Interestingly'` 0.25 (0.4), `' showcasing'` 0.24 (0.5), `'Continue'` 0.23 (0.0)

First atom chosen: `'<eos>'` 0.53, `'?'` 0.11, `' insightful'` 0.04, `'.'` 0.03, `' understandably'` 0.03, `' continued'` 0.02, `' comments'` 0.02, `':'` 0.02

Readout top-1 token: `'<eos>'` 0.95, `'...”'` 0.01, `'\n\n\n'` 0.00, `' […]'` 0.00, `'\n\n'` 0.00, `'.”'` 0.00, `'?”'` 0.00, `'...'` 0.00

### transfer_gp layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0122 | 0.0120 |
| collusion / discussion / honest | 144 | 0.0121 | 0.0125 |
| collusion / final / colluder | 72 | 0.0133 | 0.0134 |
| collusion / final / honest | 72 | 0.0140 | 0.0135 |
| collusion / private / colluder | 72 | 0.0096 | 0.0091 |
| control / discussion / honest | 288 | 0.0120 | 0.0118 |
| control / final / honest | 144 | 0.0149 | 0.0147 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.93 (1.2), `'?'` 0.66 (1.1), `':'` 0.51 (0.8), `' discussions'` 0.45 (0.7), `' thankfully'` 0.45 (0.9), `' understandably'` 0.44 (0.8), `'\n\n\n'` 0.44 (0.4), `' pungkasnya'` 0.41 (0.7), `' eloquently'` 0.39 (1.1), `'如果您'` 0.35 (0.5), `' insightful'` 0.34 (1.2), `'  '` 0.33 (0.5), `' recommendations'` 0.33 (0.7), `'\n\n\n\n'` 0.33 (0.3), `' storytelling'` 0.32 (0.8), `' continued'` 0.27 (0.9), `' Interestingly'` 0.26 (0.6), `'https'` 0.24 (0.3), `' concludes'` 0.23 (0.2), `' spokesperson'` 0.22 (0.4)

First atom chosen: `'<eos>'` 0.18, `' insightful'` 0.14, `'?'` 0.07, `' recommendations'` 0.05, `' concludes'` 0.04, `' continued'` 0.03, `' understandably'` 0.03, `' comments'` 0.03

Readout top-1 token: `'<eos>'` 0.90, `'\n\n\n'` 0.03, `'<end_of_turn>'` 0.01, `'?”'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01, `'.\\\\'` 0.00, `'.”'` 0.00

### transfer_gp layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0129 | 0.0129 |
| collusion / discussion / honest | 144 | 0.0124 | 0.0119 |
| collusion / final / colluder | 72 | 0.0146 | 0.0144 |
| collusion / final / honest | 72 | 0.0153 | 0.0149 |
| collusion / private / colluder | 72 | 0.0096 | 0.0096 |
| control / discussion / honest | 288 | 0.0123 | 0.0121 |
| control / final / honest | 144 | 0.0163 | 0.0162 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'?'` 0.71 (1.2), `'<eos>'` 0.64 (0.9), `' pungkasnya'` 0.59 (0.9), `':'` 0.46 (0.7), `' discussions'` 0.43 (0.8), `' recommendations'` 0.41 (1.0), `' thankfully'` 0.40 (0.8), `').'` 0.39 (0.9), `' comments'` 0.37 (0.3), `'  '` 0.36 (0.8), `' insightful'` 0.34 (1.0), `'https'` 0.34 (0.5), `' continued'` 0.32 (1.2), `' eloquently'` 0.30 (1.0), `' recommendation'` 0.26 (0.1), `' concludes'` 0.25 (0.3), `' further'` 0.23 (0.6), `' spokesperson'` 0.22 (0.4), `' storytelling'` 0.22 (0.6), `' showcasing'` 0.22 (0.6)

First atom chosen: `' recommendations'` 0.14, `'?'` 0.10, `' insightful'` 0.08, `' continued'` 0.08, `' concludes'` 0.07, `' comments'` 0.05, `'https'` 0.02, `' detaylı'` 0.02

Readout top-1 token: `'<eos>'` 0.70, `'<end_of_turn>'` 0.12, `'\n\n\n'` 0.04, `'?”'` 0.02, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01

### transfer_gp layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0127 | 0.0127 |
| collusion / discussion / honest | 144 | 0.0130 | 0.0133 |
| collusion / final / colluder | 72 | 0.0143 | 0.0142 |
| collusion / final / honest | 72 | 0.0150 | 0.0140 |
| collusion / private / colluder | 72 | 0.0102 | 0.0097 |
| control / discussion / honest | 288 | 0.0133 | 0.0132 |
| control / final / honest | 144 | 0.0154 | 0.0149 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.66 (0.9), `' discussions'` 0.62 (0.8), `'?'` 0.55 (1.2), `' pungkasnya'` 0.47 (0.6), `'https'` 0.42 (0.5), `' thankfully'` 0.41 (1.1), `' continued'` 0.40 (1.1), `' recommendations'` 0.37 (0.8), `' detaylı'` 0.33 (0.8), `' comments'` 0.31 (0.3), `'\n\n\n'` 0.30 (0.5), `' further'` 0.29 (0.7), `' eloquently'` 0.26 (0.8), `' would'` 0.26 (0.5), `' insightful'` 0.24 (0.8), `' storytelling'` 0.23 (0.6), `'Interestingly'` 0.23 (0.3), `' concludes'` 0.22 (0.5), `' recommendation'` 0.19 (0.1), `' Interestingly'` 0.19 (0.5)

First atom chosen: `' recommendations'` 0.09, `' continued'` 0.07, `' concludes'` 0.07, `' comments'` 0.06, `' insightful'` 0.05, `'<eos>'` 0.05, `' discussions'` 0.04, `' detaylı'` 0.04

Readout top-1 token: `'<eos>'` 0.55, `'<end_of_turn>'` 0.31, `'\n\n\n'` 0.03, `'....'` 0.01, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.00, `'...'` 0.00

### transfer_gp layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0146 | 0.0145 |
| collusion / discussion / honest | 144 | 0.0148 | 0.0151 |
| collusion / final / colluder | 72 | 0.0150 | 0.0152 |
| collusion / final / honest | 72 | 0.0156 | 0.0152 |
| collusion / private / colluder | 72 | 0.0119 | 0.0120 |
| control / discussion / honest | 288 | 0.0149 | 0.0147 |
| control / final / honest | 144 | 0.0159 | 0.0158 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' discussions'` 0.70 (1.0), `'<eos>'` 0.66 (1.0), `'?'` 0.50 (1.1), `'\n\n\n'` 0.49 (0.7), `' insightful'` 0.46 (0.9), `' continued'` 0.46 (1.3), `' recommendations'` 0.42 (0.9), `' pungkasnya'` 0.37 (0.5), `' detaylı'` 0.36 (0.7), `' further'` 0.34 (0.6), `' thankfully'` 0.34 (0.9), `' diría'` 0.30 (0.6), `' eloquently'` 0.26 (0.7), `'https'` 0.26 (0.5), `' ausführ'` 0.25 (0.3), `' policymakers'` 0.23 (0.4), `' hopefully'` 0.23 (0.5), `' LGBTQ'` 0.23 (0.7), `' liknande'` 0.21 (0.4), `' tentunya'` 0.20 (0.5)

First atom chosen: `' continued'` 0.14, `' recommendations'` 0.11, `' insightful'` 0.08, `' detaylı'` 0.07, `' discussions'` 0.07, `' concludes'` 0.05, `'<eos>'` 0.03, `' tentunya'` 0.03

Readout top-1 token: `'<eos>'` 0.48, `'<end_of_turn>'` 0.36, `'\n\n\n'` 0.09, `'\n\n\n\n'` 0.01, `'...”'` 0.01, `'\n\n'` 0.00, `' recommendations'` 0.00, `' comments'` 0.00

## core_k8 — 1700 activations, k = 8, method nnomp, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/core/20261001T012639Z/variants/jpaper_nnomp_k8`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0092 | 0.0091 | 0.0083–0.0099 | 1.00 | 0.0086 | 8.0 | 0.00 | 0.0 | 0.31 |
| 20 | 0.0093 | 0.0092 | 0.0083–0.0101 | 1.00 | 0.0083 | 8.0 | 0.00 | 0.0 | 0.48 |
| 21 | 0.0097 | 0.0096 | 0.0086–0.0107 | 1.00 | 0.0085 | 8.0 | 0.00 | 0.0 | 0.70 |
| 22 | 0.0100 | 0.0101 | 0.0091–0.0110 | 1.00 | 0.0087 | 8.0 | 0.00 | 0.0 | 0.51 |
| 23 | 0.0120 | 0.0121 | 0.0109–0.0132 | 1.00 | 0.0101 | 8.0 | 0.00 | 0.0 | 0.60 |

### core_k8 layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0089 | 0.0089 |
| collusion / discussion / honest | 300 | 0.0090 | 0.0089 |
| collusion / final / colluder | 100 | 0.0094 | 0.0092 |
| collusion / final / honest | 100 | 0.0089 | 0.0088 |
| collusion / private / colluder | 100 | 0.0093 | 0.0086 |
| control / discussion / honest | 600 | 0.0095 | 0.0093 |
| control / final / honest | 200 | 0.0093 | 0.0093 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.6), `' eloquently'` 0.74 (2.4), `' decision'` 0.63 (1.5), `' understandably'` 0.50 (1.4), `' prioritizing'` 0.37 (1.1), `' spokesperson'` 0.36 (1.1), `'.\u202f'` 0.35 (1.0), `'?'` 0.27 (0.8), `'.***'` 0.24 (0.8), `' confirmed'` 0.22 (1.3), `' bestätigt'` 0.18 (1.0), `' hatır'` 0.18 (0.8), `' hopeful'` 0.15 (1.0), `' acknowledging'` 0.14 (1.2), `' undeniably'` 0.13 (1.4), `','` 0.12 (0.8), `' confirming'` 0.12 (1.5), `' reaffirm'` 0.12 (1.1), `' 🙏'` 0.11 (0.6), `' despite'` 0.11 (1.1)

First atom chosen: `'<eos>'` 0.36, `' eloquently'` 0.23, `' decision'` 0.22, `' acknowledging'` 0.06, `' confidently'` 0.02, `' confirming'` 0.02, `' spokesperson'` 0.02, `' questioning'` 0.01

Readout top-1 token: `'<eos>'` 1.00

### core_k8 layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0081 | 0.0081 |
| collusion / discussion / honest | 300 | 0.0090 | 0.0090 |
| collusion / final / colluder | 100 | 0.0091 | 0.0090 |
| collusion / final / honest | 100 | 0.0097 | 0.0096 |
| collusion / private / colluder | 100 | 0.0092 | 0.0085 |
| control / discussion / honest | 600 | 0.0096 | 0.0095 |
| control / final / honest | 200 | 0.0102 | 0.0102 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.4), `' eloquently'` 0.67 (2.2), `' prioritizing'` 0.53 (1.4), `' decision'` 0.46 (1.8), `'.\u202f'` 0.41 (1.1), `' understandably'` 0.32 (1.6), `' empathetic'` 0.32 (1.7), `' spokesperson'` 0.30 (1.3), `' bestätigt'` 0.26 (1.4), `'.***'` 0.25 (0.8), `' reaffirm'` 0.24 (1.7), `' thankfully'` 0.20 (1.1), `'?'` 0.17 (1.0), `' confirms'` 0.15 (1.5), `' discussions'` 0.13 (1.2), `' تاکید'` 0.11 (1.1), `' 🙏'` 0.10 (0.8), `' hatır'` 0.10 (1.0), `' despite'` 0.09 (1.6), `' hopeful'` 0.09 (1.3)

First atom chosen: `' decision'` 0.25, `' eloquently'` 0.17, `' reaffirm'` 0.10, `' empathetic'` 0.08, `' confidently'` 0.05, `' spokesperson'` 0.04, `'<eos>'` 0.03, `' prioritizing'` 0.03

Readout top-1 token: `'<eos>'` 1.00, `'.”'` 0.00, `' discussions'` 0.00, `'\n\n'` 0.00, `' eloquently'` 0.00, `'?”'` 0.00, `').”'` 0.00

### core_k8 layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0083 | 0.0082 |
| collusion / discussion / honest | 300 | 0.0091 | 0.0091 |
| collusion / final / colluder | 100 | 0.0105 | 0.0103 |
| collusion / final / honest | 100 | 0.0114 | 0.0114 |
| collusion / private / colluder | 100 | 0.0095 | 0.0092 |
| control / discussion / honest | 600 | 0.0097 | 0.0097 |
| control / final / honest | 200 | 0.0118 | 0.0117 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.86 (1.0), `' prioritizing'` 0.65 (1.9), `' decision'` 0.64 (2.1), `' eloquently'` 0.55 (1.9), `' تاکید'` 0.25 (1.2), `' reaffirm'` 0.24 (1.6), `' pungkasnya'` 0.23 (0.9), `' recommendation'` 0.22 (1.8), `' spokesperson'` 0.21 (1.4), `'?'` 0.20 (0.9), `' diría'` 0.20 (0.8), `' recommendations'` 0.20 (1.4), `' despite'` 0.20 (1.8), `' confirmed'` 0.14 (1.4), `' bestätigt'` 0.13 (1.1), `' empathetic'` 0.12 (1.4), `' understandably'` 0.11 (1.5), `'.\u202f'` 0.11 (1.2), `' thankfully'` 0.09 (1.1), `' subtly'` 0.09 (1.8)

First atom chosen: `' decision'` 0.36, `' recommendation'` 0.11, `' prioritizing'` 0.06, `' reaffirm'` 0.05, `' recommendations'` 0.04, `' eloquently'` 0.03, `' advocating'` 0.02, `' convincing'` 0.02

Readout top-1 token: `'<eos>'` 0.98, `' discussions'` 0.00, `' unethical'` 0.00, `' decision'` 0.00, `' despite'` 0.00, `'<end_of_turn>'` 0.00, `' suggestions'` 0.00, `' subtly'` 0.00

### core_k8 layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0085 | 0.0084 |
| collusion / discussion / honest | 300 | 0.0099 | 0.0099 |
| collusion / final / colluder | 100 | 0.0090 | 0.0091 |
| collusion / final / honest | 100 | 0.0110 | 0.0110 |
| collusion / private / colluder | 100 | 0.0099 | 0.0096 |
| control / discussion / honest | 600 | 0.0105 | 0.0104 |
| control / final / honest | 200 | 0.0113 | 0.0110 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.68 (0.9), `' decision'` 0.63 (2.2), `' prioritizing'` 0.58 (1.8), `' تاکید'` 0.46 (1.5), `' eloquently'` 0.44 (1.7), `' diría'` 0.40 (0.9), `' recommendation'` 0.31 (1.7), `' thankfully'` 0.22 (1.0), `' spokesperson'` 0.22 (1.4), `' despite'` 0.22 (1.5), `' discussions'` 0.18 (1.5), `' likely'` 0.17 (1.0), `' recommendations'` 0.16 (1.3), `' ethically'` 0.16 (1.6), `' reaffirm'` 0.13 (1.5), `' teşekkür'` 0.13 (1.0), `' unanimously'` 0.12 (1.7), `' convincingly'` 0.10 (1.4), `' leveraging'` 0.10 (1.1), `'建议'` 0.10 (1.3)

First atom chosen: `' decision'` 0.38, `' recommendation'` 0.19, `' prioritizing'` 0.04, `' recommendations'` 0.03, `' تاکید'` 0.03, `' advocating'` 0.03, `' convincing'` 0.03, `' unanimously'` 0.02

Readout top-1 token: `'<eos>'` 0.92, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' discussions'` 0.01, `' despite'` 0.01, `' decision'` 0.00, `' leveraging'` 0.00, `' strategically'` 0.00

### core_k8 layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0111 | 0.0111 |
| collusion / discussion / honest | 300 | 0.0116 | 0.0116 |
| collusion / final / colluder | 100 | 0.0117 | 0.0115 |
| collusion / final / honest | 100 | 0.0125 | 0.0122 |
| collusion / private / colluder | 100 | 0.0114 | 0.0115 |
| control / discussion / honest | 600 | 0.0125 | 0.0124 |
| control / final / honest | 200 | 0.0129 | 0.0128 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.76 (1.0), `' prioritizing'` 0.59 (2.1), `' decision'` 0.58 (2.3), `' diría'` 0.44 (1.2), `' confirmado'` 0.31 (1.6), `' ethically'` 0.28 (2.0), `' spokesperson'` 0.23 (1.7), `' discussions'` 0.22 (1.7), `' eloquently'` 0.20 (1.6), `' unanimously'` 0.19 (2.3), `' leveraging'` 0.18 (1.5), `' recommendations'` 0.17 (1.5), `' advocating'` 0.15 (2.1), `' likely'` 0.14 (1.2), `' impactful'` 0.14 (1.2), `' consensus'` 0.13 (1.9), `' تاکید'` 0.12 (1.4), `' subtly'` 0.12 (2.1), `' dikutip'` 0.12 (1.1), `' recommendation'` 0.12 (1.9)

First atom chosen: `' decision'` 0.28, `' prioritizing'` 0.08, `' consensus'` 0.07, `' recommendation'` 0.06, `' unanimously'` 0.06, `' advocating'` 0.05, `' recommendations'` 0.04, `' spokesperson'` 0.04

Readout top-1 token: `'<eos>'` 0.89, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' decision'` 0.01, `' recommendations'` 0.01, `' discussions'` 0.01, `' hyö'` 0.00, `' ethically'` 0.00

## transfer_k8 — 936 activations, k = 8, method nnomp, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/transfer/RUNPOD/variants/jpaper_nnomp_k8`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0117 | 0.0116 | 0.0095–0.0138 | 1.00 | 0.0098 | 8.0 | 0.00 | 0.0 | 0.27 |
| 20 | 0.0119 | 0.0120 | 0.0096–0.0141 | 1.00 | 0.0095 | 8.0 | 0.00 | 0.0 | 0.32 |
| 21 | 0.0125 | 0.0127 | 0.0097–0.0150 | 1.00 | 0.0095 | 8.0 | 0.00 | 0.0 | 0.39 |
| 22 | 0.0127 | 0.0128 | 0.0105–0.0151 | 1.00 | 0.0096 | 8.0 | 0.00 | 0.0 | 0.25 |
| 23 | 0.0141 | 0.0142 | 0.0119–0.0162 | 1.00 | 0.0105 | 8.0 | 0.00 | 0.0 | 0.28 |

### transfer_k8 layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0110 | 0.0110 |
| collusion / discussion / honest | 144 | 0.0108 | 0.0110 |
| collusion / final / colluder | 72 | 0.0137 | 0.0138 |
| collusion / final / honest | 72 | 0.0135 | 0.0134 |
| collusion / private / colluder | 72 | 0.0088 | 0.0083 |
| control / discussion / honest | 288 | 0.0108 | 0.0109 |
| control / final / honest | 144 | 0.0146 | 0.0144 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.95 (1.6), `'?'` 0.47 (1.6), `' understandably'` 0.45 (1.8), `' eloquently'` 0.31 (2.1), `':'` 0.27 (1.5), `' pungkasnya'` 0.24 (0.9), `' thankfully'` 0.23 (1.4), `'<em>'` 0.20 (0.9), `' insightful'` 0.17 (1.9), `'\n\n\n'` 0.16 (0.6), `' discussions'` 0.13 (1.3), `','` 0.13 (1.1), `"'"` 0.13 (1.2), `' Webinar'` 0.12 (1.2), `'’'` 0.12 (0.9), `'https'` 0.11 (0.8), `'  '` 0.11 (0.9), `'，《'` 0.11 (0.9), `' continued'` 0.10 (1.5), `' “'` 0.10 (0.8)

First atom chosen: `'<eos>'` 0.53, `'?'` 0.11, `' insightful'` 0.04, `'.'` 0.03, `' understandably'` 0.03, `' continued'` 0.02, `' comments'` 0.02, `':'` 0.02

Readout top-1 token: `'<eos>'` 0.95, `'...”'` 0.01, `'\n\n\n'` 0.00, `' […]'` 0.00, `'\n\n'` 0.00, `'.”'` 0.00, `'?”'` 0.00, `'...'` 0.00

### transfer_k8 layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0115 | 0.0113 |
| collusion / discussion / honest | 144 | 0.0114 | 0.0119 |
| collusion / final / colluder | 72 | 0.0128 | 0.0128 |
| collusion / final / honest | 72 | 0.0134 | 0.0130 |
| collusion / private / colluder | 72 | 0.0089 | 0.0084 |
| control / discussion / honest | 288 | 0.0113 | 0.0111 |
| control / final / honest | 144 | 0.0142 | 0.0141 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.90 (1.4), `'?'` 0.45 (1.7), `' thankfully'` 0.28 (1.5), `' pungkasnya'` 0.26 (1.2), `':'` 0.26 (1.5), `' eloquently'` 0.23 (1.8), `' insightful'` 0.23 (2.3), `' understandably'` 0.22 (1.6), `'\n\n\n'` 0.22 (0.8), `' discussions'` 0.20 (1.5), `'<em>'` 0.18 (1.0), `' storytelling'` 0.14 (1.6), `' recommendations'` 0.14 (1.6), `' continued'` 0.13 (1.7), `'  '` 0.13 (1.0), `').'` 0.11 (1.2), `'如果您'` 0.10 (1.4), `'\n\n\n\n'` 0.10 (0.7), `' Interestingly'` 0.10 (1.4), `'.\u202f'` 0.09 (1.1)

First atom chosen: `'<eos>'` 0.18, `' insightful'` 0.14, `'?'` 0.07, `' recommendations'` 0.05, `' concludes'` 0.04, `' continued'` 0.03, `' understandably'` 0.03, `' comments'` 0.03

Readout top-1 token: `'<eos>'` 0.90, `'\n\n\n'` 0.03, `'<end_of_turn>'` 0.01, `'?”'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01, `'.\\\\'` 0.00, `'.”'` 0.00

### transfer_k8 layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0122 | 0.0122 |
| collusion / discussion / honest | 144 | 0.0117 | 0.0113 |
| collusion / final / colluder | 72 | 0.0139 | 0.0139 |
| collusion / final / honest | 72 | 0.0144 | 0.0139 |
| collusion / private / colluder | 72 | 0.0090 | 0.0089 |
| control / discussion / honest | 288 | 0.0116 | 0.0115 |
| control / final / honest | 144 | 0.0153 | 0.0152 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.60 (1.0), `'?'` 0.53 (1.8), `' pungkasnya'` 0.40 (1.4), `').'` 0.29 (1.3), `' thankfully'` 0.24 (1.4), `' recommendations'` 0.23 (2.1), `'  '` 0.23 (1.2), `' continued'` 0.21 (2.1), `':'` 0.20 (1.4), `' discussions'` 0.19 (1.7), `' insightful'` 0.18 (2.2), `' eloquently'` 0.18 (1.8), `'https'` 0.17 (1.0), `'<em>'` 0.16 (1.0), `'\n\n\n'` 0.14 (0.7), `' detaylı'` 0.11 (1.5), `' storytelling'` 0.10 (1.5), `' showcasing'` 0.09 (1.2), `' pungkas'` 0.09 (1.1), `' comments'` 0.08 (1.6)

First atom chosen: `' recommendations'` 0.14, `'?'` 0.10, `' insightful'` 0.08, `' continued'` 0.08, `' concludes'` 0.07, `' comments'` 0.05, `'https'` 0.02, `' detaylı'` 0.02

Readout top-1 token: `'<eos>'` 0.70, `'<end_of_turn>'` 0.12, `'\n\n\n'` 0.04, `'?”'` 0.02, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01

### transfer_k8 layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0121 | 0.0120 |
| collusion / discussion / honest | 144 | 0.0123 | 0.0125 |
| collusion / final / colluder | 72 | 0.0136 | 0.0136 |
| collusion / final / honest | 72 | 0.0142 | 0.0134 |
| collusion / private / colluder | 72 | 0.0095 | 0.0092 |
| control / discussion / honest | 288 | 0.0125 | 0.0125 |
| control / final / honest | 144 | 0.0146 | 0.0143 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.63 (1.1), `'?'` 0.46 (1.6), `' discussions'` 0.31 (1.5), `' thankfully'` 0.29 (1.6), `' pungkasnya'` 0.26 (1.2), `'https'` 0.23 (1.0), `' continued'` 0.23 (2.0), `' recommendations'` 0.20 (1.7), `'\n\n\n'` 0.19 (0.8), `' detaylı'` 0.18 (1.6), `' eloquently'` 0.14 (1.5), `' further'` 0.14 (1.3), `'  '` 0.12 (1.2), `' insightful'` 0.12 (1.9), `' storytelling'` 0.10 (1.4), `').'` 0.10 (1.3), `' would'` 0.08 (1.1), `' comments'` 0.08 (1.4), `' webinar'` 0.08 (1.4), `' concludes'` 0.08 (1.6)

First atom chosen: `' recommendations'` 0.09, `' continued'` 0.07, `' concludes'` 0.07, `' comments'` 0.06, `' insightful'` 0.05, `'<eos>'` 0.05, `' discussions'` 0.04, `' detaylı'` 0.04

Readout top-1 token: `'<eos>'` 0.55, `'<end_of_turn>'` 0.31, `'\n\n\n'` 0.03, `'....'` 0.01, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.00, `'...'` 0.00

### transfer_k8 layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0139 | 0.0138 |
| collusion / discussion / honest | 144 | 0.0140 | 0.0142 |
| collusion / final / colluder | 72 | 0.0145 | 0.0146 |
| collusion / final / honest | 72 | 0.0148 | 0.0144 |
| collusion / private / colluder | 72 | 0.0112 | 0.0113 |
| control / discussion / honest | 288 | 0.0140 | 0.0140 |
| control / final / honest | 144 | 0.0152 | 0.0150 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.61 (1.1), `' discussions'` 0.39 (1.8), `'?'` 0.39 (1.5), `' continued'` 0.34 (2.1), `'\n\n\n'` 0.32 (1.1), `' insightful'` 0.26 (2.0), `' recommendations'` 0.23 (1.9), `' thankfully'` 0.21 (1.6), `' detaylı'` 0.18 (1.8), `' diría'` 0.14 (1.3), `' pungkasnya'` 0.14 (1.2), `'.\u202f'` 0.13 (1.5), `' LGBTQ'` 0.13 (1.2), `').'` 0.12 (1.2), `'https'` 0.12 (1.0), `' further'` 0.12 (1.3), `' eloquently'` 0.11 (1.7), `' tentunya'` 0.09 (1.5), `' hopefully'` 0.08 (1.0), `':'` 0.08 (1.3)

First atom chosen: `' continued'` 0.14, `' recommendations'` 0.11, `' insightful'` 0.08, `' detaylı'` 0.07, `' discussions'` 0.07, `' concludes'` 0.05, `'<eos>'` 0.03, `' tentunya'` 0.03

Readout top-1 token: `'<eos>'` 0.48, `'<end_of_turn>'` 0.36, `'\n\n\n'` 0.09, `'\n\n\n\n'` 0.01, `'...”'` 0.01, `'\n\n'` 0.00, `' recommendations'` 0.00, `' comments'` 0.00

## core_nofill — 1700 activations, k = 25, method nnomp, select unit, candidates 1024, fill False

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/core/20261001T012639Z/variants/jpaper_nnomp_k25_nofill`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0093 | 0.0092 | 0.0084–0.0100 | 1.00 | 0.0092 | 16.4 | 1.00 | 0.0 | 0.21 |
| 20 | 0.0094 | 0.0093 | 0.0084–0.0102 | 1.00 | 0.0092 | 17.3 | 0.99 | 0.0 | 0.33 |
| 21 | 0.0098 | 0.0097 | 0.0087–0.0108 | 1.00 | 0.0096 | 16.6 | 0.99 | 0.0 | 0.56 |
| 22 | 0.0101 | 0.0102 | 0.0092–0.0111 | 1.00 | 0.0099 | 17.0 | 0.98 | 0.0 | 0.39 |
| 23 | 0.0122 | 0.0122 | 0.0110–0.0134 | 1.00 | 0.0119 | 17.7 | 0.97 | 0.0 | 0.46 |

### core_nofill layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0089 | 0.0090 |
| collusion / discussion / honest | 300 | 0.0090 | 0.0090 |
| collusion / final / colluder | 100 | 0.0095 | 0.0093 |
| collusion / final / honest | 100 | 0.0090 | 0.0089 |
| collusion / private / colluder | 100 | 0.0094 | 0.0088 |
| control / discussion / honest | 600 | 0.0095 | 0.0094 |
| control / final / honest | 200 | 0.0094 | 0.0093 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.5), `' decision'` 0.85 (1.0), `' eloquently'` 0.84 (2.0), `' understandably'` 0.83 (0.8), `' spokesperson'` 0.68 (0.6), `' prioritizing'` 0.59 (0.6), `'?'` 0.44 (0.5), `' reaffirm'` 0.44 (0.5), `' bestätigt'` 0.42 (0.4), `' confirmed'` 0.42 (0.6), `' hatır'` 0.39 (0.4), `' acknowledging'` 0.39 (0.5), `' undeniably'` 0.39 (0.6), `'.\u202f'` 0.35 (0.9), `' hopeful'` 0.34 (0.5), `' 🙏'` 0.33 (0.3), `' teşekkür'` 0.32 (0.3), `' confirming'` 0.30 (0.6), `' thankfully'` 0.30 (0.3), `'.***'` 0.26 (0.6)

First atom chosen: `'<eos>'` 0.36, `' eloquently'` 0.23, `' decision'` 0.22, `' acknowledging'` 0.06, `' confidently'` 0.02, `' confirming'` 0.02, `' spokesperson'` 0.02, `' questioning'` 0.01

Readout top-1 token: `'<eos>'` 1.00

### core_nofill layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0082 | 0.0082 |
| collusion / discussion / honest | 300 | 0.0091 | 0.0091 |
| collusion / final / colluder | 100 | 0.0092 | 0.0091 |
| collusion / final / honest | 100 | 0.0098 | 0.0098 |
| collusion / private / colluder | 100 | 0.0093 | 0.0086 |
| control / discussion / honest | 600 | 0.0097 | 0.0096 |
| control / final / honest | 200 | 0.0103 | 0.0103 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 1.00 (1.3), `' eloquently'` 0.81 (1.6), `' prioritizing'` 0.76 (0.9), `' decision'` 0.71 (1.0), `' understandably'` 0.65 (0.8), `' empathetic'` 0.63 (0.9), `' spokesperson'` 0.60 (0.7), `' bestätigt'` 0.53 (0.6), `' reaffirm'` 0.49 (0.9), `' thankfully'` 0.46 (0.5), `'.\u202f'` 0.42 (1.1), `' confirms'` 0.40 (0.6), `' تاکید'` 0.40 (0.4), `' 🙏'` 0.32 (0.3), `'?'` 0.32 (0.6), `' hatır'` 0.30 (0.4), `' discussions'` 0.28 (0.6), `' confirmed'` 0.28 (0.4), `'.***'` 0.27 (0.7), `' impactful'` 0.27 (0.3)

First atom chosen: `' decision'` 0.25, `' eloquently'` 0.17, `' reaffirm'` 0.10, `' empathetic'` 0.08, `' confidently'` 0.05, `' spokesperson'` 0.04, `'<eos>'` 0.03, `' prioritizing'` 0.03

Readout top-1 token: `'<eos>'` 1.00, `'.”'` 0.00, `' discussions'` 0.00, `'\n\n'` 0.00, `' eloquently'` 0.00, `'?”'` 0.00, `').”'` 0.00

### core_nofill layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0084 | 0.0083 |
| collusion / discussion / honest | 300 | 0.0092 | 0.0092 |
| collusion / final / colluder | 100 | 0.0106 | 0.0105 |
| collusion / final / honest | 100 | 0.0115 | 0.0116 |
| collusion / private / colluder | 100 | 0.0096 | 0.0093 |
| control / discussion / honest | 600 | 0.0098 | 0.0097 |
| control / final / honest | 200 | 0.0119 | 0.0118 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.86 (1.0), `' decision'` 0.83 (1.5), `' prioritizing'` 0.82 (1.3), `' eloquently'` 0.71 (1.4), `' تاکید'` 0.57 (0.6), `' diría'` 0.55 (0.4), `' spokesperson'` 0.46 (0.7), `' reaffirm'` 0.45 (0.9), `' recommendations'` 0.43 (0.8), `' pungkasnya'` 0.41 (0.5), `' recommendation'` 0.41 (0.8), `' confirmed'` 0.34 (0.6), `' despite'` 0.33 (1.1), `' empathetic'` 0.30 (0.7), `' dikutip'` 0.30 (0.3), `'?'` 0.29 (0.7), `' bestätigt'` 0.29 (0.4), `' impactful'` 0.25 (0.3), `' thankfully'` 0.25 (0.4), `' likely'` 0.23 (0.4)

First atom chosen: `' decision'` 0.36, `' recommendation'` 0.11, `' prioritizing'` 0.06, `' reaffirm'` 0.05, `' recommendations'` 0.04, `' eloquently'` 0.03, `' advocating'` 0.02, `' convincing'` 0.02

Readout top-1 token: `'<eos>'` 0.98, `' discussions'` 0.00, `' unethical'` 0.00, `' decision'` 0.00, `' despite'` 0.00, `'<end_of_turn>'` 0.00, `' suggestions'` 0.00, `' subtly'` 0.00

### core_nofill layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0086 | 0.0085 |
| collusion / discussion / honest | 300 | 0.0100 | 0.0100 |
| collusion / final / colluder | 100 | 0.0091 | 0.0091 |
| collusion / final / honest | 100 | 0.0111 | 0.0111 |
| collusion / private / colluder | 100 | 0.0100 | 0.0097 |
| control / discussion / honest | 600 | 0.0106 | 0.0105 |
| control / final / honest | 200 | 0.0114 | 0.0112 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' decision'` 0.83 (1.5), `' prioritizing'` 0.77 (1.2), `' diría'` 0.74 (0.5), `' تاکید'` 0.68 (1.0), `'<eos>'` 0.68 (0.8), `' eloquently'` 0.62 (1.1), `' recommendation'` 0.51 (0.9), `' thankfully'` 0.49 (0.5), `' spokesperson'` 0.43 (0.7), `' recommendations'` 0.36 (0.7), `' despite'` 0.35 (0.9), `' likely'` 0.33 (0.6), `' discussions'` 0.33 (0.8), `' teşekkür'` 0.31 (0.4), `' leveraging'` 0.31 (0.5), `' reaffirm'` 0.30 (0.7), `' ethically'` 0.28 (0.9), `' convincingly'` 0.26 (0.6), `' confirms'` 0.24 (0.5), `'REJECT'` 0.23 (0.4)

First atom chosen: `' decision'` 0.38, `' recommendation'` 0.19, `' prioritizing'` 0.04, `' recommendations'` 0.03, `' تاکید'` 0.03, `' advocating'` 0.03, `' convincing'` 0.03, `' unanimously'` 0.02

Readout top-1 token: `'<eos>'` 0.92, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' discussions'` 0.01, `' despite'` 0.01, `' decision'` 0.00, `' leveraging'` 0.00, `' strategically'` 0.00

### core_nofill layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 300 | 0.0112 | 0.0112 |
| collusion / discussion / honest | 300 | 0.0118 | 0.0117 |
| collusion / final / colluder | 100 | 0.0118 | 0.0117 |
| collusion / final / honest | 100 | 0.0126 | 0.0123 |
| collusion / private / colluder | 100 | 0.0116 | 0.0116 |
| control / discussion / honest | 600 | 0.0126 | 0.0126 |
| control / final / honest | 200 | 0.0130 | 0.0129 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' prioritizing'` 0.80 (1.4), `' diría'` 0.79 (0.7), `' decision'` 0.78 (1.6), `'<eos>'` 0.76 (0.9), `' confirmado'` 0.52 (0.8), `' ethically'` 0.45 (1.2), `' spokesperson'` 0.45 (0.9), `' leveraging'` 0.43 (0.7), `' recommendations'` 0.42 (0.7), `' discussions'` 0.39 (0.9), `' dikutip'` 0.37 (0.4), `' eloquently'` 0.37 (0.8), `' impactful'` 0.36 (0.5), `' تاکید'` 0.31 (0.6), `' unanimously'` 0.31 (1.4), `' convincingly'` 0.28 (0.8), `' advocating'` 0.27 (1.0), `' likely'` 0.26 (0.7), `' consensus'` 0.26 (0.9), `' consistent'` 0.26 (0.7)

First atom chosen: `' decision'` 0.28, `' prioritizing'` 0.08, `' consensus'` 0.07, `' recommendation'` 0.06, `' unanimously'` 0.06, `' advocating'` 0.05, `' recommendations'` 0.04, `' spokesperson'` 0.04

Readout top-1 token: `'<eos>'` 0.89, `'<end_of_turn>'` 0.04, `' unethical'` 0.01, `' decision'` 0.01, `' recommendations'` 0.01, `' discussions'` 0.01, `' hyö'` 0.00, `' ethically'` 0.00

## transfer_nofill — 936 activations, k = 25, method nnomp, select unit, candidates 1024, fill False

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/transfer/RUNPOD/variants/jpaper_nnomp_k25_nofill`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0119 | 0.0118 | 0.0096–0.0140 | 1.00 | 0.0115 | 18.5 | 0.95 | 0.0 | 0.20 |
| 20 | 0.0121 | 0.0123 | 0.0099–0.0144 | 1.00 | 0.0115 | 19.4 | 0.92 | 0.0 | 0.22 |
| 21 | 0.0127 | 0.0129 | 0.0099–0.0153 | 1.00 | 0.0116 | 19.1 | 0.90 | 0.0 | 0.27 |
| 22 | 0.0130 | 0.0131 | 0.0107–0.0154 | 1.00 | 0.0116 | 19.8 | 0.89 | 0.0 | 0.17 |
| 23 | 0.0144 | 0.0146 | 0.0122–0.0166 | 1.00 | 0.0130 | 21.0 | 0.77 | 0.0 | 0.18 |

### transfer_nofill layer 19: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0112 | 0.0112 |
| collusion / discussion / honest | 144 | 0.0110 | 0.0113 |
| collusion / final / colluder | 72 | 0.0138 | 0.0139 |
| collusion / final / honest | 72 | 0.0136 | 0.0137 |
| collusion / private / colluder | 72 | 0.0089 | 0.0084 |
| control / discussion / honest | 288 | 0.0110 | 0.0111 |
| control / final / honest | 144 | 0.0147 | 0.0146 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.96 (1.5), `' understandably'` 0.68 (1.0), `'?'` 0.65 (1.0), `':'` 0.48 (0.8), `' eloquently'` 0.48 (1.3), `' thankfully'` 0.47 (0.8), `'\n\n\n'` 0.42 (0.3), `' pungkasnya'` 0.40 (0.5), `'  '` 0.36 (0.4), `' discussions'` 0.33 (0.6), `'\n\n\n\n'` 0.33 (0.2), `"'"` 0.32 (0.6), `'https'` 0.29 (0.3), `','` 0.28 (0.6), `' showcasing'` 0.25 (0.5), `' Webinar'` 0.25 (0.6), `' insightful'` 0.23 (1.2), `'’'` 0.21 (0.5), `'<em>'` 0.21 (0.8), `'，《'` 0.20 (0.5)

First atom chosen: `'<eos>'` 0.53, `'?'` 0.11, `' insightful'` 0.04, `'.'` 0.03, `' understandably'` 0.03, `' continued'` 0.02, `' comments'` 0.02, `':'` 0.02

Readout top-1 token: `'<eos>'` 0.95, `'...”'` 0.01, `'\n\n\n'` 0.00, `' […]'` 0.00, `'\n\n'` 0.00, `'.”'` 0.00, `'?”'` 0.00, `'...'` 0.00

### transfer_nofill layer 20: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0118 | 0.0116 |
| collusion / discussion / honest | 144 | 0.0116 | 0.0121 |
| collusion / final / colluder | 72 | 0.0130 | 0.0130 |
| collusion / final / honest | 72 | 0.0135 | 0.0132 |
| collusion / private / colluder | 72 | 0.0091 | 0.0086 |
| control / discussion / honest | 288 | 0.0116 | 0.0113 |
| control / final / honest | 144 | 0.0144 | 0.0143 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.94 (1.2), `'?'` 0.67 (1.0), `':'` 0.52 (0.8), `' thankfully'` 0.46 (0.9), `' discussions'` 0.46 (0.7), `'\n\n\n'` 0.43 (0.4), `' understandably'` 0.42 (0.8), `' pungkasnya'` 0.42 (0.6), `' eloquently'` 0.41 (1.0), `'\n\n\n\n'` 0.33 (0.3), `'  '` 0.33 (0.5), `'如果您'` 0.32 (0.5), `' storytelling'` 0.32 (0.8), `' insightful'` 0.30 (1.4), `' recommendations'` 0.29 (0.8), `' continued'` 0.27 (0.9), `' Interestingly'` 0.24 (0.6), `'https'` 0.23 (0.3), `').'` 0.20 (0.7), `'<em>'` 0.18 (0.9)

First atom chosen: `'<eos>'` 0.18, `' insightful'` 0.14, `'?'` 0.07, `' recommendations'` 0.05, `' concludes'` 0.04, `' continued'` 0.03, `' understandably'` 0.03, `' comments'` 0.03

Readout top-1 token: `'<eos>'` 0.90, `'\n\n\n'` 0.03, `'<end_of_turn>'` 0.01, `'?”'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01, `'.\\\\'` 0.00, `'.”'` 0.00

### transfer_nofill layer 21: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0125 | 0.0123 |
| collusion / discussion / honest | 144 | 0.0119 | 0.0116 |
| collusion / final / colluder | 72 | 0.0141 | 0.0141 |
| collusion / final / honest | 72 | 0.0146 | 0.0141 |
| collusion / private / colluder | 72 | 0.0091 | 0.0090 |
| control / discussion / honest | 288 | 0.0119 | 0.0117 |
| control / final / honest | 144 | 0.0155 | 0.0153 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'?'` 0.71 (1.2), `'<eos>'` 0.64 (0.9), `' pungkasnya'` 0.59 (0.9), `':'` 0.47 (0.7), `' discussions'` 0.44 (0.8), `' thankfully'` 0.39 (0.8), `').'` 0.39 (0.9), `'  '` 0.36 (0.8), `' recommendations'` 0.36 (1.2), `'https'` 0.35 (0.5), `' eloquently'` 0.32 (1.0), `' continued'` 0.31 (1.2), `' insightful'` 0.28 (1.2), `' further'` 0.24 (0.5), `' pungkas'` 0.23 (0.4), `' showcasing'` 0.23 (0.6), `' storytelling'` 0.21 (0.7), `'\n\n\n'` 0.21 (0.5), `' spokesperson'` 0.19 (0.5), `' impactful'` 0.19 (0.5)

First atom chosen: `' recommendations'` 0.14, `'?'` 0.10, `' insightful'` 0.08, `' continued'` 0.08, `' concludes'` 0.07, `' comments'` 0.05, `'https'` 0.02, `' detaylı'` 0.02

Readout top-1 token: `'<eos>'` 0.70, `'<end_of_turn>'` 0.12, `'\n\n\n'` 0.04, `'?”'` 0.02, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.01, `'\n\n'` 0.01

### transfer_nofill layer 22: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0124 | 0.0123 |
| collusion / discussion / honest | 144 | 0.0126 | 0.0127 |
| collusion / final / colluder | 72 | 0.0138 | 0.0138 |
| collusion / final / honest | 72 | 0.0144 | 0.0135 |
| collusion / private / colluder | 72 | 0.0097 | 0.0093 |
| control / discussion / honest | 288 | 0.0128 | 0.0128 |
| control / final / honest | 144 | 0.0148 | 0.0145 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`'<eos>'` 0.66 (0.9), `' discussions'` 0.62 (0.8), `'?'` 0.55 (1.2), `' pungkasnya'` 0.48 (0.6), `'https'` 0.43 (0.5), `' thankfully'` 0.41 (1.1), `' continued'` 0.40 (1.1), `' recommendations'` 0.32 (0.9), `' detaylı'` 0.32 (0.8), `'\n\n\n'` 0.29 (0.5), `' further'` 0.28 (0.7), `' eloquently'` 0.27 (0.8), `' would'` 0.26 (0.5), `' storytelling'` 0.23 (0.6), `' insightful'` 0.22 (0.9), `'Interestingly'` 0.20 (0.3), `' FINALLY'` 0.19 (0.4), `' hopefully'` 0.18 (0.4), `' Interestingly'` 0.18 (0.6), `':'` 0.18 (0.6)

First atom chosen: `' recommendations'` 0.09, `' continued'` 0.07, `' concludes'` 0.07, `' comments'` 0.06, `' insightful'` 0.05, `'<eos>'` 0.05, `' discussions'` 0.04, `' detaylı'` 0.04

Readout top-1 token: `'<eos>'` 0.55, `'<end_of_turn>'` 0.31, `'\n\n\n'` 0.03, `'....'` 0.01, `'https'` 0.01, `'.\\\\'` 0.01, `'\n\n\n\n'` 0.00, `'...'` 0.00

### transfer_nofill layer 23: FVE_J by mode / phase / role

| group | n | mean | median |
|---|---|---|---|
| collusion / discussion / colluder | 144 | 0.0143 | 0.0141 |
| collusion / discussion / honest | 144 | 0.0144 | 0.0148 |
| collusion / final / colluder | 72 | 0.0148 | 0.0148 |
| collusion / final / honest | 72 | 0.0151 | 0.0146 |
| collusion / private / colluder | 72 | 0.0115 | 0.0115 |
| control / discussion / honest | 288 | 0.0144 | 0.0144 |
| control / final / honest | 144 | 0.0154 | 0.0154 |

Most chosen atoms (share of activations choosing the token; mean nonnegative coefficient):

`' discussions'` 0.71 (0.9), `'<eos>'` 0.66 (0.9), `'?'` 0.51 (1.1), `'\n\n\n'` 0.49 (0.7), `' continued'` 0.46 (1.3), `' insightful'` 0.43 (1.0), `' recommendations'` 0.39 (1.0), `' pungkasnya'` 0.38 (0.5), `' further'` 0.35 (0.6), `' thankfully'` 0.33 (0.9), `' diría'` 0.30 (0.6), `' detaylı'` 0.29 (0.8), `' eloquently'` 0.29 (0.7), `'https'` 0.25 (0.5), `' hopefully'` 0.24 (0.5), `' policymakers'` 0.23 (0.4), `' LGBTQ'` 0.23 (0.6), `' liknande'` 0.21 (0.4), `'articolo'` 0.20 (0.3), `' mentions'` 0.18 (0.5)

First atom chosen: `' continued'` 0.14, `' recommendations'` 0.11, `' insightful'` 0.08, `' detaylı'` 0.07, `' discussions'` 0.07, `' concludes'` 0.05, `'<eos>'` 0.03, `' tentunya'` 0.03

Readout top-1 token: `'<eos>'` 0.48, `'<end_of_turn>'` 0.36, `'\n\n\n'` 0.09, `'\n\n\n\n'` 0.01, `'...”'` 0.01, `'\n\n'` 0.00, `' recommendations'` 0.00, `' comments'` 0.00

