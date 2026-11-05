# J-space decomposition summary (paper-faithful split)

Written 2026-11-05 08:06 UTC by `scripts/jspace_decompose_summary.py`.

## core — 1700 activations, k = 25, method nnomp, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/core/20261001T012639Z/variants/jpaper_nnomp_k25`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0098 | 0.0097 | 0.0089–0.0105 | 1.00 | 0.0103 | 16.4 | 1.00 | 8.6 | 0.14 |
| 20 | 0.0097 | 0.0096 | 0.0088–0.0106 | 1.00 | 0.0100 | 17.3 | 0.99 | 7.7 | 0.27 |
| 21 | 0.0104 | 0.0103 | 0.0092–0.0116 | 1.00 | 0.0108 | 16.6 | 0.99 | 8.4 | 0.53 |
| 22 | 0.0108 | 0.0109 | 0.0098–0.0120 | 1.00 | 0.0111 | 17.0 | 0.98 | 8.0 | 0.39 |
| 23 | 0.0128 | 0.0128 | 0.0116–0.0141 | 1.00 | 0.0132 | 17.7 | 0.97 | 7.3 | 0.47 |

### core layer 19: FVE_J by mode / phase / role

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

### core layer 20: FVE_J by mode / phase / role

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

### core layer 21: FVE_J by mode / phase / role

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

### core layer 22: FVE_J by mode / phase / role

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

### core layer 23: FVE_J by mode / phase / role

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

## transfer — 936 activations, k = 25, method nnomp, select unit, candidates 1024, fill True

`/Users/dehualiang/Documents/claude_projects/narcbench-core-demo/data/activations/gemma2_9b/transfer/RUNPOD/variants/jpaper_nnomp_k25`

| layer | FVE_J mean | median | q25–q75 | share FVE_J < 0.10 | shuffled FVE mean | steps mean | early stop | filled mean | chosen ∩ readout top-25 |
|---|---|---|---|---|---|---|---|---|---|
| 19 | 0.0124 | 0.0123 | 0.0101–0.0145 | 1.00 | 0.0127 | 18.5 | 0.95 | 6.5 | 0.18 |
| 20 | 0.0125 | 0.0127 | 0.0103–0.0148 | 1.00 | 0.0126 | 19.4 | 0.92 | 5.6 | 0.20 |
| 21 | 0.0132 | 0.0134 | 0.0104–0.0159 | 1.00 | 0.0127 | 19.1 | 0.90 | 5.9 | 0.25 |
| 22 | 0.0134 | 0.0136 | 0.0111–0.0158 | 1.00 | 0.0126 | 19.8 | 0.89 | 5.2 | 0.16 |
| 23 | 0.0148 | 0.0150 | 0.0126–0.0170 | 1.00 | 0.0138 | 21.0 | 0.77 | 4.0 | 0.18 |

### transfer layer 19: FVE_J by mode / phase / role

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

### transfer layer 20: FVE_J by mode / phase / role

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

### transfer layer 21: FVE_J by mode / phase / role

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

### transfer layer 22: FVE_J by mode / phase / role

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

### transfer layer 23: FVE_J by mode / phase / role

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

