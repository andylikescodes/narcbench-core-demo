# Stage 2: probes on recorded episodes

**Dry run: every label was shuffled inside its matched set before anything was fit. No number below says anything about the experiment; 0.5 is expected everywhere.**

Every number is a matched-prompt AUROC: over all pairs of units with word-for-word identical prompts and different labels, the share in which the probe scores the unit of class 1 higher (0.5 = chance; ties count half). Brackets: 95% interval from resampled scenarios. Unless a layer is named, a score is the mean over the layers [19, 20, 21, 22, 23]. Probes are fit on other domains than the one they are tested on.

Positions. `m2.pre` is the token just before room message 2 starts, `m2.t0` its first token, `m2.q1`, `m2.q2`, `m2.q3` the tokens at 25%, 50% and 75% of it, `m2.end` its last token. Messages 1 and 3 are written by the member who opens the room, 2 and 4 by the one who answers; at a room position the unit is the member who writes there. `p1.pre` and `p1.end` are the token before a member's first public statement and its last token (`p2`, `p3` likewise); `v.pre` and `v.end` the same for the vote.

Text baselines. *Words so far*: a bag of words (TF-IDF, logistic regression) of everything said up to that token. *Words by speaker*: the same with every word marked as the unit's own or someone else's, because an activation belongs to one member and a fair reader of the words has to know which of them are that member's. *Neutral reader*: the same model reads the words alone, without the prompt, and a probe of the same kind is fit on its activations. "More than the words" means above all three.

Units: 976 members and 488 rooms of ['pilot3_I1', 'pilot3_T7', 'pilot4_T7', 'pilot5_K1', 'pilot5_K2']; fitted on 880 members, 96 members of the instructed control are only scored.

## vote: votes for the stake option (1) against votes for the other option (0)

Unit: the member. No coder is involved.

| Position | Units 1 / 0 | Pairs | Activation | Identical-prompt mean removed | Words so far | Words by speaker | Neutral reader | Activation minus words | Activation minus words by speaker | Activation minus neutral reader |
|---|---|---|---|---|---|---|---|---|---|---|
| `m1.pre` | 257 / 183 | 139 | 0.468 [0.390, 0.552] | – | 0.500 [0.500, 0.500] | – | – | -0.032 [-0.110, +0.052] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.484 [0.411, 0.559] | – | 0.460 [0.428, 0.500] | – | – | +0.024 [-0.063, +0.107] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.527 [0.461, 0.591] | – | 0.536 [0.446, 0.625] | – | – | -0.009 [-0.115, +0.093] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.426 [0.325, 0.520] | – | 0.518 [0.386, 0.626] | – | – | -0.092 [-0.195, +0.024] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.436 [0.350, 0.526] | – | 0.489 [0.389, 0.588] | – | – | -0.053 [-0.179, +0.084] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.496 [0.408, 0.594] | – | 0.554 [0.468, 0.632] | – | – | -0.058 [-0.170, +0.065] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.499 [0.429, 0.568] | – | 0.567 [0.456, 0.679] | – | – | -0.068 [-0.188, +0.051] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.479 [0.405, 0.550] | – | 0.567 [0.451, 0.683] | – | – | -0.088 [-0.211, +0.042] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.567 [0.507, 0.634] | – | 0.518 [0.446, 0.607] | – | – | +0.050 [-0.050, +0.137] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.509 [0.428, 0.581] | – | 0.546 [0.466, 0.626] | – | – | -0.037 [-0.155, +0.058] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.460 [0.379, 0.545] | – | 0.496 [0.394, 0.609] | – | – | -0.037 [-0.157, +0.075] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.502 [0.415, 0.583] | – | 0.532 [0.435, 0.631] | – | – | -0.030 [-0.149, +0.082] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.501 [0.396, 0.602] | – | 0.561 [0.456, 0.656] | – | – | -0.060 [-0.184, +0.056] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.499 [0.411, 0.596] | – | 0.554 [0.450, 0.648] | – | – | -0.055 [-0.167, +0.071] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.583 [0.510, 0.650] | – | 0.540 [0.458, 0.616] | – | – | +0.043 [-0.050, +0.135] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.590 [0.509, 0.685] | – | 0.547 [0.453, 0.636] | – | – | +0.043 [-0.043, +0.140] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.498 [0.400, 0.590] | – | 0.489 [0.397, 0.578] | – | – | +0.009 [-0.097, +0.121] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.452 [0.373, 0.530] | – | 0.504 [0.396, 0.611] | – | – | -0.052 [-0.155, +0.065] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.523 [0.459, 0.591] | – | 0.511 [0.401, 0.615] | – | – | +0.013 [-0.103, +0.135] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.505 [0.426, 0.590] | – | 0.511 [0.401, 0.615] | – | – | -0.006 [-0.132, +0.136] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.485 [0.414, 0.556] | – | 0.511 [0.394, 0.618] | – | – | -0.026 [-0.099, +0.055] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.549 [0.452, 0.657] | – | 0.475 [0.351, 0.599] | – | – | +0.074 [-0.012, +0.174] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.519 [0.444, 0.594] | – | 0.532 [0.406, 0.657] | – | – | -0.013 [-0.160, +0.141] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.532 [0.453, 0.610] | – | 0.504 [0.380, 0.619] | – | – | +0.028 [-0.098, +0.168] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.551 [0.511, 0.591] | – | 0.500 [0.413, 0.577] | – | – | +0.051 [-0.023, +0.129] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.474 [0.419, 0.523] | – | 0.518 [0.447, 0.589] | – | – | -0.044 [-0.129, +0.030] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.514 [0.458, 0.573] | – | 0.457 [0.393, 0.524] | – | – | +0.056 [-0.034, +0.151] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.503 [0.444, 0.557] | – | 0.471 [0.404, 0.537] | – | – | +0.031 [-0.035, +0.097] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.506 [0.461, 0.555] | – | 0.443 [0.387, 0.502] | – | – | +0.063 [+0.011, +0.121] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.504 [0.431, 0.567] | – | 0.436 [0.382, 0.489] | – | – | +0.068 [-0.012, +0.146] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.479 [0.429, 0.534] | – | 0.461 [0.397, 0.520] | – | – | +0.018 [-0.052, +0.100] | – | – |
| `v.end` | 518 / 362 | 280 | 0.473 [0.402, 0.542] | – | 0.471 [0.401, 0.540] | – | – | +0.001 [-0.083, +0.086] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.532 [0.453, 0.610]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.496, `m2.end` 0.502, `m3.end` 0.452, `m4.end` 0.532. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `p3.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.504 [0.483, 0.526]; activation minus words -0.018 [-0.065, +0.029]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.500 [0.468, 0.532]; activation minus words +0.031 [-0.022, +0.081]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

