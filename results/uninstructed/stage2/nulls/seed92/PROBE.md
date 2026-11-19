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
| `m1.pre` | 257 / 183 | 139 | 0.529 [0.462, 0.600] | – | 0.500 [0.500, 0.500] | – | – | +0.029 [-0.038, +0.100] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.527 [0.448, 0.594] | – | 0.460 [0.418, 0.506] | – | – | +0.066 [-0.026, +0.142] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.404 [0.344, 0.476] | – | 0.486 [0.387, 0.584] | – | – | -0.081 [-0.169, +0.016] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.506 [0.416, 0.607] | – | 0.446 [0.348, 0.550] | – | – | +0.060 [-0.057, +0.188] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.455 [0.361, 0.569] | – | 0.561 [0.452, 0.682] | – | – | -0.106 [-0.239, +0.028] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.449 [0.356, 0.545] | – | 0.576 [0.483, 0.676] | – | – | -0.127 [-0.254, +0.011] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.562 [0.501, 0.622] | – | 0.468 [0.358, 0.579] | – | – | +0.094 [-0.012, +0.197] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.470 [0.380, 0.561] | – | 0.461 [0.351, 0.573] | – | – | +0.009 [-0.092, +0.112] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.562 [0.486, 0.632] | – | 0.461 [0.345, 0.569] | – | – | +0.101 [-0.045, +0.244] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.430 [0.342, 0.526] | – | 0.489 [0.377, 0.599] | – | – | -0.060 [-0.200, +0.082] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.603 [0.501, 0.701] | – | 0.475 [0.380, 0.578] | – | – | +0.128 [+0.012, +0.236] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.450 [0.364, 0.527] | – | 0.525 [0.403, 0.638] | – | – | -0.075 [-0.225, +0.082] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.485 [0.400, 0.570] | – | 0.597 [0.490, 0.703] | – | – | -0.112 [-0.211, -0.018] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.442 [0.384, 0.505] | – | 0.597 [0.490, 0.703] | – | – | -0.155 [-0.253, -0.055] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.504 [0.422, 0.587] | – | 0.547 [0.433, 0.659] | – | – | -0.043 [-0.148, +0.055] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.574 [0.464, 0.675] | – | 0.496 [0.364, 0.626] | – | – | +0.078 [-0.071, +0.230] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.502 [0.408, 0.594] | – | 0.532 [0.415, 0.644] | – | – | -0.030 [-0.136, +0.087] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.459 [0.399, 0.531] | – | 0.540 [0.409, 0.655] | – | – | -0.081 [-0.199, +0.041] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.474 [0.379, 0.551] | – | 0.475 [0.386, 0.569] | – | – | -0.001 [-0.100, +0.084] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.440 [0.335, 0.534] | – | 0.482 [0.395, 0.572] | – | – | -0.043 [-0.183, +0.100] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.539 [0.457, 0.629] | – | 0.496 [0.402, 0.588] | – | – | +0.043 [-0.073, +0.145] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.482 [0.392, 0.577] | – | 0.518 [0.412, 0.623] | – | – | -0.035 [-0.176, +0.096] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.523 [0.447, 0.596] | – | 0.532 [0.428, 0.631] | – | – | -0.009 [-0.130, +0.120] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.528 [0.441, 0.618] | – | 0.553 [0.466, 0.640] | – | – | -0.026 [-0.168, +0.114] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.520 [0.452, 0.584] | – | 0.586 [0.520, 0.644] | – | – | -0.066 [-0.158, +0.025] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.463 [0.405, 0.515] | – | 0.618 [0.545, 0.675] | – | – | -0.155 [-0.241, -0.056] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.523 [0.448, 0.600] | – | 0.557 [0.460, 0.637] | – | – | -0.034 [-0.140, +0.081] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.479 [0.403, 0.550] | – | 0.529 [0.422, 0.608] | – | – | -0.050 [-0.117, +0.027] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.459 [0.390, 0.523] | – | 0.557 [0.455, 0.633] | – | – | -0.099 [-0.208, +0.010] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.547 [0.483, 0.598] | – | 0.564 [0.459, 0.645] | – | – | -0.017 [-0.116, +0.078] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.496 [0.434, 0.551] | – | 0.571 [0.467, 0.650] | – | – | -0.076 [-0.164, +0.030] | – | – |
| `v.end` | 518 / 362 | 280 | 0.499 [0.448, 0.549] | – | 0.557 [0.456, 0.631] | – | – | -0.058 [-0.136, +0.037] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.528 [0.441, 0.618]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.449, `m2.end` 0.450, `m3.end` 0.459, `m4.end` 0.528. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.q3`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.496 [0.469, 0.525]; activation minus words -0.016 [-0.069, +0.032]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.498 [0.466, 0.525]; activation minus words -0.069 [-0.126, -0.002]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

