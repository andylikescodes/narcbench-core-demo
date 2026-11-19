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
| `m1.pre` | 257 / 183 | 139 | 0.537 [0.471, 0.610] | – | 0.500 [0.500, 0.500] | – | – | +0.037 [-0.029, +0.110] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.464 [0.395, 0.536] | – | 0.511 [0.467, 0.554] | – | – | -0.047 [-0.114, +0.019] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.489 [0.415, 0.560] | – | 0.554 [0.420, 0.669] | – | – | -0.065 [-0.187, +0.077] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.409 [0.343, 0.482] | – | 0.576 [0.456, 0.677] | – | – | -0.167 [-0.267, -0.035] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.456 [0.377, 0.540] | – | 0.468 [0.366, 0.571] | – | – | -0.012 [-0.156, +0.146] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.515 [0.440, 0.591] | – | 0.460 [0.345, 0.571] | – | – | +0.055 [-0.034, +0.152] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.498 [0.414, 0.600] | – | 0.638 [0.539, 0.742] | – | – | -0.140 [-0.282, +0.004] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.521 [0.438, 0.594] | – | 0.638 [0.534, 0.742] | – | – | -0.118 [-0.253, +0.006] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.529 [0.446, 0.622] | – | 0.560 [0.463, 0.667] | – | – | -0.031 [-0.191, +0.129] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.485 [0.404, 0.581] | – | 0.603 [0.514, 0.701] | – | – | -0.118 [-0.237, -0.013] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.502 [0.417, 0.593] | – | 0.582 [0.465, 0.695] | – | – | -0.079 [-0.228, +0.053] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.464 [0.388, 0.554] | – | 0.567 [0.444, 0.690] | – | – | -0.104 [-0.268, +0.043] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.548 [0.465, 0.636] | – | 0.468 [0.339, 0.588] | – | – | +0.081 [-0.062, +0.208] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.560 [0.458, 0.642] | – | 0.468 [0.339, 0.588] | – | – | +0.092 [-0.021, +0.195] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.466 [0.370, 0.549] | – | 0.446 [0.292, 0.585] | – | – | +0.020 [-0.113, +0.161] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.505 [0.418, 0.592] | – | 0.468 [0.320, 0.599] | – | – | +0.037 [-0.087, +0.185] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.548 [0.455, 0.624] | – | 0.475 [0.341, 0.595] | – | – | +0.073 [-0.058, +0.218] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.388 [0.291, 0.487] | – | 0.496 [0.380, 0.607] | – | – | -0.108 [-0.223, +0.031] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.472 [0.394, 0.552] | – | 0.539 [0.415, 0.652] | – | – | -0.067 [-0.184, +0.066] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.502 [0.385, 0.610] | – | 0.539 [0.415, 0.652] | – | – | -0.037 [-0.161, +0.075] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.525 [0.440, 0.625] | – | 0.511 [0.370, 0.644] | – | – | +0.014 [-0.145, +0.192] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.504 [0.427, 0.582] | – | 0.468 [0.330, 0.609] | – | – | +0.035 [-0.112, +0.190] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.504 [0.427, 0.592] | – | 0.475 [0.338, 0.601] | – | – | +0.028 [-0.118, +0.192] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.509 [0.429, 0.590] | – | 0.511 [0.384, 0.635] | – | – | -0.001 [-0.167, +0.149] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.514 [0.466, 0.563] | – | 0.493 [0.405, 0.575] | – | – | +0.021 [-0.071, +0.110] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.520 [0.455, 0.575] | – | 0.446 [0.362, 0.528] | – | – | +0.074 [-0.021, +0.160] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.524 [0.467, 0.578] | – | 0.450 [0.374, 0.533] | – | – | +0.074 [-0.024, +0.180] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.488 [0.430, 0.553] | – | 0.429 [0.352, 0.517] | – | – | +0.059 [-0.050, +0.165] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.474 [0.411, 0.532] | – | 0.439 [0.373, 0.503] | – | – | +0.034 [-0.058, +0.132] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.529 [0.454, 0.587] | – | 0.429 [0.348, 0.503] | – | – | +0.101 [-0.002, +0.186] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.554 [0.496, 0.606] | – | 0.436 [0.360, 0.506] | – | – | +0.119 [+0.034, +0.203] | – | – |
| `v.end` | 518 / 362 | 280 | 0.536 [0.466, 0.595] | – | 0.443 [0.361, 0.517] | – | – | +0.093 [-0.021, +0.190] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.509 [0.429, 0.590]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.515, `m2.end` 0.464, `m3.end` 0.388, `m4.end` 0.509. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `v.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.496 [0.476, 0.517]; activation minus words -0.026 [-0.081, +0.034]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.517 [0.490, 0.539]; activation minus words +0.072 [-0.001, +0.145]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

