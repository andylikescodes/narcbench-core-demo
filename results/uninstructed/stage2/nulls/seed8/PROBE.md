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
| `m1.pre` | 257 / 183 | 139 | 0.419 [0.336, 0.498] | – | 0.500 [0.500, 0.500] | – | – | -0.081 [-0.164, -0.002] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.559 [0.501, 0.618] | – | 0.547 [0.519, 0.574] | – | – | +0.012 [-0.054, +0.079] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.465 [0.365, 0.556] | – | 0.424 [0.333, 0.517] | – | – | +0.040 [-0.047, +0.119] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.496 [0.411, 0.583] | – | 0.475 [0.355, 0.595] | – | – | +0.022 [-0.062, +0.101] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.540 [0.459, 0.602] | – | 0.424 [0.317, 0.531] | – | – | +0.115 [-0.016, +0.224] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.437 [0.339, 0.535] | – | 0.403 [0.296, 0.512] | – | – | +0.035 [-0.093, +0.158] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.546 [0.432, 0.657] | – | 0.511 [0.409, 0.601] | – | – | +0.035 [-0.088, +0.167] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.501 [0.395, 0.609] | – | 0.518 [0.411, 0.613] | – | – | -0.017 [-0.145, +0.109] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.443 [0.320, 0.556] | – | 0.482 [0.356, 0.582] | – | – | -0.040 [-0.183, +0.110] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.529 [0.415, 0.652] | – | 0.496 [0.372, 0.612] | – | – | +0.033 [-0.088, +0.173] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.488 [0.381, 0.590] | – | 0.504 [0.395, 0.616] | – | – | -0.016 [-0.124, +0.071] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.538 [0.423, 0.644] | – | 0.532 [0.410, 0.661] | – | – | +0.006 [-0.156, +0.130] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.458 [0.368, 0.548] | – | 0.489 [0.393, 0.579] | – | – | -0.032 [-0.123, +0.068] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.475 [0.389, 0.568] | – | 0.496 [0.396, 0.588] | – | – | -0.022 [-0.133, +0.107] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.456 [0.352, 0.550] | – | 0.489 [0.405, 0.582] | – | – | -0.033 [-0.142, +0.060] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.458 [0.397, 0.511] | – | 0.475 [0.363, 0.594] | – | – | -0.017 [-0.143, +0.090] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.462 [0.404, 0.520] | – | 0.475 [0.358, 0.571] | – | – | -0.013 [-0.114, +0.105] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.447 [0.373, 0.525] | – | 0.417 [0.320, 0.500] | – | – | +0.030 [-0.067, +0.130] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.509 [0.427, 0.592] | – | 0.504 [0.376, 0.636] | – | – | +0.006 [-0.104, +0.104] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.559 [0.472, 0.640] | – | 0.511 [0.380, 0.644] | – | – | +0.048 [-0.074, +0.178] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.496 [0.426, 0.575] | – | 0.504 [0.374, 0.636] | – | – | -0.007 [-0.118, +0.117] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.539 [0.429, 0.637] | – | 0.518 [0.371, 0.660] | – | – | +0.021 [-0.153, +0.184] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.633 [0.537, 0.722] | – | 0.511 [0.362, 0.659] | – | – | +0.122 [-0.035, +0.269] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.546 [0.460, 0.618] | – | 0.539 [0.390, 0.690] | – | – | +0.007 [-0.101, +0.107] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.514 [0.444, 0.584] | – | 0.582 [0.484, 0.657] | – | – | -0.068 [-0.151, +0.038] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.519 [0.469, 0.576] | – | 0.600 [0.503, 0.677] | – | – | -0.081 [-0.165, +0.017] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.486 [0.417, 0.551] | – | 0.604 [0.522, 0.686] | – | – | -0.118 [-0.199, -0.044] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.505 [0.429, 0.578] | – | 0.607 [0.530, 0.677] | – | – | -0.102 [-0.215, -0.001] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.478 [0.418, 0.538] | – | 0.582 [0.494, 0.668] | – | – | -0.104 [-0.182, -0.022] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.526 [0.469, 0.588] | – | 0.579 [0.492, 0.666] | – | – | -0.052 [-0.151, +0.056] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.617 [0.556, 0.680] | – | 0.596 [0.521, 0.669] | – | – | +0.021 [-0.059, +0.101] | – | – |
| `v.end` | 518 / 362 | 280 | 0.539 [0.464, 0.607] | – | 0.589 [0.511, 0.660] | – | – | -0.050 [-0.123, +0.014] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.546 [0.460, 0.618]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.437, `m2.end` 0.538, `m3.end` 0.447, `m4.end` 0.546. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: `p1.end`.
  - In one number, the mean over the 24 room positions: activation 0.500 [0.460, 0.536]; activation minus words +0.011 [-0.026, +0.048]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.523 [0.484, 0.560]; activation minus words -0.069 [-0.129, -0.004]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

