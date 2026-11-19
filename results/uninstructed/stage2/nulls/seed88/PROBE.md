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
| `m1.pre` | 257 / 183 | 139 | 0.509 [0.441, 0.574] | – | 0.500 [0.500, 0.500] | – | – | +0.009 [-0.059, +0.074] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.471 [0.409, 0.540] | – | 0.446 [0.388, 0.503] | – | – | +0.025 [-0.055, +0.102] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.591 [0.516, 0.662] | – | 0.608 [0.512, 0.707] | – | – | -0.017 [-0.138, +0.104] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.413 [0.342, 0.483] | – | 0.532 [0.463, 0.617] | – | – | -0.119 [-0.221, -0.025] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.422 [0.309, 0.534] | – | 0.453 [0.343, 0.570] | – | – | -0.032 [-0.177, +0.107] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.410 [0.307, 0.511] | – | 0.453 [0.340, 0.563] | – | – | -0.043 [-0.182, +0.098] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.454 [0.384, 0.513] | – | 0.582 [0.504, 0.656] | – | – | -0.128 [-0.208, -0.060] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.530 [0.453, 0.604] | – | 0.582 [0.504, 0.656] | – | – | -0.051 [-0.138, +0.025] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.508 [0.435, 0.592] | – | 0.596 [0.488, 0.706] | – | – | -0.088 [-0.202, +0.032] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.475 [0.384, 0.576] | – | 0.532 [0.446, 0.635] | – | – | -0.057 [-0.215, +0.084] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.540 [0.464, 0.623] | – | 0.496 [0.428, 0.578] | – | – | +0.044 [-0.062, +0.146] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.523 [0.415, 0.647] | – | 0.489 [0.434, 0.553] | – | – | +0.034 [-0.068, +0.139] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.485 [0.387, 0.579] | – | 0.511 [0.428, 0.607] | – | – | -0.026 [-0.165, +0.113] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.473 [0.378, 0.568] | – | 0.518 [0.430, 0.616] | – | – | -0.045 [-0.158, +0.064] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.535 [0.445, 0.640] | – | 0.554 [0.457, 0.653] | – | – | -0.019 [-0.152, +0.126] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.492 [0.430, 0.574] | – | 0.532 [0.433, 0.634] | – | – | -0.040 [-0.161, +0.095] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.473 [0.405, 0.538] | – | 0.532 [0.447, 0.632] | – | – | -0.059 [-0.172, +0.040] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.505 [0.426, 0.600] | – | 0.561 [0.456, 0.659] | – | – | -0.056 [-0.175, +0.073] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.535 [0.465, 0.607] | – | 0.567 [0.497, 0.646] | – | – | -0.033 [-0.134, +0.069] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.506 [0.429, 0.582] | – | 0.560 [0.495, 0.632] | – | – | -0.054 [-0.176, +0.065] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.553 [0.469, 0.656] | – | 0.546 [0.482, 0.619] | – | – | +0.007 [-0.101, +0.124] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.481 [0.398, 0.565] | – | 0.553 [0.486, 0.625] | – | – | -0.072 [-0.209, +0.046] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.559 [0.464, 0.654] | – | 0.496 [0.406, 0.585] | – | – | +0.062 [-0.067, +0.183] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.522 [0.440, 0.603] | – | 0.496 [0.412, 0.596] | – | – | +0.026 [-0.078, +0.130] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.527 [0.482, 0.563] | – | 0.489 [0.437, 0.566] | – | – | +0.038 [-0.050, +0.103] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.480 [0.420, 0.546] | – | 0.500 [0.436, 0.571] | – | – | -0.020 [-0.098, +0.060] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.519 [0.484, 0.556] | – | 0.539 [0.479, 0.606] | – | – | -0.021 [-0.085, +0.043] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.471 [0.413, 0.533] | – | 0.507 [0.439, 0.583] | – | – | -0.036 [-0.107, +0.034] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.561 [0.518, 0.606] | – | 0.539 [0.485, 0.607] | – | – | +0.021 [-0.027, +0.073] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.522 [0.473, 0.570] | – | 0.529 [0.470, 0.596] | – | – | -0.006 [-0.077, +0.060] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.459 [0.406, 0.509] | – | 0.539 [0.486, 0.597] | – | – | -0.081 [-0.165, -0.012] | – | – |
| `v.end` | 518 / 362 | 280 | 0.510 [0.459, 0.562] | – | 0.521 [0.463, 0.579] | – | – | -0.011 [-0.083, +0.058] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.522 [0.440, 0.603]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.410, `m2.end` 0.523, `m3.end` 0.505, `m4.end` 0.522. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.499 [0.477, 0.526]; activation minus words -0.030 [-0.075, +0.011]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.506 [0.481, 0.531]; activation minus words -0.014 [-0.060, +0.025]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

