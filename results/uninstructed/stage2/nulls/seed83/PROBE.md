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
| `m1.pre` | 257 / 183 | 139 | 0.562 [0.486, 0.637] | – | 0.500 [0.500, 0.500] | – | – | +0.062 [-0.014, +0.137] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.397 [0.329, 0.455] | – | 0.489 [0.444, 0.530] | – | – | -0.092 [-0.164, -0.029] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.517 [0.403, 0.600] | – | 0.601 [0.516, 0.692] | – | – | -0.084 [-0.192, +0.011] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.481 [0.389, 0.564] | – | 0.597 [0.477, 0.712] | – | – | -0.117 [-0.252, +0.021] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.439 [0.358, 0.517] | – | 0.496 [0.380, 0.607] | – | – | -0.058 [-0.184, +0.063] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.442 [0.353, 0.518] | – | 0.590 [0.471, 0.702] | – | – | -0.148 [-0.263, -0.034] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.552 [0.470, 0.625] | – | 0.532 [0.435, 0.635] | – | – | +0.020 [-0.127, +0.147] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.546 [0.461, 0.634] | – | 0.532 [0.440, 0.628] | – | – | +0.014 [-0.090, +0.116] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.533 [0.429, 0.633] | – | 0.532 [0.424, 0.639] | – | – | +0.001 [-0.169, +0.162] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.543 [0.481, 0.604] | – | 0.489 [0.382, 0.589] | – | – | +0.054 [-0.027, +0.139] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.553 [0.460, 0.639] | – | 0.496 [0.381, 0.609] | – | – | +0.057 [-0.091, +0.179] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.596 [0.528, 0.669] | – | 0.567 [0.475, 0.664] | – | – | +0.028 [-0.076, +0.142] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.541 [0.465, 0.614] | – | 0.547 [0.414, 0.663] | – | – | -0.006 [-0.137, +0.137] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.505 [0.404, 0.598] | – | 0.540 [0.413, 0.653] | – | – | -0.035 [-0.172, +0.094] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.427 [0.346, 0.528] | – | 0.525 [0.417, 0.637] | – | – | -0.098 [-0.245, +0.058] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.522 [0.448, 0.597] | – | 0.496 [0.367, 0.624] | – | – | +0.026 [-0.098, +0.173] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.430 [0.347, 0.517] | – | 0.468 [0.331, 0.600] | – | – | -0.037 [-0.175, +0.122] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.476 [0.385, 0.570] | – | 0.468 [0.345, 0.587] | – | – | +0.009 [-0.137, +0.163] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.553 [0.435, 0.660] | – | 0.574 [0.488, 0.653] | – | – | -0.021 [-0.147, +0.125] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.518 [0.417, 0.630] | – | 0.574 [0.488, 0.653] | – | – | -0.057 [-0.169, +0.085] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.390 [0.289, 0.516] | – | 0.511 [0.420, 0.607] | – | – | -0.121 [-0.226, -0.010] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.499 [0.430, 0.587] | – | 0.511 [0.409, 0.611] | – | – | -0.011 [-0.145, +0.137] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.556 [0.481, 0.636] | – | 0.553 [0.470, 0.648] | – | – | +0.003 [-0.113, +0.118] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.496 [0.442, 0.567] | – | 0.532 [0.455, 0.622] | – | – | -0.035 [-0.123, +0.056] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.504 [0.448, 0.560] | – | 0.468 [0.414, 0.523] | – | – | +0.036 [-0.024, +0.091] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.484 [0.427, 0.545] | – | 0.468 [0.408, 0.531] | – | – | +0.016 [-0.062, +0.094] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.568 [0.503, 0.634] | – | 0.414 [0.352, 0.488] | – | – | +0.154 [+0.065, +0.225] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.557 [0.478, 0.629] | – | 0.414 [0.361, 0.474] | – | – | +0.143 [+0.058, +0.223] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.514 [0.451, 0.576] | – | 0.457 [0.407, 0.512] | – | – | +0.056 [-0.014, +0.127] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.424 [0.353, 0.507] | – | 0.471 [0.419, 0.532] | – | – | -0.048 [-0.147, +0.044] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.498 [0.443, 0.541] | – | 0.468 [0.402, 0.541] | – | – | +0.030 [-0.074, +0.114] | – | – |
| `v.end` | 518 / 362 | 280 | 0.472 [0.416, 0.525] | – | 0.457 [0.392, 0.530] | – | – | +0.015 [-0.082, +0.102] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.496 [0.442, 0.567]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.442, `m2.end` 0.596, `m3.end` 0.476, `m4.end` 0.496. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `p2.pre`, `p2.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.503 [0.482, 0.526]; activation minus words -0.027 [-0.072, +0.026]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.502 [0.475, 0.535]; activation minus words +0.050 [+0.001, +0.093]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

