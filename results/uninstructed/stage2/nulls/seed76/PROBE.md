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
| `m1.pre` | 257 / 183 | 139 | 0.620 [0.545, 0.692] | – | 0.500 [0.500, 0.500] | – | – | +0.120 [+0.045, +0.192] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.521 [0.447, 0.590] | – | 0.511 [0.486, 0.537] | – | – | +0.010 [-0.065, +0.081] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.524 [0.424, 0.624] | – | 0.442 [0.327, 0.574] | – | – | +0.081 [-0.095, +0.234] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.571 [0.477, 0.681] | – | 0.489 [0.400, 0.592] | – | – | +0.082 [-0.071, +0.220] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.511 [0.409, 0.606] | – | 0.424 [0.339, 0.513] | – | – | +0.086 [-0.036, +0.187] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.577 [0.482, 0.658] | – | 0.388 [0.307, 0.475] | – | – | +0.188 [+0.085, +0.276] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.479 [0.383, 0.578] | – | 0.525 [0.448, 0.616] | – | – | -0.045 [-0.162, +0.066] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.494 [0.398, 0.598] | – | 0.518 [0.426, 0.620] | – | – | -0.024 [-0.152, +0.102] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.437 [0.330, 0.531] | – | 0.553 [0.462, 0.653] | – | – | -0.116 [-0.261, +0.008] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.572 [0.513, 0.634] | – | 0.539 [0.453, 0.626] | – | – | +0.033 [-0.075, +0.126] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.496 [0.424, 0.567] | – | 0.468 [0.386, 0.561] | – | – | +0.028 [-0.098, +0.137] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.397 [0.311, 0.506] | – | 0.418 [0.339, 0.497] | – | – | -0.021 [-0.115, +0.087] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.570 [0.468, 0.681] | – | 0.424 [0.317, 0.517] | – | – | +0.145 [-0.016, +0.337] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.517 [0.450, 0.590] | – | 0.424 [0.317, 0.517] | – | – | +0.092 [-0.017, +0.219] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.558 [0.504, 0.612] | – | 0.432 [0.329, 0.532] | – | – | +0.127 [+0.020, +0.238] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.528 [0.457, 0.619] | – | 0.403 [0.299, 0.508] | – | – | +0.125 [-0.011, +0.267] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.514 [0.443, 0.594] | – | 0.353 [0.289, 0.414] | – | – | +0.161 [+0.069, +0.259] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.488 [0.386, 0.574] | – | 0.381 [0.305, 0.465] | – | – | +0.106 [+0.003, +0.195] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.384 [0.311, 0.457] | – | 0.496 [0.431, 0.563] | – | – | -0.112 [-0.197, -0.025] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.440 [0.340, 0.551] | – | 0.496 [0.431, 0.568] | – | – | -0.057 [-0.158, +0.064] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.502 [0.395, 0.608] | – | 0.482 [0.410, 0.551] | – | – | +0.020 [-0.098, +0.139] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.550 [0.465, 0.624] | – | 0.475 [0.392, 0.554] | – | – | +0.075 [-0.043, +0.185] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.452 [0.378, 0.536] | – | 0.475 [0.378, 0.567] | – | – | -0.023 [-0.143, +0.090] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.455 [0.340, 0.546] | – | 0.482 [0.393, 0.559] | – | – | -0.027 [-0.133, +0.071] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.469 [0.419, 0.521] | – | 0.486 [0.425, 0.541] | – | – | -0.017 [-0.092, +0.059] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.541 [0.449, 0.621] | – | 0.479 [0.418, 0.534] | – | – | +0.062 [-0.044, +0.160] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.489 [0.438, 0.541] | – | 0.471 [0.412, 0.531] | – | – | +0.018 [-0.063, +0.098] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.475 [0.420, 0.527] | – | 0.482 [0.424, 0.538] | – | – | -0.007 [-0.096, +0.075] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.501 [0.453, 0.543] | – | 0.489 [0.444, 0.535] | – | – | +0.012 [-0.036, +0.053] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.444 [0.397, 0.498] | – | 0.496 [0.450, 0.541] | – | – | -0.053 [-0.103, +0.003] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.501 [0.451, 0.555] | – | 0.511 [0.459, 0.557] | – | – | -0.010 [-0.065, +0.055] | – | – |
| `v.end` | 518 / 362 | 280 | 0.494 [0.434, 0.554] | – | 0.493 [0.437, 0.543] | – | – | +0.001 [-0.083, +0.085] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.455 [0.340, 0.546]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.577, `m2.end` 0.397, `m3.end` 0.488, `m4.end` 0.455. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.pre`, `m1.end`, `m3.q1`, `m3.q3`, `m3.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.507 [0.488, 0.526]; activation minus words +0.044 [+0.004, +0.086]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.489 [0.462, 0.513]; activation minus words +0.001 [-0.041, +0.042]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

