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
| `m1.pre` | 257 / 183 | 139 | 0.542 [0.466, 0.613] | – | 0.500 [0.500, 0.500] | – | – | +0.042 [-0.034, +0.113] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.424 [0.358, 0.493] | – | 0.468 [0.428, 0.500] | – | – | -0.043 [-0.125, +0.040] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.440 [0.361, 0.523] | – | 0.486 [0.351, 0.617] | – | – | -0.045 [-0.203, +0.108] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.544 [0.440, 0.640] | – | 0.561 [0.488, 0.634] | – | – | -0.017 [-0.140, +0.088] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.463 [0.377, 0.562] | – | 0.439 [0.321, 0.557] | – | – | +0.024 [-0.115, +0.170] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.514 [0.420, 0.604] | – | 0.468 [0.374, 0.565] | – | – | +0.046 [-0.076, +0.164] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.515 [0.457, 0.585] | – | 0.482 [0.395, 0.576] | – | – | +0.033 [-0.086, +0.160] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.441 [0.342, 0.546] | – | 0.475 [0.389, 0.573] | – | – | -0.034 [-0.179, +0.117] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.475 [0.340, 0.594] | – | 0.447 [0.367, 0.531] | – | – | +0.028 [-0.169, +0.192] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.437 [0.335, 0.537] | – | 0.482 [0.364, 0.599] | – | – | -0.045 [-0.206, +0.086] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.409 [0.336, 0.486] | – | 0.461 [0.345, 0.573] | – | – | -0.052 [-0.173, +0.079] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.472 [0.385, 0.563] | – | 0.504 [0.426, 0.582] | – | – | -0.031 [-0.167, +0.096] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.561 [0.473, 0.644] | – | 0.453 [0.348, 0.568] | – | – | +0.108 [-0.028, +0.244] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.466 [0.410, 0.526] | – | 0.453 [0.348, 0.568] | – | – | +0.013 [-0.111, +0.145] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.563 [0.477, 0.637] | – | 0.446 [0.325, 0.575] | – | – | +0.117 [-0.030, +0.243] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.495 [0.397, 0.600] | – | 0.453 [0.349, 0.567] | – | – | +0.042 [-0.065, +0.147] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.442 [0.364, 0.507] | – | 0.417 [0.309, 0.540] | – | – | +0.024 [-0.127, +0.145] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.439 [0.367, 0.511] | – | 0.403 [0.299, 0.506] | – | – | +0.036 [-0.091, +0.175] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.481 [0.410, 0.549] | – | 0.496 [0.410, 0.589] | – | – | -0.016 [-0.119, +0.083] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.433 [0.379, 0.489] | – | 0.496 [0.410, 0.589] | – | – | -0.064 [-0.152, +0.023] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.586 [0.492, 0.676] | – | 0.525 [0.455, 0.598] | – | – | +0.061 [-0.024, +0.160] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.495 [0.394, 0.603] | – | 0.511 [0.443, 0.579] | – | – | -0.016 [-0.131, +0.110] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.506 [0.408, 0.612] | – | 0.532 [0.457, 0.610] | – | – | -0.026 [-0.143, +0.095] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.475 [0.383, 0.580] | – | 0.525 [0.444, 0.605] | – | – | -0.050 [-0.163, +0.071] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.486 [0.443, 0.536] | – | 0.461 [0.403, 0.520] | – | – | +0.026 [-0.048, +0.107] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.519 [0.450, 0.598] | – | 0.454 [0.385, 0.521] | – | – | +0.066 [-0.014, +0.154] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.496 [0.445, 0.560] | – | 0.493 [0.441, 0.556] | – | – | +0.003 [-0.073, +0.081] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.511 [0.460, 0.564] | – | 0.489 [0.423, 0.561] | – | – | +0.022 [-0.065, +0.112] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.515 [0.465, 0.566] | – | 0.464 [0.413, 0.523] | – | – | +0.051 [-0.021, +0.114] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.526 [0.471, 0.585] | – | 0.486 [0.429, 0.540] | – | – | +0.040 [-0.026, +0.110] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.520 [0.461, 0.574] | – | 0.500 [0.432, 0.567] | – | – | +0.020 [-0.051, +0.092] | – | – |
| `v.end` | 518 / 362 | 280 | 0.496 [0.438, 0.566] | – | 0.496 [0.430, 0.556] | – | – | -0.001 [-0.065, +0.075] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.475 [0.383, 0.580]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.514, `m2.end` 0.472, `m3.end` 0.439, `m4.end` 0.475. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.484 [0.460, 0.508]; activation minus words +0.006 [-0.051, +0.056]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.509 [0.477, 0.545]; activation minus words +0.028 [-0.025, +0.088]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

