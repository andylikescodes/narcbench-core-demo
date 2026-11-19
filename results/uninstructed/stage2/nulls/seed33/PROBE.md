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
| `m1.pre` | 257 / 183 | 139 | 0.401 [0.361, 0.454] | – | 0.500 [0.500, 0.500] | – | – | -0.099 [-0.139, -0.046] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.565 [0.502, 0.626] | – | 0.504 [0.463, 0.538] | – | – | +0.062 [-0.002, +0.124] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.432 [0.357, 0.504] | – | 0.475 [0.375, 0.566] | – | – | -0.043 [-0.154, +0.065] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.504 [0.411, 0.597] | – | 0.561 [0.439, 0.667] | – | – | -0.058 [-0.182, +0.085] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.527 [0.436, 0.607] | – | 0.583 [0.500, 0.663] | – | – | -0.056 [-0.168, +0.043] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.570 [0.468, 0.662] | – | 0.482 [0.395, 0.557] | – | – | +0.088 [-0.043, +0.207] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.553 [0.465, 0.650] | – | 0.496 [0.418, 0.566] | – | – | +0.057 [-0.039, +0.178] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.563 [0.496, 0.630] | – | 0.496 [0.412, 0.569] | – | – | +0.067 [-0.005, +0.152] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.498 [0.420, 0.580] | – | 0.546 [0.452, 0.630] | – | – | -0.048 [-0.164, +0.081] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.590 [0.502, 0.691] | – | 0.603 [0.518, 0.686] | – | – | -0.013 [-0.144, +0.126] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.556 [0.464, 0.654] | – | 0.574 [0.471, 0.670] | – | – | -0.018 [-0.139, +0.115] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.576 [0.471, 0.671] | – | 0.504 [0.413, 0.589] | – | – | +0.072 [-0.033, +0.171] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.545 [0.449, 0.628] | – | 0.554 [0.435, 0.660] | – | – | -0.009 [-0.126, +0.119] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.580 [0.464, 0.676] | – | 0.554 [0.435, 0.660] | – | – | +0.026 [-0.110, +0.161] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.492 [0.390, 0.598] | – | 0.518 [0.388, 0.632] | – | – | -0.026 [-0.137, +0.110] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.509 [0.410, 0.598] | – | 0.554 [0.450, 0.637] | – | – | -0.045 [-0.173, +0.100] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.506 [0.408, 0.602] | – | 0.619 [0.522, 0.693] | – | – | -0.112 [-0.219, +0.005] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.564 [0.486, 0.656] | – | 0.561 [0.446, 0.656] | – | – | +0.003 [-0.116, +0.138] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.518 [0.410, 0.634] | – | 0.504 [0.398, 0.603] | – | – | +0.014 [-0.084, +0.136] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.501 [0.440, 0.572] | – | 0.504 [0.403, 0.595] | – | – | -0.003 [-0.095, +0.106] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.515 [0.400, 0.632] | – | 0.560 [0.460, 0.642] | – | – | -0.045 [-0.145, +0.085] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.489 [0.400, 0.586] | – | 0.582 [0.478, 0.671] | – | – | -0.092 [-0.218, +0.053] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.433 [0.357, 0.510] | – | 0.567 [0.465, 0.660] | – | – | -0.135 [-0.203, -0.041] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.475 [0.392, 0.568] | – | 0.567 [0.454, 0.669] | – | – | -0.092 [-0.188, +0.044] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.411 [0.350, 0.468] | – | 0.561 [0.490, 0.618] | – | – | -0.149 [-0.197, -0.090] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.459 [0.409, 0.501] | – | 0.561 [0.492, 0.619] | – | – | -0.101 [-0.171, -0.024] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.480 [0.423, 0.534] | – | 0.600 [0.539, 0.651] | – | – | -0.120 [-0.193, -0.042] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.521 [0.465, 0.578] | – | 0.589 [0.523, 0.641] | – | – | -0.068 [-0.140, +0.013] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.446 [0.403, 0.491] | – | 0.571 [0.504, 0.629] | – | – | -0.126 [-0.197, -0.052] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.504 [0.453, 0.556] | – | 0.554 [0.492, 0.609] | – | – | -0.050 [-0.129, +0.043] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.521 [0.446, 0.591] | – | 0.554 [0.484, 0.611] | – | – | -0.033 [-0.099, +0.045] | – | – |
| `v.end` | 518 / 362 | 280 | 0.505 [0.443, 0.566] | – | 0.546 [0.481, 0.606] | – | – | -0.041 [-0.137, +0.067] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.475 [0.392, 0.568]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.570, `m2.end` 0.576, `m3.end` 0.564, `m4.end` 0.475. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: `p2.pre`.
  - In one number, the mean over the 24 room positions: activation 0.519 [0.494, 0.546]; activation minus words -0.021 [-0.052, +0.022]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.481 [0.460, 0.502]; activation minus words -0.086 [-0.128, -0.032]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

