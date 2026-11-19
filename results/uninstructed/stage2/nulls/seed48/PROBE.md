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
| `m1.pre` | 257 / 183 | 139 | 0.526 [0.428, 0.626] | – | 0.500 [0.500, 0.500] | – | – | +0.026 [-0.072, +0.126] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.440 [0.371, 0.508] | – | 0.482 [0.438, 0.524] | – | – | -0.042 [-0.123, +0.035] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.466 [0.384, 0.549] | – | 0.442 [0.353, 0.524] | – | – | +0.024 [-0.084, +0.136] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.468 [0.393, 0.530] | – | 0.496 [0.398, 0.590] | – | – | -0.029 [-0.125, +0.065] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.504 [0.417, 0.583] | – | 0.576 [0.500, 0.657] | – | – | -0.072 [-0.177, +0.034] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.565 [0.481, 0.654] | – | 0.568 [0.470, 0.656] | – | – | -0.003 [-0.128, +0.138] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.576 [0.527, 0.628] | – | 0.482 [0.385, 0.593] | – | – | +0.094 [-0.004, +0.179] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.501 [0.436, 0.567] | – | 0.482 [0.385, 0.593] | – | – | +0.018 [-0.086, +0.103] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.488 [0.407, 0.571] | – | 0.525 [0.396, 0.664] | – | – | -0.037 [-0.152, +0.067] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.565 [0.456, 0.666] | – | 0.532 [0.433, 0.650] | – | – | +0.033 [-0.084, +0.136] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.658 [0.572, 0.730] | – | 0.589 [0.478, 0.694] | – | – | +0.070 [-0.051, +0.183] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.523 [0.462, 0.578] | – | 0.589 [0.491, 0.691] | – | – | -0.065 [-0.186, +0.044] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.517 [0.438, 0.585] | – | 0.619 [0.490, 0.727] | – | – | -0.102 [-0.205, +0.007] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.488 [0.405, 0.558] | – | 0.619 [0.490, 0.727] | – | – | -0.131 [-0.238, -0.004] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.481 [0.386, 0.584] | – | 0.590 [0.479, 0.701] | – | – | -0.109 [-0.275, +0.061] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.489 [0.394, 0.562] | – | 0.554 [0.459, 0.649] | – | – | -0.065 [-0.202, +0.069] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.494 [0.382, 0.605] | – | 0.525 [0.420, 0.630] | – | – | -0.032 [-0.136, +0.109] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.495 [0.432, 0.561] | – | 0.590 [0.476, 0.707] | – | – | -0.095 [-0.233, +0.039] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.481 [0.402, 0.569] | – | 0.553 [0.481, 0.625] | – | – | -0.072 [-0.161, +0.021] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.533 [0.461, 0.593] | – | 0.546 [0.478, 0.612] | – | – | -0.013 [-0.108, +0.062] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.557 [0.491, 0.630] | – | 0.589 [0.512, 0.663] | – | – | -0.031 [-0.121, +0.060] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.532 [0.453, 0.596] | – | 0.532 [0.459, 0.613] | – | – | -0.000 [-0.110, +0.093] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.521 [0.429, 0.607] | – | 0.518 [0.437, 0.590] | – | – | +0.003 [-0.087, +0.084] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.506 [0.420, 0.602] | – | 0.525 [0.423, 0.619] | – | – | -0.018 [-0.143, +0.112] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.501 [0.444, 0.550] | – | 0.568 [0.488, 0.639] | – | – | -0.067 [-0.144, +0.006] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.491 [0.433, 0.553] | – | 0.532 [0.456, 0.606] | – | – | -0.041 [-0.112, +0.042] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.517 [0.472, 0.559] | – | 0.507 [0.436, 0.572] | – | – | +0.010 [-0.039, +0.057] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.494 [0.418, 0.561] | – | 0.514 [0.436, 0.589] | – | – | -0.020 [-0.134, +0.082] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.447 [0.396, 0.494] | – | 0.521 [0.446, 0.600] | – | – | -0.074 [-0.145, +0.008] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.509 [0.451, 0.564] | – | 0.518 [0.451, 0.589] | – | – | -0.009 [-0.108, +0.084] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.531 [0.470, 0.590] | – | 0.529 [0.460, 0.600] | – | – | +0.002 [-0.091, +0.094] | – | – |
| `v.end` | 518 / 362 | 280 | 0.550 [0.481, 0.618] | – | 0.532 [0.459, 0.607] | – | – | +0.018 [-0.073, +0.104] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.506 [0.420, 0.602]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.565, `m2.end` 0.523, `m3.end` 0.495, `m4.end` 0.506. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.516 [0.489, 0.536]; activation minus words -0.027 [-0.063, +0.004]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.505 [0.474, 0.537]; activation minus words -0.023 [-0.082, +0.038]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

