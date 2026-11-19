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
| `m1.pre` | 257 / 183 | 139 | 0.529 [0.438, 0.609] | – | 0.500 [0.500, 0.500] | – | – | +0.029 [-0.062, +0.109] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.540 [0.448, 0.629] | – | 0.525 [0.492, 0.563] | – | – | +0.015 [-0.102, +0.124] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.466 [0.398, 0.539] | – | 0.392 [0.274, 0.504] | – | – | +0.074 [-0.028, +0.187] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.528 [0.444, 0.613] | – | 0.525 [0.409, 0.624] | – | – | +0.003 [-0.102, +0.110] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.446 [0.374, 0.527] | – | 0.468 [0.363, 0.576] | – | – | -0.022 [-0.141, +0.102] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.502 [0.403, 0.610] | – | 0.525 [0.415, 0.636] | – | – | -0.023 [-0.128, +0.084] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.506 [0.392, 0.617] | – | 0.489 [0.371, 0.596] | – | – | +0.017 [-0.103, +0.152] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.499 [0.401, 0.581] | – | 0.482 [0.373, 0.582] | – | – | +0.017 [-0.067, +0.117] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.440 [0.352, 0.540] | – | 0.468 [0.356, 0.571] | – | – | -0.028 [-0.159, +0.124] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.479 [0.397, 0.563] | – | 0.440 [0.359, 0.507] | – | – | +0.040 [-0.046, +0.144] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.495 [0.403, 0.600] | – | 0.411 [0.310, 0.518] | – | – | +0.084 [-0.028, +0.200] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.535 [0.448, 0.609] | – | 0.340 [0.254, 0.434] | – | – | +0.194 [+0.094, +0.284] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.541 [0.467, 0.621] | – | 0.396 [0.290, 0.500] | – | – | +0.145 [+0.035, +0.265] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.551 [0.467, 0.632] | – | 0.396 [0.290, 0.500] | – | – | +0.155 [+0.016, +0.287] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.437 [0.375, 0.497] | – | 0.460 [0.368, 0.552] | – | – | -0.023 [-0.144, +0.101] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.473 [0.383, 0.562] | – | 0.489 [0.385, 0.609] | – | – | -0.016 [-0.171, +0.123] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.492 [0.406, 0.578] | – | 0.496 [0.386, 0.597] | – | – | -0.004 [-0.126, +0.136] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.498 [0.412, 0.585] | – | 0.504 [0.401, 0.599] | – | – | -0.006 [-0.139, +0.118] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.579 [0.454, 0.683] | – | 0.440 [0.336, 0.545] | – | – | +0.139 [+0.027, +0.229] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.559 [0.481, 0.639] | – | 0.440 [0.336, 0.545] | – | – | +0.119 [+0.009, +0.229] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.543 [0.469, 0.626] | – | 0.454 [0.347, 0.554] | – | – | +0.089 [-0.037, +0.247] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.526 [0.448, 0.600] | – | 0.411 [0.306, 0.525] | – | – | +0.115 [-0.011, +0.239] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.475 [0.373, 0.583] | – | 0.433 [0.327, 0.545] | – | – | +0.043 [-0.104, +0.197] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.529 [0.446, 0.610] | – | 0.440 [0.333, 0.543] | – | – | +0.089 [-0.017, +0.208] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.489 [0.436, 0.531] | – | 0.471 [0.402, 0.539] | – | – | +0.018 [-0.069, +0.104] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.536 [0.476, 0.592] | – | 0.454 [0.387, 0.514] | – | – | +0.082 [+0.020, +0.148] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.496 [0.448, 0.547] | – | 0.418 [0.353, 0.475] | – | – | +0.079 [+0.003, +0.157] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.494 [0.432, 0.549] | – | 0.436 [0.364, 0.505] | – | – | +0.058 [-0.027, +0.142] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.484 [0.431, 0.533] | – | 0.429 [0.351, 0.503] | – | – | +0.056 [-0.033, +0.145] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.548 [0.456, 0.631] | – | 0.457 [0.374, 0.540] | – | – | +0.091 [+0.005, +0.185] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.457 [0.384, 0.527] | – | 0.454 [0.375, 0.526] | – | – | +0.004 [-0.065, +0.082] | – | – |
| `v.end` | 518 / 362 | 280 | 0.499 [0.423, 0.585] | – | 0.454 [0.378, 0.528] | – | – | +0.046 [-0.049, +0.126] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.529 [0.446, 0.610]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.502, `m2.end` 0.535, `m3.end` 0.498, `m4.end` 0.529. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.end`, `m3.pre`, `m3.t0`, `m4.pre`, `m4.t0`, `p1.end`, `p2.pre`, `p3.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.507 [0.478, 0.538]; activation minus words +0.052 [+0.002, +0.103]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.500 [0.468, 0.529]; activation minus words +0.054 [+0.007, +0.105]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

