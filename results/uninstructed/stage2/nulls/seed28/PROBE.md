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
| `m1.pre` | 257 / 183 | 139 | 0.548 [0.461, 0.637] | – | 0.500 [0.500, 0.500] | – | – | +0.048 [-0.039, +0.137] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.444 [0.339, 0.541] | – | 0.489 [0.448, 0.525] | – | – | -0.045 [-0.138, +0.046] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.573 [0.478, 0.672] | – | 0.450 [0.383, 0.525] | – | – | +0.123 [+0.024, +0.234] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.532 [0.464, 0.603] | – | 0.540 [0.420, 0.657] | – | – | -0.007 [-0.133, +0.126] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.468 [0.368, 0.562] | – | 0.590 [0.480, 0.698] | – | – | -0.122 [-0.247, +0.001] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.481 [0.387, 0.582] | – | 0.576 [0.480, 0.673] | – | – | -0.095 [-0.176, -0.021] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.530 [0.433, 0.636] | – | 0.440 [0.373, 0.508] | – | – | +0.091 [-0.028, +0.205] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.482 [0.368, 0.590] | – | 0.433 [0.366, 0.500] | – | – | +0.050 [-0.062, +0.164] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.536 [0.460, 0.619] | – | 0.482 [0.419, 0.547] | – | – | +0.054 [-0.046, +0.148] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.548 [0.470, 0.624] | – | 0.496 [0.436, 0.571] | – | – | +0.051 [-0.047, +0.139] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.540 [0.459, 0.627] | – | 0.461 [0.389, 0.533] | – | – | +0.079 [-0.027, +0.195] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.599 [0.510, 0.681] | – | 0.475 [0.355, 0.587] | – | – | +0.123 [+0.013, +0.225] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.463 [0.397, 0.536] | – | 0.482 [0.403, 0.556] | – | – | -0.019 [-0.099, +0.068] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.447 [0.357, 0.530] | – | 0.482 [0.403, 0.556] | – | – | -0.035 [-0.140, +0.057] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.563 [0.505, 0.625] | – | 0.511 [0.401, 0.612] | – | – | +0.052 [-0.061, +0.167] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.453 [0.343, 0.555] | – | 0.532 [0.426, 0.630] | – | – | -0.079 [-0.225, +0.057] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.514 [0.437, 0.592] | – | 0.576 [0.482, 0.667] | – | – | -0.062 [-0.150, +0.035] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.447 [0.352, 0.531] | – | 0.525 [0.418, 0.620] | – | – | -0.078 [-0.198, +0.037] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.511 [0.446, 0.573] | – | 0.433 [0.332, 0.545] | – | – | +0.078 [-0.046, +0.185] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.533 [0.454, 0.610] | – | 0.433 [0.332, 0.545] | – | – | +0.101 [-0.067, +0.241] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.489 [0.391, 0.571] | – | 0.468 [0.375, 0.570] | – | – | +0.021 [-0.105, +0.139] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.546 [0.433, 0.634] | – | 0.447 [0.347, 0.537] | – | – | +0.099 [-0.048, +0.239] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.383 [0.314, 0.449] | – | 0.496 [0.394, 0.586] | – | – | -0.113 [-0.220, -0.003] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.539 [0.470, 0.605] | – | 0.489 [0.385, 0.585] | – | – | +0.050 [-0.042, +0.157] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.493 [0.429, 0.556] | – | 0.479 [0.383, 0.562] | – | – | +0.014 [-0.075, +0.123] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.447 [0.382, 0.516] | – | 0.500 [0.415, 0.583] | – | – | -0.053 [-0.145, +0.039] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.509 [0.449, 0.563] | – | 0.468 [0.384, 0.548] | – | – | +0.041 [-0.028, +0.114] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.504 [0.445, 0.560] | – | 0.468 [0.382, 0.545] | – | – | +0.036 [-0.040, +0.114] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.533 [0.481, 0.583] | – | 0.446 [0.355, 0.533] | – | – | +0.086 [-0.004, +0.189] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.465 [0.424, 0.510] | – | 0.439 [0.352, 0.524] | – | – | +0.026 [-0.066, +0.123] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.577 [0.495, 0.651] | – | 0.439 [0.357, 0.519] | – | – | +0.138 [+0.051, +0.233] | – | – |
| `v.end` | 518 / 362 | 280 | 0.516 [0.444, 0.594] | – | 0.450 [0.371, 0.516] | – | – | +0.066 [-0.030, +0.170] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.539 [0.470, 0.605]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.481, `m2.end` 0.599, `m3.end` 0.447, `m4.end` 0.539. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.q1`, `m2.end`, `v.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.507 [0.475, 0.538]; activation minus words +0.015 [-0.017, +0.048]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.505 [0.469, 0.543]; activation minus words +0.044 [-0.017, +0.112]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

