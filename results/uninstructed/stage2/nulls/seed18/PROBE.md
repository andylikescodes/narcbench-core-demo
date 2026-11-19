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
| `m1.pre` | 257 / 183 | 139 | 0.595 [0.516, 0.663] | – | 0.500 [0.500, 0.500] | – | – | +0.095 [+0.016, +0.163] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.509 [0.429, 0.592] | – | 0.460 [0.428, 0.500] | – | – | +0.049 [-0.033, +0.121] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.499 [0.414, 0.569] | – | 0.622 [0.520, 0.723] | – | – | -0.123 [-0.263, -0.001] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.443 [0.357, 0.521] | – | 0.576 [0.495, 0.660] | – | – | -0.132 [-0.263, -0.011] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.519 [0.404, 0.627] | – | 0.504 [0.417, 0.596] | – | – | +0.016 [-0.126, +0.133] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.450 [0.345, 0.535] | – | 0.460 [0.358, 0.558] | – | – | -0.010 [-0.142, +0.116] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.478 [0.390, 0.558] | – | 0.489 [0.400, 0.578] | – | – | -0.011 [-0.116, +0.096] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.526 [0.434, 0.621] | – | 0.475 [0.387, 0.562] | – | – | +0.051 [-0.039, +0.157] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.545 [0.459, 0.613] | – | 0.496 [0.404, 0.579] | – | – | +0.048 [-0.058, +0.147] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.549 [0.445, 0.654] | – | 0.504 [0.416, 0.584] | – | – | +0.045 [-0.094, +0.204] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.532 [0.416, 0.642] | – | 0.504 [0.402, 0.598] | – | – | +0.028 [-0.103, +0.168] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.569 [0.477, 0.655] | – | 0.475 [0.362, 0.594] | – | – | +0.094 [-0.035, +0.234] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.481 [0.385, 0.580] | – | 0.511 [0.423, 0.603] | – | – | -0.030 [-0.144, +0.073] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.439 [0.365, 0.513] | – | 0.518 [0.428, 0.610] | – | – | -0.079 [-0.193, +0.018] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.473 [0.393, 0.556] | – | 0.561 [0.473, 0.642] | – | – | -0.088 [-0.165, -0.013] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.437 [0.350, 0.542] | – | 0.518 [0.445, 0.591] | – | – | -0.081 [-0.185, +0.033] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.488 [0.405, 0.564] | – | 0.511 [0.435, 0.587] | – | – | -0.023 [-0.133, +0.074] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.512 [0.414, 0.600] | – | 0.482 [0.431, 0.533] | – | – | +0.030 [-0.073, +0.127] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.637 [0.541, 0.717] | – | 0.567 [0.447, 0.677] | – | – | +0.070 [-0.057, +0.194] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.572 [0.493, 0.658] | – | 0.553 [0.426, 0.663] | – | – | +0.018 [-0.102, +0.149] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.538 [0.422, 0.638] | – | 0.539 [0.416, 0.639] | – | – | -0.001 [-0.108, +0.100] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.433 [0.357, 0.514] | – | 0.518 [0.400, 0.624] | – | – | -0.085 [-0.238, +0.073] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.489 [0.385, 0.596] | – | 0.539 [0.422, 0.645] | – | – | -0.050 [-0.218, +0.122] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.609 [0.498, 0.717] | – | 0.582 [0.476, 0.675] | – | – | +0.027 [-0.065, +0.133] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.559 [0.516, 0.597] | – | 0.493 [0.416, 0.556] | – | – | +0.066 [-0.002, +0.145] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.504 [0.437, 0.558] | – | 0.475 [0.395, 0.545] | – | – | +0.029 [-0.065, +0.121] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.528 [0.464, 0.589] | – | 0.507 [0.440, 0.569] | – | – | +0.021 [-0.074, +0.122] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.523 [0.459, 0.582] | – | 0.514 [0.442, 0.584] | – | – | +0.009 [-0.092, +0.093] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.553 [0.503, 0.598] | – | 0.518 [0.452, 0.589] | – | – | +0.035 [-0.049, +0.113] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.501 [0.445, 0.562] | – | 0.507 [0.448, 0.567] | – | – | -0.006 [-0.083, +0.070] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.484 [0.427, 0.529] | – | 0.554 [0.486, 0.611] | – | – | -0.069 [-0.152, +0.015] | – | – |
| `v.end` | 518 / 362 | 280 | 0.491 [0.433, 0.542] | – | 0.529 [0.455, 0.594] | – | – | -0.038 [-0.109, +0.024] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.609 [0.498, 0.717]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.450, `m2.end` 0.569, `m3.end` 0.512, `m4.end` 0.609. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.513 [0.479, 0.543]; activation minus words -0.006 [-0.041, +0.028]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.518 [0.484, 0.546]; activation minus words +0.006 [-0.058, +0.067]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

