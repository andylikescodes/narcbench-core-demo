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
| `m1.pre` | 257 / 183 | 139 | 0.460 [0.372, 0.551] | – | 0.500 [0.500, 0.500] | – | – | -0.040 [-0.128, +0.051] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.460 [0.407, 0.510] | – | 0.489 [0.457, 0.522] | – | – | -0.029 [-0.081, +0.027] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.537 [0.479, 0.606] | – | 0.417 [0.339, 0.490] | – | – | +0.119 [+0.039, +0.207] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.535 [0.456, 0.607] | – | 0.504 [0.397, 0.613] | – | – | +0.032 [-0.121, +0.187] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.529 [0.420, 0.638] | – | 0.460 [0.355, 0.553] | – | – | +0.069 [-0.070, +0.206] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.570 [0.469, 0.657] | – | 0.518 [0.405, 0.629] | – | – | +0.052 [-0.079, +0.189] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.530 [0.438, 0.614] | – | 0.624 [0.514, 0.733] | – | – | -0.094 [-0.224, +0.033] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.488 [0.397, 0.577] | – | 0.624 [0.510, 0.739] | – | – | -0.136 [-0.238, -0.037] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.482 [0.373, 0.607] | – | 0.652 [0.554, 0.754] | – | – | -0.170 [-0.303, -0.031] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.454 [0.373, 0.542] | – | 0.574 [0.480, 0.674] | – | – | -0.121 [-0.232, +0.007] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.565 [0.467, 0.647] | – | 0.638 [0.514, 0.752] | – | – | -0.074 [-0.196, +0.037] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.539 [0.417, 0.650] | – | 0.645 [0.542, 0.748] | – | – | -0.106 [-0.240, +0.020] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.496 [0.417, 0.579] | – | 0.482 [0.409, 0.554] | – | – | +0.014 [-0.076, +0.105] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.508 [0.411, 0.613] | – | 0.482 [0.409, 0.554] | – | – | +0.026 [-0.079, +0.125] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.557 [0.458, 0.654] | – | 0.511 [0.433, 0.591] | – | – | +0.046 [-0.068, +0.152] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.482 [0.374, 0.592] | – | 0.468 [0.402, 0.538] | – | – | +0.014 [-0.088, +0.116] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.466 [0.378, 0.557] | – | 0.446 [0.366, 0.526] | – | – | +0.020 [-0.104, +0.140] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.449 [0.366, 0.546] | – | 0.388 [0.301, 0.474] | – | – | +0.060 [-0.032, +0.161] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.495 [0.426, 0.571] | – | 0.553 [0.457, 0.646] | – | – | -0.058 [-0.149, +0.044] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.410 [0.323, 0.504] | – | 0.553 [0.457, 0.646] | – | – | -0.143 [-0.275, -0.004] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.478 [0.382, 0.570] | – | 0.525 [0.435, 0.636] | – | – | -0.047 [-0.205, +0.098] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.508 [0.438, 0.590] | – | 0.518 [0.436, 0.609] | – | – | -0.010 [-0.131, +0.112] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.532 [0.436, 0.636] | – | 0.511 [0.406, 0.617] | – | – | +0.021 [-0.115, +0.132] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.545 [0.439, 0.648] | – | 0.525 [0.430, 0.630] | – | – | +0.020 [-0.116, +0.148] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.540 [0.481, 0.595] | – | 0.486 [0.432, 0.549] | – | – | +0.054 [-0.046, +0.146] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.526 [0.478, 0.589] | – | 0.479 [0.405, 0.561] | – | – | +0.047 [-0.047, +0.127] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.539 [0.484, 0.586] | – | 0.475 [0.400, 0.559] | – | – | +0.064 [-0.024, +0.145] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.529 [0.483, 0.584] | – | 0.496 [0.413, 0.584] | – | – | +0.032 [-0.048, +0.103] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.536 [0.475, 0.593] | – | 0.504 [0.423, 0.588] | – | – | +0.033 [-0.079, +0.138] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.491 [0.433, 0.537] | – | 0.496 [0.419, 0.582] | – | – | -0.005 [-0.110, +0.089] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.471 [0.414, 0.531] | – | 0.518 [0.447, 0.594] | – | – | -0.047 [-0.121, +0.020] | – | – |
| `v.end` | 518 / 362 | 280 | 0.539 [0.488, 0.586] | – | 0.514 [0.447, 0.591] | – | – | +0.024 [-0.041, +0.090] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.545 [0.439, 0.648]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.570, `m2.end` 0.539, `m3.end` 0.449, `m4.end` 0.545. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.q1`. The bag of words' own onset: `m2.pre`.
  - In one number, the mean over the 24 room positions: activation 0.503 [0.488, 0.521]; activation minus words -0.022 [-0.061, +0.019]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.521 [0.498, 0.542]; activation minus words +0.025 [-0.040, +0.088]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

