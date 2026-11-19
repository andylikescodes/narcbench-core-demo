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
| `m1.pre` | 257 / 183 | 139 | 0.570 [0.518, 0.621] | – | 0.500 [0.500, 0.500] | – | – | +0.070 [+0.018, +0.121] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.576 [0.509, 0.651] | – | 0.489 [0.455, 0.520] | – | – | +0.086 [+0.010, +0.172] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.447 [0.370, 0.537] | – | 0.532 [0.441, 0.638] | – | – | -0.085 [-0.193, +0.017] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.488 [0.409, 0.576] | – | 0.597 [0.497, 0.699] | – | – | -0.109 [-0.200, -0.016] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.417 [0.328, 0.497] | – | 0.540 [0.444, 0.629] | – | – | -0.122 [-0.246, +0.012] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.540 [0.455, 0.620] | – | 0.540 [0.421, 0.639] | – | – | +0.000 [-0.132, +0.143] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.485 [0.409, 0.582] | – | 0.468 [0.381, 0.550] | – | – | +0.017 [-0.066, +0.109] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.543 [0.468, 0.620] | – | 0.468 [0.388, 0.550] | – | – | +0.075 [-0.028, +0.184] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.496 [0.404, 0.576] | – | 0.489 [0.397, 0.583] | – | – | +0.007 [-0.095, +0.091] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.521 [0.419, 0.645] | – | 0.404 [0.327, 0.500] | – | – | +0.116 [+0.007, +0.248] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.470 [0.364, 0.589] | – | 0.390 [0.295, 0.495] | – | – | +0.079 [-0.061, +0.218] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.479 [0.403, 0.580] | – | 0.411 [0.320, 0.508] | – | – | +0.068 [-0.059, +0.211] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.506 [0.412, 0.602] | – | 0.475 [0.364, 0.571] | – | – | +0.032 [-0.079, +0.169] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.534 [0.454, 0.626] | – | 0.482 [0.367, 0.579] | – | – | +0.052 [-0.051, +0.183] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.535 [0.465, 0.619] | – | 0.475 [0.377, 0.564] | – | – | +0.060 [-0.057, +0.211] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.512 [0.440, 0.591] | – | 0.504 [0.403, 0.588] | – | – | +0.009 [-0.097, +0.145] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.458 [0.374, 0.556] | – | 0.518 [0.423, 0.607] | – | – | -0.060 [-0.181, +0.081] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.473 [0.385, 0.553] | – | 0.489 [0.411, 0.564] | – | – | -0.016 [-0.113, +0.079] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.530 [0.442, 0.625] | – | 0.404 [0.340, 0.470] | – | – | +0.126 [+0.045, +0.228] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.526 [0.438, 0.611] | – | 0.411 [0.345, 0.476] | – | – | +0.115 [+0.024, +0.212] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.498 [0.419, 0.573] | – | 0.454 [0.379, 0.536] | – | – | +0.044 [-0.068, +0.154] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.498 [0.416, 0.575] | – | 0.418 [0.322, 0.515] | – | – | +0.079 [-0.026, +0.197] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.499 [0.405, 0.611] | – | 0.440 [0.355, 0.519] | – | – | +0.060 [-0.055, +0.197] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.443 [0.370, 0.519] | – | 0.454 [0.370, 0.529] | – | – | -0.011 [-0.088, +0.087] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.473 [0.425, 0.520] | – | 0.464 [0.397, 0.521] | – | – | +0.009 [-0.068, +0.091] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.538 [0.484, 0.591] | – | 0.468 [0.392, 0.540] | – | – | +0.070 [-0.019, +0.157] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.524 [0.479, 0.563] | – | 0.479 [0.423, 0.548] | – | – | +0.046 [-0.027, +0.106] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.497 [0.429, 0.566] | – | 0.489 [0.434, 0.551] | – | – | +0.008 [-0.078, +0.097] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.485 [0.430, 0.536] | – | 0.486 [0.420, 0.563] | – | – | -0.001 [-0.095, +0.083] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.465 [0.406, 0.531] | – | 0.486 [0.424, 0.561] | – | – | -0.021 [-0.109, +0.067] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.518 [0.465, 0.584] | – | 0.496 [0.440, 0.556] | – | – | +0.021 [-0.040, +0.080] | – | – |
| `v.end` | 518 / 362 | 280 | 0.526 [0.473, 0.568] | – | 0.504 [0.437, 0.575] | – | – | +0.022 [-0.063, +0.092] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.443 [0.370, 0.519]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.540, `m2.end` 0.479, `m3.end` 0.473, `m4.end` 0.443. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.pre`, `m1.t0`, `m2.q2`, `m4.pre`, `m4.t0`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.502 [0.473, 0.539]; activation minus words +0.029 [-0.018, +0.090]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.503 [0.479, 0.524]; activation minus words +0.019 [-0.038, +0.073]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

