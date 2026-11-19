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
| `m1.pre` | 257 / 183 | 139 | 0.524 [0.422, 0.605] | – | 0.500 [0.500, 0.500] | – | – | +0.024 [-0.078, +0.105] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.498 [0.440, 0.562] | – | 0.482 [0.439, 0.526] | – | – | +0.016 [-0.054, +0.087] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.465 [0.359, 0.568] | – | 0.486 [0.394, 0.581] | – | – | -0.021 [-0.157, +0.113] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.518 [0.414, 0.626] | – | 0.453 [0.374, 0.524] | – | – | +0.065 [-0.060, +0.182] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.453 [0.354, 0.538] | – | 0.489 [0.380, 0.587] | – | – | -0.036 [-0.170, +0.089] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.564 [0.472, 0.654] | – | 0.604 [0.494, 0.708] | – | – | -0.040 [-0.152, +0.066] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.518 [0.450, 0.597] | – | 0.496 [0.375, 0.628] | – | – | +0.021 [-0.143, +0.182] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.508 [0.441, 0.573] | – | 0.496 [0.377, 0.625] | – | – | +0.011 [-0.144, +0.149] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.553 [0.464, 0.642] | – | 0.454 [0.344, 0.576] | – | – | +0.099 [-0.046, +0.244] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.417 [0.323, 0.507] | – | 0.411 [0.317, 0.523] | – | – | +0.006 [-0.121, +0.101] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.573 [0.485, 0.652] | – | 0.461 [0.350, 0.593] | – | – | +0.112 [-0.027, +0.239] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.546 [0.455, 0.642] | – | 0.447 [0.345, 0.568] | – | – | +0.099 [-0.042, +0.216] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.540 [0.476, 0.606] | – | 0.511 [0.392, 0.619] | – | – | +0.029 [-0.089, +0.159] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.534 [0.434, 0.629] | – | 0.504 [0.389, 0.610] | – | – | +0.030 [-0.088, +0.144] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.481 [0.379, 0.565] | – | 0.496 [0.387, 0.601] | – | – | -0.016 [-0.126, +0.097] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.528 [0.432, 0.632] | – | 0.496 [0.388, 0.604] | – | – | +0.032 [-0.103, +0.168] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.365 [0.287, 0.434] | – | 0.468 [0.358, 0.574] | – | – | -0.102 [-0.235, +0.014] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.387 [0.283, 0.498] | – | 0.460 [0.339, 0.576] | – | – | -0.073 [-0.188, +0.039] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.502 [0.423, 0.579] | – | 0.567 [0.457, 0.686] | – | – | -0.065 [-0.190, +0.060] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.470 [0.367, 0.555] | – | 0.567 [0.466, 0.680] | – | – | -0.098 [-0.230, +0.015] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.535 [0.464, 0.615] | – | 0.582 [0.472, 0.706] | – | – | -0.047 [-0.149, +0.053] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.430 [0.355, 0.509] | – | 0.546 [0.430, 0.677] | – | – | -0.116 [-0.279, +0.026] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.390 [0.312, 0.469] | – | 0.518 [0.414, 0.646] | – | – | -0.128 [-0.293, +0.011] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.491 [0.411, 0.566] | – | 0.518 [0.414, 0.637] | – | – | -0.027 [-0.140, +0.086] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.561 [0.502, 0.615] | – | 0.475 [0.392, 0.548] | – | – | +0.086 [+0.002, +0.175] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.483 [0.423, 0.531] | – | 0.511 [0.448, 0.570] | – | – | -0.028 [-0.104, +0.042] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.580 [0.515, 0.639] | – | 0.482 [0.410, 0.549] | – | – | +0.098 [+0.037, +0.169] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.460 [0.392, 0.534] | – | 0.479 [0.402, 0.552] | – | – | -0.019 [-0.096, +0.055] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.560 [0.507, 0.605] | – | 0.482 [0.396, 0.562] | – | – | +0.078 [+0.007, +0.153] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.487 [0.425, 0.554] | – | 0.486 [0.412, 0.565] | – | – | +0.001 [-0.069, +0.078] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.547 [0.508, 0.585] | – | 0.461 [0.393, 0.526] | – | – | +0.086 [+0.012, +0.157] | – | – |
| `v.end` | 518 / 362 | 280 | 0.490 [0.444, 0.540] | – | 0.443 [0.373, 0.513] | – | – | +0.047 [-0.036, +0.135] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.491 [0.411, 0.566]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.564, `m2.end` 0.546, `m3.end` 0.387, `m4.end` 0.491. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `p1.pre`, `p2.pre`, `p3.pre`, `v.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.491 [0.458, 0.523]; activation minus words -0.009 [-0.069, +0.040]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.521 [0.488, 0.551]; activation minus words +0.044 [+0.001, +0.089]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

