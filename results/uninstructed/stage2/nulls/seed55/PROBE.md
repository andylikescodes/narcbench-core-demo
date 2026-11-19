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
| `m1.pre` | 257 / 183 | 139 | 0.526 [0.445, 0.607] | – | 0.500 [0.500, 0.500] | – | – | +0.026 [-0.055, +0.107] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.610 [0.519, 0.693] | – | 0.561 [0.520, 0.605] | – | – | +0.049 [-0.034, +0.141] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.576 [0.490, 0.660] | – | 0.572 [0.487, 0.664] | – | – | +0.004 [-0.123, +0.106] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.521 [0.436, 0.594] | – | 0.432 [0.339, 0.518] | – | – | +0.089 [-0.023, +0.187] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.483 [0.398, 0.573] | – | 0.403 [0.317, 0.500] | – | – | +0.081 [-0.028, +0.199] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.388 [0.317, 0.476] | – | 0.374 [0.286, 0.455] | – | – | +0.014 [-0.088, +0.126] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.434 [0.367, 0.505] | – | 0.461 [0.385, 0.541] | – | – | -0.027 [-0.133, +0.084] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.553 [0.470, 0.638] | – | 0.461 [0.381, 0.545] | – | – | +0.092 [-0.027, +0.213] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.599 [0.516, 0.687] | – | 0.496 [0.406, 0.595] | – | – | +0.102 [-0.017, +0.228] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.550 [0.446, 0.651] | – | 0.496 [0.417, 0.571] | – | – | +0.054 [-0.075, +0.178] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.546 [0.486, 0.601] | – | 0.454 [0.347, 0.548] | – | – | +0.092 [+0.003, +0.196] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.518 [0.430, 0.601] | – | 0.482 [0.386, 0.571] | – | – | +0.035 [-0.094, +0.162] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.505 [0.446, 0.564] | – | 0.475 [0.383, 0.570] | – | – | +0.030 [-0.097, +0.154] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.489 [0.392, 0.566] | – | 0.482 [0.395, 0.577] | – | – | +0.007 [-0.144, +0.136] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.553 [0.450, 0.661] | – | 0.468 [0.367, 0.565] | – | – | +0.085 [-0.025, +0.195] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.531 [0.438, 0.613] | – | 0.453 [0.346, 0.551] | – | – | +0.078 [-0.023, +0.185] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.468 [0.364, 0.585] | – | 0.468 [0.351, 0.575] | – | – | +0.000 [-0.177, +0.205] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.496 [0.398, 0.586] | – | 0.453 [0.337, 0.551] | – | – | +0.043 [-0.077, +0.170] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.485 [0.392, 0.567] | – | 0.511 [0.417, 0.618] | – | – | -0.026 [-0.168, +0.098] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.417 [0.308, 0.513] | – | 0.511 [0.417, 0.618] | – | – | -0.094 [-0.253, +0.056] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.556 [0.466, 0.635] | – | 0.504 [0.390, 0.623] | – | – | +0.052 [-0.078, +0.159] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.383 [0.283, 0.475] | – | 0.525 [0.418, 0.641] | – | – | -0.142 [-0.299, -0.000] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.498 [0.417, 0.569] | – | 0.546 [0.452, 0.632] | – | – | -0.048 [-0.158, +0.054] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.396 [0.316, 0.471] | – | 0.511 [0.410, 0.598] | – | – | -0.115 [-0.200, -0.021] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.474 [0.424, 0.515] | – | 0.514 [0.422, 0.594] | – | – | -0.041 [-0.121, +0.042] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.414 [0.344, 0.491] | – | 0.529 [0.455, 0.591] | – | – | -0.115 [-0.208, -0.016] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.529 [0.473, 0.574] | – | 0.504 [0.416, 0.579] | – | – | +0.026 [-0.049, +0.105] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.579 [0.513, 0.639] | – | 0.514 [0.431, 0.586] | – | – | +0.064 [+0.016, +0.122] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.505 [0.451, 0.550] | – | 0.525 [0.437, 0.610] | – | – | -0.020 [-0.106, +0.069] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.575 [0.519, 0.624] | – | 0.507 [0.427, 0.586] | – | – | +0.068 [-0.022, +0.160] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.556 [0.497, 0.603] | – | 0.521 [0.436, 0.599] | – | – | +0.034 [-0.059, +0.133] | – | – |
| `v.end` | 518 / 362 | 280 | 0.559 [0.466, 0.641] | – | 0.525 [0.441, 0.603] | – | – | +0.034 [-0.049, +0.105] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.396 [0.316, 0.471]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.388, `m2.end` 0.518, `m3.end` 0.496, `m4.end` 0.396. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.q3`, `p2.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.503 [0.472, 0.534]; activation minus words +0.020 [-0.027, +0.062]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.524 [0.485, 0.558]; activation minus words +0.006 [-0.048, +0.062]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

