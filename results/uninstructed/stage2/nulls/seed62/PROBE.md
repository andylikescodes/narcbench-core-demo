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
| `m1.pre` | 257 / 183 | 139 | 0.537 [0.472, 0.608] | – | 0.500 [0.500, 0.500] | – | – | +0.037 [-0.028, +0.108] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.581 [0.485, 0.664] | – | 0.525 [0.481, 0.563] | – | – | +0.055 [-0.018, +0.126] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.583 [0.490, 0.679] | – | 0.547 [0.461, 0.632] | – | – | +0.036 [-0.060, +0.145] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.522 [0.422, 0.613] | – | 0.482 [0.391, 0.570] | – | – | +0.040 [-0.088, +0.161] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.578 [0.492, 0.680] | – | 0.590 [0.517, 0.658] | – | – | -0.012 [-0.105, +0.101] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.525 [0.426, 0.630] | – | 0.525 [0.463, 0.592] | – | – | +0.000 [-0.092, +0.098] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.525 [0.426, 0.627] | – | 0.489 [0.426, 0.562] | – | – | +0.035 [-0.086, +0.156] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.519 [0.451, 0.587] | – | 0.475 [0.404, 0.551] | – | – | +0.044 [-0.030, +0.110] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.518 [0.443, 0.607] | – | 0.589 [0.511, 0.671] | – | – | -0.071 [-0.173, +0.045] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.434 [0.350, 0.517] | – | 0.511 [0.432, 0.598] | – | – | -0.077 [-0.197, +0.032] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.512 [0.440, 0.581] | – | 0.496 [0.420, 0.579] | – | – | +0.016 [-0.105, +0.131] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.498 [0.427, 0.564] | – | 0.418 [0.356, 0.487] | – | – | +0.079 [-0.015, +0.163] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.518 [0.423, 0.602] | – | 0.576 [0.505, 0.647] | – | – | -0.058 [-0.177, +0.049] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.588 [0.497, 0.676] | – | 0.583 [0.510, 0.659] | – | – | +0.006 [-0.123, +0.133] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.469 [0.381, 0.552] | – | 0.568 [0.500, 0.636] | – | – | -0.099 [-0.197, -0.009] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.498 [0.397, 0.593] | – | 0.568 [0.489, 0.650] | – | – | -0.071 [-0.216, +0.068] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.571 [0.487, 0.675] | – | 0.604 [0.514, 0.694] | – | – | -0.033 [-0.119, +0.079] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.472 [0.391, 0.562] | – | 0.604 [0.510, 0.707] | – | – | -0.132 [-0.223, -0.037] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.522 [0.457, 0.594] | – | 0.426 [0.358, 0.492] | – | – | +0.096 [+0.006, +0.198] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.502 [0.451, 0.557] | – | 0.426 [0.358, 0.492] | – | – | +0.077 [+0.014, +0.140] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.431 [0.367, 0.488] | – | 0.454 [0.396, 0.520] | – | – | -0.023 [-0.115, +0.060] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.499 [0.394, 0.601] | – | 0.426 [0.339, 0.527] | – | – | +0.074 [-0.053, +0.180] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.484 [0.400, 0.563] | – | 0.475 [0.369, 0.583] | – | – | +0.009 [-0.134, +0.152] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.519 [0.432, 0.609] | – | 0.426 [0.339, 0.521] | – | – | +0.094 [-0.046, +0.241] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.542 [0.478, 0.598] | – | 0.507 [0.452, 0.585] | – | – | +0.035 [-0.065, +0.118] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.484 [0.434, 0.533] | – | 0.486 [0.430, 0.549] | – | – | -0.001 [-0.093, +0.088] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.520 [0.461, 0.581] | – | 0.482 [0.400, 0.575] | – | – | +0.038 [-0.046, +0.114] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.546 [0.491, 0.593] | – | 0.482 [0.402, 0.571] | – | – | +0.064 [-0.038, +0.162] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.489 [0.430, 0.561] | – | 0.479 [0.403, 0.570] | – | – | +0.011 [-0.082, +0.101] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.500 [0.434, 0.565] | – | 0.461 [0.388, 0.551] | – | – | +0.039 [-0.061, +0.138] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.444 [0.389, 0.512] | – | 0.461 [0.395, 0.538] | – | – | -0.017 [-0.077, +0.039] | – | – |
| `v.end` | 518 / 362 | 280 | 0.485 [0.434, 0.536] | – | 0.479 [0.403, 0.567] | – | – | +0.006 [-0.107, +0.105] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.519 [0.432, 0.609]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.525, `m2.end` 0.498, `m3.end` 0.472, `m4.end` 0.519. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.pre`, `m4.t0`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.517 [0.494, 0.538]; activation minus words +0.005 [-0.034, +0.041]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.501 [0.469, 0.536]; activation minus words +0.022 [-0.051, +0.087]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

