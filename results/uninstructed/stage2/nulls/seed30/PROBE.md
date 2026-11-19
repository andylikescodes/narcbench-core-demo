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
| `m1.pre` | 257 / 183 | 139 | 0.393 [0.314, 0.462] | – | 0.500 [0.500, 0.500] | – | – | -0.107 [-0.186, -0.038] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.456 [0.380, 0.531] | – | 0.496 [0.463, 0.527] | – | – | -0.040 [-0.131, +0.045] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.462 [0.397, 0.531] | – | 0.475 [0.376, 0.568] | – | – | -0.013 [-0.102, +0.091] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.531 [0.417, 0.631] | – | 0.468 [0.366, 0.578] | – | – | +0.063 [-0.047, +0.163] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.587 [0.499, 0.670] | – | 0.460 [0.325, 0.591] | – | – | +0.127 [+0.003, +0.245] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.637 [0.557, 0.708] | – | 0.460 [0.355, 0.561] | – | – | +0.177 [+0.077, +0.279] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.384 [0.327, 0.435] | – | 0.525 [0.426, 0.611] | – | – | -0.140 [-0.227, -0.052] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.382 [0.300, 0.445] | – | 0.532 [0.429, 0.619] | – | – | -0.150 [-0.243, -0.058] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.409 [0.329, 0.507] | – | 0.539 [0.435, 0.646] | – | – | -0.130 [-0.225, -0.010] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.418 [0.318, 0.502] | – | 0.539 [0.425, 0.656] | – | – | -0.121 [-0.226, -0.021] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.457 [0.404, 0.515] | – | 0.489 [0.379, 0.582] | – | – | -0.033 [-0.121, +0.069] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.535 [0.447, 0.625] | – | 0.447 [0.344, 0.541] | – | – | +0.088 [-0.033, +0.204] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.545 [0.469, 0.627] | – | 0.388 [0.300, 0.492] | – | – | +0.157 [+0.051, +0.265] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.558 [0.461, 0.673] | – | 0.388 [0.300, 0.492] | – | – | +0.170 [+0.071, +0.285] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.512 [0.428, 0.610] | – | 0.453 [0.366, 0.551] | – | – | +0.059 [-0.064, +0.207] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.494 [0.410, 0.577] | – | 0.489 [0.408, 0.584] | – | – | +0.004 [-0.112, +0.117] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.498 [0.416, 0.588] | – | 0.532 [0.444, 0.626] | – | – | -0.035 [-0.180, +0.092] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.560 [0.476, 0.639] | – | 0.554 [0.467, 0.644] | – | – | +0.006 [-0.097, +0.109] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.414 [0.357, 0.488] | – | 0.454 [0.373, 0.530] | – | – | -0.040 [-0.130, +0.064] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.400 [0.334, 0.480] | – | 0.461 [0.377, 0.536] | – | – | -0.061 [-0.135, +0.033] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.468 [0.379, 0.560] | – | 0.475 [0.377, 0.570] | – | – | -0.007 [-0.129, +0.117] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.495 [0.391, 0.599] | – | 0.504 [0.406, 0.590] | – | – | -0.009 [-0.120, +0.111] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.502 [0.413, 0.580] | – | 0.489 [0.388, 0.583] | – | – | +0.013 [-0.065, +0.092] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.465 [0.380, 0.556] | – | 0.504 [0.416, 0.583] | – | – | -0.038 [-0.138, +0.073] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.465 [0.403, 0.532] | – | 0.439 [0.390, 0.493] | – | – | +0.026 [-0.061, +0.112] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.447 [0.401, 0.507] | – | 0.475 [0.414, 0.540] | – | – | -0.028 [-0.100, +0.055] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.514 [0.451, 0.567] | – | 0.468 [0.391, 0.534] | – | – | +0.046 [-0.025, +0.118] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.547 [0.495, 0.597] | – | 0.450 [0.377, 0.517] | – | – | +0.097 [+0.020, +0.186] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.510 [0.444, 0.578] | – | 0.457 [0.399, 0.507] | – | – | +0.053 [-0.023, +0.132] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.506 [0.458, 0.564] | – | 0.446 [0.380, 0.505] | – | – | +0.060 [-0.012, +0.134] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.431 [0.388, 0.470] | – | 0.446 [0.374, 0.507] | – | – | -0.015 [-0.060, +0.035] | – | – |
| `v.end` | 518 / 362 | 280 | 0.530 [0.482, 0.585] | – | 0.457 [0.386, 0.513] | – | – | +0.073 [-0.009, +0.171] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.465 [0.380, 0.556]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.637, `m2.end` 0.535, `m3.end` 0.560, `m4.end` 0.465. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.q3`, `m1.end`, `m3.pre`, `m3.t0`, `p2.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.482 [0.458, 0.506]; activation minus words -0.003 [-0.036, +0.035]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.494 [0.472, 0.519]; activation minus words +0.039 [-0.013, +0.099]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

