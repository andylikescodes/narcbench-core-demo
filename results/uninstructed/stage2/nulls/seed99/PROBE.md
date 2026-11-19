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
| `m1.pre` | 257 / 183 | 139 | 0.519 [0.436, 0.611] | – | 0.500 [0.500, 0.500] | – | – | +0.019 [-0.064, +0.111] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.527 [0.448, 0.606] | – | 0.496 [0.464, 0.531] | – | – | +0.031 [-0.044, +0.105] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.512 [0.435, 0.591] | – | 0.424 [0.323, 0.541] | – | – | +0.088 [-0.047, +0.206] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.501 [0.403, 0.601] | – | 0.496 [0.381, 0.618] | – | – | +0.004 [-0.122, +0.143] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.460 [0.386, 0.539] | – | 0.439 [0.333, 0.558] | – | – | +0.022 [-0.137, +0.167] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.544 [0.462, 0.621] | – | 0.475 [0.399, 0.569] | – | – | +0.069 [-0.044, +0.159] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.550 [0.469, 0.650] | – | 0.560 [0.482, 0.659] | – | – | -0.010 [-0.123, +0.100] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.492 [0.398, 0.573] | – | 0.560 [0.482, 0.659] | – | – | -0.068 [-0.211, +0.052] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.521 [0.448, 0.585] | – | 0.496 [0.418, 0.578] | – | – | +0.024 [-0.099, +0.135] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.538 [0.423, 0.655] | – | 0.504 [0.419, 0.596] | – | – | +0.034 [-0.106, +0.186] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.556 [0.456, 0.667] | – | 0.511 [0.411, 0.619] | – | – | +0.045 [-0.068, +0.181] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.509 [0.443, 0.577] | – | 0.418 [0.317, 0.528] | – | – | +0.091 [-0.032, +0.203] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.492 [0.416, 0.569] | – | 0.518 [0.426, 0.631] | – | – | -0.026 [-0.135, +0.069] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.518 [0.445, 0.593] | – | 0.518 [0.426, 0.631] | – | – | +0.000 [-0.098, +0.094] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.498 [0.428, 0.572] | – | 0.504 [0.419, 0.599] | – | – | -0.006 [-0.084, +0.068] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.505 [0.398, 0.602] | – | 0.489 [0.403, 0.582] | – | – | +0.016 [-0.093, +0.115] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.541 [0.439, 0.633] | – | 0.511 [0.416, 0.631] | – | – | +0.030 [-0.106, +0.138] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.515 [0.445, 0.585] | – | 0.504 [0.403, 0.621] | – | – | +0.012 [-0.107, +0.120] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.487 [0.400, 0.580] | – | 0.426 [0.320, 0.549] | – | – | +0.061 [-0.090, +0.210] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.461 [0.404, 0.527] | – | 0.411 [0.308, 0.532] | – | – | +0.050 [-0.077, +0.173] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.489 [0.412, 0.573] | – | 0.440 [0.330, 0.558] | – | – | +0.050 [-0.098, +0.198] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.546 [0.462, 0.646] | – | 0.433 [0.336, 0.542] | – | – | +0.113 [-0.016, +0.257] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.467 [0.388, 0.551] | – | 0.426 [0.339, 0.540] | – | – | +0.041 [-0.091, +0.158] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.468 [0.402, 0.543] | – | 0.418 [0.331, 0.535] | – | – | +0.050 [-0.046, +0.132] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.551 [0.498, 0.611] | – | 0.475 [0.413, 0.548] | – | – | +0.076 [+0.001, +0.150] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.486 [0.421, 0.552] | – | 0.471 [0.410, 0.536] | – | – | +0.014 [-0.079, +0.100] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.416 [0.374, 0.466] | – | 0.486 [0.433, 0.549] | – | – | -0.070 [-0.130, -0.014] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.541 [0.495, 0.588] | – | 0.471 [0.423, 0.528] | – | – | +0.070 [-0.008, +0.139] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.491 [0.441, 0.541] | – | 0.500 [0.441, 0.568] | – | – | -0.009 [-0.080, +0.064] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.551 [0.499, 0.605] | – | 0.486 [0.426, 0.558] | – | – | +0.065 [-0.007, +0.133] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.544 [0.482, 0.618] | – | 0.468 [0.413, 0.534] | – | – | +0.076 [+0.004, +0.162] | – | – |
| `v.end` | 518 / 362 | 280 | 0.469 [0.425, 0.520] | – | 0.468 [0.409, 0.540] | – | – | +0.001 [-0.067, +0.065] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.468 [0.402, 0.543]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.544, `m2.end` 0.509, `m3.end` 0.515, `m4.end` 0.468. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `p1.pre`, `v.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.509 [0.486, 0.535]; activation minus words +0.031 [-0.021, +0.077]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.506 [0.479, 0.536]; activation minus words +0.028 [-0.022, +0.077]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

