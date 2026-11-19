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
| `m1.pre` | 257 / 183 | 139 | 0.441 [0.364, 0.518] | – | 0.500 [0.500, 0.500] | – | – | -0.059 [-0.136, +0.018] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.494 [0.426, 0.568] | – | 0.511 [0.471, 0.548] | – | – | -0.017 [-0.093, +0.065] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.426 [0.359, 0.501] | – | 0.536 [0.412, 0.656] | – | – | -0.110 [-0.242, +0.013] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.508 [0.421, 0.592] | – | 0.633 [0.517, 0.746] | – | – | -0.125 [-0.254, +0.018] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.529 [0.428, 0.624] | – | 0.554 [0.478, 0.624] | – | – | -0.024 [-0.137, +0.087] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.534 [0.425, 0.646] | – | 0.540 [0.466, 0.612] | – | – | -0.006 [-0.154, +0.153] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.516 [0.444, 0.580] | – | 0.496 [0.402, 0.595] | – | – | +0.020 [-0.099, +0.121] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.546 [0.453, 0.626] | – | 0.496 [0.388, 0.602] | – | – | +0.050 [-0.100, +0.197] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.484 [0.407, 0.559] | – | 0.511 [0.393, 0.630] | – | – | -0.027 [-0.154, +0.107] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.495 [0.411, 0.585] | – | 0.574 [0.439, 0.703] | – | – | -0.079 [-0.236, +0.084] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.599 [0.492, 0.685] | – | 0.489 [0.367, 0.615] | – | – | +0.109 [-0.043, +0.239] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.487 [0.420, 0.549] | – | 0.496 [0.368, 0.625] | – | – | -0.010 [-0.143, +0.131] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.430 [0.361, 0.507] | – | 0.576 [0.490, 0.667] | – | – | -0.145 [-0.260, -0.029] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.456 [0.390, 0.533] | – | 0.568 [0.484, 0.659] | – | – | -0.112 [-0.215, -0.005] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.466 [0.370, 0.559] | – | 0.554 [0.485, 0.634] | – | – | -0.088 [-0.207, +0.015] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.465 [0.384, 0.549] | – | 0.554 [0.479, 0.629] | – | – | -0.089 [-0.196, +0.022] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.531 [0.454, 0.639] | – | 0.518 [0.447, 0.595] | – | – | +0.013 [-0.091, +0.137] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.489 [0.402, 0.595] | – | 0.540 [0.473, 0.618] | – | – | -0.050 [-0.162, +0.068] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.539 [0.458, 0.611] | – | 0.496 [0.388, 0.604] | – | – | +0.043 [-0.080, +0.149] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.485 [0.370, 0.583] | – | 0.504 [0.398, 0.613] | – | – | -0.018 [-0.180, +0.131] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.570 [0.494, 0.634] | – | 0.475 [0.382, 0.567] | – | – | +0.095 [-0.032, +0.219] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.468 [0.349, 0.574] | – | 0.496 [0.394, 0.587] | – | – | -0.028 [-0.210, +0.139] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.465 [0.376, 0.541] | – | 0.496 [0.392, 0.597] | – | – | -0.031 [-0.118, +0.055] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.526 [0.443, 0.599] | – | 0.518 [0.411, 0.615] | – | – | +0.009 [-0.097, +0.105] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.534 [0.485, 0.585] | – | 0.543 [0.473, 0.607] | – | – | -0.009 [-0.089, +0.076] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.528 [0.471, 0.591] | – | 0.525 [0.453, 0.590] | – | – | +0.003 [-0.060, +0.073] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.529 [0.474, 0.585] | – | 0.554 [0.493, 0.611] | – | – | -0.024 [-0.090, +0.040] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.504 [0.452, 0.546] | – | 0.568 [0.518, 0.619] | – | – | -0.064 [-0.132, -0.009] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.549 [0.483, 0.616] | – | 0.582 [0.520, 0.641] | – | – | -0.033 [-0.091, +0.026] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.459 [0.406, 0.508] | – | 0.575 [0.518, 0.632] | – | – | -0.116 [-0.193, -0.045] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.494 [0.437, 0.552] | – | 0.568 [0.513, 0.627] | – | – | -0.074 [-0.137, -0.014] | – | – |
| `v.end` | 518 / 362 | 280 | 0.539 [0.473, 0.595] | – | 0.568 [0.513, 0.624] | – | – | -0.029 [-0.094, +0.021] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.526 [0.443, 0.599]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.534, `m2.end` 0.487, `m3.end` 0.489, `m4.end` 0.526. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: `p2.end`.
  - In one number, the mean over the 24 room positions: activation 0.498 [0.475, 0.517]; activation minus words -0.028 [-0.075, +0.016]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.517 [0.483, 0.549]; activation minus words -0.043 [-0.080, -0.008]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

