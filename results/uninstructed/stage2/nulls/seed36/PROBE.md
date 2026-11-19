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
| `m1.pre` | 257 / 183 | 139 | 0.515 [0.406, 0.618] | – | 0.500 [0.500, 0.500] | – | – | +0.015 [-0.094, +0.118] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.499 [0.431, 0.575] | – | 0.453 [0.411, 0.496] | – | – | +0.046 [-0.022, +0.112] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.468 [0.388, 0.561] | – | 0.536 [0.451, 0.631] | – | – | -0.068 [-0.168, +0.047] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.472 [0.392, 0.538] | – | 0.561 [0.473, 0.646] | – | – | -0.089 [-0.207, +0.025] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.607 [0.535, 0.689] | – | 0.619 [0.519, 0.726] | – | – | -0.012 [-0.146, +0.105] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.580 [0.508, 0.658] | – | 0.590 [0.486, 0.703] | – | – | -0.010 [-0.120, +0.095] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.428 [0.341, 0.517] | – | 0.518 [0.404, 0.641] | – | – | -0.089 [-0.211, +0.023] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.420 [0.345, 0.498] | – | 0.511 [0.395, 0.639] | – | – | -0.091 [-0.232, +0.036] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.506 [0.416, 0.597] | – | 0.447 [0.351, 0.549] | – | – | +0.060 [-0.086, +0.207] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.508 [0.404, 0.613] | – | 0.397 [0.343, 0.454] | – | – | +0.111 [-0.000, +0.226] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.489 [0.413, 0.567] | – | 0.426 [0.325, 0.529] | – | – | +0.064 [-0.039, +0.169] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.562 [0.453, 0.682] | – | 0.440 [0.357, 0.525] | – | – | +0.122 [+0.003, +0.243] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.508 [0.420, 0.591] | – | 0.561 [0.469, 0.654] | – | – | -0.053 [-0.168, +0.041] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.538 [0.440, 0.643] | – | 0.561 [0.469, 0.654] | – | – | -0.023 [-0.154, +0.106] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.542 [0.443, 0.648] | – | 0.547 [0.438, 0.645] | – | – | -0.004 [-0.130, +0.120] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.555 [0.452, 0.648] | – | 0.532 [0.432, 0.634] | – | – | +0.023 [-0.133, +0.169] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.594 [0.487, 0.693] | – | 0.547 [0.448, 0.647] | – | – | +0.047 [-0.064, +0.160] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.643 [0.565, 0.732] | – | 0.554 [0.472, 0.644] | – | – | +0.089 [-0.008, +0.193] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.616 [0.526, 0.712] | – | 0.511 [0.405, 0.612] | – | – | +0.105 [+0.007, +0.209] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.546 [0.471, 0.618] | – | 0.511 [0.405, 0.612] | – | – | +0.035 [-0.079, +0.161] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.577 [0.456, 0.681] | – | 0.468 [0.371, 0.568] | – | – | +0.109 [-0.018, +0.234] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.489 [0.400, 0.580] | – | 0.496 [0.403, 0.590] | – | – | -0.007 [-0.135, +0.117] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.489 [0.406, 0.581] | – | 0.496 [0.394, 0.597] | – | – | -0.007 [-0.107, +0.095] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.487 [0.402, 0.582] | – | 0.511 [0.410, 0.622] | – | – | -0.024 [-0.126, +0.084] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.513 [0.453, 0.569] | – | 0.629 [0.558, 0.702] | – | – | -0.116 [-0.207, -0.024] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.462 [0.395, 0.541] | – | 0.586 [0.509, 0.666] | – | – | -0.124 [-0.236, -0.010] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.450 [0.398, 0.506] | – | 0.611 [0.539, 0.677] | – | – | -0.161 [-0.229, -0.086] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.514 [0.462, 0.572] | – | 0.607 [0.538, 0.674] | – | – | -0.093 [-0.167, -0.007] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.471 [0.410, 0.529] | – | 0.607 [0.547, 0.667] | – | – | -0.136 [-0.206, -0.069] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.458 [0.413, 0.507] | – | 0.593 [0.523, 0.665] | – | – | -0.135 [-0.205, -0.054] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.506 [0.453, 0.567] | – | 0.611 [0.537, 0.680] | – | – | -0.104 [-0.193, -0.012] | – | – |
| `v.end` | 518 / 362 | 280 | 0.524 [0.473, 0.584] | – | 0.611 [0.531, 0.682] | – | – | -0.086 [-0.178, +0.013] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.487 [0.402, 0.582]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.580, `m2.end` 0.562, `m3.end` 0.643, `m4.end` 0.487. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.end`, `m4.pre`. The bag of words' own onset: `p1.pre`.
  - In one number, the mean over the 24 room positions: activation 0.527 [0.510, 0.545]; activation minus words +0.015 [-0.040, +0.058]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.487 [0.458, 0.523]; activation minus words -0.119 [-0.180, -0.061]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

