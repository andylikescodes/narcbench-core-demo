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
| `m1.pre` | 257 / 183 | 139 | 0.531 [0.457, 0.605] | – | 0.500 [0.500, 0.500] | – | – | +0.031 [-0.043, +0.105] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.520 [0.448, 0.596] | – | 0.504 [0.448, 0.557] | – | – | +0.017 [-0.058, +0.109] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.527 [0.443, 0.604] | – | 0.558 [0.458, 0.676] | – | – | -0.031 [-0.142, +0.068] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.590 [0.510, 0.676] | – | 0.511 [0.412, 0.607] | – | – | +0.079 [-0.020, +0.167] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.557 [0.475, 0.641] | – | 0.496 [0.422, 0.569] | – | – | +0.060 [-0.047, +0.170] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.482 [0.377, 0.594] | – | 0.619 [0.555, 0.685] | – | – | -0.137 [-0.251, -0.024] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.589 [0.487, 0.687] | – | 0.525 [0.425, 0.636] | – | – | +0.064 [-0.093, +0.214] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.582 [0.512, 0.648] | – | 0.539 [0.433, 0.655] | – | – | +0.043 [-0.105, +0.179] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.502 [0.430, 0.593] | – | 0.511 [0.443, 0.585] | – | – | -0.009 [-0.110, +0.104] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.519 [0.436, 0.599] | – | 0.532 [0.441, 0.632] | – | – | -0.013 [-0.103, +0.080] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.526 [0.424, 0.631] | – | 0.539 [0.439, 0.652] | – | – | -0.013 [-0.145, +0.089] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.484 [0.401, 0.564] | – | 0.574 [0.497, 0.667] | – | – | -0.091 [-0.208, +0.008] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.518 [0.400, 0.632] | – | 0.475 [0.377, 0.561] | – | – | +0.043 [-0.099, +0.199] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.555 [0.467, 0.654] | – | 0.482 [0.383, 0.565] | – | – | +0.073 [-0.042, +0.210] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.512 [0.432, 0.600] | – | 0.532 [0.432, 0.615] | – | – | -0.020 [-0.115, +0.103] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.593 [0.517, 0.657] | – | 0.518 [0.437, 0.595] | – | – | +0.075 [-0.029, +0.165] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.557 [0.451, 0.678] | – | 0.568 [0.496, 0.627] | – | – | -0.012 [-0.128, +0.117] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.541 [0.482, 0.602] | – | 0.554 [0.473, 0.625] | – | – | -0.013 [-0.087, +0.071] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.536 [0.452, 0.622] | – | 0.582 [0.507, 0.679] | – | – | -0.045 [-0.142, +0.034] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.489 [0.424, 0.557] | – | 0.589 [0.513, 0.686] | – | – | -0.099 [-0.214, -0.004] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.518 [0.437, 0.602] | – | 0.589 [0.497, 0.703] | – | – | -0.071 [-0.188, +0.018] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.530 [0.455, 0.615] | – | 0.589 [0.496, 0.701] | – | – | -0.058 [-0.162, +0.042] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.583 [0.491, 0.675] | – | 0.603 [0.519, 0.694] | – | – | -0.020 [-0.130, +0.089] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.539 [0.456, 0.618] | – | 0.617 [0.535, 0.706] | – | – | -0.078 [-0.183, +0.012] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.524 [0.459, 0.599] | – | 0.593 [0.521, 0.663] | – | – | -0.069 [-0.148, +0.003] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.524 [0.464, 0.579] | – | 0.586 [0.515, 0.662] | – | – | -0.061 [-0.158, +0.019] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.561 [0.513, 0.608] | – | 0.561 [0.505, 0.623] | – | – | +0.000 [-0.052, +0.049] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.583 [0.539, 0.631] | – | 0.550 [0.502, 0.602] | – | – | +0.033 [-0.029, +0.092] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.525 [0.472, 0.588] | – | 0.529 [0.457, 0.597] | – | – | -0.004 [-0.089, +0.089] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.516 [0.466, 0.560] | – | 0.561 [0.491, 0.625] | – | – | -0.044 [-0.125, +0.037] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.541 [0.480, 0.599] | – | 0.546 [0.484, 0.603] | – | – | -0.006 [-0.076, +0.056] | – | – |
| `v.end` | 518 / 362 | 280 | 0.470 [0.405, 0.542] | – | 0.557 [0.496, 0.615] | – | – | -0.087 [-0.162, -0.005] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.539 [0.456, 0.618]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.482, `m2.end` 0.484, `m3.end` 0.541, `m4.end` 0.539. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: `m4.q3`.
  - In one number, the mean over the 24 room positions: activation 0.537 [0.504, 0.567]; activation minus words -0.009 [-0.058, +0.035]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.530 [0.507, 0.558]; activation minus words -0.030 [-0.076, +0.017]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

