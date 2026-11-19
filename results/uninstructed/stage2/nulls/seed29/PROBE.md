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
| `m1.pre` | 257 / 183 | 139 | 0.484 [0.401, 0.556] | – | 0.500 [0.500, 0.500] | – | – | -0.016 [-0.099, +0.056] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.499 [0.440, 0.563] | – | 0.511 [0.487, 0.537] | – | – | -0.012 [-0.085, +0.060] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.515 [0.447, 0.583] | – | 0.518 [0.422, 0.613] | – | – | -0.003 [-0.116, +0.101] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.511 [0.420, 0.602] | – | 0.647 [0.534, 0.757] | – | – | -0.137 [-0.272, -0.006] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.593 [0.509, 0.673] | – | 0.619 [0.511, 0.714] | – | – | -0.026 [-0.097, +0.064] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.495 [0.420, 0.592] | – | 0.590 [0.504, 0.673] | – | – | -0.095 [-0.189, +0.024] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.582 [0.515, 0.651] | – | 0.504 [0.410, 0.603] | – | – | +0.078 [-0.030, +0.186] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.574 [0.497, 0.654] | – | 0.504 [0.411, 0.603] | – | – | +0.071 [-0.040, +0.166] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.475 [0.385, 0.576] | – | 0.553 [0.438, 0.662] | – | – | -0.078 [-0.181, +0.031] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.457 [0.333, 0.574] | – | 0.539 [0.435, 0.657] | – | – | -0.082 [-0.237, +0.060] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.569 [0.505, 0.633] | – | 0.553 [0.435, 0.674] | – | – | +0.016 [-0.131, +0.145] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.427 [0.367, 0.488] | – | 0.539 [0.417, 0.685] | – | – | -0.112 [-0.267, +0.028] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.525 [0.437, 0.610] | – | 0.626 [0.525, 0.735] | – | – | -0.101 [-0.179, -0.012] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.529 [0.432, 0.626] | – | 0.619 [0.516, 0.726] | – | – | -0.089 [-0.181, +0.020] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.524 [0.455, 0.603] | – | 0.619 [0.498, 0.748] | – | – | -0.095 [-0.211, +0.036] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.550 [0.463, 0.644] | – | 0.626 [0.488, 0.757] | – | – | -0.076 [-0.177, +0.034] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.442 [0.335, 0.550] | – | 0.590 [0.460, 0.726] | – | – | -0.148 [-0.276, -0.022] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.550 [0.444, 0.663] | – | 0.604 [0.468, 0.753] | – | – | -0.055 [-0.160, +0.065] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.560 [0.502, 0.634] | – | 0.589 [0.500, 0.692] | – | – | -0.028 [-0.109, +0.045] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.521 [0.437, 0.589] | – | 0.582 [0.480, 0.695] | – | – | -0.061 [-0.209, +0.061] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.600 [0.534, 0.654] | – | 0.596 [0.484, 0.710] | – | – | +0.004 [-0.127, +0.126] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.583 [0.522, 0.645] | – | 0.567 [0.453, 0.694] | – | – | +0.016 [-0.100, +0.110] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.519 [0.413, 0.612] | – | 0.539 [0.439, 0.660] | – | – | -0.020 [-0.200, +0.117] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.538 [0.423, 0.643] | – | 0.539 [0.438, 0.659] | – | – | -0.001 [-0.134, +0.110] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.521 [0.462, 0.587] | – | 0.586 [0.514, 0.657] | – | – | -0.065 [-0.135, +0.014] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.543 [0.483, 0.606] | – | 0.554 [0.474, 0.626] | – | – | -0.011 [-0.088, +0.074] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.518 [0.458, 0.582] | – | 0.554 [0.464, 0.641] | – | – | -0.036 [-0.121, +0.055] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.477 [0.422, 0.548] | – | 0.539 [0.450, 0.634] | – | – | -0.062 [-0.123, +0.014] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.536 [0.486, 0.589] | – | 0.550 [0.467, 0.635] | – | – | -0.014 [-0.088, +0.069] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.544 [0.478, 0.607] | – | 0.532 [0.442, 0.625] | – | – | +0.012 [-0.072, +0.105] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.534 [0.454, 0.618] | – | 0.521 [0.452, 0.596] | – | – | +0.012 [-0.076, +0.100] | – | – |
| `v.end` | 518 / 362 | 280 | 0.527 [0.476, 0.590] | – | 0.525 [0.453, 0.608] | – | – | +0.002 [-0.056, +0.066] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.538 [0.423, 0.643]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.495, `m2.end` 0.427, `m3.end` 0.550, `m4.end` 0.538. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: `m1.q2`.
  - In one number, the mean over the 24 room positions: activation 0.526 [0.503, 0.549]; activation minus words -0.044 [-0.084, -0.008]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.525 [0.484, 0.572]; activation minus words -0.020 [-0.070, +0.040]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

