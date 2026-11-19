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
| `m1.pre` | 257 / 183 | 139 | 0.502 [0.431, 0.589] | – | 0.500 [0.500, 0.500] | – | – | +0.002 [-0.069, +0.089] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.503 [0.414, 0.587] | – | 0.482 [0.440, 0.522] | – | – | +0.021 [-0.073, +0.117] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.551 [0.484, 0.617] | – | 0.464 [0.335, 0.571] | – | – | +0.087 [-0.034, +0.224] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.545 [0.448, 0.633] | – | 0.489 [0.377, 0.593] | – | – | +0.056 [-0.119, +0.222] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.443 [0.347, 0.548] | – | 0.453 [0.333, 0.563] | – | – | -0.010 [-0.128, +0.136] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.508 [0.412, 0.584] | – | 0.518 [0.407, 0.616] | – | – | -0.010 [-0.165, +0.134] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.546 [0.473, 0.600] | – | 0.447 [0.355, 0.540] | – | – | +0.099 [-0.018, +0.198] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.518 [0.427, 0.616] | – | 0.447 [0.362, 0.533] | – | – | +0.071 [-0.032, +0.177] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.550 [0.469, 0.634] | – | 0.489 [0.409, 0.561] | – | – | +0.061 [-0.059, +0.180] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.519 [0.429, 0.612] | – | 0.504 [0.429, 0.584] | – | – | +0.016 [-0.080, +0.118] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.590 [0.495, 0.685] | – | 0.496 [0.409, 0.585] | – | – | +0.094 [+0.003, +0.185] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.599 [0.474, 0.708] | – | 0.489 [0.394, 0.586] | – | – | +0.109 [-0.015, +0.225] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.491 [0.426, 0.549] | – | 0.460 [0.352, 0.559] | – | – | +0.030 [-0.087, +0.136] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.495 [0.435, 0.543] | – | 0.460 [0.352, 0.559] | – | – | +0.035 [-0.065, +0.145] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.452 [0.324, 0.563] | – | 0.475 [0.376, 0.578] | – | – | -0.023 [-0.194, +0.128] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.445 [0.349, 0.538] | – | 0.468 [0.373, 0.551] | – | – | -0.023 [-0.151, +0.094] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.403 [0.337, 0.481] | – | 0.482 [0.381, 0.571] | – | – | -0.079 [-0.174, +0.039] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.491 [0.411, 0.575] | – | 0.504 [0.386, 0.601] | – | – | -0.013 [-0.126, +0.120] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.448 [0.320, 0.570] | – | 0.532 [0.419, 0.640] | – | – | -0.084 [-0.237, +0.062] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.414 [0.343, 0.487] | – | 0.525 [0.412, 0.636] | – | – | -0.111 [-0.241, +0.018] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.461 [0.365, 0.557] | – | 0.525 [0.423, 0.621] | – | – | -0.064 [-0.185, +0.038] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.484 [0.358, 0.603] | – | 0.489 [0.400, 0.581] | – | – | -0.006 [-0.146, +0.125] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.498 [0.377, 0.610] | – | 0.489 [0.386, 0.588] | – | – | +0.009 [-0.134, +0.148] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.450 [0.341, 0.566] | – | 0.504 [0.395, 0.617] | – | – | -0.054 [-0.178, +0.053] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.513 [0.447, 0.577] | – | 0.446 [0.381, 0.502] | – | – | +0.066 [-0.034, +0.174] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.522 [0.470, 0.575] | – | 0.464 [0.388, 0.536] | – | – | +0.058 [-0.016, +0.139] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.464 [0.413, 0.517] | – | 0.475 [0.409, 0.540] | – | – | -0.011 [-0.089, +0.084] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.504 [0.452, 0.557] | – | 0.500 [0.438, 0.565] | – | – | +0.004 [-0.086, +0.088] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.489 [0.458, 0.523] | – | 0.486 [0.419, 0.554] | – | – | +0.004 [-0.060, +0.069] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.461 [0.393, 0.512] | – | 0.471 [0.402, 0.543] | – | – | -0.010 [-0.081, +0.062] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.492 [0.442, 0.540] | – | 0.468 [0.396, 0.546] | – | – | +0.024 [-0.079, +0.113] | – | – |
| `v.end` | 518 / 362 | 280 | 0.494 [0.434, 0.558] | – | 0.457 [0.385, 0.536] | – | – | +0.036 [-0.027, +0.101] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.450 [0.341, 0.566]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.508, `m2.end` 0.599, `m3.end` 0.491, `m4.end` 0.450. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.q3`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.496 [0.466, 0.524]; activation minus words +0.009 [-0.038, +0.052]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.492 [0.465, 0.514]; activation minus words +0.021 [-0.032, +0.073]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

