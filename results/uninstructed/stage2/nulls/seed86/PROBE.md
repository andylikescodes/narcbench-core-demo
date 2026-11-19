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
| `m1.pre` | 257 / 183 | 139 | 0.451 [0.397, 0.497] | – | 0.500 [0.500, 0.500] | – | – | -0.049 [-0.103, -0.003] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.494 [0.420, 0.579] | – | 0.489 [0.455, 0.521] | – | – | +0.004 [-0.075, +0.088] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.445 [0.364, 0.527] | – | 0.435 [0.314, 0.549] | – | – | +0.009 [-0.105, +0.145] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.427 [0.344, 0.493] | – | 0.446 [0.352, 0.532] | – | – | -0.019 [-0.133, +0.093] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.489 [0.387, 0.594] | – | 0.525 [0.446, 0.602] | – | – | -0.036 [-0.141, +0.084] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.514 [0.391, 0.621] | – | 0.489 [0.382, 0.603] | – | – | +0.024 [-0.118, +0.167] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.559 [0.464, 0.644] | – | 0.454 [0.356, 0.542] | – | – | +0.105 [-0.038, +0.247] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.563 [0.458, 0.659] | – | 0.454 [0.358, 0.539] | – | – | +0.109 [-0.034, +0.260] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.482 [0.391, 0.561] | – | 0.447 [0.328, 0.573] | – | – | +0.035 [-0.080, +0.142] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.506 [0.432, 0.596] | – | 0.454 [0.327, 0.590] | – | – | +0.052 [-0.092, +0.203] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.434 [0.357, 0.521] | – | 0.475 [0.355, 0.607] | – | – | -0.041 [-0.179, +0.078] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.475 [0.407, 0.546] | – | 0.461 [0.312, 0.599] | – | – | +0.014 [-0.135, +0.172] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.525 [0.458, 0.604] | – | 0.489 [0.377, 0.593] | – | – | +0.036 [-0.072, +0.146] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.583 [0.512, 0.654] | – | 0.489 [0.377, 0.593] | – | – | +0.094 [-0.039, +0.229] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.567 [0.494, 0.649] | – | 0.518 [0.405, 0.632] | – | – | +0.049 [-0.071, +0.172] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.468 [0.404, 0.534] | – | 0.547 [0.444, 0.655] | – | – | -0.079 [-0.189, +0.047] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.588 [0.519, 0.653] | – | 0.540 [0.447, 0.639] | – | – | +0.049 [-0.059, +0.157] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.433 [0.366, 0.502] | – | 0.547 [0.442, 0.649] | – | – | -0.114 [-0.245, +0.021] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.431 [0.351, 0.514] | – | 0.504 [0.359, 0.642] | – | – | -0.072 [-0.214, +0.073] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.451 [0.376, 0.518] | – | 0.496 [0.356, 0.632] | – | – | -0.045 [-0.224, +0.118] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.532 [0.447, 0.611] | – | 0.525 [0.380, 0.655] | – | – | +0.007 [-0.161, +0.172] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.528 [0.452, 0.600] | – | 0.518 [0.372, 0.643] | – | – | +0.010 [-0.156, +0.200] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.472 [0.396, 0.544] | – | 0.504 [0.395, 0.607] | – | – | -0.031 [-0.167, +0.109] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.423 [0.334, 0.518] | – | 0.496 [0.381, 0.599] | – | – | -0.074 [-0.168, +0.027] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.506 [0.453, 0.561] | – | 0.496 [0.396, 0.582] | – | – | +0.009 [-0.084, +0.124] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.463 [0.407, 0.522] | – | 0.457 [0.389, 0.526] | – | – | +0.006 [-0.065, +0.081] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.486 [0.422, 0.544] | – | 0.500 [0.400, 0.600] | – | – | -0.014 [-0.093, +0.066] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.509 [0.435, 0.575] | – | 0.475 [0.382, 0.580] | – | – | +0.034 [-0.070, +0.129] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.509 [0.464, 0.560] | – | 0.500 [0.405, 0.600] | – | – | +0.009 [-0.080, +0.097] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.499 [0.440, 0.556] | – | 0.504 [0.414, 0.592] | – | – | -0.005 [-0.099, +0.091] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.474 [0.405, 0.551] | – | 0.486 [0.407, 0.567] | – | – | -0.012 [-0.075, +0.060] | – | – |
| `v.end` | 518 / 362 | 280 | 0.444 [0.373, 0.515] | – | 0.479 [0.403, 0.559] | – | – | -0.035 [-0.146, +0.063] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.423 [0.334, 0.518]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.514, `m2.end` 0.475, `m3.end` 0.433, `m4.end` 0.423. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.493 [0.468, 0.518]; activation minus words +0.002 [-0.058, +0.067]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.486 [0.451, 0.522]; activation minus words -0.001 [-0.059, +0.062]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

