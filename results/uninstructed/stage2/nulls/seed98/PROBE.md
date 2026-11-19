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
| `m1.pre` | 257 / 183 | 139 | 0.552 [0.461, 0.643] | – | 0.500 [0.500, 0.500] | – | – | +0.052 [-0.039, +0.143] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.464 [0.377, 0.550] | – | 0.518 [0.487, 0.546] | – | – | -0.054 [-0.142, +0.047] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.406 [0.295, 0.511] | – | 0.522 [0.411, 0.653] | – | – | -0.116 [-0.223, -0.016] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.517 [0.439, 0.596] | – | 0.518 [0.475, 0.571] | – | – | -0.001 [-0.083, +0.079] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.407 [0.323, 0.498] | – | 0.504 [0.433, 0.585] | – | – | -0.096 [-0.185, -0.002] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.525 [0.437, 0.602] | – | 0.396 [0.314, 0.480] | – | – | +0.129 [+0.018, +0.227] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.484 [0.408, 0.567] | – | 0.489 [0.409, 0.569] | – | – | -0.006 [-0.107, +0.093] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.499 [0.441, 0.561] | – | 0.489 [0.406, 0.569] | – | – | +0.010 [-0.074, +0.096] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.539 [0.474, 0.606] | – | 0.518 [0.408, 0.626] | – | – | +0.021 [-0.109, +0.142] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.481 [0.384, 0.579] | – | 0.496 [0.407, 0.592] | – | – | -0.016 [-0.179, +0.137] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.440 [0.339, 0.548] | – | 0.496 [0.421, 0.584] | – | – | -0.057 [-0.185, +0.077] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.555 [0.444, 0.646] | – | 0.504 [0.444, 0.573] | – | – | +0.051 [-0.059, +0.141] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.558 [0.494, 0.621] | – | 0.475 [0.382, 0.569] | – | – | +0.083 [-0.032, +0.191] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.551 [0.475, 0.635] | – | 0.475 [0.382, 0.569] | – | – | +0.076 [-0.039, +0.188] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.554 [0.460, 0.640] | – | 0.460 [0.379, 0.544] | – | – | +0.094 [-0.009, +0.198] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.522 [0.449, 0.609] | – | 0.496 [0.417, 0.583] | – | – | +0.026 [-0.076, +0.113] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.489 [0.388, 0.586] | – | 0.489 [0.404, 0.584] | – | – | -0.000 [-0.144, +0.135] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.518 [0.408, 0.625] | – | 0.547 [0.460, 0.629] | – | – | -0.029 [-0.132, +0.069] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.475 [0.388, 0.564] | – | 0.496 [0.428, 0.570] | – | – | -0.021 [-0.138, +0.084] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.496 [0.417, 0.586] | – | 0.482 [0.416, 0.556] | – | – | +0.014 [-0.084, +0.127] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.455 [0.379, 0.539] | – | 0.504 [0.424, 0.594] | – | – | -0.048 [-0.179, +0.078] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.495 [0.392, 0.580] | – | 0.525 [0.434, 0.627] | – | – | -0.030 [-0.164, +0.077] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.542 [0.469, 0.622] | – | 0.496 [0.408, 0.590] | – | – | +0.045 [-0.047, +0.141] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.495 [0.398, 0.596] | – | 0.468 [0.397, 0.542] | – | – | +0.027 [-0.079, +0.125] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.502 [0.445, 0.562] | – | 0.489 [0.430, 0.559] | – | – | +0.013 [-0.082, +0.111] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.508 [0.449, 0.571] | – | 0.471 [0.403, 0.545] | – | – | +0.036 [-0.061, +0.144] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.459 [0.401, 0.514] | – | 0.454 [0.382, 0.528] | – | – | +0.005 [-0.114, +0.108] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.488 [0.405, 0.584] | – | 0.464 [0.389, 0.538] | – | – | +0.024 [-0.062, +0.115] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.464 [0.420, 0.514] | – | 0.518 [0.438, 0.594] | – | – | -0.054 [-0.144, +0.038] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.583 [0.523, 0.650] | – | 0.511 [0.431, 0.585] | – | – | +0.072 [+0.000, +0.154] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.479 [0.411, 0.548] | – | 0.514 [0.436, 0.590] | – | – | -0.036 [-0.148, +0.073] | – | – |
| `v.end` | 518 / 362 | 280 | 0.519 [0.462, 0.588] | – | 0.514 [0.425, 0.597] | – | – | +0.005 [-0.082, +0.094] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.495 [0.398, 0.596]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.525, `m2.end` 0.555, `m3.end` 0.518, `m4.end` 0.495. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.501 [0.480, 0.524]; activation minus words +0.006 [-0.039, +0.048]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.500 [0.475, 0.530]; activation minus words +0.008 [-0.058, +0.079]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

