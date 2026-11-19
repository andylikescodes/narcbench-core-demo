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
| `m1.pre` | 257 / 183 | 139 | 0.490 [0.422, 0.554] | – | 0.500 [0.500, 0.500] | – | – | -0.010 [-0.078, +0.054] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.481 [0.427, 0.545] | – | 0.482 [0.449, 0.513] | – | – | -0.001 [-0.064, +0.072] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.524 [0.452, 0.593] | – | 0.471 [0.375, 0.567] | – | – | +0.053 [-0.079, +0.193] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.557 [0.472, 0.635] | – | 0.489 [0.380, 0.587] | – | – | +0.068 [-0.035, +0.169] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.447 [0.365, 0.532] | – | 0.504 [0.427, 0.586] | – | – | -0.056 [-0.179, +0.069] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.514 [0.416, 0.614] | – | 0.475 [0.376, 0.571] | – | – | +0.039 [-0.082, +0.160] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.474 [0.415, 0.544] | – | 0.511 [0.403, 0.614] | – | – | -0.037 [-0.170, +0.100] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.566 [0.490, 0.635] | – | 0.518 [0.414, 0.621] | – | – | +0.048 [-0.090, +0.186] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.482 [0.429, 0.547] | – | 0.518 [0.414, 0.614] | – | – | -0.035 [-0.155, +0.098] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.464 [0.375, 0.553] | – | 0.489 [0.374, 0.600] | – | – | -0.026 [-0.147, +0.105] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.496 [0.404, 0.597] | – | 0.468 [0.384, 0.555] | – | – | +0.028 [-0.091, +0.150] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.577 [0.519, 0.647] | – | 0.454 [0.365, 0.558] | – | – | +0.123 [+0.008, +0.234] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.525 [0.459, 0.589] | – | 0.504 [0.412, 0.608] | – | – | +0.022 [-0.084, +0.125] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.537 [0.465, 0.612] | – | 0.504 [0.412, 0.608] | – | – | +0.033 [-0.079, +0.145] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.465 [0.392, 0.542] | – | 0.504 [0.402, 0.602] | – | – | -0.039 [-0.150, +0.082] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.399 [0.329, 0.479] | – | 0.511 [0.402, 0.623] | – | – | -0.112 [-0.254, +0.032] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.396 [0.318, 0.480] | – | 0.489 [0.386, 0.601] | – | – | -0.094 [-0.225, +0.049] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.453 [0.391, 0.525] | – | 0.576 [0.455, 0.692] | – | – | -0.122 [-0.255, +0.020] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.444 [0.353, 0.528] | – | 0.511 [0.407, 0.608] | – | – | -0.067 [-0.146, +0.015] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.516 [0.437, 0.606] | – | 0.511 [0.407, 0.608] | – | – | +0.006 [-0.088, +0.135] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.409 [0.333, 0.487] | – | 0.482 [0.398, 0.571] | – | – | -0.074 [-0.171, +0.025] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.400 [0.319, 0.482] | – | 0.482 [0.385, 0.589] | – | – | -0.082 [-0.194, +0.030] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.580 [0.508, 0.657] | – | 0.482 [0.401, 0.580] | – | – | +0.098 [-0.005, +0.199] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.479 [0.371, 0.583] | – | 0.489 [0.414, 0.578] | – | – | -0.010 [-0.149, +0.113] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.477 [0.423, 0.538] | – | 0.486 [0.417, 0.573] | – | – | -0.009 [-0.107, +0.081] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.437 [0.372, 0.509] | – | 0.464 [0.408, 0.534] | – | – | -0.027 [-0.125, +0.062] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.448 [0.396, 0.504] | – | 0.486 [0.420, 0.552] | – | – | -0.038 [-0.122, +0.048] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.505 [0.450, 0.559] | – | 0.486 [0.430, 0.543] | – | – | +0.019 [-0.058, +0.089] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.516 [0.476, 0.554] | – | 0.486 [0.416, 0.557] | – | – | +0.030 [-0.055, +0.115] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.422 [0.361, 0.494] | – | 0.489 [0.426, 0.551] | – | – | -0.067 [-0.125, +0.000] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.496 [0.449, 0.548] | – | 0.479 [0.419, 0.538] | – | – | +0.017 [-0.058, +0.093] | – | – |
| `v.end` | 518 / 362 | 280 | 0.431 [0.367, 0.495] | – | 0.475 [0.410, 0.542] | – | – | -0.044 [-0.145, +0.040] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.479 [0.371, 0.583]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.514, `m2.end` 0.577, `m3.end` 0.453, `m4.end` 0.479. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.486 [0.463, 0.509]; activation minus words -0.010 [-0.054, +0.036]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.466 [0.445, 0.489]; activation minus words -0.015 [-0.069, +0.041]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

