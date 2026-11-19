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
| `m1.pre` | 257 / 183 | 139 | 0.490 [0.431, 0.549] | – | 0.500 [0.500, 0.500] | – | – | -0.010 [-0.069, +0.049] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.517 [0.448, 0.582] | – | 0.489 [0.447, 0.527] | – | – | +0.028 [-0.052, +0.096] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.525 [0.418, 0.631] | – | 0.522 [0.447, 0.611] | – | – | +0.004 [-0.136, +0.139] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.506 [0.418, 0.610] | – | 0.410 [0.315, 0.517] | – | – | +0.096 [-0.013, +0.198] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.504 [0.445, 0.566] | – | 0.424 [0.309, 0.561] | – | – | +0.079 [-0.071, +0.210] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.413 [0.335, 0.513] | – | 0.410 [0.298, 0.545] | – | – | +0.003 [-0.097, +0.090] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.366 [0.310, 0.433] | – | 0.532 [0.439, 0.634] | – | – | -0.166 [-0.262, -0.054] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.399 [0.324, 0.485] | – | 0.518 [0.423, 0.621] | – | – | -0.119 [-0.223, +0.011] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.458 [0.376, 0.540] | – | 0.525 [0.419, 0.626] | – | – | -0.067 [-0.190, +0.069] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.470 [0.375, 0.562] | – | 0.518 [0.395, 0.614] | – | – | -0.048 [-0.190, +0.115] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.458 [0.370, 0.531] | – | 0.532 [0.382, 0.639] | – | – | -0.074 [-0.200, +0.078] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.623 [0.557, 0.700] | – | 0.489 [0.353, 0.594] | – | – | +0.133 [+0.015, +0.294] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.537 [0.446, 0.641] | – | 0.446 [0.343, 0.550] | – | – | +0.091 [+0.000, +0.178] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.565 [0.484, 0.658] | – | 0.446 [0.343, 0.550] | – | – | +0.119 [+0.021, +0.228] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.588 [0.497, 0.671] | – | 0.410 [0.308, 0.509] | – | – | +0.178 [+0.083, +0.262] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.466 [0.366, 0.579] | – | 0.417 [0.307, 0.512] | – | – | +0.049 [-0.079, +0.186] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.506 [0.411, 0.595] | – | 0.410 [0.306, 0.520] | – | – | +0.096 [-0.030, +0.209] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.459 [0.343, 0.557] | – | 0.381 [0.267, 0.490] | – | – | +0.078 [-0.016, +0.154] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.437 [0.373, 0.520] | – | 0.518 [0.406, 0.606] | – | – | -0.081 [-0.188, +0.065] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.428 [0.339, 0.532] | – | 0.511 [0.386, 0.606] | – | – | -0.082 [-0.228, +0.115] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.438 [0.331, 0.554] | – | 0.454 [0.343, 0.548] | – | – | -0.016 [-0.141, +0.129] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.479 [0.371, 0.577] | – | 0.447 [0.333, 0.548] | – | – | +0.033 [-0.095, +0.159] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.523 [0.456, 0.591] | – | 0.411 [0.278, 0.520] | – | – | +0.112 [+0.003, +0.243] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.365 [0.294, 0.439] | – | 0.426 [0.310, 0.510] | – | – | -0.061 [-0.167, +0.072] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.500 [0.442, 0.551] | – | 0.489 [0.422, 0.559] | – | – | +0.011 [-0.090, +0.102] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.511 [0.475, 0.556] | – | 0.511 [0.443, 0.584] | – | – | +0.001 [-0.085, +0.095] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.478 [0.431, 0.527] | – | 0.479 [0.386, 0.566] | – | – | -0.001 [-0.114, +0.100] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.524 [0.467, 0.582] | – | 0.482 [0.400, 0.556] | – | – | +0.041 [-0.056, +0.137] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.505 [0.439, 0.571] | – | 0.489 [0.413, 0.556] | – | – | +0.016 [-0.094, +0.123] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.467 [0.404, 0.537] | – | 0.496 [0.429, 0.560] | – | – | -0.029 [-0.115, +0.066] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.484 [0.441, 0.529] | – | 0.493 [0.428, 0.553] | – | – | -0.009 [-0.064, +0.056] | – | – |
| `v.end` | 518 / 362 | 280 | 0.494 [0.433, 0.557] | – | 0.489 [0.428, 0.547] | – | – | +0.005 [-0.056, +0.068] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.365 [0.294, 0.439]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.413, `m2.end` 0.623, `m3.end` 0.459, `m4.end` 0.365. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.end`, `m3.t0`, `m3.q1`, `m4.q3`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.480 [0.458, 0.506]; activation minus words +0.016 [-0.037, +0.086]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.495 [0.463, 0.525]; activation minus words +0.004 [-0.065, +0.075]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

