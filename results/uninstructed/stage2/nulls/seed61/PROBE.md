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
| `m1.pre` | 257 / 183 | 139 | 0.562 [0.477, 0.648] | – | 0.500 [0.500, 0.500] | – | – | +0.062 [-0.023, +0.148] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.496 [0.434, 0.547] | – | 0.504 [0.477, 0.530] | – | – | -0.007 [-0.067, +0.040] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.482 [0.372, 0.596] | – | 0.486 [0.379, 0.577] | – | – | -0.004 [-0.135, +0.134] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.517 [0.432, 0.602] | – | 0.561 [0.496, 0.642] | – | – | -0.045 [-0.147, +0.053] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.465 [0.353, 0.570] | – | 0.453 [0.380, 0.529] | – | – | +0.012 [-0.122, +0.127] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.496 [0.403, 0.602] | – | 0.432 [0.336, 0.529] | – | – | +0.065 [-0.018, +0.149] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.433 [0.352, 0.511] | – | 0.525 [0.428, 0.629] | – | – | -0.092 [-0.213, +0.022] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.452 [0.356, 0.548] | – | 0.518 [0.414, 0.626] | – | – | -0.065 [-0.210, +0.078] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.450 [0.363, 0.530] | – | 0.511 [0.400, 0.623] | – | – | -0.061 [-0.179, +0.053] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.396 [0.312, 0.480] | – | 0.454 [0.352, 0.569] | – | – | -0.058 [-0.206, +0.090] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.506 [0.423, 0.590] | – | 0.390 [0.290, 0.518] | – | – | +0.116 [-0.016, +0.215] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.427 [0.367, 0.483] | – | 0.383 [0.257, 0.516] | – | – | +0.044 [-0.081, +0.168] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.564 [0.474, 0.646] | – | 0.403 [0.302, 0.494] | – | – | +0.161 [+0.014, +0.292] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.551 [0.478, 0.621] | – | 0.403 [0.302, 0.494] | – | – | +0.148 [+0.049, +0.251] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.460 [0.383, 0.545] | – | 0.453 [0.364, 0.538] | – | – | +0.007 [-0.100, +0.132] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.473 [0.400, 0.560] | – | 0.410 [0.315, 0.512] | – | – | +0.063 [-0.056, +0.197] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.387 [0.308, 0.469] | – | 0.432 [0.333, 0.527] | – | – | -0.045 [-0.175, +0.082] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.553 [0.471, 0.624] | – | 0.475 [0.374, 0.587] | – | – | +0.078 [-0.054, +0.197] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.513 [0.444, 0.577] | – | 0.333 [0.230, 0.452] | – | – | +0.180 [+0.060, +0.291] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.552 [0.472, 0.632] | – | 0.333 [0.230, 0.452] | – | – | +0.218 [+0.091, +0.346] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.495 [0.401, 0.585] | – | 0.348 [0.247, 0.455] | – | – | +0.148 [+0.026, +0.269] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.400 [0.340, 0.458] | – | 0.369 [0.250, 0.487] | – | – | +0.031 [-0.088, +0.160] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.448 [0.349, 0.532] | – | 0.418 [0.318, 0.511] | – | – | +0.030 [-0.092, +0.147] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.431 [0.379, 0.490] | – | 0.440 [0.348, 0.540] | – | – | -0.009 [-0.137, +0.105] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.491 [0.440, 0.548] | – | 0.486 [0.408, 0.558] | – | – | +0.006 [-0.075, +0.093] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.494 [0.439, 0.536] | – | 0.518 [0.437, 0.599] | – | – | -0.024 [-0.120, +0.062] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.552 [0.493, 0.612] | – | 0.561 [0.479, 0.635] | – | – | -0.009 [-0.058, +0.052] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.527 [0.469, 0.598] | – | 0.529 [0.436, 0.602] | – | – | -0.001 [-0.075, +0.100] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.494 [0.445, 0.534] | – | 0.493 [0.404, 0.580] | – | – | +0.001 [-0.081, +0.080] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.537 [0.459, 0.611] | – | 0.507 [0.423, 0.580] | – | – | +0.030 [-0.033, +0.101] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.497 [0.440, 0.556] | – | 0.504 [0.436, 0.573] | – | – | -0.006 [-0.084, +0.074] | – | – |
| `v.end` | 518 / 362 | 280 | 0.497 [0.429, 0.572] | – | 0.489 [0.423, 0.561] | – | – | +0.008 [-0.069, +0.101] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.431 [0.379, 0.490]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.496, `m2.end` 0.427, `m3.end` 0.553, `m4.end` 0.431. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m3.pre`, `m3.t0`, `m4.pre`, `m4.t0`, `m4.q1`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.480 [0.456, 0.501]; activation minus words +0.041 [-0.007, +0.087]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.511 [0.483, 0.541]; activation minus words +0.001 [-0.045, +0.056]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

