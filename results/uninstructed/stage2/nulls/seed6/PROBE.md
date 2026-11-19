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
| `m1.pre` | 257 / 183 | 139 | 0.578 [0.500, 0.667] | – | 0.500 [0.500, 0.500] | – | – | +0.078 [+0.000, +0.167] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.553 [0.473, 0.625] | – | 0.504 [0.477, 0.528] | – | – | +0.049 [-0.029, +0.124] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.524 [0.429, 0.603] | – | 0.529 [0.445, 0.606] | – | – | -0.005 [-0.097, +0.084] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.482 [0.413, 0.549] | – | 0.453 [0.370, 0.535] | – | – | +0.029 [-0.063, +0.127] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.508 [0.417, 0.591] | – | 0.482 [0.405, 0.550] | – | – | +0.026 [-0.097, +0.155] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.522 [0.440, 0.610] | – | 0.532 [0.430, 0.634] | – | – | -0.010 [-0.118, +0.097] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.445 [0.370, 0.531] | – | 0.447 [0.353, 0.536] | – | – | -0.001 [-0.110, +0.119] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.491 [0.404, 0.568] | – | 0.447 [0.357, 0.540] | – | – | +0.044 [-0.079, +0.172] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.460 [0.376, 0.551] | – | 0.404 [0.283, 0.522] | – | – | +0.055 [-0.062, +0.177] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.530 [0.462, 0.595] | – | 0.390 [0.292, 0.479] | – | – | +0.140 [+0.039, +0.239] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.444 [0.370, 0.510] | – | 0.433 [0.315, 0.542] | – | – | +0.011 [-0.146, +0.171] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.406 [0.307, 0.506] | – | 0.404 [0.297, 0.504] | – | – | +0.001 [-0.132, +0.142] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.504 [0.432, 0.579] | – | 0.554 [0.481, 0.634] | – | – | -0.050 [-0.130, +0.024] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.485 [0.401, 0.565] | – | 0.561 [0.484, 0.644] | – | – | -0.076 [-0.160, +0.005] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.482 [0.396, 0.559] | – | 0.576 [0.491, 0.651] | – | – | -0.094 [-0.202, +0.018] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.553 [0.485, 0.612] | – | 0.554 [0.446, 0.650] | – | – | -0.001 [-0.116, +0.121] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.555 [0.456, 0.640] | – | 0.518 [0.431, 0.601] | – | – | +0.037 [-0.091, +0.157] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.506 [0.429, 0.590] | – | 0.504 [0.425, 0.587] | – | – | +0.003 [-0.094, +0.123] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.455 [0.400, 0.517] | – | 0.482 [0.374, 0.600] | – | – | -0.027 [-0.140, +0.078] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.540 [0.480, 0.602] | – | 0.482 [0.374, 0.600] | – | – | +0.058 [-0.046, +0.154] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.487 [0.403, 0.566] | – | 0.489 [0.380, 0.607] | – | – | -0.003 [-0.140, +0.125] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.502 [0.408, 0.598] | – | 0.496 [0.391, 0.606] | – | – | +0.006 [-0.144, +0.143] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.427 [0.341, 0.514] | – | 0.454 [0.348, 0.558] | – | – | -0.027 [-0.184, +0.134] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.470 [0.395, 0.544] | – | 0.482 [0.374, 0.587] | – | – | -0.013 [-0.144, +0.106] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.485 [0.434, 0.540] | – | 0.493 [0.431, 0.570] | – | – | -0.008 [-0.091, +0.066] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.525 [0.475, 0.574] | – | 0.446 [0.396, 0.512] | – | – | +0.079 [+0.015, +0.136] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.465 [0.419, 0.522] | – | 0.471 [0.411, 0.540] | – | – | -0.006 [-0.071, +0.067] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.521 [0.475, 0.577] | – | 0.482 [0.419, 0.553] | – | – | +0.039 [-0.014, +0.094] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.512 [0.463, 0.573] | – | 0.496 [0.437, 0.565] | – | – | +0.016 [-0.061, +0.088] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.486 [0.429, 0.542] | – | 0.489 [0.427, 0.561] | – | – | -0.004 [-0.112, +0.092] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.484 [0.442, 0.528] | – | 0.486 [0.429, 0.554] | – | – | -0.001 [-0.076, +0.057] | – | – |
| `v.end` | 518 / 362 | 280 | 0.470 [0.409, 0.535] | – | 0.482 [0.423, 0.556] | – | – | -0.012 [-0.078, +0.050] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.470 [0.395, 0.544]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.522, `m2.end` 0.406, `m3.end` 0.506, `m4.end` 0.470. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.q2`, `p1.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.496 [0.477, 0.515]; activation minus words +0.010 [-0.037, +0.056]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.493 [0.467, 0.528]; activation minus words +0.013 [-0.039, +0.053]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

