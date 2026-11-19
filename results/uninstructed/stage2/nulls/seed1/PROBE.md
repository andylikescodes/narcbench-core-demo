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
| `m1.pre` | 257 / 183 | 139 | 0.561 [0.481, 0.645] | – | 0.500 [0.500, 0.500] | – | – | +0.061 [-0.019, +0.145] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.545 [0.481, 0.610] | – | 0.504 [0.464, 0.538] | – | – | +0.041 [-0.037, +0.124] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.483 [0.431, 0.542] | – | 0.612 [0.511, 0.711] | – | – | -0.128 [-0.234, -0.011] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.499 [0.405, 0.581] | – | 0.647 [0.539, 0.743] | – | – | -0.148 [-0.275, -0.025] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.541 [0.454, 0.635] | – | 0.540 [0.460, 0.613] | – | – | +0.001 [-0.089, +0.117] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.548 [0.483, 0.626] | – | 0.460 [0.369, 0.543] | – | – | +0.088 [+0.008, +0.180] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.546 [0.463, 0.636] | – | 0.525 [0.442, 0.608] | – | – | +0.021 [-0.055, +0.104] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.553 [0.474, 0.627] | – | 0.525 [0.442, 0.608] | – | – | +0.028 [-0.060, +0.106] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.631 [0.520, 0.741] | – | 0.582 [0.489, 0.676] | – | – | +0.050 [-0.058, +0.164] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.363 [0.287, 0.433] | – | 0.638 [0.549, 0.738] | – | – | -0.275 [-0.406, -0.160] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.540 [0.450, 0.639] | – | 0.617 [0.525, 0.700] | – | – | -0.077 [-0.187, +0.035] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.528 [0.421, 0.647] | – | 0.574 [0.467, 0.667] | – | – | -0.047 [-0.188, +0.110] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.479 [0.394, 0.567] | – | 0.468 [0.358, 0.575] | – | – | +0.012 [-0.126, +0.135] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.528 [0.429, 0.620] | – | 0.475 [0.369, 0.582] | – | – | +0.053 [-0.111, +0.198] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.488 [0.417, 0.561] | – | 0.504 [0.394, 0.619] | – | – | -0.016 [-0.113, +0.089] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.506 [0.424, 0.580] | – | 0.511 [0.407, 0.619] | – | – | -0.004 [-0.147, +0.125] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.397 [0.313, 0.494] | – | 0.475 [0.374, 0.589] | – | – | -0.078 [-0.193, +0.033] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.449 [0.367, 0.529] | – | 0.489 [0.396, 0.589] | – | – | -0.040 [-0.142, +0.077] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.512 [0.433, 0.589] | – | 0.645 [0.531, 0.757] | – | – | -0.133 [-0.253, -0.007] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.484 [0.397, 0.580] | – | 0.660 [0.549, 0.765] | – | – | -0.176 [-0.301, -0.046] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.511 [0.411, 0.601] | – | 0.610 [0.492, 0.722] | – | – | -0.099 [-0.197, +0.002] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.509 [0.411, 0.595] | – | 0.638 [0.505, 0.763] | – | – | -0.129 [-0.295, +0.028] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.471 [0.393, 0.542] | – | 0.610 [0.503, 0.714] | – | – | -0.139 [-0.259, -0.007] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.516 [0.433, 0.587] | – | 0.567 [0.456, 0.667] | – | – | -0.051 [-0.155, +0.059] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.466 [0.412, 0.509] | – | 0.493 [0.413, 0.559] | – | – | -0.026 [-0.101, +0.054] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.465 [0.401, 0.535] | – | 0.507 [0.438, 0.573] | – | – | -0.042 [-0.141, +0.056] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.439 [0.386, 0.491] | – | 0.482 [0.415, 0.549] | – | – | -0.043 [-0.142, +0.056] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.481 [0.425, 0.532] | – | 0.468 [0.395, 0.538] | – | – | +0.013 [-0.085, +0.102] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.489 [0.431, 0.548] | – | 0.468 [0.398, 0.536] | – | – | +0.021 [-0.046, +0.094] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.479 [0.420, 0.531] | – | 0.468 [0.407, 0.521] | – | – | +0.011 [-0.060, +0.093] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.495 [0.436, 0.549] | – | 0.439 [0.383, 0.494] | – | – | +0.056 [+0.002, +0.112] | – | – |
| `v.end` | 518 / 362 | 280 | 0.448 [0.390, 0.508] | – | 0.464 [0.406, 0.518] | – | – | -0.016 [-0.082, +0.059] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.516 [0.433, 0.587]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.548, `m2.end` 0.528, `m3.end` 0.449, `m4.end` 0.516. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.end`, `v.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.508 [0.483, 0.533]; activation minus words -0.049 [-0.077, -0.018]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.470 [0.442, 0.498]; activation minus words -0.003 [-0.060, +0.053]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

