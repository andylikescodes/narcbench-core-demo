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
| `m1.pre` | 257 / 183 | 139 | 0.493 [0.424, 0.559] | – | 0.500 [0.500, 0.500] | – | – | -0.007 [-0.076, +0.059] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.482 [0.423, 0.557] | – | 0.475 [0.441, 0.509] | – | – | +0.007 [-0.052, +0.074] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.522 [0.442, 0.595] | – | 0.493 [0.407, 0.584] | – | – | +0.029 [-0.051, +0.095] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.450 [0.393, 0.522] | – | 0.525 [0.432, 0.631] | – | – | -0.075 [-0.169, +0.011] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.489 [0.381, 0.584] | – | 0.504 [0.418, 0.605] | – | – | -0.014 [-0.131, +0.104] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.442 [0.359, 0.532] | – | 0.460 [0.384, 0.551] | – | – | -0.019 [-0.113, +0.083] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.420 [0.319, 0.536] | – | 0.489 [0.396, 0.600] | – | – | -0.070 [-0.231, +0.083] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.475 [0.402, 0.573] | – | 0.489 [0.394, 0.598] | – | – | -0.014 [-0.129, +0.103] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.488 [0.383, 0.587] | – | 0.461 [0.351, 0.571] | – | – | +0.027 [-0.140, +0.189] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.397 [0.283, 0.531] | – | 0.489 [0.387, 0.582] | – | – | -0.092 [-0.230, +0.062] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.460 [0.373, 0.538] | – | 0.475 [0.375, 0.571] | – | – | -0.016 [-0.134, +0.095] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.521 [0.434, 0.601] | – | 0.482 [0.364, 0.589] | – | – | +0.038 [-0.060, +0.150] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.525 [0.464, 0.590] | – | 0.489 [0.406, 0.582] | – | – | +0.036 [-0.075, +0.135] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.509 [0.442, 0.590] | – | 0.496 [0.410, 0.590] | – | – | +0.013 [-0.063, +0.104] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.527 [0.456, 0.594] | – | 0.547 [0.455, 0.640] | – | – | -0.020 [-0.106, +0.055] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.545 [0.450, 0.635] | – | 0.590 [0.481, 0.706] | – | – | -0.045 [-0.167, +0.061] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.483 [0.394, 0.576] | – | 0.576 [0.453, 0.711] | – | – | -0.092 [-0.233, +0.047] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.557 [0.480, 0.627] | – | 0.496 [0.396, 0.599] | – | – | +0.060 [-0.067, +0.179] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.504 [0.397, 0.612] | – | 0.397 [0.268, 0.508] | – | – | +0.106 [-0.067, +0.300] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.504 [0.415, 0.596] | – | 0.397 [0.268, 0.508] | – | – | +0.106 [-0.051, +0.292] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.413 [0.336, 0.508] | – | 0.355 [0.250, 0.455] | – | – | +0.058 [-0.075, +0.205] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.516 [0.422, 0.597] | – | 0.383 [0.260, 0.511] | – | – | +0.133 [+0.005, +0.276] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.633 [0.517, 0.736] | – | 0.411 [0.290, 0.537] | – | – | +0.221 [+0.005, +0.428] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.593 [0.525, 0.662] | – | 0.390 [0.294, 0.495] | – | – | +0.203 [+0.098, +0.305] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.486 [0.424, 0.554] | – | 0.500 [0.436, 0.576] | – | – | -0.014 [-0.090, +0.067] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.451 [0.394, 0.509] | – | 0.454 [0.378, 0.532] | – | – | -0.003 [-0.123, +0.110] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.519 [0.470, 0.570] | – | 0.450 [0.381, 0.517] | – | – | +0.069 [-0.028, +0.166] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.479 [0.428, 0.535] | – | 0.450 [0.382, 0.519] | – | – | +0.029 [-0.056, +0.115] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.520 [0.464, 0.585] | – | 0.489 [0.427, 0.547] | – | – | +0.031 [-0.057, +0.134] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.484 [0.423, 0.544] | – | 0.482 [0.416, 0.540] | – | – | +0.002 [-0.072, +0.079] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.461 [0.421, 0.505] | – | 0.486 [0.431, 0.543] | – | – | -0.024 [-0.092, +0.048] | – | – |
| `v.end` | 518 / 362 | 280 | 0.494 [0.411, 0.573] | – | 0.479 [0.431, 0.529] | – | – | +0.015 [-0.072, +0.104] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.593 [0.525, 0.662]. Yes: the interval is above 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.442, `m2.end` 0.521, `m3.end` 0.557, `m4.end` 0.593. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.q2`, `m4.q3`, `m4.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.498 [0.469, 0.534]; activation minus words +0.024 [-0.028, +0.080]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.487 [0.457, 0.517]; activation minus words +0.013 [-0.046, +0.078]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

