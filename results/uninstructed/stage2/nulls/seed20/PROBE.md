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
| `m1.pre` | 257 / 183 | 139 | 0.580 [0.498, 0.659] | – | 0.500 [0.500, 0.500] | – | – | +0.080 [-0.002, +0.159] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.479 [0.422, 0.542] | – | 0.489 [0.457, 0.521] | – | – | -0.010 [-0.082, +0.071] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.501 [0.428, 0.600] | – | 0.554 [0.453, 0.639] | – | – | -0.053 [-0.165, +0.090] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.486 [0.419, 0.555] | – | 0.532 [0.415, 0.635] | – | – | -0.046 [-0.150, +0.090] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.535 [0.439, 0.629] | – | 0.504 [0.432, 0.577] | – | – | +0.032 [-0.063, +0.120] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.468 [0.376, 0.556] | – | 0.460 [0.365, 0.558] | – | – | +0.007 [-0.124, +0.144] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.477 [0.408, 0.556] | – | 0.461 [0.369, 0.569] | – | – | +0.016 [-0.109, +0.135] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.461 [0.395, 0.529] | – | 0.454 [0.358, 0.560] | – | – | +0.007 [-0.142, +0.145] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.559 [0.469, 0.642] | – | 0.426 [0.354, 0.505] | – | – | +0.133 [-0.007, +0.263] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.498 [0.422, 0.580] | – | 0.418 [0.340, 0.507] | – | – | +0.079 [-0.022, +0.163] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.451 [0.375, 0.522] | – | 0.397 [0.306, 0.489] | – | – | +0.054 [-0.043, +0.152] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.539 [0.450, 0.627] | – | 0.411 [0.286, 0.543] | – | – | +0.128 [-0.031, +0.269] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.478 [0.398, 0.559] | – | 0.496 [0.398, 0.590] | – | – | -0.019 [-0.099, +0.065] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.453 [0.363, 0.543] | – | 0.496 [0.398, 0.590] | – | – | -0.043 [-0.138, +0.064] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.527 [0.442, 0.617] | – | 0.453 [0.360, 0.556] | – | – | +0.073 [-0.072, +0.208] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.514 [0.421, 0.602] | – | 0.475 [0.393, 0.551] | – | – | +0.039 [-0.081, +0.147] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.483 [0.392, 0.585] | – | 0.482 [0.367, 0.575] | – | – | +0.001 [-0.122, +0.167] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.508 [0.421, 0.591] | – | 0.468 [0.338, 0.571] | – | – | +0.040 [-0.055, +0.146] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.515 [0.434, 0.597] | – | 0.404 [0.292, 0.527] | – | – | +0.111 [+0.008, +0.211] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.501 [0.418, 0.587] | – | 0.397 [0.287, 0.520] | – | – | +0.104 [-0.038, +0.236] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.599 [0.484, 0.692] | – | 0.411 [0.298, 0.535] | – | – | +0.187 [+0.031, +0.318] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.559 [0.473, 0.645] | – | 0.390 [0.276, 0.517] | – | – | +0.169 [-0.007, +0.339] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.494 [0.392, 0.597] | – | 0.433 [0.324, 0.554] | – | – | +0.061 [-0.105, +0.211] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.399 [0.300, 0.493] | – | 0.454 [0.338, 0.570] | – | – | -0.055 [-0.181, +0.077] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.505 [0.460, 0.556] | – | 0.496 [0.385, 0.594] | – | – | +0.009 [-0.103, +0.142] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.499 [0.432, 0.560] | – | 0.493 [0.388, 0.586] | – | – | +0.006 [-0.078, +0.092] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.490 [0.441, 0.542] | – | 0.500 [0.412, 0.575] | – | – | -0.010 [-0.083, +0.078] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.499 [0.457, 0.537] | – | 0.489 [0.405, 0.555] | – | – | +0.010 [-0.058, +0.097] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.461 [0.423, 0.512] | – | 0.493 [0.411, 0.561] | – | – | -0.031 [-0.114, +0.063] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.485 [0.433, 0.538] | – | 0.479 [0.404, 0.538] | – | – | +0.006 [-0.065, +0.094] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.461 [0.406, 0.511] | – | 0.496 [0.425, 0.555] | – | – | -0.036 [-0.118, +0.054] | – | – |
| `v.end` | 518 / 362 | 280 | 0.523 [0.475, 0.572] | – | 0.511 [0.435, 0.570] | – | – | +0.012 [-0.056, +0.092] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.399 [0.300, 0.493]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.468, `m2.end` 0.539, `m3.end` 0.508, `m4.end` 0.399. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.pre`, `m4.q1`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.503 [0.483, 0.523]; activation minus words +0.046 [-0.000, +0.091]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.490 [0.467, 0.512]; activation minus words -0.004 [-0.065, +0.072]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

