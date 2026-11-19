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
| `m1.pre` | 257 / 183 | 139 | 0.420 [0.366, 0.467] | – | 0.500 [0.500, 0.500] | – | – | -0.080 [-0.134, -0.033] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.465 [0.392, 0.543] | – | 0.496 [0.457, 0.535] | – | – | -0.032 [-0.104, +0.034] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.439 [0.332, 0.561] | – | 0.453 [0.329, 0.581] | – | – | -0.014 [-0.131, +0.107] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.403 [0.327, 0.503] | – | 0.482 [0.328, 0.624] | – | – | -0.079 [-0.215, +0.092] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.468 [0.389, 0.558] | – | 0.424 [0.314, 0.541] | – | – | +0.043 [-0.109, +0.188] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.471 [0.357, 0.591] | – | 0.453 [0.331, 0.573] | – | – | +0.017 [-0.080, +0.121] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.406 [0.326, 0.485] | – | 0.475 [0.364, 0.597] | – | – | -0.070 [-0.186, +0.051] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.457 [0.386, 0.528] | – | 0.468 [0.361, 0.583] | – | – | -0.011 [-0.125, +0.108] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.550 [0.464, 0.652] | – | 0.482 [0.364, 0.600] | – | – | +0.068 [-0.054, +0.199] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.552 [0.448, 0.648] | – | 0.504 [0.402, 0.608] | – | – | +0.048 [-0.049, +0.157] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.441 [0.366, 0.534] | – | 0.504 [0.405, 0.600] | – | – | -0.062 [-0.156, +0.037] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.423 [0.361, 0.483] | – | 0.504 [0.395, 0.601] | – | – | -0.081 [-0.180, +0.021] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.563 [0.453, 0.686] | – | 0.475 [0.372, 0.593] | – | – | +0.088 [-0.058, +0.221] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.481 [0.406, 0.553] | – | 0.475 [0.372, 0.593] | – | – | +0.006 [-0.123, +0.119] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.456 [0.382, 0.546] | – | 0.453 [0.344, 0.574] | – | – | +0.003 [-0.151, +0.133] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.427 [0.354, 0.499] | – | 0.424 [0.312, 0.541] | – | – | +0.003 [-0.111, +0.114] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.504 [0.425, 0.585] | – | 0.381 [0.261, 0.520] | – | – | +0.122 [-0.019, +0.250] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.544 [0.475, 0.615] | – | 0.446 [0.316, 0.577] | – | – | +0.098 [-0.043, +0.243] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.566 [0.481, 0.662] | – | 0.496 [0.361, 0.618] | – | – | +0.070 [-0.083, +0.241] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.530 [0.450, 0.619] | – | 0.496 [0.361, 0.618] | – | – | +0.034 [-0.103, +0.188] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.391 [0.309, 0.470] | – | 0.504 [0.383, 0.616] | – | – | -0.112 [-0.266, +0.047] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.424 [0.351, 0.492] | – | 0.511 [0.390, 0.628] | – | – | -0.087 [-0.235, +0.081] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.457 [0.377, 0.541] | – | 0.511 [0.401, 0.619] | – | – | -0.054 [-0.176, +0.088] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.567 [0.483, 0.643] | – | 0.482 [0.372, 0.592] | – | – | +0.085 [-0.038, +0.210] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.509 [0.456, 0.571] | – | 0.511 [0.440, 0.577] | – | – | -0.001 [-0.071, +0.083] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.521 [0.452, 0.594] | – | 0.525 [0.440, 0.602] | – | – | -0.004 [-0.107, +0.107] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.531 [0.474, 0.586] | – | 0.496 [0.404, 0.583] | – | – | +0.034 [-0.046, +0.122] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.514 [0.435, 0.592] | – | 0.489 [0.402, 0.568] | – | – | +0.024 [-0.102, +0.142] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.485 [0.437, 0.538] | – | 0.489 [0.389, 0.579] | – | – | -0.004 [-0.107, +0.103] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.515 [0.452, 0.578] | – | 0.486 [0.394, 0.568] | – | – | +0.029 [-0.076, +0.140] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.531 [0.466, 0.602] | – | 0.493 [0.411, 0.570] | – | – | +0.038 [-0.070, +0.161] | – | – |
| `v.end` | 518 / 362 | 280 | 0.504 [0.440, 0.562] | – | 0.507 [0.429, 0.575] | – | – | -0.003 [-0.082, +0.077] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.567 [0.483, 0.643]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.471, `m2.end` 0.423, `m3.end` 0.544, `m4.end` 0.567. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.475 [0.450, 0.503]; activation minus words +0.000 [-0.042, +0.049]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.514 [0.477, 0.547]; activation minus words +0.014 [-0.065, +0.094]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

