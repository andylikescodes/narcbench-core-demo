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
| `m1.pre` | 257 / 183 | 139 | 0.500 [0.431, 0.567] | – | 0.500 [0.500, 0.500] | – | – | +0.000 [-0.069, +0.067] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.526 [0.453, 0.598] | – | 0.511 [0.460, 0.564] | – | – | +0.015 [-0.041, +0.074] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.586 [0.481, 0.682] | – | 0.543 [0.468, 0.623] | – | – | +0.042 [-0.070, +0.165] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.524 [0.445, 0.618] | – | 0.388 [0.258, 0.517] | – | – | +0.135 [-0.042, +0.323] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.458 [0.359, 0.581] | – | 0.482 [0.386, 0.589] | – | – | -0.024 [-0.176, +0.150] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.423 [0.338, 0.499] | – | 0.410 [0.307, 0.523] | – | – | +0.013 [-0.106, +0.121] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.521 [0.458, 0.580] | – | 0.496 [0.387, 0.597] | – | – | +0.024 [-0.103, +0.149] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.482 [0.393, 0.576] | – | 0.504 [0.395, 0.603] | – | – | -0.021 [-0.162, +0.111] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.428 [0.328, 0.521] | – | 0.461 [0.365, 0.578] | – | – | -0.033 [-0.176, +0.080] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.542 [0.456, 0.633] | – | 0.525 [0.445, 0.621] | – | – | +0.017 [-0.082, +0.114] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.560 [0.501, 0.629] | – | 0.539 [0.463, 0.625] | – | – | +0.021 [-0.079, +0.114] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.467 [0.365, 0.557] | – | 0.482 [0.384, 0.600] | – | – | -0.016 [-0.183, +0.130] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.450 [0.376, 0.521] | – | 0.417 [0.320, 0.514] | – | – | +0.033 [-0.063, +0.133] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.472 [0.389, 0.562] | – | 0.417 [0.320, 0.514] | – | – | +0.055 [-0.060, +0.179] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.459 [0.355, 0.555] | – | 0.446 [0.352, 0.533] | – | – | +0.013 [-0.125, +0.143] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.545 [0.452, 0.616] | – | 0.439 [0.351, 0.515] | – | – | +0.106 [-0.000, +0.214] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.443 [0.365, 0.520] | – | 0.468 [0.371, 0.557] | – | – | -0.024 [-0.126, +0.095] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.517 [0.442, 0.591] | – | 0.468 [0.362, 0.572] | – | – | +0.049 [-0.081, +0.176] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.529 [0.454, 0.596] | – | 0.440 [0.322, 0.550] | – | – | +0.089 [-0.023, +0.213] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.577 [0.471, 0.664] | – | 0.447 [0.329, 0.559] | – | – | +0.130 [-0.043, +0.301] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.552 [0.449, 0.642] | – | 0.496 [0.403, 0.580] | – | – | +0.055 [-0.051, +0.153] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.518 [0.433, 0.606] | – | 0.489 [0.384, 0.581] | – | – | +0.028 [-0.122, +0.190] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.513 [0.443, 0.575] | – | 0.553 [0.440, 0.649] | – | – | -0.040 [-0.144, +0.083] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.543 [0.461, 0.634] | – | 0.560 [0.452, 0.653] | – | – | -0.017 [-0.138, +0.123] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.413 [0.359, 0.470] | – | 0.532 [0.475, 0.589] | – | – | -0.119 [-0.184, -0.047] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.562 [0.509, 0.617] | – | 0.521 [0.459, 0.591] | – | – | +0.041 [-0.025, +0.109] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.517 [0.467, 0.569] | – | 0.543 [0.466, 0.610] | – | – | -0.026 [-0.092, +0.045] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.486 [0.436, 0.536] | – | 0.529 [0.448, 0.600] | – | – | -0.042 [-0.115, +0.033] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.506 [0.464, 0.547] | – | 0.536 [0.437, 0.623] | – | – | -0.029 [-0.140, +0.082] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.470 [0.426, 0.526] | – | 0.529 [0.436, 0.609] | – | – | -0.059 [-0.139, +0.030] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.514 [0.465, 0.573] | – | 0.525 [0.432, 0.614] | – | – | -0.011 [-0.111, +0.101] | – | – |
| `v.end` | 518 / 362 | 280 | 0.548 [0.498, 0.592] | – | 0.514 [0.421, 0.601] | – | – | +0.034 [-0.063, +0.133] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.543 [0.461, 0.634]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.423, `m2.end` 0.467, `m3.end` 0.517, `m4.end` 0.543. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.506 [0.487, 0.524]; activation minus words +0.027 [-0.019, +0.075]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.502 [0.480, 0.530]; activation minus words -0.027 [-0.093, +0.043]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

