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
| `m1.pre` | 257 / 183 | 139 | 0.591 [0.516, 0.650] | – | 0.500 [0.500, 0.500] | – | – | +0.091 [+0.016, +0.150] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.640 [0.564, 0.718] | – | 0.511 [0.477, 0.537] | – | – | +0.129 [+0.050, +0.218] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.558 [0.490, 0.633] | – | 0.540 [0.436, 0.640] | – | – | +0.019 [-0.086, +0.130] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.580 [0.502, 0.672] | – | 0.532 [0.451, 0.617] | – | – | +0.047 [-0.057, +0.163] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.409 [0.322, 0.491] | – | 0.453 [0.389, 0.514] | – | – | -0.045 [-0.148, +0.061] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.498 [0.389, 0.599] | – | 0.496 [0.413, 0.579] | – | – | +0.001 [-0.122, +0.106] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.580 [0.495, 0.665] | – | 0.525 [0.444, 0.600] | – | – | +0.055 [-0.046, +0.163] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.542 [0.457, 0.623] | – | 0.532 [0.451, 0.606] | – | – | +0.010 [-0.091, +0.116] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.556 [0.462, 0.650] | – | 0.504 [0.398, 0.610] | – | – | +0.052 [-0.085, +0.202] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.582 [0.502, 0.653] | – | 0.489 [0.400, 0.565] | – | – | +0.092 [-0.006, +0.181] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.506 [0.426, 0.604] | – | 0.525 [0.439, 0.603] | – | – | -0.018 [-0.113, +0.095] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.548 [0.465, 0.637] | – | 0.567 [0.490, 0.641] | – | – | -0.020 [-0.117, +0.080] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.551 [0.475, 0.633] | – | 0.532 [0.446, 0.600] | – | – | +0.019 [-0.079, +0.129] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.545 [0.483, 0.607] | – | 0.525 [0.440, 0.591] | – | – | +0.020 [-0.073, +0.123] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.476 [0.377, 0.577] | – | 0.525 [0.427, 0.603] | – | – | -0.049 [-0.142, +0.049] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.531 [0.456, 0.607] | – | 0.547 [0.460, 0.623] | – | – | -0.016 [-0.122, +0.099] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.485 [0.394, 0.551] | – | 0.554 [0.470, 0.628] | – | – | -0.069 [-0.171, +0.024] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.463 [0.390, 0.537] | – | 0.511 [0.426, 0.593] | – | – | -0.047 [-0.121, +0.030] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.552 [0.483, 0.622] | – | 0.596 [0.481, 0.692] | – | – | -0.044 [-0.148, +0.075] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.586 [0.523, 0.643] | – | 0.596 [0.481, 0.692] | – | – | -0.010 [-0.089, +0.086] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.604 [0.526, 0.694] | – | 0.567 [0.467, 0.653] | – | – | +0.037 [-0.069, +0.174] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.502 [0.390, 0.595] | – | 0.567 [0.473, 0.647] | – | – | -0.065 [-0.219, +0.096] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.667 [0.564, 0.755] | – | 0.574 [0.483, 0.655] | – | – | +0.092 [-0.046, +0.226] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.540 [0.483, 0.613] | – | 0.560 [0.470, 0.643] | – | – | -0.020 [-0.120, +0.095] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.527 [0.476, 0.581] | – | 0.532 [0.466, 0.590] | – | – | -0.005 [-0.078, +0.075] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.500 [0.458, 0.541] | – | 0.536 [0.485, 0.586] | – | – | -0.036 [-0.086, +0.020] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.560 [0.494, 0.620] | – | 0.561 [0.500, 0.620] | – | – | -0.001 [-0.081, +0.091] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.483 [0.434, 0.532] | – | 0.557 [0.486, 0.621] | – | – | -0.074 [-0.153, +0.003] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.544 [0.484, 0.606] | – | 0.550 [0.478, 0.616] | – | – | -0.006 [-0.098, +0.086] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.542 [0.465, 0.610] | – | 0.550 [0.476, 0.614] | – | – | -0.008 [-0.087, +0.066] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.555 [0.506, 0.602] | – | 0.536 [0.460, 0.601] | – | – | +0.019 [-0.042, +0.084] | – | – |
| `v.end` | 518 / 362 | 280 | 0.546 [0.489, 0.601] | – | 0.536 [0.470, 0.594] | – | – | +0.011 [-0.045, +0.066] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.540 [0.483, 0.613]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.498, `m2.end` 0.548, `m3.end` 0.463, `m4.end` 0.540. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.pre`, `m1.t0`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.545 [0.524, 0.566]; activation minus words +0.011 [-0.028, +0.061]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.532 [0.507, 0.557]; activation minus words -0.013 [-0.054, +0.036]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** layer 21, anchor `m4.pre` (score there 0.582).

