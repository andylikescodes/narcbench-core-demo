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
| `m1.pre` | 257 / 183 | 139 | 0.495 [0.415, 0.561] | – | 0.500 [0.500, 0.500] | – | – | -0.005 [-0.085, +0.061] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.527 [0.460, 0.585] | – | 0.504 [0.457, 0.545] | – | – | +0.023 [-0.066, +0.098] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.419 [0.335, 0.503] | – | 0.493 [0.366, 0.622] | – | – | -0.074 [-0.225, +0.074] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.491 [0.406, 0.559] | – | 0.446 [0.366, 0.533] | – | – | +0.045 [-0.094, +0.158] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.501 [0.414, 0.574] | – | 0.518 [0.418, 0.621] | – | – | -0.017 [-0.159, +0.113] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.475 [0.399, 0.551] | – | 0.561 [0.457, 0.664] | – | – | -0.086 [-0.184, +0.016] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.557 [0.485, 0.637] | – | 0.468 [0.367, 0.578] | – | – | +0.089 [-0.055, +0.237] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.543 [0.448, 0.651] | – | 0.475 [0.375, 0.591] | – | – | +0.068 [-0.057, +0.214] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.454 [0.369, 0.526] | – | 0.489 [0.385, 0.583] | – | – | -0.035 [-0.161, +0.102] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.504 [0.418, 0.593] | – | 0.482 [0.371, 0.583] | – | – | +0.021 [-0.095, +0.159] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.535 [0.441, 0.616] | – | 0.504 [0.387, 0.602] | – | – | +0.031 [-0.092, +0.152] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.550 [0.471, 0.616] | – | 0.518 [0.420, 0.615] | – | – | +0.033 [-0.095, +0.154] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.492 [0.431, 0.550] | – | 0.532 [0.412, 0.652] | – | – | -0.040 [-0.165, +0.086] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.574 [0.502, 0.651] | – | 0.525 [0.402, 0.648] | – | – | +0.049 [-0.054, +0.170] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.532 [0.469, 0.598] | – | 0.583 [0.487, 0.683] | – | – | -0.050 [-0.131, +0.034] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.521 [0.425, 0.620] | – | 0.568 [0.472, 0.675] | – | – | -0.047 [-0.145, +0.050] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.492 [0.405, 0.582] | – | 0.561 [0.450, 0.677] | – | – | -0.069 [-0.206, +0.054] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.560 [0.494, 0.636] | – | 0.576 [0.460, 0.691] | – | – | -0.016 [-0.148, +0.116] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.508 [0.417, 0.600] | – | 0.589 [0.515, 0.670] | – | – | -0.081 [-0.163, +0.002] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.536 [0.437, 0.640] | – | 0.574 [0.508, 0.656] | – | – | -0.038 [-0.155, +0.057] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.552 [0.479, 0.618] | – | 0.553 [0.470, 0.644] | – | – | -0.001 [-0.111, +0.082] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.572 [0.473, 0.654] | – | 0.553 [0.477, 0.640] | – | – | +0.018 [-0.110, +0.119] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.441 [0.346, 0.534] | – | 0.567 [0.483, 0.657] | – | – | -0.126 [-0.258, -0.003] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.570 [0.470, 0.660] | – | 0.489 [0.415, 0.566] | – | – | +0.081 [-0.007, +0.150] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.498 [0.430, 0.554] | – | 0.500 [0.429, 0.569] | – | – | -0.002 [-0.095, +0.084] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.485 [0.413, 0.546] | – | 0.507 [0.443, 0.579] | – | – | -0.022 [-0.117, +0.055] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.526 [0.452, 0.579] | – | 0.500 [0.446, 0.559] | – | – | +0.026 [-0.058, +0.090] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.576 [0.529, 0.622] | – | 0.496 [0.443, 0.553] | – | – | +0.079 [+0.016, +0.150] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.484 [0.428, 0.533] | – | 0.493 [0.446, 0.545] | – | – | -0.009 [-0.065, +0.039] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.506 [0.432, 0.575] | – | 0.507 [0.461, 0.557] | – | – | -0.001 [-0.093, +0.088] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.574 [0.519, 0.620] | – | 0.525 [0.479, 0.572] | – | – | +0.049 [-0.017, +0.109] | – | – |
| `v.end` | 518 / 362 | 280 | 0.511 [0.440, 0.569] | – | 0.536 [0.494, 0.581] | – | – | -0.025 [-0.089, +0.031] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.570 [0.470, 0.660]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.475, `m2.end` 0.550, `m3.end` 0.560, `m4.end` 0.570. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `p2.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.517 [0.497, 0.536]; activation minus words -0.010 [-0.053, +0.030]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.520 [0.486, 0.547]; activation minus words +0.012 [-0.032, +0.048]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

