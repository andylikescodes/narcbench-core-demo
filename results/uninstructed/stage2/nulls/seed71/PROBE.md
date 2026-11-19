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
| `m1.pre` | 257 / 183 | 139 | 0.413 [0.345, 0.485] | – | 0.500 [0.500, 0.500] | – | – | -0.087 [-0.155, -0.015] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.473 [0.410, 0.531] | – | 0.511 [0.486, 0.536] | – | – | -0.038 [-0.095, +0.014] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.506 [0.412, 0.608] | – | 0.446 [0.333, 0.576] | – | – | +0.060 [-0.048, +0.166] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.593 [0.513, 0.671] | – | 0.453 [0.341, 0.558] | – | – | +0.140 [+0.027, +0.249] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.558 [0.467, 0.658] | – | 0.475 [0.397, 0.550] | – | – | +0.083 [-0.023, +0.200] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.568 [0.494, 0.644] | – | 0.396 [0.321, 0.474] | – | – | +0.173 [+0.090, +0.257] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.471 [0.400, 0.550] | – | 0.411 [0.313, 0.514] | – | – | +0.060 [-0.039, +0.160] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.450 [0.358, 0.531] | – | 0.404 [0.311, 0.504] | – | – | +0.045 [-0.057, +0.129] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.528 [0.435, 0.611] | – | 0.418 [0.344, 0.484] | – | – | +0.109 [-0.016, +0.238] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.485 [0.390, 0.589] | – | 0.468 [0.383, 0.550] | – | – | +0.017 [-0.075, +0.120] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.496 [0.380, 0.629] | – | 0.426 [0.319, 0.525] | – | – | +0.071 [-0.006, +0.169] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.467 [0.389, 0.546] | – | 0.390 [0.279, 0.508] | – | – | +0.077 [-0.048, +0.214] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.553 [0.489, 0.623] | – | 0.453 [0.331, 0.574] | – | – | +0.099 [-0.029, +0.245] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.495 [0.428, 0.559] | – | 0.453 [0.331, 0.574] | – | – | +0.042 [-0.089, +0.185] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.403 [0.327, 0.496] | – | 0.489 [0.381, 0.594] | – | – | -0.086 [-0.212, +0.044] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.456 [0.326, 0.593] | – | 0.468 [0.369, 0.569] | – | – | -0.012 [-0.114, +0.097] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.501 [0.414, 0.595] | – | 0.496 [0.409, 0.592] | – | – | +0.004 [-0.112, +0.127] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.508 [0.400, 0.611] | – | 0.460 [0.367, 0.558] | – | – | +0.047 [-0.077, +0.176] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.416 [0.341, 0.486] | – | 0.376 [0.263, 0.488] | – | – | +0.040 [-0.080, +0.158] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.485 [0.420, 0.539] | – | 0.383 [0.264, 0.500] | – | – | +0.102 [-0.028, +0.239] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.506 [0.392, 0.632] | – | 0.397 [0.276, 0.504] | – | – | +0.109 [-0.069, +0.307] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.468 [0.385, 0.553] | – | 0.397 [0.283, 0.493] | – | – | +0.071 [-0.056, +0.211] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.509 [0.429, 0.602] | – | 0.418 [0.307, 0.518] | – | – | +0.091 [-0.039, +0.253] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.489 [0.397, 0.585] | – | 0.426 [0.317, 0.525] | – | – | +0.064 [-0.074, +0.207] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.469 [0.419, 0.519] | – | 0.429 [0.357, 0.485] | – | – | +0.040 [-0.018, +0.111] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.508 [0.452, 0.567] | – | 0.479 [0.404, 0.534] | – | – | +0.029 [-0.058, +0.128] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.484 [0.417, 0.542] | – | 0.486 [0.413, 0.546] | – | – | -0.001 [-0.068, +0.068] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.551 [0.494, 0.609] | – | 0.489 [0.432, 0.541] | – | – | +0.062 [-0.012, +0.147] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.512 [0.472, 0.558] | – | 0.479 [0.408, 0.541] | – | – | +0.034 [-0.034, +0.114] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.501 [0.458, 0.547] | – | 0.486 [0.418, 0.551] | – | – | +0.016 [-0.057, +0.086] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.523 [0.472, 0.568] | – | 0.500 [0.438, 0.555] | – | – | +0.023 [-0.039, +0.077] | – | – |
| `v.end` | 518 / 362 | 280 | 0.491 [0.439, 0.539] | – | 0.489 [0.425, 0.551] | – | – | +0.001 [-0.086, +0.080] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.489 [0.397, 0.585]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.568, `m2.end` 0.467, `m3.end` 0.508, `m4.end` 0.489. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.q2`, `m1.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.492 [0.468, 0.517]; activation minus words +0.053 [+0.020, +0.095]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.505 [0.476, 0.535]; activation minus words +0.025 [-0.023, +0.083]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

