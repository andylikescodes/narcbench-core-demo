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
| `m1.pre` | 257 / 183 | 139 | 0.503 [0.426, 0.578] | – | 0.500 [0.500, 0.500] | – | – | +0.003 [-0.074, +0.078] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.474 [0.404, 0.540] | – | 0.453 [0.410, 0.496] | – | – | +0.021 [-0.048, +0.086] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.466 [0.360, 0.570] | – | 0.424 [0.320, 0.513] | – | – | +0.042 [-0.063, +0.148] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.479 [0.392, 0.558] | – | 0.511 [0.385, 0.617] | – | – | -0.032 [-0.156, +0.097] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.471 [0.396, 0.544] | – | 0.583 [0.475, 0.682] | – | – | -0.112 [-0.218, +0.002] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.496 [0.432, 0.565] | – | 0.597 [0.486, 0.695] | – | – | -0.101 [-0.202, +0.015] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.543 [0.456, 0.619] | – | 0.596 [0.506, 0.682] | – | – | -0.052 [-0.163, +0.041] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.570 [0.483, 0.662] | – | 0.596 [0.495, 0.698] | – | – | -0.026 [-0.184, +0.124] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.359 [0.275, 0.445] | – | 0.489 [0.409, 0.585] | – | – | -0.130 [-0.222, -0.041] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.474 [0.398, 0.557] | – | 0.482 [0.400, 0.579] | – | – | -0.009 [-0.124, +0.105] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.491 [0.400, 0.576] | – | 0.475 [0.391, 0.569] | – | – | +0.016 [-0.108, +0.129] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.495 [0.396, 0.597] | – | 0.532 [0.422, 0.643] | – | – | -0.037 [-0.182, +0.113] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.508 [0.413, 0.603] | – | 0.633 [0.558, 0.714] | – | – | -0.125 [-0.249, -0.017] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.439 [0.370, 0.505] | – | 0.633 [0.558, 0.714] | – | – | -0.194 [-0.284, -0.105] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.568 [0.470, 0.665] | – | 0.554 [0.453, 0.664] | – | – | +0.014 [-0.097, +0.138] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.469 [0.357, 0.571] | – | 0.590 [0.481, 0.704] | – | – | -0.121 [-0.254, +0.016] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.485 [0.402, 0.583] | – | 0.590 [0.472, 0.710] | – | – | -0.105 [-0.236, +0.026] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.492 [0.404, 0.583] | – | 0.561 [0.461, 0.657] | – | – | -0.069 [-0.173, +0.051] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.451 [0.367, 0.535] | – | 0.546 [0.418, 0.652] | – | – | -0.095 [-0.245, +0.061] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.367 [0.307, 0.426] | – | 0.546 [0.418, 0.652] | – | – | -0.179 [-0.288, -0.062] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.569 [0.468, 0.677] | – | 0.525 [0.404, 0.640] | – | – | +0.044 [-0.103, +0.197] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.509 [0.424, 0.602] | – | 0.546 [0.421, 0.654] | – | – | -0.037 [-0.162, +0.106] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.438 [0.341, 0.527] | – | 0.475 [0.363, 0.565] | – | – | -0.037 [-0.177, +0.127] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.526 [0.475, 0.583] | – | 0.504 [0.403, 0.590] | – | – | +0.023 [-0.059, +0.121] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.525 [0.490, 0.563] | – | 0.525 [0.466, 0.588] | – | – | +0.000 [-0.077, +0.073] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.555 [0.505, 0.608] | – | 0.561 [0.511, 0.617] | – | – | -0.006 [-0.086, +0.072] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.501 [0.457, 0.539] | – | 0.500 [0.439, 0.560] | – | – | +0.001 [-0.070, +0.067] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.584 [0.538, 0.630] | – | 0.504 [0.435, 0.565] | – | – | +0.080 [+0.010, +0.152] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.535 [0.471, 0.598] | – | 0.493 [0.426, 0.547] | – | – | +0.042 [-0.049, +0.140] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.502 [0.446, 0.553] | – | 0.479 [0.413, 0.536] | – | – | +0.024 [-0.056, +0.103] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.526 [0.478, 0.563] | – | 0.489 [0.425, 0.542] | – | – | +0.036 [-0.034, +0.104] | – | – |
| `v.end` | 518 / 362 | 280 | 0.549 [0.484, 0.609] | – | 0.504 [0.438, 0.557] | – | – | +0.045 [-0.032, +0.126] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.526 [0.475, 0.583]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.496, `m2.end` 0.495, `m3.end` 0.492, `m4.end` 0.526. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `p2.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.485 [0.460, 0.512]; activation minus words -0.054 [-0.101, +0.003]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.534 [0.515, 0.552]; activation minus words +0.028 [-0.021, +0.076]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

