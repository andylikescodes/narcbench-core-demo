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
| `m1.pre` | 257 / 183 | 139 | 0.483 [0.415, 0.542] | – | 0.500 [0.500, 0.500] | – | – | -0.017 [-0.085, +0.042] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.487 [0.409, 0.576] | – | 0.489 [0.457, 0.520] | – | – | -0.002 [-0.090, +0.105] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.491 [0.388, 0.594] | – | 0.421 [0.315, 0.531] | – | – | +0.070 [-0.019, +0.162] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.524 [0.436, 0.611] | – | 0.388 [0.302, 0.493] | – | – | +0.135 [+0.017, +0.247] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.476 [0.411, 0.558] | – | 0.475 [0.387, 0.571] | – | – | +0.001 [-0.127, +0.131] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.521 [0.428, 0.616] | – | 0.489 [0.393, 0.600] | – | – | +0.032 [-0.085, +0.126] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.421 [0.339, 0.504] | – | 0.582 [0.492, 0.686] | – | – | -0.160 [-0.293, -0.037] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.489 [0.414, 0.578] | – | 0.582 [0.491, 0.692] | – | – | -0.092 [-0.203, +0.016] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.434 [0.349, 0.516] | – | 0.546 [0.427, 0.670] | – | – | -0.112 [-0.272, +0.047] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.562 [0.476, 0.648] | – | 0.525 [0.418, 0.636] | – | – | +0.037 [-0.120, +0.187] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.529 [0.418, 0.648] | – | 0.525 [0.419, 0.634] | – | – | +0.004 [-0.144, +0.166] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.458 [0.369, 0.545] | – | 0.539 [0.444, 0.638] | – | – | -0.081 [-0.214, +0.047] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.514 [0.424, 0.594] | – | 0.446 [0.356, 0.543] | – | – | +0.068 [-0.034, +0.149] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.417 [0.321, 0.524] | – | 0.446 [0.356, 0.543] | – | – | -0.029 [-0.112, +0.043] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.469 [0.374, 0.568] | – | 0.424 [0.340, 0.521] | – | – | +0.045 [-0.091, +0.176] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.522 [0.434, 0.603] | – | 0.446 [0.352, 0.546] | – | – | +0.076 [-0.025, +0.164] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.545 [0.480, 0.625] | – | 0.403 [0.302, 0.508] | – | – | +0.142 [+0.001, +0.280] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.455 [0.359, 0.564] | – | 0.432 [0.322, 0.545] | – | – | +0.023 [-0.112, +0.149] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.559 [0.468, 0.656] | – | 0.553 [0.473, 0.649] | – | – | +0.006 [-0.111, +0.111] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.496 [0.431, 0.569] | – | 0.567 [0.491, 0.655] | – | – | -0.071 [-0.167, +0.015] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.338 [0.239, 0.435] | – | 0.496 [0.406, 0.600] | – | – | -0.159 [-0.314, -0.029] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.478 [0.408, 0.556] | – | 0.489 [0.400, 0.588] | – | – | -0.011 [-0.128, +0.108] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.437 [0.359, 0.520] | – | 0.461 [0.384, 0.545] | – | – | -0.024 [-0.119, +0.053] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.418 [0.351, 0.483] | – | 0.426 [0.350, 0.518] | – | – | -0.007 [-0.115, +0.082] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.495 [0.451, 0.541] | – | 0.489 [0.422, 0.563] | – | – | +0.006 [-0.061, +0.075] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.479 [0.396, 0.552] | – | 0.493 [0.424, 0.577] | – | – | -0.014 [-0.103, +0.072] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.522 [0.471, 0.585] | – | 0.511 [0.432, 0.603] | – | – | +0.011 [-0.106, +0.116] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.499 [0.438, 0.565] | – | 0.486 [0.398, 0.585] | – | – | +0.014 [-0.092, +0.113] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.474 [0.413, 0.539] | – | 0.496 [0.419, 0.585] | – | – | -0.022 [-0.101, +0.059] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.526 [0.473, 0.577] | – | 0.511 [0.425, 0.604] | – | – | +0.015 [-0.080, +0.103] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.472 [0.418, 0.531] | – | 0.504 [0.422, 0.594] | – | – | -0.031 [-0.154, +0.078] | – | – |
| `v.end` | 518 / 362 | 280 | 0.504 [0.438, 0.562] | – | 0.514 [0.438, 0.601] | – | – | -0.011 [-0.113, +0.088] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.418 [0.351, 0.483]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.521, `m2.end` 0.458, `m3.end` 0.455, `m4.end` 0.418. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.q2`, `m3.q3`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.480 [0.457, 0.504]; activation minus words -0.005 [-0.052, +0.030]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.496 [0.475, 0.519]; activation minus words -0.004 [-0.077, +0.062]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

