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
| `m1.pre` | 257 / 183 | 139 | 0.399 [0.314, 0.493] | – | 0.500 [0.500, 0.500] | – | – | -0.101 [-0.186, -0.007] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.441 [0.385, 0.496] | – | 0.468 [0.433, 0.500] | – | – | -0.027 [-0.086, +0.032] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.485 [0.389, 0.577] | – | 0.424 [0.345, 0.500] | – | – | +0.060 [-0.037, +0.167] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.554 [0.464, 0.639] | – | 0.576 [0.461, 0.688] | – | – | -0.022 [-0.185, +0.120] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.506 [0.405, 0.603] | – | 0.576 [0.463, 0.667] | – | – | -0.069 [-0.201, +0.079] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.471 [0.367, 0.567] | – | 0.590 [0.475, 0.693] | – | – | -0.119 [-0.247, +0.017] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.519 [0.449, 0.591] | – | 0.468 [0.386, 0.566] | – | – | +0.051 [-0.071, +0.166] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.421 [0.339, 0.505] | – | 0.468 [0.386, 0.566] | – | – | -0.047 [-0.173, +0.071] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.499 [0.375, 0.611] | – | 0.489 [0.406, 0.578] | – | – | +0.010 [-0.158, +0.158] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.464 [0.389, 0.529] | – | 0.496 [0.412, 0.596] | – | – | -0.033 [-0.160, +0.065] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.418 [0.368, 0.466] | – | 0.504 [0.406, 0.625] | – | – | -0.085 [-0.228, +0.033] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.589 [0.480, 0.693] | – | 0.546 [0.445, 0.671] | – | – | +0.043 [-0.102, +0.176] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.469 [0.379, 0.563] | – | 0.525 [0.424, 0.619] | – | – | -0.056 [-0.177, +0.075] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.509 [0.446, 0.585] | – | 0.532 [0.432, 0.625] | – | – | -0.023 [-0.129, +0.091] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.479 [0.375, 0.574] | – | 0.554 [0.468, 0.646] | – | – | -0.075 [-0.188, +0.053] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.561 [0.490, 0.624] | – | 0.540 [0.448, 0.627] | – | – | +0.022 [-0.074, +0.111] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.527 [0.445, 0.600] | – | 0.561 [0.465, 0.658] | – | – | -0.035 [-0.143, +0.079] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.524 [0.426, 0.596] | – | 0.583 [0.479, 0.691] | – | – | -0.059 [-0.220, +0.072] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.416 [0.339, 0.484] | – | 0.546 [0.436, 0.672] | – | – | -0.130 [-0.290, +0.009] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.484 [0.419, 0.541] | – | 0.553 [0.448, 0.674] | – | – | -0.070 [-0.209, +0.045] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.511 [0.398, 0.614] | – | 0.504 [0.397, 0.625] | – | – | +0.007 [-0.130, +0.122] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.587 [0.512, 0.658] | – | 0.532 [0.408, 0.663] | – | – | +0.055 [-0.076, +0.182] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.418 [0.336, 0.503] | – | 0.553 [0.439, 0.665] | – | – | -0.135 [-0.275, +0.001] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.552 [0.485, 0.620] | – | 0.560 [0.448, 0.667] | – | – | -0.009 [-0.134, +0.129] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.466 [0.417, 0.515] | – | 0.518 [0.466, 0.576] | – | – | -0.051 [-0.118, +0.008] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.561 [0.509, 0.611] | – | 0.550 [0.470, 0.629] | – | – | +0.011 [-0.086, +0.100] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.515 [0.472, 0.558] | – | 0.529 [0.468, 0.597] | – | – | -0.014 [-0.105, +0.068] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.446 [0.402, 0.486] | – | 0.532 [0.473, 0.601] | – | – | -0.086 [-0.150, -0.025] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.491 [0.453, 0.531] | – | 0.496 [0.434, 0.566] | – | – | -0.006 [-0.094, +0.080] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.509 [0.437, 0.575] | – | 0.496 [0.435, 0.570] | – | – | +0.013 [-0.088, +0.097] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.511 [0.455, 0.565] | – | 0.521 [0.447, 0.593] | – | – | -0.011 [-0.097, +0.065] | – | – |
| `v.end` | 518 / 362 | 280 | 0.524 [0.470, 0.589] | – | 0.511 [0.439, 0.579] | – | – | +0.014 [-0.073, +0.114] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.552 [0.485, 0.620]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.471, `m2.end` 0.589, `m3.end` 0.524, `m4.end` 0.552. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.492 [0.465, 0.516]; activation minus words -0.035 [-0.084, +0.010]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.503 [0.480, 0.526]; activation minus words -0.016 [-0.083, +0.043]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

