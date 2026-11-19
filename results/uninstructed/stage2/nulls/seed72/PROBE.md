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
| `m1.pre` | 257 / 183 | 139 | 0.501 [0.429, 0.568] | – | 0.500 [0.500, 0.500] | – | – | +0.001 [-0.071, +0.068] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.376 [0.305, 0.444] | – | 0.489 [0.446, 0.522] | – | – | -0.114 [-0.179, -0.043] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.571 [0.507, 0.635] | – | 0.525 [0.422, 0.636] | – | – | +0.046 [-0.103, +0.181] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.436 [0.369, 0.501] | – | 0.590 [0.511, 0.667] | – | – | -0.154 [-0.257, -0.057] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.468 [0.389, 0.554] | – | 0.511 [0.408, 0.607] | – | – | -0.043 [-0.167, +0.081] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.416 [0.339, 0.506] | – | 0.576 [0.464, 0.664] | – | – | -0.160 [-0.271, -0.035] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.529 [0.448, 0.615] | – | 0.489 [0.366, 0.606] | – | – | +0.040 [-0.105, +0.199] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.533 [0.481, 0.594] | – | 0.496 [0.381, 0.611] | – | – | +0.037 [-0.089, +0.166] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.586 [0.513, 0.665] | – | 0.553 [0.447, 0.649] | – | – | +0.033 [-0.097, +0.172] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.528 [0.440, 0.614] | – | 0.525 [0.419, 0.627] | – | – | +0.003 [-0.118, +0.132] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.528 [0.442, 0.613] | – | 0.447 [0.358, 0.526] | – | – | +0.081 [-0.031, +0.189] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.623 [0.497, 0.734] | – | 0.475 [0.362, 0.584] | – | – | +0.148 [-0.033, +0.321] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.469 [0.398, 0.546] | – | 0.489 [0.376, 0.596] | – | – | -0.020 [-0.137, +0.103] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.482 [0.400, 0.558] | – | 0.489 [0.376, 0.596] | – | – | -0.007 [-0.133, +0.109] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.481 [0.387, 0.564] | – | 0.482 [0.377, 0.583] | – | – | -0.001 [-0.140, +0.120] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.514 [0.447, 0.587] | – | 0.460 [0.366, 0.557] | – | – | +0.053 [-0.062, +0.179] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.422 [0.342, 0.495] | – | 0.432 [0.336, 0.531] | – | – | -0.010 [-0.129, +0.099] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.442 [0.355, 0.516] | – | 0.460 [0.371, 0.556] | – | – | -0.019 [-0.144, +0.087] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.434 [0.355, 0.517] | – | 0.454 [0.380, 0.524] | – | – | -0.020 [-0.114, +0.088] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.574 [0.489, 0.673] | – | 0.468 [0.392, 0.536] | – | – | +0.106 [+0.000, +0.233] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.484 [0.412, 0.568] | – | 0.482 [0.405, 0.561] | – | – | +0.001 [-0.110, +0.125] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.577 [0.497, 0.670] | – | 0.489 [0.409, 0.573] | – | – | +0.088 [-0.029, +0.213] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.569 [0.490, 0.662] | – | 0.454 [0.360, 0.554] | – | – | +0.115 [+0.008, +0.247] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.542 [0.470, 0.608] | – | 0.496 [0.390, 0.603] | – | – | +0.045 [-0.087, +0.179] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.575 [0.534, 0.615] | – | 0.482 [0.420, 0.539] | – | – | +0.093 [+0.040, +0.150] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.489 [0.440, 0.533] | – | 0.457 [0.402, 0.509] | – | – | +0.031 [-0.033, +0.097] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.534 [0.475, 0.594] | – | 0.443 [0.391, 0.492] | – | – | +0.091 [+0.023, +0.172] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.474 [0.425, 0.522] | – | 0.464 [0.412, 0.512] | – | – | +0.010 [-0.055, +0.072] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.506 [0.443, 0.573] | – | 0.461 [0.403, 0.515] | – | – | +0.046 [-0.045, +0.137] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.435 [0.386, 0.478] | – | 0.443 [0.380, 0.504] | – | – | -0.008 [-0.088, +0.069] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.454 [0.402, 0.505] | – | 0.454 [0.388, 0.516] | – | – | +0.000 [-0.060, +0.064] | – | – |
| `v.end` | 518 / 362 | 280 | 0.449 [0.389, 0.506] | – | 0.464 [0.390, 0.524] | – | – | -0.015 [-0.096, +0.079] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.542 [0.470, 0.608]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.416, `m2.end` 0.623, `m3.end` 0.442, `m4.end` 0.542. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.q3`, `p1.pre`, `p2.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.503 [0.481, 0.529]; activation minus words +0.010 [-0.028, +0.054]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.490 [0.463, 0.512]; activation minus words +0.031 [-0.015, +0.078]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

