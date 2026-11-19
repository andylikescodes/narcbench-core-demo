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
| `m1.pre` | 257 / 183 | 139 | 0.523 [0.433, 0.607] | – | 0.500 [0.500, 0.500] | – | – | +0.023 [-0.067, +0.107] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.422 [0.348, 0.496] | – | 0.453 [0.409, 0.496] | – | – | -0.031 [-0.116, +0.048] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.555 [0.451, 0.641] | – | 0.514 [0.421, 0.608] | – | – | +0.041 [-0.071, +0.161] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.465 [0.386, 0.548] | – | 0.446 [0.333, 0.555] | – | – | +0.019 [-0.135, +0.176] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.499 [0.407, 0.585] | – | 0.460 [0.336, 0.580] | – | – | +0.039 [-0.092, +0.183] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.424 [0.370, 0.482] | – | 0.482 [0.385, 0.568] | – | – | -0.058 [-0.152, +0.058] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.484 [0.400, 0.580] | – | 0.496 [0.384, 0.600] | – | – | -0.013 [-0.119, +0.092] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.532 [0.468, 0.597] | – | 0.489 [0.378, 0.593] | – | – | +0.043 [-0.086, +0.180] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.474 [0.381, 0.567] | – | 0.511 [0.386, 0.622] | – | – | -0.037 [-0.141, +0.064] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.501 [0.427, 0.579] | – | 0.511 [0.397, 0.612] | – | – | -0.010 [-0.121, +0.129] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.523 [0.458, 0.587] | – | 0.553 [0.440, 0.640] | – | – | -0.030 [-0.120, +0.073] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.559 [0.483, 0.641] | – | 0.560 [0.422, 0.664] | – | – | -0.001 [-0.101, +0.136] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.483 [0.401, 0.584] | – | 0.453 [0.342, 0.551] | – | – | +0.030 [-0.112, +0.195] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.443 [0.360, 0.531] | – | 0.446 [0.341, 0.538] | – | – | -0.003 [-0.145, +0.147] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.501 [0.413, 0.585] | – | 0.453 [0.364, 0.522] | – | – | +0.047 [-0.051, +0.159] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.535 [0.432, 0.633] | – | 0.475 [0.375, 0.554] | – | – | +0.060 [-0.047, +0.174] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.469 [0.390, 0.550] | – | 0.489 [0.400, 0.562] | – | – | -0.020 [-0.107, +0.070] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.471 [0.415, 0.534] | – | 0.511 [0.440, 0.576] | – | – | -0.040 [-0.113, +0.043] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.518 [0.404, 0.622] | – | 0.525 [0.420, 0.617] | – | – | -0.007 [-0.135, +0.119] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.472 [0.374, 0.575] | – | 0.525 [0.420, 0.617] | – | – | -0.052 [-0.163, +0.063] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.519 [0.442, 0.590] | – | 0.511 [0.413, 0.604] | – | – | +0.009 [-0.106, +0.117] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.498 [0.416, 0.570] | – | 0.496 [0.402, 0.589] | – | – | +0.001 [-0.107, +0.107] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.543 [0.453, 0.642] | – | 0.532 [0.420, 0.639] | – | – | +0.011 [-0.111, +0.131] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.485 [0.412, 0.565] | – | 0.496 [0.376, 0.616] | – | – | -0.011 [-0.147, +0.140] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.569 [0.513, 0.623] | – | 0.500 [0.446, 0.556] | – | – | +0.069 [-0.025, +0.157] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.412 [0.361, 0.466] | – | 0.482 [0.422, 0.541] | – | – | -0.070 [-0.135, -0.004] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.514 [0.465, 0.566] | – | 0.493 [0.420, 0.554] | – | – | +0.021 [-0.068, +0.115] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.524 [0.470, 0.585] | – | 0.479 [0.407, 0.543] | – | – | +0.045 [-0.038, +0.137] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.560 [0.497, 0.616] | – | 0.446 [0.382, 0.510] | – | – | +0.114 [+0.027, +0.199] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.469 [0.412, 0.530] | – | 0.436 [0.370, 0.500] | – | – | +0.033 [-0.049, +0.128] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.511 [0.467, 0.553] | – | 0.439 [0.377, 0.506] | – | – | +0.071 [-0.012, +0.147] | – | – |
| `v.end` | 518 / 362 | 280 | 0.494 [0.442, 0.557] | – | 0.432 [0.372, 0.495] | – | – | +0.062 [-0.008, +0.142] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.485 [0.412, 0.565]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.424, `m2.end` 0.559, `m3.end` 0.471, `m4.end` 0.485. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `p3.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.496 [0.480, 0.512]; activation minus words +0.000 [-0.045, +0.054]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.507 [0.481, 0.536]; activation minus words +0.043 [-0.010, +0.102]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

