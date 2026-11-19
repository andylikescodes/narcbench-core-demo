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
| `m1.pre` | 257 / 183 | 139 | 0.550 [0.459, 0.632] | – | 0.500 [0.500, 0.500] | – | – | +0.050 [-0.041, +0.132] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.578 [0.504, 0.647] | – | 0.511 [0.472, 0.546] | – | – | +0.067 [-0.019, +0.147] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.560 [0.472, 0.670] | – | 0.525 [0.426, 0.617] | – | – | +0.035 [-0.093, +0.184] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.420 [0.356, 0.484] | – | 0.424 [0.322, 0.526] | – | – | -0.004 [-0.134, +0.128] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.547 [0.439, 0.646] | – | 0.446 [0.350, 0.533] | – | – | +0.101 [-0.037, +0.258] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.483 [0.383, 0.574] | – | 0.496 [0.400, 0.589] | – | – | -0.013 [-0.149, +0.112] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.552 [0.489, 0.614] | – | 0.610 [0.517, 0.701] | – | – | -0.058 [-0.171, +0.074] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.485 [0.421, 0.547] | – | 0.589 [0.472, 0.698] | – | – | -0.104 [-0.238, +0.025] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.489 [0.423, 0.558] | – | 0.596 [0.503, 0.681] | – | – | -0.106 [-0.206, -0.002] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.623 [0.528, 0.715] | – | 0.553 [0.448, 0.638] | – | – | +0.070 [-0.016, +0.174] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.523 [0.415, 0.620] | – | 0.532 [0.423, 0.639] | – | – | -0.009 [-0.155, +0.130] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.580 [0.524, 0.638] | – | 0.546 [0.435, 0.651] | – | – | +0.034 [-0.067, +0.138] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.432 [0.359, 0.509] | – | 0.475 [0.381, 0.569] | – | – | -0.043 [-0.165, +0.071] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.453 [0.397, 0.502] | – | 0.475 [0.381, 0.569] | – | – | -0.022 [-0.118, +0.080] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.476 [0.381, 0.600] | – | 0.482 [0.382, 0.581] | – | – | -0.006 [-0.124, +0.134] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.547 [0.430, 0.652] | – | 0.525 [0.436, 0.610] | – | – | +0.022 [-0.117, +0.137] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.476 [0.414, 0.538] | – | 0.540 [0.465, 0.617] | – | – | -0.063 [-0.159, +0.027] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.540 [0.448, 0.630] | – | 0.504 [0.428, 0.586] | – | – | +0.036 [-0.107, +0.169] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.543 [0.472, 0.615] | – | 0.532 [0.441, 0.611] | – | – | +0.011 [-0.103, +0.129] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.557 [0.485, 0.642] | – | 0.532 [0.431, 0.617] | – | – | +0.026 [-0.091, +0.156] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.484 [0.415, 0.550] | – | 0.553 [0.465, 0.624] | – | – | -0.070 [-0.165, +0.042] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.502 [0.412, 0.581] | – | 0.525 [0.425, 0.611] | – | – | -0.023 [-0.116, +0.083] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.455 [0.385, 0.525] | – | 0.539 [0.446, 0.629] | – | – | -0.084 [-0.205, +0.049] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.566 [0.497, 0.629] | – | 0.532 [0.447, 0.631] | – | – | +0.034 [-0.078, +0.123] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.499 [0.458, 0.547] | – | 0.461 [0.396, 0.521] | – | – | +0.038 [-0.042, +0.136] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.420 [0.351, 0.502] | – | 0.450 [0.378, 0.524] | – | – | -0.030 [-0.133, +0.082] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.451 [0.406, 0.491] | – | 0.482 [0.400, 0.561] | – | – | -0.031 [-0.110, +0.056] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.476 [0.426, 0.520] | – | 0.457 [0.373, 0.543] | – | – | +0.019 [-0.071, +0.103] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.499 [0.448, 0.542] | – | 0.468 [0.394, 0.541] | – | – | +0.031 [-0.031, +0.089] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.509 [0.448, 0.563] | – | 0.479 [0.403, 0.555] | – | – | +0.030 [-0.056, +0.114] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.489 [0.443, 0.541] | – | 0.454 [0.375, 0.538] | – | – | +0.036 [-0.061, +0.133] | – | – |
| `v.end` | 518 / 362 | 280 | 0.482 [0.417, 0.539] | – | 0.482 [0.404, 0.565] | – | – | -0.000 [-0.076, +0.064] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.566 [0.497, 0.629]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.483, `m2.end` 0.580, `m3.end` 0.540, `m4.end` 0.566. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.518 [0.491, 0.543]; activation minus words -0.005 [-0.047, +0.040]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.478 [0.455, 0.498]; activation minus words +0.012 [-0.043, +0.067]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

