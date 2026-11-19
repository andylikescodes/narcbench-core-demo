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
| `m1.pre` | 257 / 183 | 139 | 0.539 [0.462, 0.606] | – | 0.500 [0.500, 0.500] | – | – | +0.039 [-0.038, +0.106] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.554 [0.495, 0.618] | – | 0.525 [0.486, 0.564] | – | – | +0.029 [-0.039, +0.099] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.447 [0.354, 0.532] | – | 0.532 [0.409, 0.644] | – | – | -0.085 [-0.220, +0.064] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.460 [0.385, 0.527] | – | 0.482 [0.374, 0.594] | – | – | -0.022 [-0.153, +0.110] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.370 [0.294, 0.439] | – | 0.511 [0.400, 0.616] | – | – | -0.141 [-0.259, -0.004] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.534 [0.423, 0.646] | – | 0.504 [0.415, 0.588] | – | – | +0.030 [-0.101, +0.164] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.546 [0.471, 0.630] | – | 0.589 [0.507, 0.667] | – | – | -0.043 [-0.136, +0.054] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.557 [0.489, 0.632] | – | 0.589 [0.507, 0.667] | – | – | -0.031 [-0.130, +0.073] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.565 [0.441, 0.674] | – | 0.567 [0.474, 0.664] | – | – | -0.003 [-0.138, +0.101] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.509 [0.416, 0.591] | – | 0.617 [0.528, 0.711] | – | – | -0.108 [-0.261, +0.020] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.479 [0.376, 0.600] | – | 0.603 [0.510, 0.692] | – | – | -0.123 [-0.235, -0.005] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.488 [0.381, 0.587] | – | 0.553 [0.465, 0.636] | – | – | -0.065 [-0.178, +0.040] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.529 [0.438, 0.613] | – | 0.489 [0.418, 0.557] | – | – | +0.040 [-0.077, +0.153] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.542 [0.479, 0.593] | – | 0.489 [0.418, 0.557] | – | – | +0.053 [-0.037, +0.143] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.380 [0.310, 0.457] | – | 0.496 [0.424, 0.575] | – | – | -0.117 [-0.206, -0.025] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.472 [0.420, 0.528] | – | 0.525 [0.429, 0.621] | – | – | -0.053 [-0.183, +0.078] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.463 [0.374, 0.570] | – | 0.496 [0.397, 0.608] | – | – | -0.033 [-0.191, +0.145] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.446 [0.358, 0.563] | – | 0.489 [0.395, 0.604] | – | – | -0.043 [-0.182, +0.102] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.542 [0.465, 0.614] | – | 0.461 [0.372, 0.540] | – | – | +0.081 [-0.004, +0.167] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.574 [0.508, 0.639] | – | 0.468 [0.384, 0.542] | – | – | +0.106 [+0.035, +0.196] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.539 [0.467, 0.588] | – | 0.461 [0.369, 0.543] | – | – | +0.078 [-0.010, +0.169] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.567 [0.496, 0.640] | – | 0.461 [0.366, 0.547] | – | – | +0.106 [+0.005, +0.221] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.472 [0.374, 0.568] | – | 0.426 [0.318, 0.522] | – | – | +0.047 [-0.093, +0.206] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.542 [0.459, 0.624] | – | 0.482 [0.396, 0.556] | – | – | +0.060 [-0.059, +0.180] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.474 [0.424, 0.525] | – | 0.525 [0.455, 0.592] | – | – | -0.051 [-0.117, +0.023] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.496 [0.444, 0.548] | – | 0.521 [0.462, 0.571] | – | – | -0.025 [-0.082, +0.046] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.473 [0.400, 0.542] | – | 0.518 [0.433, 0.602] | – | – | -0.045 [-0.148, +0.068] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.510 [0.436, 0.579] | – | 0.525 [0.448, 0.605] | – | – | -0.015 [-0.120, +0.088] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.492 [0.427, 0.556] | – | 0.493 [0.414, 0.578] | – | – | -0.001 [-0.099, +0.113] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.527 [0.473, 0.591] | – | 0.511 [0.426, 0.595] | – | – | +0.016 [-0.071, +0.122] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.473 [0.419, 0.514] | – | 0.496 [0.404, 0.589] | – | – | -0.024 [-0.126, +0.064] | – | – |
| `v.end` | 518 / 362 | 280 | 0.512 [0.464, 0.554] | – | 0.500 [0.411, 0.588] | – | – | +0.012 [-0.061, +0.090] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.542 [0.459, 0.624]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.534, `m2.end` 0.488, `m3.end` 0.446, `m4.end` 0.542. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.t0`, `m4.q2`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.505 [0.482, 0.526]; activation minus words -0.008 [-0.047, +0.038]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.495 [0.458, 0.528]; activation minus words -0.016 [-0.083, +0.054]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

