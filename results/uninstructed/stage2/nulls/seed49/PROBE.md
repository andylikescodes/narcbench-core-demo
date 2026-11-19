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
| `m1.pre` | 257 / 183 | 139 | 0.518 [0.456, 0.591] | – | 0.500 [0.500, 0.500] | – | – | +0.018 [-0.044, +0.091] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.511 [0.450, 0.572] | – | 0.496 [0.468, 0.521] | – | – | +0.014 [-0.038, +0.074] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.517 [0.417, 0.598] | – | 0.561 [0.463, 0.662] | – | – | -0.045 [-0.203, +0.092] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.458 [0.381, 0.554] | – | 0.468 [0.379, 0.571] | – | – | -0.010 [-0.129, +0.112] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.414 [0.341, 0.489] | – | 0.446 [0.338, 0.561] | – | – | -0.032 [-0.189, +0.124] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.532 [0.460, 0.588] | – | 0.468 [0.352, 0.589] | – | – | +0.065 [-0.070, +0.184] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.496 [0.426, 0.562] | – | 0.525 [0.426, 0.647] | – | – | -0.028 [-0.169, +0.088] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.470 [0.387, 0.552] | – | 0.539 [0.445, 0.654] | – | – | -0.070 [-0.205, +0.040] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.562 [0.458, 0.645] | – | 0.525 [0.419, 0.647] | – | – | +0.037 [-0.101, +0.157] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.543 [0.455, 0.630] | – | 0.567 [0.477, 0.664] | – | – | -0.024 [-0.148, +0.106] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.488 [0.390, 0.581] | – | 0.525 [0.428, 0.628] | – | – | -0.037 [-0.142, +0.062] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.451 [0.361, 0.540] | – | 0.454 [0.342, 0.571] | – | – | -0.003 [-0.146, +0.140] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.502 [0.420, 0.596] | – | 0.482 [0.395, 0.584] | – | – | +0.020 [-0.107, +0.152] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.502 [0.424, 0.581] | – | 0.475 [0.389, 0.579] | – | – | +0.027 [-0.116, +0.152] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.486 [0.410, 0.563] | – | 0.532 [0.460, 0.611] | – | – | -0.046 [-0.165, +0.065] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.437 [0.329, 0.539] | – | 0.453 [0.369, 0.534] | – | – | -0.016 [-0.170, +0.130] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.529 [0.413, 0.641] | – | 0.468 [0.394, 0.550] | – | – | +0.062 [-0.071, +0.193] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.414 [0.320, 0.516] | – | 0.468 [0.395, 0.551] | – | – | -0.053 [-0.165, +0.065] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.582 [0.498, 0.660] | – | 0.468 [0.352, 0.585] | – | – | +0.113 [-0.054, +0.273] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.594 [0.511, 0.675] | – | 0.468 [0.352, 0.585] | – | – | +0.126 [-0.014, +0.270] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.516 [0.440, 0.588] | – | 0.496 [0.373, 0.613] | – | – | +0.020 [-0.128, +0.165] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.523 [0.460, 0.584] | – | 0.518 [0.402, 0.623] | – | – | +0.006 [-0.115, +0.134] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.577 [0.518, 0.652] | – | 0.532 [0.426, 0.630] | – | – | +0.045 [-0.053, +0.161] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.523 [0.435, 0.609] | – | 0.525 [0.414, 0.628] | – | – | -0.001 [-0.092, +0.100] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.496 [0.435, 0.562] | – | 0.525 [0.444, 0.594] | – | – | -0.029 [-0.126, +0.066] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.519 [0.477, 0.566] | – | 0.500 [0.422, 0.564] | – | – | +0.019 [-0.048, +0.102] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.473 [0.412, 0.542] | – | 0.521 [0.471, 0.579] | – | – | -0.049 [-0.116, +0.028] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.540 [0.472, 0.601] | – | 0.507 [0.454, 0.569] | – | – | +0.033 [-0.057, +0.114] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.465 [0.412, 0.509] | – | 0.504 [0.442, 0.570] | – | – | -0.039 [-0.121, +0.035] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.486 [0.427, 0.551] | – | 0.489 [0.433, 0.552] | – | – | -0.003 [-0.070, +0.071] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.506 [0.468, 0.546] | – | 0.482 [0.414, 0.562] | – | – | +0.024 [-0.056, +0.096] | – | – |
| `v.end` | 518 / 362 | 280 | 0.494 [0.424, 0.565] | – | 0.475 [0.413, 0.548] | – | – | +0.019 [-0.078, +0.121] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.523 [0.435, 0.609]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.532, `m2.end` 0.451, `m3.end` 0.414, `m4.end` 0.523. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.506 [0.490, 0.522]; activation minus words +0.008 [-0.059, +0.067]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.497 [0.476, 0.520]; activation minus words -0.003 [-0.056, +0.050]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

