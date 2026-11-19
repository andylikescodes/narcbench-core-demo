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
| `m1.pre` | 257 / 183 | 139 | 0.472 [0.389, 0.553] | – | 0.500 [0.500, 0.500] | – | – | -0.028 [-0.111, +0.053] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.506 [0.448, 0.559] | – | 0.489 [0.438, 0.548] | – | – | +0.017 [-0.078, +0.098] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.479 [0.402, 0.551] | – | 0.478 [0.375, 0.581] | – | – | +0.001 [-0.125, +0.106] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.551 [0.461, 0.639] | – | 0.504 [0.401, 0.603] | – | – | +0.047 [-0.070, +0.177] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.518 [0.453, 0.588] | – | 0.496 [0.394, 0.594] | – | – | +0.022 [-0.087, +0.133] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.505 [0.404, 0.586] | – | 0.525 [0.425, 0.646] | – | – | -0.020 [-0.132, +0.079] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.511 [0.419, 0.599] | – | 0.553 [0.471, 0.632] | – | – | -0.043 [-0.127, +0.040] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.535 [0.461, 0.611] | – | 0.560 [0.479, 0.638] | – | – | -0.026 [-0.112, +0.067] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.499 [0.432, 0.561] | – | 0.560 [0.458, 0.664] | – | – | -0.061 [-0.187, +0.053] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.470 [0.388, 0.552] | – | 0.574 [0.452, 0.690] | – | – | -0.105 [-0.238, +0.015] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.394 [0.326, 0.451] | – | 0.504 [0.384, 0.625] | – | – | -0.109 [-0.239, -0.004] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.609 [0.506, 0.701] | – | 0.560 [0.469, 0.648] | – | – | +0.048 [-0.080, +0.180] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.557 [0.486, 0.630] | – | 0.532 [0.447, 0.631] | – | – | +0.024 [-0.085, +0.126] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.529 [0.467, 0.596] | – | 0.525 [0.440, 0.625] | – | – | +0.004 [-0.113, +0.111] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.564 [0.472, 0.638] | – | 0.532 [0.434, 0.628] | – | – | +0.032 [-0.094, +0.140] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.422 [0.347, 0.496] | – | 0.504 [0.400, 0.599] | – | – | -0.082 [-0.189, +0.051] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.597 [0.477, 0.691] | – | 0.446 [0.353, 0.533] | – | – | +0.151 [+0.012, +0.269] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.581 [0.484, 0.680] | – | 0.475 [0.370, 0.574] | – | – | +0.106 [-0.007, +0.210] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.516 [0.447, 0.586] | – | 0.546 [0.462, 0.617] | – | – | -0.030 [-0.106, +0.071] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.494 [0.417, 0.563] | – | 0.553 [0.475, 0.622] | – | – | -0.060 [-0.152, +0.041] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.394 [0.302, 0.498] | – | 0.553 [0.479, 0.624] | – | – | -0.159 [-0.288, -0.022] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.515 [0.428, 0.592] | – | 0.560 [0.508, 0.612] | – | – | -0.045 [-0.149, +0.044] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.393 [0.299, 0.481] | – | 0.525 [0.463, 0.583] | – | – | -0.132 [-0.226, -0.030] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.435 [0.343, 0.541] | – | 0.525 [0.446, 0.589] | – | – | -0.089 [-0.191, +0.033] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.439 [0.396, 0.487] | – | 0.471 [0.411, 0.528] | – | – | -0.033 [-0.103, +0.048] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.521 [0.460, 0.580] | – | 0.468 [0.403, 0.534] | – | – | +0.054 [-0.031, +0.137] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.456 [0.401, 0.516] | – | 0.479 [0.422, 0.534] | – | – | -0.022 [-0.085, +0.043] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.504 [0.461, 0.557] | – | 0.454 [0.393, 0.517] | – | – | +0.051 [-0.038, +0.146] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.507 [0.439, 0.576] | – | 0.464 [0.407, 0.522] | – | – | +0.043 [-0.033, +0.121] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.412 [0.350, 0.473] | – | 0.457 [0.400, 0.515] | – | – | -0.045 [-0.129, +0.034] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.406 [0.351, 0.469] | – | 0.446 [0.381, 0.512] | – | – | -0.040 [-0.109, +0.036] | – | – |
| `v.end` | 518 / 362 | 280 | 0.507 [0.443, 0.574] | – | 0.443 [0.377, 0.510] | – | – | +0.064 [+0.008, +0.128] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.435 [0.343, 0.541]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.505, `m2.end` 0.609, `m3.end` 0.581, `m4.end` 0.435. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m3.q3`, `v.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.502 [0.475, 0.527]; activation minus words -0.022 [-0.065, +0.017]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.469 [0.442, 0.500]; activation minus words +0.009 [-0.034, +0.055]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

