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
| `m1.pre` | 257 / 183 | 139 | 0.510 [0.421, 0.595] | – | 0.500 [0.500, 0.500] | – | – | +0.010 [-0.079, +0.095] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.524 [0.432, 0.611] | – | 0.468 [0.422, 0.506] | – | – | +0.056 [-0.038, +0.157] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.458 [0.357, 0.559] | – | 0.493 [0.415, 0.584] | – | – | -0.035 [-0.170, +0.093] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.558 [0.475, 0.638] | – | 0.482 [0.397, 0.553] | – | – | +0.076 [-0.025, +0.183] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.492 [0.416, 0.576] | – | 0.518 [0.405, 0.622] | – | – | -0.026 [-0.133, +0.084] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.548 [0.465, 0.624] | – | 0.547 [0.448, 0.629] | – | – | +0.001 [-0.111, +0.135] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.563 [0.456, 0.651] | – | 0.518 [0.410, 0.631] | – | – | +0.045 [-0.094, +0.165] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.461 [0.383, 0.529] | – | 0.511 [0.398, 0.627] | – | – | -0.050 [-0.176, +0.064] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.447 [0.352, 0.539] | – | 0.475 [0.377, 0.573] | – | – | -0.028 [-0.172, +0.091] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.522 [0.416, 0.624] | – | 0.475 [0.337, 0.596] | – | – | +0.047 [-0.073, +0.172] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.451 [0.378, 0.525] | – | 0.489 [0.340, 0.619] | – | – | -0.038 [-0.169, +0.119] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.499 [0.403, 0.598] | – | 0.504 [0.359, 0.639] | – | – | -0.004 [-0.158, +0.158] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.426 [0.367, 0.480] | – | 0.532 [0.422, 0.626] | – | – | -0.106 [-0.195, -0.004] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.525 [0.466, 0.590] | – | 0.532 [0.422, 0.626] | – | – | -0.007 [-0.101, +0.097] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.540 [0.439, 0.626] | – | 0.511 [0.389, 0.612] | – | – | +0.029 [-0.098, +0.167] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.432 [0.367, 0.506] | – | 0.532 [0.391, 0.647] | – | – | -0.101 [-0.223, +0.051] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.447 [0.371, 0.526] | – | 0.576 [0.438, 0.685] | – | – | -0.128 [-0.268, +0.029] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.517 [0.410, 0.606] | – | 0.561 [0.436, 0.667] | – | – | -0.045 [-0.172, +0.094] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.495 [0.381, 0.600] | – | 0.475 [0.344, 0.595] | – | – | +0.020 [-0.116, +0.178] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.499 [0.406, 0.602] | – | 0.468 [0.343, 0.582] | – | – | +0.031 [-0.102, +0.186] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.423 [0.372, 0.479] | – | 0.447 [0.328, 0.566] | – | – | -0.024 [-0.143, +0.100] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.460 [0.346, 0.561] | – | 0.461 [0.327, 0.595] | – | – | -0.001 [-0.169, +0.174] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.488 [0.364, 0.603] | – | 0.496 [0.356, 0.631] | – | – | -0.009 [-0.179, +0.170] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.565 [0.492, 0.635] | – | 0.475 [0.343, 0.599] | – | – | +0.089 [-0.072, +0.255] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.502 [0.429, 0.568] | – | 0.486 [0.401, 0.561] | – | – | +0.016 [-0.116, +0.137] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.468 [0.420, 0.524] | – | 0.457 [0.370, 0.540] | – | – | +0.011 [-0.094, +0.124] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.485 [0.407, 0.553] | – | 0.493 [0.390, 0.584] | – | – | -0.008 [-0.131, +0.118] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.504 [0.430, 0.576] | – | 0.514 [0.429, 0.592] | – | – | -0.010 [-0.086, +0.057] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.479 [0.414, 0.538] | – | 0.504 [0.404, 0.597] | – | – | -0.024 [-0.138, +0.107] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.554 [0.493, 0.617] | – | 0.511 [0.425, 0.593] | – | – | +0.043 [-0.058, +0.146] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.510 [0.462, 0.560] | – | 0.479 [0.391, 0.560] | – | – | +0.031 [-0.068, +0.141] | – | – |
| `v.end` | 518 / 362 | 280 | 0.487 [0.413, 0.556] | – | 0.486 [0.396, 0.569] | – | – | +0.001 [-0.082, +0.084] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.565 [0.492, 0.635]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.548, `m2.end` 0.499, `m3.end` 0.517, `m4.end` 0.565. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.494 [0.470, 0.515]; activation minus words -0.008 [-0.049, +0.038]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.499 [0.470, 0.521]; activation minus words +0.008 [-0.068, +0.092]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

