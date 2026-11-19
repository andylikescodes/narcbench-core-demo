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
| `m1.pre` | 257 / 183 | 139 | 0.480 [0.402, 0.571] | – | 0.500 [0.500, 0.500] | – | – | -0.020 [-0.098, +0.071] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.466 [0.389, 0.543] | – | 0.489 [0.450, 0.528] | – | – | -0.023 [-0.115, +0.066] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.504 [0.400, 0.612] | – | 0.558 [0.433, 0.685] | – | – | -0.054 [-0.179, +0.064] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.558 [0.488, 0.647] | – | 0.496 [0.382, 0.620] | – | – | +0.062 [-0.089, +0.232] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.466 [0.397, 0.535] | – | 0.439 [0.346, 0.527] | – | – | +0.027 [-0.066, +0.125] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.554 [0.477, 0.627] | – | 0.504 [0.394, 0.616] | – | – | +0.050 [-0.059, +0.148] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.540 [0.463, 0.620] | – | 0.624 [0.527, 0.721] | – | – | -0.084 [-0.204, +0.031] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.582 [0.484, 0.676] | – | 0.624 [0.527, 0.721] | – | – | -0.043 [-0.132, +0.062] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.513 [0.432, 0.584] | – | 0.610 [0.532, 0.690] | – | – | -0.096 [-0.213, +0.003] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.451 [0.351, 0.546] | – | 0.617 [0.496, 0.715] | – | – | -0.166 [-0.303, -0.013] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.522 [0.420, 0.616] | – | 0.603 [0.503, 0.698] | – | – | -0.081 [-0.209, +0.048] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.532 [0.438, 0.616] | – | 0.589 [0.496, 0.671] | – | – | -0.057 [-0.170, +0.048] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.509 [0.441, 0.585] | – | 0.619 [0.522, 0.707] | – | – | -0.109 [-0.209, +0.000] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.485 [0.425, 0.556] | – | 0.619 [0.522, 0.707] | – | – | -0.134 [-0.247, -0.014] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.599 [0.499, 0.688] | – | 0.619 [0.542, 0.705] | – | – | -0.020 [-0.129, +0.094] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.489 [0.388, 0.587] | – | 0.619 [0.548, 0.684] | – | – | -0.129 [-0.251, -0.014] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.541 [0.448, 0.628] | – | 0.597 [0.543, 0.650] | – | – | -0.056 [-0.161, +0.044] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.433 [0.354, 0.524] | – | 0.612 [0.537, 0.679] | – | – | -0.178 [-0.280, -0.065] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.496 [0.419, 0.572] | – | 0.560 [0.446, 0.650] | – | – | -0.064 [-0.193, +0.085] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.478 [0.383, 0.560] | – | 0.560 [0.446, 0.650] | – | – | -0.082 [-0.196, +0.045] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.481 [0.414, 0.551] | – | 0.596 [0.483, 0.686] | – | – | -0.115 [-0.227, +0.013] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.468 [0.372, 0.566] | – | 0.539 [0.427, 0.629] | – | – | -0.071 [-0.184, +0.053] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.433 [0.359, 0.515] | – | 0.525 [0.413, 0.622] | – | – | -0.092 [-0.223, +0.070] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.411 [0.344, 0.471] | – | 0.489 [0.398, 0.567] | – | – | -0.078 [-0.177, +0.029] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.469 [0.423, 0.518] | – | 0.543 [0.477, 0.618] | – | – | -0.074 [-0.134, -0.018] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.539 [0.471, 0.600] | – | 0.539 [0.472, 0.619] | – | – | -0.001 [-0.077, +0.076] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.453 [0.390, 0.520] | – | 0.575 [0.500, 0.655] | – | – | -0.122 [-0.190, -0.044] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.486 [0.411, 0.553] | – | 0.571 [0.494, 0.650] | – | – | -0.085 [-0.173, +0.011] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.461 [0.407, 0.520] | – | 0.564 [0.498, 0.629] | – | – | -0.104 [-0.184, -0.017] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.454 [0.401, 0.509] | – | 0.564 [0.500, 0.628] | – | – | -0.110 [-0.176, -0.040] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.511 [0.465, 0.560] | – | 0.561 [0.491, 0.629] | – | – | -0.049 [-0.130, +0.037] | – | – |
| `v.end` | 518 / 362 | 280 | 0.519 [0.455, 0.585] | – | 0.561 [0.487, 0.630] | – | – | -0.042 [-0.120, +0.033] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.411 [0.344, 0.471]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.554, `m2.end` 0.532, `m3.end` 0.433, `m4.end` 0.411. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: `m2.pre`.
  - In one number, the mean over the 24 room positions: activation 0.500 [0.475, 0.522]; activation minus words -0.067 [-0.100, -0.032]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.486 [0.452, 0.521]; activation minus words -0.073 [-0.121, -0.026]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

