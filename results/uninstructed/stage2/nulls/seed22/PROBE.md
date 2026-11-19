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
| `m1.pre` | 257 / 183 | 139 | 0.503 [0.423, 0.574] | – | 0.500 [0.500, 0.500] | – | – | +0.003 [-0.077, +0.074] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.529 [0.439, 0.609] | – | 0.532 [0.505, 0.560] | – | – | -0.004 [-0.085, +0.074] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.619 [0.540, 0.692] | – | 0.572 [0.440, 0.708] | – | – | +0.047 [-0.096, +0.177] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.501 [0.419, 0.584] | – | 0.511 [0.399, 0.628] | – | – | -0.010 [-0.164, +0.155] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.486 [0.371, 0.593] | – | 0.496 [0.389, 0.615] | – | – | -0.010 [-0.148, +0.108] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.512 [0.417, 0.606] | – | 0.475 [0.357, 0.595] | – | – | +0.037 [-0.057, +0.134] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.501 [0.419, 0.587] | – | 0.518 [0.442, 0.592] | – | – | -0.017 [-0.130, +0.095] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.472 [0.412, 0.542] | – | 0.518 [0.442, 0.592] | – | – | -0.045 [-0.135, +0.051] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.438 [0.372, 0.512] | – | 0.496 [0.415, 0.577] | – | – | -0.058 [-0.152, +0.058] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.548 [0.462, 0.623] | – | 0.489 [0.402, 0.582] | – | – | +0.058 [-0.059, +0.157] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.455 [0.365, 0.534] | – | 0.560 [0.477, 0.636] | – | – | -0.105 [-0.200, -0.024] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.532 [0.423, 0.652] | – | 0.553 [0.465, 0.648] | – | – | -0.021 [-0.137, +0.099] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.528 [0.451, 0.608] | – | 0.489 [0.372, 0.611] | – | – | +0.039 [-0.062, +0.144] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.517 [0.434, 0.601] | – | 0.489 [0.372, 0.611] | – | – | +0.027 [-0.088, +0.142] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.560 [0.480, 0.638] | – | 0.482 [0.344, 0.619] | – | – | +0.078 [-0.047, +0.197] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.417 [0.323, 0.518] | – | 0.504 [0.400, 0.607] | – | – | -0.086 [-0.181, +0.006] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.515 [0.431, 0.613] | – | 0.525 [0.416, 0.625] | – | – | -0.010 [-0.132, +0.125] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.577 [0.517, 0.637] | – | 0.496 [0.391, 0.599] | – | – | +0.081 [-0.036, +0.203] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.515 [0.458, 0.567] | – | 0.496 [0.413, 0.580] | – | – | +0.018 [-0.071, +0.102] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.470 [0.400, 0.535] | – | 0.496 [0.413, 0.580] | – | – | -0.027 [-0.128, +0.067] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.470 [0.361, 0.594] | – | 0.468 [0.374, 0.562] | – | – | +0.001 [-0.095, +0.109] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.498 [0.418, 0.571] | – | 0.475 [0.386, 0.586] | – | – | +0.023 [-0.105, +0.122] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.609 [0.553, 0.667] | – | 0.468 [0.364, 0.591] | – | – | +0.140 [+0.010, +0.245] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.457 [0.335, 0.573] | – | 0.489 [0.385, 0.612] | – | – | -0.033 [-0.164, +0.090] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.510 [0.455, 0.569] | – | 0.521 [0.453, 0.579] | – | – | -0.011 [-0.078, +0.067] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.546 [0.468, 0.618] | – | 0.561 [0.477, 0.633] | – | – | -0.014 [-0.118, +0.093] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.505 [0.458, 0.551] | – | 0.561 [0.471, 0.642] | – | – | -0.056 [-0.152, +0.033] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.549 [0.507, 0.601] | – | 0.579 [0.489, 0.661] | – | – | -0.029 [-0.109, +0.075] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.508 [0.461, 0.558] | – | 0.586 [0.492, 0.674] | – | – | -0.078 [-0.162, +0.018] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.534 [0.472, 0.602] | – | 0.586 [0.496, 0.665] | – | – | -0.052 [-0.130, +0.022] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.567 [0.525, 0.615] | – | 0.607 [0.530, 0.682] | – | – | -0.040 [-0.118, +0.041] | – | – |
| `v.end` | 518 / 362 | 280 | 0.539 [0.467, 0.613] | – | 0.582 [0.492, 0.667] | – | – | -0.044 [-0.128, +0.037] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.457 [0.335, 0.573]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.512, `m2.end` 0.532, `m3.end` 0.577, `m4.end` 0.457. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.q3`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.509 [0.482, 0.538]; activation minus words +0.005 [-0.034, +0.041]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.532 [0.501, 0.570]; activation minus words -0.041 [-0.106, +0.033]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

