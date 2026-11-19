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
| `m1.pre` | 257 / 183 | 139 | 0.449 [0.376, 0.514] | – | 0.500 [0.500, 0.500] | – | – | -0.051 [-0.124, +0.014] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.478 [0.396, 0.555] | – | 0.489 [0.456, 0.521] | – | – | -0.011 [-0.102, +0.073] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.502 [0.417, 0.581] | – | 0.496 [0.413, 0.580] | – | – | +0.006 [-0.087, +0.095] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.603 [0.511, 0.666] | – | 0.547 [0.429, 0.664] | – | – | +0.056 [-0.081, +0.179] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.515 [0.434, 0.600] | – | 0.547 [0.430, 0.649] | – | – | -0.032 [-0.137, +0.080] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.542 [0.430, 0.642] | – | 0.518 [0.393, 0.634] | – | – | +0.024 [-0.095, +0.139] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.545 [0.453, 0.626] | – | 0.461 [0.358, 0.565] | – | – | +0.084 [-0.057, +0.222] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.576 [0.481, 0.669] | – | 0.461 [0.358, 0.565] | – | – | +0.115 [-0.004, +0.240] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.499 [0.389, 0.619] | – | 0.440 [0.325, 0.560] | – | – | +0.060 [-0.083, +0.211] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.458 [0.375, 0.539] | – | 0.447 [0.340, 0.549] | – | – | +0.011 [-0.089, +0.117] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.477 [0.365, 0.581] | – | 0.433 [0.329, 0.542] | – | – | +0.044 [-0.046, +0.148] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.543 [0.438, 0.645] | – | 0.433 [0.320, 0.544] | – | – | +0.111 [-0.033, +0.271] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.502 [0.402, 0.595] | – | 0.561 [0.430, 0.676] | – | – | -0.059 [-0.147, +0.046] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.573 [0.493, 0.652] | – | 0.561 [0.430, 0.676] | – | – | +0.012 [-0.086, +0.131] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.436 [0.352, 0.510] | – | 0.540 [0.395, 0.678] | – | – | -0.104 [-0.254, +0.053] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.560 [0.492, 0.614] | – | 0.489 [0.333, 0.633] | – | – | +0.071 [-0.061, +0.211] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.583 [0.495, 0.668] | – | 0.525 [0.384, 0.658] | – | – | +0.058 [-0.060, +0.194] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.593 [0.503, 0.691] | – | 0.532 [0.381, 0.676] | – | – | +0.060 [-0.087, +0.214] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.434 [0.338, 0.512] | – | 0.447 [0.366, 0.539] | – | – | -0.013 [-0.129, +0.080] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.530 [0.436, 0.615] | – | 0.454 [0.368, 0.546] | – | – | +0.077 [-0.056, +0.190] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.556 [0.491, 0.630] | – | 0.454 [0.368, 0.552] | – | – | +0.102 [-0.007, +0.208] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.452 [0.365, 0.536] | – | 0.433 [0.349, 0.525] | – | – | +0.020 [-0.125, +0.156] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.434 [0.342, 0.540] | – | 0.468 [0.360, 0.575] | – | – | -0.034 [-0.180, +0.098] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.505 [0.416, 0.592] | – | 0.447 [0.353, 0.545] | – | – | +0.058 [-0.077, +0.187] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.525 [0.481, 0.566] | – | 0.464 [0.395, 0.534] | – | – | +0.061 [-0.011, +0.131] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.503 [0.450, 0.561] | – | 0.489 [0.410, 0.565] | – | – | +0.014 [-0.069, +0.107] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.466 [0.416, 0.515] | – | 0.536 [0.486, 0.590] | – | – | -0.069 [-0.145, +0.000] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.518 [0.461, 0.579] | – | 0.514 [0.461, 0.567] | – | – | +0.004 [-0.068, +0.090] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.471 [0.433, 0.506] | – | 0.521 [0.468, 0.575] | – | – | -0.050 [-0.109, +0.009] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.462 [0.392, 0.521] | – | 0.529 [0.469, 0.582] | – | – | -0.066 [-0.158, +0.018] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.500 [0.440, 0.547] | – | 0.514 [0.453, 0.571] | – | – | -0.014 [-0.102, +0.065] | – | – |
| `v.end` | 518 / 362 | 280 | 0.518 [0.441, 0.585] | – | 0.521 [0.464, 0.573] | – | – | -0.004 [-0.094, +0.083] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.505 [0.416, 0.592]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.542, `m2.end` 0.543, `m3.end` 0.593, `m4.end` 0.505. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.514 [0.480, 0.541]; activation minus words +0.028 [-0.011, +0.073]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.495 [0.468, 0.520]; activation minus words -0.016 [-0.058, +0.026]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

