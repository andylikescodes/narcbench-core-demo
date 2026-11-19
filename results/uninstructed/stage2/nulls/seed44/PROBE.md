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
| `m1.pre` | 257 / 183 | 139 | 0.519 [0.440, 0.594] | – | 0.500 [0.500, 0.500] | – | – | +0.019 [-0.060, +0.094] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.494 [0.445, 0.545] | – | 0.496 [0.456, 0.532] | – | – | -0.002 [-0.055, +0.056] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.525 [0.459, 0.593] | – | 0.514 [0.448, 0.604] | – | – | +0.011 [-0.103, +0.102] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.460 [0.387, 0.538] | – | 0.439 [0.361, 0.523] | – | – | +0.022 [-0.102, +0.137] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.511 [0.434, 0.575] | – | 0.475 [0.356, 0.579] | – | – | +0.036 [-0.063, +0.134] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.475 [0.394, 0.539] | – | 0.554 [0.468, 0.638] | – | – | -0.079 [-0.149, -0.018] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.557 [0.465, 0.638] | – | 0.560 [0.450, 0.669] | – | – | -0.003 [-0.111, +0.099] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.600 [0.516, 0.685] | – | 0.560 [0.450, 0.669] | – | – | +0.040 [-0.069, +0.156] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.543 [0.466, 0.615] | – | 0.560 [0.453, 0.660] | – | – | -0.017 [-0.159, +0.129] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.499 [0.392, 0.600] | – | 0.617 [0.514, 0.724] | – | – | -0.118 [-0.241, +0.004] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.505 [0.407, 0.590] | – | 0.582 [0.486, 0.674] | – | – | -0.077 [-0.207, +0.027] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.574 [0.492, 0.648] | – | 0.560 [0.463, 0.660] | – | – | +0.014 [-0.100, +0.119] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.479 [0.387, 0.561] | – | 0.590 [0.496, 0.686] | – | – | -0.111 [-0.229, -0.007] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.475 [0.364, 0.576] | – | 0.583 [0.492, 0.673] | – | – | -0.108 [-0.218, -0.017] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.521 [0.429, 0.614] | – | 0.518 [0.422, 0.622] | – | – | +0.003 [-0.114, +0.108] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.439 [0.366, 0.519] | – | 0.496 [0.388, 0.602] | – | – | -0.058 [-0.188, +0.074] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.488 [0.414, 0.567] | – | 0.525 [0.428, 0.629] | – | – | -0.037 [-0.166, +0.092] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.466 [0.388, 0.537] | – | 0.496 [0.391, 0.598] | – | – | -0.030 [-0.139, +0.086] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.535 [0.473, 0.602] | – | 0.532 [0.432, 0.629] | – | – | +0.003 [-0.107, +0.127] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.557 [0.485, 0.621] | – | 0.525 [0.422, 0.619] | – | – | +0.033 [-0.095, +0.154] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.504 [0.417, 0.585] | – | 0.532 [0.429, 0.636] | – | – | -0.028 [-0.176, +0.106] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.555 [0.463, 0.663] | – | 0.518 [0.413, 0.619] | – | – | +0.037 [-0.083, +0.169] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.535 [0.451, 0.617] | – | 0.461 [0.360, 0.567] | – | – | +0.074 [-0.042, +0.199] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.511 [0.421, 0.589] | – | 0.496 [0.404, 0.581] | – | – | +0.014 [-0.080, +0.119] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.535 [0.479, 0.587] | – | 0.457 [0.401, 0.510] | – | – | +0.078 [+0.010, +0.143] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.541 [0.459, 0.600] | – | 0.518 [0.461, 0.569] | – | – | +0.024 [-0.056, +0.089] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.494 [0.449, 0.534] | – | 0.479 [0.407, 0.550] | – | – | +0.016 [-0.070, +0.104] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.509 [0.423, 0.593] | – | 0.511 [0.430, 0.592] | – | – | -0.001 [-0.091, +0.085] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.489 [0.427, 0.541] | – | 0.511 [0.425, 0.598] | – | – | -0.022 [-0.103, +0.067] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.570 [0.515, 0.625] | – | 0.496 [0.418, 0.575] | – | – | +0.074 [-0.024, +0.169] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.514 [0.466, 0.553] | – | 0.529 [0.446, 0.615] | – | – | -0.015 [-0.104, +0.061] | – | – |
| `v.end` | 518 / 362 | 280 | 0.474 [0.406, 0.526] | – | 0.550 [0.466, 0.640] | – | – | -0.076 [-0.189, +0.031] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.511 [0.421, 0.589]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.475, `m2.end` 0.574, `m3.end` 0.466, `m4.end` 0.511. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `p1.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.514 [0.487, 0.538]; activation minus words -0.015 [-0.066, +0.033]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.516 [0.476, 0.549]; activation minus words +0.010 [-0.055, +0.077]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

