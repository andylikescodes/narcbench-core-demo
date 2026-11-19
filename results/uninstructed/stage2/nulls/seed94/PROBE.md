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
| `m1.pre` | 257 / 183 | 139 | 0.553 [0.489, 0.625] | – | 0.500 [0.500, 0.500] | – | – | +0.053 [-0.011, +0.125] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.508 [0.454, 0.567] | – | 0.518 [0.487, 0.547] | – | – | -0.010 [-0.067, +0.060] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.471 [0.366, 0.565] | – | 0.576 [0.458, 0.694] | – | – | -0.105 [-0.232, +0.039] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.443 [0.364, 0.526] | – | 0.540 [0.448, 0.625] | – | – | -0.096 [-0.200, +0.026] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.594 [0.510, 0.678] | – | 0.525 [0.408, 0.637] | – | – | +0.069 [-0.064, +0.214] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.494 [0.393, 0.597] | – | 0.489 [0.379, 0.595] | – | – | +0.004 [-0.102, +0.107] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.501 [0.425, 0.586] | – | 0.532 [0.395, 0.646] | – | – | -0.031 [-0.172, +0.133] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.454 [0.380, 0.536] | – | 0.539 [0.394, 0.657] | – | – | -0.085 [-0.216, +0.086] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.477 [0.373, 0.575] | – | 0.596 [0.479, 0.709] | – | – | -0.119 [-0.250, +0.026] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.484 [0.376, 0.583] | – | 0.574 [0.417, 0.717] | – | – | -0.091 [-0.250, +0.079] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.525 [0.438, 0.607] | – | 0.574 [0.455, 0.680] | – | – | -0.050 [-0.171, +0.091] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.590 [0.500, 0.654] | – | 0.553 [0.440, 0.671] | – | – | +0.037 [-0.104, +0.155] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.386 [0.307, 0.463] | – | 0.482 [0.360, 0.615] | – | – | -0.096 [-0.221, +0.027] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.442 [0.378, 0.514] | – | 0.482 [0.360, 0.615] | – | – | -0.040 [-0.168, +0.079] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.529 [0.433, 0.624] | – | 0.511 [0.391, 0.636] | – | – | +0.019 [-0.127, +0.162] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.485 [0.410, 0.565] | – | 0.525 [0.406, 0.649] | – | – | -0.040 [-0.145, +0.069] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.419 [0.337, 0.513] | – | 0.511 [0.386, 0.644] | – | – | -0.092 [-0.212, +0.061] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.519 [0.428, 0.625] | – | 0.518 [0.390, 0.651] | – | – | +0.001 [-0.135, +0.150] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.426 [0.348, 0.496] | – | 0.468 [0.373, 0.563] | – | – | -0.043 [-0.170, +0.069] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.452 [0.374, 0.525] | – | 0.461 [0.364, 0.557] | – | – | -0.009 [-0.127, +0.111] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.496 [0.434, 0.554] | – | 0.546 [0.420, 0.667] | – | – | -0.050 [-0.178, +0.087] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.548 [0.458, 0.617] | – | 0.539 [0.409, 0.663] | – | – | +0.009 [-0.148, +0.157] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.501 [0.415, 0.577] | – | 0.546 [0.437, 0.649] | – | – | -0.045 [-0.153, +0.062] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.572 [0.479, 0.645] | – | 0.525 [0.428, 0.625] | – | – | +0.047 [-0.067, +0.152] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.538 [0.476, 0.595] | – | 0.529 [0.455, 0.598] | – | – | +0.009 [-0.082, +0.113] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.549 [0.496, 0.602] | – | 0.504 [0.418, 0.586] | – | – | +0.046 [-0.023, +0.115] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.496 [0.442, 0.542] | – | 0.496 [0.409, 0.580] | – | – | -0.000 [-0.093, +0.087] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.464 [0.406, 0.519] | – | 0.489 [0.411, 0.567] | – | – | -0.025 [-0.112, +0.046] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.494 [0.436, 0.549] | – | 0.529 [0.439, 0.615] | – | – | -0.034 [-0.151, +0.082] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.476 [0.414, 0.542] | – | 0.529 [0.449, 0.611] | – | – | -0.053 [-0.129, +0.030] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.555 [0.513, 0.600] | – | 0.536 [0.452, 0.617] | – | – | +0.019 [-0.077, +0.119] | – | – |
| `v.end` | 518 / 362 | 280 | 0.564 [0.505, 0.614] | – | 0.529 [0.436, 0.619] | – | – | +0.036 [-0.039, +0.108] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.572 [0.479, 0.645]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.494, `m2.end` 0.590, `m3.end` 0.519, `m4.end` 0.572. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.494 [0.474, 0.515]; activation minus words -0.032 [-0.078, +0.025]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.517 [0.496, 0.537]; activation minus words -0.000 [-0.068, +0.065]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

