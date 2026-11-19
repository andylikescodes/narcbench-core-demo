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
| `m1.pre` | 257 / 183 | 139 | 0.464 [0.393, 0.554] | – | 0.500 [0.500, 0.500] | – | – | -0.036 [-0.107, +0.054] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.593 [0.522, 0.680] | – | 0.511 [0.487, 0.539] | – | – | +0.082 [+0.005, +0.168] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.453 [0.373, 0.529] | – | 0.460 [0.383, 0.550] | – | – | -0.007 [-0.144, +0.111] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.377 [0.296, 0.463] | – | 0.453 [0.368, 0.556] | – | – | -0.076 [-0.215, +0.051] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.417 [0.355, 0.479] | – | 0.504 [0.394, 0.608] | – | – | -0.086 [-0.191, +0.028] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.509 [0.443, 0.574] | – | 0.482 [0.384, 0.570] | – | – | +0.027 [-0.074, +0.142] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.451 [0.384, 0.531] | – | 0.553 [0.416, 0.667] | – | – | -0.102 [-0.233, +0.038] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.505 [0.426, 0.583] | – | 0.560 [0.425, 0.670] | – | – | -0.055 [-0.168, +0.061] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.488 [0.416, 0.569] | – | 0.567 [0.453, 0.667] | – | – | -0.079 [-0.207, +0.063] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.538 [0.448, 0.610] | – | 0.546 [0.442, 0.639] | – | – | -0.009 [-0.134, +0.113] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.512 [0.405, 0.619] | – | 0.518 [0.411, 0.598] | – | – | -0.006 [-0.127, +0.121] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.548 [0.456, 0.630] | – | 0.553 [0.447, 0.639] | – | – | -0.006 [-0.104, +0.105] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.505 [0.397, 0.609] | – | 0.460 [0.372, 0.549] | – | – | +0.045 [-0.088, +0.171] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.560 [0.460, 0.645] | – | 0.460 [0.372, 0.549] | – | – | +0.099 [-0.017, +0.218] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.537 [0.435, 0.637] | – | 0.518 [0.427, 0.602] | – | – | +0.019 [-0.100, +0.135] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.534 [0.436, 0.620] | – | 0.525 [0.442, 0.611] | – | – | +0.009 [-0.112, +0.121] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.452 [0.374, 0.516] | – | 0.504 [0.434, 0.562] | – | – | -0.052 [-0.120, +0.017] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.564 [0.472, 0.635] | – | 0.475 [0.381, 0.557] | – | – | +0.089 [-0.022, +0.184] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.580 [0.511, 0.652] | – | 0.525 [0.405, 0.634] | – | – | +0.055 [-0.043, +0.160] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.590 [0.498, 0.683] | – | 0.518 [0.400, 0.629] | – | – | +0.072 [-0.051, +0.204] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.491 [0.397, 0.584] | – | 0.553 [0.460, 0.649] | – | – | -0.062 [-0.182, +0.078] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.489 [0.398, 0.581] | – | 0.574 [0.478, 0.663] | – | – | -0.085 [-0.211, +0.046] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.460 [0.388, 0.527] | – | 0.582 [0.471, 0.672] | – | – | -0.122 [-0.213, -0.004] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.477 [0.396, 0.544] | – | 0.496 [0.369, 0.602] | – | – | -0.020 [-0.122, +0.098] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.579 [0.522, 0.631] | – | 0.525 [0.457, 0.581] | – | – | +0.054 [-0.023, +0.130] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.484 [0.420, 0.542] | – | 0.521 [0.464, 0.574] | – | – | -0.038 [-0.113, +0.038] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.571 [0.518, 0.639] | – | 0.514 [0.455, 0.567] | – | – | +0.056 [-0.009, +0.145] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.515 [0.441, 0.586] | – | 0.514 [0.453, 0.565] | – | – | +0.001 [-0.077, +0.079] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.466 [0.404, 0.525] | – | 0.529 [0.465, 0.596] | – | – | -0.062 [-0.155, +0.026] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.493 [0.431, 0.555] | – | 0.532 [0.463, 0.603] | – | – | -0.039 [-0.138, +0.054] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.453 [0.400, 0.507] | – | 0.518 [0.448, 0.592] | – | – | -0.065 [-0.148, +0.027] | – | – |
| `v.end` | 518 / 362 | 280 | 0.497 [0.433, 0.561] | – | 0.525 [0.459, 0.593] | – | – | -0.028 [-0.110, +0.058] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.477 [0.396, 0.544]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.509, `m2.end` 0.548, `m3.end` 0.564, `m4.end` 0.477. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.t0`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.504 [0.474, 0.532]; activation minus words -0.013 [-0.050, +0.026]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.507 [0.474, 0.541]; activation minus words -0.015 [-0.062, +0.039]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

