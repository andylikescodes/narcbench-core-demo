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
| `m1.pre` | 257 / 183 | 139 | 0.482 [0.418, 0.552] | – | 0.500 [0.500, 0.500] | – | – | -0.018 [-0.082, +0.052] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.525 [0.458, 0.597] | – | 0.489 [0.447, 0.527] | – | – | +0.036 [-0.042, +0.122] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.462 [0.384, 0.559] | – | 0.504 [0.405, 0.600] | – | – | -0.042 [-0.170, +0.091] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.435 [0.339, 0.533] | – | 0.540 [0.469, 0.618] | – | – | -0.105 [-0.231, +0.021] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.424 [0.350, 0.513] | – | 0.511 [0.415, 0.620] | – | – | -0.086 [-0.202, +0.047] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.494 [0.414, 0.575] | – | 0.525 [0.428, 0.628] | – | – | -0.032 [-0.151, +0.089] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.443 [0.375, 0.508] | – | 0.553 [0.458, 0.636] | – | – | -0.111 [-0.200, -0.030] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.522 [0.442, 0.600] | – | 0.553 [0.458, 0.636] | – | – | -0.031 [-0.120, +0.063] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.590 [0.476, 0.675] | – | 0.468 [0.360, 0.573] | – | – | +0.122 [-0.010, +0.249] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.478 [0.389, 0.581] | – | 0.468 [0.347, 0.582] | – | – | +0.010 [-0.107, +0.168] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.479 [0.391, 0.571] | – | 0.461 [0.356, 0.561] | – | – | +0.018 [-0.082, +0.138] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.479 [0.405, 0.562] | – | 0.440 [0.349, 0.528] | – | – | +0.040 [-0.070, +0.164] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.479 [0.396, 0.568] | – | 0.475 [0.361, 0.589] | – | – | +0.004 [-0.116, +0.147] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.486 [0.396, 0.584] | – | 0.475 [0.371, 0.578] | – | – | +0.012 [-0.107, +0.147] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.445 [0.360, 0.531] | – | 0.468 [0.362, 0.572] | – | – | -0.023 [-0.138, +0.091] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.527 [0.445, 0.606] | – | 0.511 [0.412, 0.615] | – | – | +0.016 [-0.127, +0.147] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.481 [0.393, 0.577] | – | 0.453 [0.371, 0.538] | – | – | +0.027 [-0.076, +0.144] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.455 [0.384, 0.534] | – | 0.446 [0.348, 0.533] | – | – | +0.009 [-0.112, +0.139] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.478 [0.418, 0.548] | – | 0.433 [0.341, 0.514] | – | – | +0.045 [-0.064, +0.173] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.487 [0.413, 0.565] | – | 0.433 [0.341, 0.514] | – | – | +0.054 [-0.025, +0.149] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.465 [0.380, 0.547] | – | 0.440 [0.356, 0.515] | – | – | +0.026 [-0.104, +0.148] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.468 [0.374, 0.578] | – | 0.418 [0.315, 0.511] | – | – | +0.050 [-0.057, +0.160] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.404 [0.332, 0.479] | – | 0.440 [0.353, 0.523] | – | – | -0.035 [-0.146, +0.076] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.460 [0.367, 0.554] | – | 0.482 [0.387, 0.566] | – | – | -0.023 [-0.113, +0.087] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.441 [0.397, 0.491] | – | 0.411 [0.337, 0.483] | – | – | +0.031 [-0.055, +0.122] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.474 [0.412, 0.543] | – | 0.393 [0.325, 0.469] | – | – | +0.081 [-0.012, +0.180] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.492 [0.428, 0.554] | – | 0.411 [0.358, 0.467] | – | – | +0.081 [+0.004, +0.159] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.533 [0.459, 0.631] | – | 0.418 [0.360, 0.490] | – | – | +0.115 [+0.038, +0.207] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.490 [0.431, 0.544] | – | 0.436 [0.380, 0.489] | – | – | +0.054 [-0.038, +0.140] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.488 [0.433, 0.543] | – | 0.429 [0.371, 0.484] | – | – | +0.059 [-0.004, +0.126] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.446 [0.401, 0.491] | – | 0.436 [0.367, 0.504] | – | – | +0.010 [-0.061, +0.082] | – | – |
| `v.end` | 518 / 362 | 280 | 0.531 [0.476, 0.592] | – | 0.439 [0.377, 0.502] | – | – | +0.091 [+0.007, +0.172] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.460 [0.367, 0.554]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.494, `m2.end` 0.479, `m3.end` 0.455, `m4.end` 0.460. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `p2.pre`, `p2.end`, `v.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.477 [0.460, 0.498]; activation minus words -0.002 [-0.049, +0.055]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.487 [0.459, 0.517]; activation minus words +0.065 [+0.016, +0.118]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

