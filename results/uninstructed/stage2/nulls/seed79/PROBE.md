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
| `m1.pre` | 257 / 183 | 139 | 0.505 [0.415, 0.588] | – | 0.500 [0.500, 0.500] | – | – | +0.005 [-0.085, +0.088] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.530 [0.454, 0.593] | – | 0.496 [0.445, 0.555] | – | – | +0.034 [-0.048, +0.110] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.488 [0.394, 0.573] | – | 0.558 [0.479, 0.642] | – | – | -0.070 [-0.217, +0.055] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.521 [0.414, 0.624] | – | 0.532 [0.443, 0.619] | – | – | -0.012 [-0.130, +0.107] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.528 [0.443, 0.626] | – | 0.554 [0.467, 0.647] | – | – | -0.026 [-0.117, +0.077] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.519 [0.396, 0.640] | – | 0.511 [0.419, 0.619] | – | – | +0.009 [-0.132, +0.149] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.474 [0.395, 0.562] | – | 0.454 [0.349, 0.567] | – | – | +0.020 [-0.093, +0.136] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.407 [0.338, 0.486] | – | 0.454 [0.347, 0.567] | – | – | -0.047 [-0.152, +0.069] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.543 [0.442, 0.641] | – | 0.489 [0.393, 0.595] | – | – | +0.054 [-0.058, +0.166] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.505 [0.410, 0.598] | – | 0.454 [0.355, 0.567] | – | – | +0.051 [-0.100, +0.193] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.614 [0.552, 0.677] | – | 0.461 [0.358, 0.571] | – | – | +0.153 [+0.058, +0.249] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.484 [0.377, 0.593] | – | 0.433 [0.333, 0.540] | – | – | +0.051 [-0.070, +0.185] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.498 [0.426, 0.587] | – | 0.446 [0.357, 0.533] | – | – | +0.052 [-0.076, +0.200] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.485 [0.416, 0.560] | – | 0.446 [0.364, 0.524] | – | – | +0.039 [-0.070, +0.163] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.455 [0.368, 0.550] | – | 0.489 [0.403, 0.569] | – | – | -0.035 [-0.137, +0.076] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.525 [0.412, 0.647] | – | 0.460 [0.380, 0.533] | – | – | +0.065 [-0.069, +0.209] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.547 [0.481, 0.610] | – | 0.439 [0.339, 0.543] | – | – | +0.108 [-0.026, +0.225] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.445 [0.374, 0.526] | – | 0.475 [0.380, 0.577] | – | – | -0.030 [-0.128, +0.076] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.506 [0.422, 0.583] | – | 0.511 [0.395, 0.630] | – | – | -0.004 [-0.128, +0.120] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.518 [0.430, 0.606] | – | 0.511 [0.395, 0.630] | – | – | +0.007 [-0.152, +0.186] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.461 [0.365, 0.550] | – | 0.511 [0.389, 0.627] | – | – | -0.050 [-0.161, +0.054] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.533 [0.408, 0.650] | – | 0.504 [0.414, 0.592] | – | – | +0.030 [-0.124, +0.178] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.614 [0.530, 0.707] | – | 0.504 [0.432, 0.561] | – | – | +0.111 [+0.014, +0.222] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.545 [0.438, 0.634] | – | 0.518 [0.429, 0.608] | – | – | +0.027 [-0.077, +0.122] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.468 [0.400, 0.528] | – | 0.529 [0.466, 0.604] | – | – | -0.061 [-0.163, +0.027] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.511 [0.434, 0.585] | – | 0.493 [0.440, 0.549] | – | – | +0.019 [-0.060, +0.091] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.504 [0.448, 0.567] | – | 0.514 [0.418, 0.619] | – | – | -0.010 [-0.098, +0.079] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.471 [0.404, 0.551] | – | 0.514 [0.430, 0.603] | – | – | -0.044 [-0.128, +0.044] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.529 [0.494, 0.563] | – | 0.464 [0.381, 0.553] | – | – | +0.065 [-0.019, +0.146] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.461 [0.406, 0.515] | – | 0.468 [0.386, 0.553] | – | – | -0.007 [-0.118, +0.098] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.484 [0.432, 0.531] | – | 0.507 [0.447, 0.567] | – | – | -0.024 [-0.084, +0.034] | – | – |
| `v.end` | 518 / 362 | 280 | 0.457 [0.403, 0.510] | – | 0.493 [0.422, 0.568] | – | – | -0.036 [-0.137, +0.062] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.545 [0.438, 0.634]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.519, `m2.end` 0.484, `m3.end` 0.445, `m4.end` 0.545. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.q3`, `m4.q3`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.510 [0.491, 0.533]; activation minus words +0.023 [-0.011, +0.058]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.486 [0.462, 0.512]; activation minus words -0.012 [-0.071, +0.043]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

