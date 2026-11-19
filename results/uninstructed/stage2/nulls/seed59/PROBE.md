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
| `m1.pre` | 257 / 183 | 139 | 0.447 [0.385, 0.505] | – | 0.500 [0.500, 0.500] | – | – | -0.053 [-0.115, +0.005] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.503 [0.424, 0.571] | – | 0.518 [0.480, 0.552] | – | – | -0.015 [-0.102, +0.069] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.498 [0.416, 0.593] | – | 0.504 [0.417, 0.582] | – | – | -0.006 [-0.104, +0.103] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.514 [0.432, 0.585] | – | 0.511 [0.415, 0.605] | – | – | +0.003 [-0.083, +0.086] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.483 [0.407, 0.583] | – | 0.540 [0.420, 0.652] | – | – | -0.056 [-0.168, +0.063] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.485 [0.396, 0.577] | – | 0.446 [0.333, 0.571] | – | – | +0.039 [-0.093, +0.175] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.505 [0.426, 0.576] | – | 0.454 [0.333, 0.565] | – | – | +0.051 [-0.075, +0.184] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.451 [0.362, 0.539] | – | 0.454 [0.333, 0.565] | – | – | -0.003 [-0.160, +0.135] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.460 [0.398, 0.530] | – | 0.496 [0.371, 0.603] | – | – | -0.037 [-0.149, +0.095] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.543 [0.473, 0.609] | – | 0.489 [0.375, 0.595] | – | – | +0.054 [-0.058, +0.177] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.522 [0.449, 0.585] | – | 0.468 [0.354, 0.581] | – | – | +0.054 [-0.046, +0.157] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.528 [0.439, 0.615] | – | 0.475 [0.368, 0.575] | – | – | +0.052 [-0.085, +0.188] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.443 [0.365, 0.527] | – | 0.460 [0.369, 0.558] | – | – | -0.017 [-0.131, +0.100] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.511 [0.444, 0.580] | – | 0.453 [0.359, 0.551] | – | – | +0.058 [-0.064, +0.175] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.469 [0.378, 0.561] | – | 0.489 [0.389, 0.589] | – | – | -0.020 [-0.159, +0.115] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.435 [0.353, 0.532] | – | 0.453 [0.353, 0.553] | – | – | -0.019 [-0.119, +0.095] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.527 [0.431, 0.616] | – | 0.424 [0.333, 0.520] | – | – | +0.102 [-0.029, +0.218] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.439 [0.370, 0.511] | – | 0.439 [0.321, 0.548] | – | – | +0.000 [-0.132, +0.131] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.536 [0.447, 0.619] | – | 0.482 [0.379, 0.578] | – | – | +0.054 [-0.062, +0.167] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.445 [0.317, 0.567] | – | 0.482 [0.379, 0.578] | – | – | -0.037 [-0.166, +0.102] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.407 [0.329, 0.496] | – | 0.511 [0.388, 0.616] | – | – | -0.104 [-0.221, +0.057] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.499 [0.390, 0.593] | – | 0.475 [0.382, 0.557] | – | – | +0.024 [-0.105, +0.150] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.411 [0.341, 0.491] | – | 0.475 [0.378, 0.571] | – | – | -0.064 [-0.162, +0.045] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.545 [0.428, 0.654] | – | 0.525 [0.437, 0.607] | – | – | +0.020 [-0.113, +0.153] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.514 [0.473, 0.551] | – | 0.471 [0.383, 0.573] | – | – | +0.043 [-0.077, +0.147] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.483 [0.433, 0.550] | – | 0.464 [0.386, 0.550] | – | – | +0.019 [-0.059, +0.110] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.494 [0.440, 0.546] | – | 0.468 [0.397, 0.538] | – | – | +0.026 [-0.042, +0.090] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.478 [0.426, 0.535] | – | 0.464 [0.386, 0.548] | – | – | +0.014 [-0.062, +0.090] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.490 [0.458, 0.524] | – | 0.518 [0.423, 0.624] | – | – | -0.028 [-0.121, +0.054] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.465 [0.415, 0.509] | – | 0.514 [0.422, 0.617] | – | – | -0.049 [-0.140, +0.027] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.502 [0.464, 0.542] | – | 0.521 [0.425, 0.623] | – | – | -0.019 [-0.117, +0.070] | – | – |
| `v.end` | 518 / 362 | 280 | 0.441 [0.379, 0.505] | – | 0.511 [0.421, 0.608] | – | – | -0.069 [-0.146, +0.003] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.545 [0.428, 0.654]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.485, `m2.end` 0.528, `m3.end` 0.439, `m4.end` 0.545. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.484 [0.454, 0.511]; activation minus words +0.003 [-0.049, +0.052]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.483 [0.455, 0.516]; activation minus words -0.008 [-0.072, +0.047]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

