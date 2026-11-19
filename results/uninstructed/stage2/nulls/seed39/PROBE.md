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
| `m1.pre` | 257 / 183 | 139 | 0.532 [0.438, 0.618] | – | 0.500 [0.500, 0.500] | – | – | +0.032 [-0.062, +0.118] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.588 [0.526, 0.639] | – | 0.496 [0.451, 0.537] | – | – | +0.091 [+0.028, +0.150] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.446 [0.365, 0.515] | – | 0.439 [0.355, 0.522] | – | – | +0.007 [-0.112, +0.113] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.455 [0.394, 0.525] | – | 0.532 [0.427, 0.621] | – | – | -0.078 [-0.164, +0.032] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.442 [0.348, 0.544] | – | 0.475 [0.360, 0.570] | – | – | -0.033 [-0.144, +0.097] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.609 [0.529, 0.683] | – | 0.504 [0.386, 0.606] | – | – | +0.105 [-0.030, +0.248] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.421 [0.344, 0.489] | – | 0.574 [0.468, 0.692] | – | – | -0.153 [-0.306, -0.030] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.494 [0.428, 0.561] | – | 0.582 [0.481, 0.697] | – | – | -0.088 [-0.212, +0.010] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.569 [0.453, 0.662] | – | 0.567 [0.461, 0.676] | – | – | +0.001 [-0.141, +0.118] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.506 [0.407, 0.620] | – | 0.539 [0.458, 0.623] | – | – | -0.033 [-0.167, +0.091] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.468 [0.401, 0.542] | – | 0.582 [0.467, 0.693] | – | – | -0.113 [-0.218, -0.001] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.502 [0.400, 0.619] | – | 0.496 [0.417, 0.580] | – | – | +0.006 [-0.101, +0.115] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.475 [0.403, 0.540] | – | 0.496 [0.391, 0.600] | – | – | -0.022 [-0.157, +0.122] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.498 [0.435, 0.563] | – | 0.496 [0.391, 0.600] | – | – | +0.001 [-0.127, +0.136] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.486 [0.411, 0.566] | – | 0.496 [0.399, 0.594] | – | – | -0.010 [-0.096, +0.083] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.429 [0.350, 0.512] | – | 0.518 [0.425, 0.609] | – | – | -0.089 [-0.179, +0.000] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.482 [0.400, 0.563] | – | 0.547 [0.447, 0.636] | – | – | -0.065 [-0.183, +0.079] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.488 [0.371, 0.595] | – | 0.540 [0.441, 0.628] | – | – | -0.052 [-0.161, +0.062] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.515 [0.422, 0.593] | – | 0.482 [0.390, 0.573] | – | – | +0.033 [-0.096, +0.135] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.505 [0.422, 0.587] | – | 0.475 [0.378, 0.571] | – | – | +0.030 [-0.095, +0.147] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.513 [0.394, 0.641] | – | 0.496 [0.401, 0.582] | – | – | +0.017 [-0.127, +0.171] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.468 [0.376, 0.567] | – | 0.546 [0.437, 0.634] | – | – | -0.078 [-0.210, +0.071] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.464 [0.371, 0.559] | – | 0.489 [0.375, 0.585] | – | – | -0.026 [-0.138, +0.099] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.513 [0.422, 0.595] | – | 0.511 [0.391, 0.607] | – | – | +0.003 [-0.117, +0.126] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.541 [0.501, 0.582] | – | 0.450 [0.377, 0.519] | – | – | +0.091 [+0.011, +0.168] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.506 [0.450, 0.558] | – | 0.461 [0.382, 0.542] | – | – | +0.046 [-0.043, +0.136] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.484 [0.424, 0.538] | – | 0.489 [0.418, 0.574] | – | – | -0.005 [-0.090, +0.067] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.463 [0.429, 0.495] | – | 0.500 [0.422, 0.591] | – | – | -0.037 [-0.131, +0.042] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.514 [0.473, 0.562] | – | 0.493 [0.412, 0.584] | – | – | +0.021 [-0.069, +0.115] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.521 [0.464, 0.578] | – | 0.493 [0.416, 0.581] | – | – | +0.028 [-0.066, +0.117] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.476 [0.423, 0.535] | – | 0.486 [0.412, 0.570] | – | – | -0.009 [-0.098, +0.089] | – | – |
| `v.end` | 518 / 362 | 280 | 0.541 [0.472, 0.604] | – | 0.482 [0.408, 0.571] | – | – | +0.059 [-0.035, +0.147] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.513 [0.422, 0.595]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.609, `m2.end` 0.502, `m3.end` 0.488, `m4.end` 0.513. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.t0`, `p1.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.494 [0.461, 0.530]; activation minus words -0.021 [-0.064, +0.027]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.506 [0.487, 0.524]; activation minus words +0.024 [-0.045, +0.086]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

