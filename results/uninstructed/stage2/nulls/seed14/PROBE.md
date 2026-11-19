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
| `m1.pre` | 257 / 183 | 139 | 0.496 [0.429, 0.563] | – | 0.500 [0.500, 0.500] | – | – | -0.004 [-0.071, +0.063] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.515 [0.443, 0.583] | – | 0.453 [0.412, 0.496] | – | – | +0.062 [-0.007, +0.126] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.422 [0.327, 0.519] | – | 0.464 [0.330, 0.591] | – | – | -0.042 [-0.200, +0.096] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.504 [0.431, 0.584] | – | 0.482 [0.397, 0.579] | – | – | +0.022 [-0.096, +0.127] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.463 [0.352, 0.585] | – | 0.525 [0.444, 0.614] | – | – | -0.062 [-0.221, +0.096] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.429 [0.360, 0.490] | – | 0.446 [0.329, 0.567] | – | – | -0.017 [-0.154, +0.118] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.441 [0.362, 0.526] | – | 0.433 [0.301, 0.566] | – | – | +0.009 [-0.152, +0.176] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.440 [0.350, 0.530] | – | 0.433 [0.301, 0.566] | – | – | +0.007 [-0.173, +0.182] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.523 [0.455, 0.604] | – | 0.454 [0.347, 0.548] | – | – | +0.070 [-0.054, +0.222] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.428 [0.339, 0.528] | – | 0.411 [0.315, 0.500] | – | – | +0.017 [-0.095, +0.158] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.435 [0.338, 0.530] | – | 0.369 [0.283, 0.452] | – | – | +0.067 [-0.019, +0.142] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.450 [0.347, 0.562] | – | 0.390 [0.299, 0.485] | – | – | +0.060 [-0.055, +0.184] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.537 [0.425, 0.629] | – | 0.576 [0.492, 0.664] | – | – | -0.039 [-0.206, +0.106] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.488 [0.409, 0.568] | – | 0.568 [0.489, 0.655] | – | – | -0.081 [-0.206, +0.038] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.540 [0.418, 0.635] | – | 0.597 [0.494, 0.697] | – | – | -0.058 [-0.218, +0.081] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.479 [0.376, 0.589] | – | 0.576 [0.467, 0.672] | – | – | -0.096 [-0.194, +0.007] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.620 [0.531, 0.715] | – | 0.590 [0.487, 0.688] | – | – | +0.030 [-0.127, +0.181] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.560 [0.474, 0.648] | – | 0.511 [0.405, 0.621] | – | – | +0.049 [-0.069, +0.159] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.505 [0.393, 0.632] | – | 0.482 [0.400, 0.583] | – | – | +0.023 [-0.084, +0.118] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.508 [0.424, 0.603] | – | 0.468 [0.390, 0.568] | – | – | +0.040 [-0.084, +0.162] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.479 [0.377, 0.576] | – | 0.468 [0.369, 0.591] | – | – | +0.011 [-0.111, +0.122] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.468 [0.400, 0.542] | – | 0.447 [0.342, 0.571] | – | – | +0.021 [-0.121, +0.161] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.529 [0.428, 0.637] | – | 0.440 [0.353, 0.554] | – | – | +0.089 [-0.065, +0.227] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.569 [0.488, 0.647] | – | 0.418 [0.335, 0.518] | – | – | +0.150 [+0.025, +0.263] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.510 [0.448, 0.572] | – | 0.511 [0.453, 0.587] | – | – | -0.001 [-0.109, +0.083] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.452 [0.371, 0.524] | – | 0.468 [0.397, 0.549] | – | – | -0.016 [-0.122, +0.071] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.490 [0.441, 0.550] | – | 0.507 [0.436, 0.578] | – | – | -0.017 [-0.093, +0.069] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.490 [0.433, 0.541] | – | 0.493 [0.427, 0.558] | – | – | -0.003 [-0.083, +0.079] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.473 [0.424, 0.519] | – | 0.489 [0.422, 0.555] | – | – | -0.016 [-0.083, +0.057] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.504 [0.453, 0.558] | – | 0.486 [0.416, 0.558] | – | – | +0.019 [-0.049, +0.087] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.464 [0.414, 0.515] | – | 0.461 [0.394, 0.531] | – | – | +0.004 [-0.079, +0.090] | – | – |
| `v.end` | 518 / 362 | 280 | 0.454 [0.394, 0.523] | – | 0.479 [0.412, 0.550] | – | – | -0.025 [-0.102, +0.071] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.569 [0.488, 0.647]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.429, `m2.end` 0.450, `m3.end` 0.560, `m4.end` 0.569. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.493 [0.467, 0.518]; activation minus words +0.014 [-0.042, +0.059]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.480 [0.457, 0.502]; activation minus words -0.007 [-0.058, +0.046]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

