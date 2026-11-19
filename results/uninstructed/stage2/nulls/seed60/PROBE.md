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
| `m1.pre` | 257 / 183 | 139 | 0.527 [0.462, 0.593] | – | 0.500 [0.500, 0.500] | – | – | +0.027 [-0.038, +0.093] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.411 [0.339, 0.476] | – | 0.532 [0.476, 0.592] | – | – | -0.122 [-0.207, -0.038] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.489 [0.416, 0.583] | – | 0.460 [0.373, 0.556] | – | – | +0.029 [-0.075, +0.145] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.518 [0.435, 0.597] | – | 0.482 [0.382, 0.579] | – | – | +0.036 [-0.094, +0.154] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.527 [0.440, 0.600] | – | 0.417 [0.340, 0.500] | – | – | +0.109 [-0.014, +0.232] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.508 [0.424, 0.606] | – | 0.453 [0.371, 0.534] | – | – | +0.055 [-0.050, +0.168] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.475 [0.402, 0.551] | – | 0.418 [0.355, 0.500] | – | – | +0.057 [-0.036, +0.136] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.417 [0.342, 0.493] | – | 0.433 [0.366, 0.518] | – | – | -0.016 [-0.120, +0.074] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.464 [0.376, 0.547] | – | 0.426 [0.351, 0.518] | – | – | +0.038 [-0.070, +0.131] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.441 [0.338, 0.543] | – | 0.447 [0.362, 0.543] | – | – | -0.006 [-0.125, +0.111] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.499 [0.419, 0.581] | – | 0.475 [0.376, 0.584] | – | – | +0.024 [-0.079, +0.128] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.540 [0.461, 0.607] | – | 0.482 [0.405, 0.575] | – | – | +0.058 [-0.057, +0.148] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.554 [0.478, 0.633] | – | 0.482 [0.395, 0.556] | – | – | +0.072 [-0.036, +0.189] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.468 [0.398, 0.534] | – | 0.489 [0.399, 0.563] | – | – | -0.022 [-0.104, +0.075] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.496 [0.430, 0.564] | – | 0.532 [0.434, 0.626] | – | – | -0.036 [-0.148, +0.088] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.587 [0.465, 0.707] | – | 0.525 [0.429, 0.600] | – | – | +0.062 [-0.068, +0.188] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.525 [0.453, 0.594] | – | 0.468 [0.375, 0.552] | – | – | +0.058 [-0.054, +0.170] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.417 [0.334, 0.501] | – | 0.432 [0.336, 0.519] | – | – | -0.014 [-0.139, +0.119] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.438 [0.339, 0.518] | – | 0.482 [0.379, 0.581] | – | – | -0.044 [-0.183, +0.085] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.420 [0.352, 0.488] | – | 0.468 [0.375, 0.558] | – | – | -0.048 [-0.154, +0.066] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.454 [0.366, 0.543] | – | 0.418 [0.318, 0.533] | – | – | +0.035 [-0.094, +0.167] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.594 [0.516, 0.674] | – | 0.418 [0.314, 0.545] | – | – | +0.176 [+0.040, +0.304] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.569 [0.465, 0.641] | – | 0.433 [0.312, 0.566] | – | – | +0.136 [-0.057, +0.281] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.523 [0.427, 0.600] | – | 0.404 [0.303, 0.523] | – | – | +0.119 [-0.015, +0.228] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.569 [0.512, 0.621] | – | 0.425 [0.350, 0.488] | – | – | +0.144 [+0.064, +0.233] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.489 [0.429, 0.540] | – | 0.432 [0.368, 0.496] | – | – | +0.056 [-0.041, +0.147] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.464 [0.418, 0.508] | – | 0.464 [0.390, 0.538] | – | – | -0.001 [-0.068, +0.065] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.433 [0.383, 0.486] | – | 0.468 [0.395, 0.537] | – | – | -0.035 [-0.119, +0.057] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.472 [0.439, 0.507] | – | 0.475 [0.408, 0.556] | – | – | -0.003 [-0.076, +0.063] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.525 [0.465, 0.579] | – | 0.457 [0.389, 0.536] | – | – | +0.068 [-0.035, +0.159] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.439 [0.365, 0.503] | – | 0.446 [0.398, 0.503] | – | – | -0.007 [-0.103, +0.071] | – | – |
| `v.end` | 518 / 362 | 280 | 0.469 [0.412, 0.522] | – | 0.450 [0.389, 0.525] | – | – | +0.019 [-0.071, +0.099] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.523 [0.427, 0.600]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.508, `m2.end` 0.540, `m3.end` 0.417, `m4.end` 0.523. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.q2`, `p1.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.494 [0.464, 0.522]; activation minus words +0.033 [-0.021, +0.082]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.482 [0.459, 0.503]; activation minus words +0.030 [-0.037, +0.089]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

