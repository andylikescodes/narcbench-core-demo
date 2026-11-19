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
| `m1.pre` | 257 / 183 | 139 | 0.517 [0.444, 0.580] | – | 0.500 [0.500, 0.500] | – | – | +0.017 [-0.056, +0.080] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.485 [0.398, 0.560] | – | 0.518 [0.485, 0.546] | – | – | -0.033 [-0.126, +0.050] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.472 [0.383, 0.568] | – | 0.522 [0.389, 0.646] | – | – | -0.050 [-0.138, +0.051] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.496 [0.410, 0.575] | – | 0.540 [0.391, 0.673] | – | – | -0.043 [-0.186, +0.101] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.485 [0.404, 0.561] | – | 0.554 [0.437, 0.667] | – | – | -0.069 [-0.214, +0.077] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.485 [0.409, 0.558] | – | 0.561 [0.437, 0.686] | – | – | -0.076 [-0.205, +0.032] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.509 [0.412, 0.595] | – | 0.376 [0.262, 0.495] | – | – | +0.133 [-0.026, +0.282] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.519 [0.462, 0.566] | – | 0.390 [0.277, 0.506] | – | – | +0.129 [+0.009, +0.243] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.464 [0.390, 0.536] | – | 0.404 [0.302, 0.513] | – | – | +0.060 [-0.105, +0.204] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.496 [0.410, 0.588] | – | 0.418 [0.308, 0.533] | – | – | +0.078 [-0.090, +0.234] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.538 [0.430, 0.636] | – | 0.369 [0.262, 0.470] | – | – | +0.169 [+0.021, +0.292] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.538 [0.445, 0.628] | – | 0.454 [0.330, 0.576] | – | – | +0.084 [-0.082, +0.233] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.512 [0.439, 0.586] | – | 0.482 [0.377, 0.602] | – | – | +0.030 [-0.091, +0.132] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.522 [0.439, 0.615] | – | 0.489 [0.385, 0.605] | – | – | +0.033 [-0.068, +0.133] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.430 [0.355, 0.519] | – | 0.468 [0.364, 0.589] | – | – | -0.037 [-0.170, +0.091] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.463 [0.391, 0.523] | – | 0.489 [0.354, 0.652] | – | – | -0.026 [-0.228, +0.141] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.524 [0.446, 0.615] | – | 0.504 [0.379, 0.644] | – | – | +0.020 [-0.127, +0.159] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.440 [0.353, 0.527] | – | 0.532 [0.404, 0.681] | – | – | -0.092 [-0.241, +0.047] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.461 [0.388, 0.531] | – | 0.426 [0.307, 0.541] | – | – | +0.035 [-0.061, +0.141] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.549 [0.477, 0.619] | – | 0.426 [0.307, 0.541] | – | – | +0.123 [+0.012, +0.252] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.549 [0.462, 0.642] | – | 0.475 [0.351, 0.587] | – | – | +0.074 [-0.071, +0.227] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.498 [0.403, 0.581] | – | 0.482 [0.364, 0.595] | – | – | +0.016 [-0.157, +0.182] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.475 [0.405, 0.549] | – | 0.454 [0.350, 0.548] | – | – | +0.021 [-0.113, +0.165] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.523 [0.457, 0.592] | – | 0.482 [0.384, 0.587] | – | – | +0.041 [-0.088, +0.163] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.495 [0.449, 0.536] | – | 0.518 [0.427, 0.598] | – | – | -0.023 [-0.119, +0.064] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.513 [0.450, 0.577] | – | 0.500 [0.426, 0.582] | – | – | +0.013 [-0.067, +0.091] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.505 [0.451, 0.553] | – | 0.489 [0.413, 0.573] | – | – | +0.016 [-0.073, +0.087] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.503 [0.454, 0.545] | – | 0.475 [0.403, 0.544] | – | – | +0.028 [-0.056, +0.107] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.490 [0.440, 0.530] | – | 0.464 [0.382, 0.539] | – | – | +0.026 [-0.076, +0.113] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.566 [0.503, 0.624] | – | 0.464 [0.388, 0.533] | – | – | +0.101 [+0.019, +0.191] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.537 [0.492, 0.575] | – | 0.454 [0.381, 0.519] | – | – | +0.084 [+0.012, +0.158] | – | – |
| `v.end` | 518 / 362 | 280 | 0.442 [0.398, 0.481] | – | 0.454 [0.380, 0.522] | – | – | -0.011 [-0.065, +0.044] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.523 [0.457, 0.592]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.485, `m2.end` 0.538, `m3.end` 0.440, `m4.end` 0.523. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.t0`, `m2.q3`, `m4.t0`, `p3.end`, `v.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.498 [0.476, 0.516]; activation minus words +0.027 [-0.044, +0.096]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.506 [0.483, 0.525]; activation minus words +0.029 [-0.026, +0.084]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

