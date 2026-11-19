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
| `m1.pre` | 257 / 183 | 139 | 0.428 [0.360, 0.491] | – | 0.500 [0.500, 0.500] | – | – | -0.072 [-0.140, -0.009] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.515 [0.448, 0.581] | – | 0.532 [0.500, 0.559] | – | – | -0.017 [-0.097, +0.067] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.469 [0.354, 0.569] | – | 0.550 [0.434, 0.655] | – | – | -0.081 [-0.230, +0.032] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.528 [0.427, 0.630] | – | 0.496 [0.389, 0.602] | – | – | +0.032 [-0.085, +0.153] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.504 [0.412, 0.602] | – | 0.475 [0.382, 0.561] | – | – | +0.029 [-0.092, +0.168] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.479 [0.360, 0.604] | – | 0.453 [0.338, 0.549] | – | – | +0.026 [-0.063, +0.136] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.482 [0.408, 0.565] | – | 0.433 [0.298, 0.565] | – | – | +0.050 [-0.083, +0.202] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.460 [0.391, 0.531] | – | 0.440 [0.306, 0.574] | – | – | +0.020 [-0.102, +0.158] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.445 [0.356, 0.536] | – | 0.532 [0.415, 0.654] | – | – | -0.087 [-0.267, +0.094] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.389 [0.298, 0.491] | – | 0.482 [0.388, 0.582] | – | – | -0.094 [-0.219, +0.033] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.474 [0.393, 0.554] | – | 0.411 [0.314, 0.495] | – | – | +0.062 [-0.080, +0.211] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.424 [0.325, 0.524] | – | 0.468 [0.389, 0.545] | – | – | -0.044 [-0.173, +0.085] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.518 [0.437, 0.596] | – | 0.525 [0.402, 0.623] | – | – | -0.007 [-0.108, +0.104] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.453 [0.393, 0.505] | – | 0.525 [0.402, 0.623] | – | – | -0.072 [-0.170, +0.049] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.488 [0.403, 0.568] | – | 0.518 [0.417, 0.609] | – | – | -0.030 [-0.183, +0.125] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.542 [0.458, 0.621] | – | 0.547 [0.448, 0.629] | – | – | -0.004 [-0.109, +0.106] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.519 [0.444, 0.598] | – | 0.532 [0.403, 0.638] | – | – | -0.013 [-0.147, +0.138] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.498 [0.411, 0.590] | – | 0.525 [0.399, 0.635] | – | – | -0.027 [-0.150, +0.107] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.448 [0.335, 0.559] | – | 0.454 [0.342, 0.552] | – | – | -0.006 [-0.141, +0.113] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.502 [0.414, 0.584] | – | 0.447 [0.339, 0.542] | – | – | +0.055 [-0.063, +0.173] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.391 [0.283, 0.487] | – | 0.440 [0.345, 0.533] | – | – | -0.048 [-0.202, +0.096] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.540 [0.434, 0.645] | – | 0.447 [0.354, 0.538] | – | – | +0.094 [-0.073, +0.246] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.509 [0.435, 0.571] | – | 0.475 [0.383, 0.559] | – | – | +0.034 [-0.079, +0.141] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.487 [0.382, 0.579] | – | 0.461 [0.366, 0.538] | – | – | +0.026 [-0.068, +0.131] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.493 [0.444, 0.547] | – | 0.511 [0.455, 0.575] | – | – | -0.018 [-0.083, +0.046] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.531 [0.470, 0.586] | – | 0.525 [0.459, 0.581] | – | – | +0.006 [-0.072, +0.087] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.489 [0.444, 0.530] | – | 0.486 [0.433, 0.540] | – | – | +0.004 [-0.071, +0.071] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.519 [0.481, 0.554] | – | 0.496 [0.432, 0.555] | – | – | +0.022 [-0.050, +0.095] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.481 [0.440, 0.527] | – | 0.518 [0.455, 0.584] | – | – | -0.036 [-0.117, +0.049] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.490 [0.429, 0.541] | – | 0.521 [0.464, 0.583] | – | – | -0.031 [-0.121, +0.055] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.461 [0.423, 0.496] | – | 0.536 [0.471, 0.592] | – | – | -0.074 [-0.146, +0.003] | – | – |
| `v.end` | 518 / 362 | 280 | 0.516 [0.453, 0.583] | – | 0.550 [0.483, 0.610] | – | – | -0.034 [-0.116, +0.058] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.487 [0.382, 0.579]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.479, `m2.end` 0.424, `m3.end` 0.498, `m4.end` 0.487. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.479 [0.454, 0.501]; activation minus words -0.007 [-0.069, +0.057]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.497 [0.479, 0.515]; activation minus words -0.020 [-0.062, +0.025]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

