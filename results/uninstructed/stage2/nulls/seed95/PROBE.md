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
| `m1.pre` | 257 / 183 | 139 | 0.485 [0.407, 0.566] | – | 0.500 [0.500, 0.500] | – | – | -0.015 [-0.093, +0.066] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.468 [0.387, 0.544] | – | 0.511 [0.470, 0.553] | – | – | -0.042 [-0.155, +0.058] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.593 [0.481, 0.698] | – | 0.500 [0.387, 0.611] | – | – | +0.093 [-0.070, +0.249] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.505 [0.394, 0.620] | – | 0.525 [0.434, 0.624] | – | – | -0.020 [-0.170, +0.116] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.565 [0.511, 0.625] | – | 0.424 [0.342, 0.515] | – | – | +0.141 [+0.061, +0.211] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.465 [0.378, 0.547] | – | 0.410 [0.313, 0.526] | – | – | +0.055 [-0.098, +0.186] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.529 [0.440, 0.620] | – | 0.489 [0.413, 0.558] | – | – | +0.040 [-0.062, +0.147] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.498 [0.405, 0.593] | – | 0.482 [0.410, 0.553] | – | – | +0.016 [-0.103, +0.131] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.488 [0.397, 0.565] | – | 0.468 [0.388, 0.538] | – | – | +0.020 [-0.088, +0.129] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.496 [0.385, 0.607] | – | 0.447 [0.360, 0.538] | – | – | +0.050 [-0.094, +0.190] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.479 [0.410, 0.543] | – | 0.496 [0.403, 0.584] | – | – | -0.017 [-0.101, +0.079] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.472 [0.380, 0.567] | – | 0.532 [0.451, 0.613] | – | – | -0.060 [-0.180, +0.055] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.442 [0.344, 0.531] | – | 0.482 [0.402, 0.574] | – | – | -0.040 [-0.177, +0.078] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.475 [0.406, 0.540] | – | 0.475 [0.396, 0.568] | – | – | +0.000 [-0.125, +0.108] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.564 [0.479, 0.642] | – | 0.453 [0.381, 0.539] | – | – | +0.111 [-0.017, +0.222] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.499 [0.414, 0.597] | – | 0.475 [0.383, 0.587] | – | – | +0.024 [-0.066, +0.115] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.469 [0.347, 0.602] | – | 0.475 [0.389, 0.569] | – | – | -0.006 [-0.132, +0.110] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.502 [0.424, 0.581] | – | 0.453 [0.365, 0.556] | – | – | +0.049 [-0.063, +0.141] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.560 [0.462, 0.661] | – | 0.511 [0.440, 0.591] | – | – | +0.050 [-0.034, +0.136] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.502 [0.429, 0.579] | – | 0.496 [0.417, 0.585] | – | – | +0.006 [-0.114, +0.126] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.610 [0.509, 0.712] | – | 0.511 [0.456, 0.579] | – | – | +0.099 [-0.006, +0.202] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.467 [0.394, 0.545] | – | 0.496 [0.427, 0.578] | – | – | -0.030 [-0.140, +0.073] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.569 [0.461, 0.674] | – | 0.518 [0.444, 0.592] | – | – | +0.051 [-0.102, +0.197] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.610 [0.526, 0.689] | – | 0.546 [0.482, 0.617] | – | – | +0.064 [-0.045, +0.176] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.568 [0.524, 0.608] | – | 0.521 [0.461, 0.575] | – | – | +0.046 [-0.024, +0.110] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.552 [0.500, 0.601] | – | 0.507 [0.439, 0.569] | – | – | +0.045 [-0.014, +0.115] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.551 [0.475, 0.617] | – | 0.539 [0.471, 0.607] | – | – | +0.011 [-0.076, +0.083] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.543 [0.487, 0.599] | – | 0.550 [0.486, 0.620] | – | – | -0.007 [-0.081, +0.065] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.504 [0.440, 0.561] | – | 0.539 [0.475, 0.599] | – | – | -0.036 [-0.114, +0.045] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.504 [0.447, 0.558] | – | 0.529 [0.475, 0.577] | – | – | -0.025 [-0.073, +0.026] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.499 [0.447, 0.556] | – | 0.507 [0.458, 0.560] | – | – | -0.008 [-0.068, +0.050] | – | – |
| `v.end` | 518 / 362 | 280 | 0.501 [0.445, 0.552] | – | 0.521 [0.471, 0.574] | – | – | -0.020 [-0.097, +0.050] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.610 [0.526, 0.689]. Yes: the interval is above 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.465, `m2.end` 0.472, `m3.end` 0.502, `m4.end` 0.610. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.q3`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.513 [0.486, 0.539]; activation minus words +0.027 [-0.021, +0.076]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.528 [0.497, 0.555]; activation minus words +0.001 [-0.037, +0.036]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

