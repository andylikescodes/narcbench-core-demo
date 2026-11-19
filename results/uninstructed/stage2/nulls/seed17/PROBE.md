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
| `m1.pre` | 257 / 183 | 139 | 0.494 [0.436, 0.558] | – | 0.500 [0.500, 0.500] | – | – | -0.006 [-0.064, +0.058] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.446 [0.376, 0.519] | – | 0.504 [0.471, 0.536] | – | – | -0.058 [-0.124, +0.013] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.576 [0.496, 0.659] | – | 0.540 [0.435, 0.651] | – | – | +0.036 [-0.069, +0.141] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.519 [0.433, 0.613] | – | 0.518 [0.412, 0.628] | – | – | +0.001 [-0.118, +0.114] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.463 [0.369, 0.573] | – | 0.424 [0.349, 0.505] | – | – | +0.039 [-0.059, +0.138] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.439 [0.368, 0.515] | – | 0.460 [0.375, 0.544] | – | – | -0.022 [-0.124, +0.081] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.518 [0.434, 0.613] | – | 0.546 [0.444, 0.651] | – | – | -0.028 [-0.166, +0.141] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.489 [0.427, 0.557] | – | 0.553 [0.458, 0.655] | – | – | -0.064 [-0.177, +0.043] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.587 [0.508, 0.667] | – | 0.511 [0.426, 0.601] | – | – | +0.077 [-0.004, +0.153] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.391 [0.311, 0.476] | – | 0.574 [0.482, 0.671] | – | – | -0.183 [-0.308, -0.057] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.586 [0.491, 0.682] | – | 0.567 [0.472, 0.667] | – | – | +0.018 [-0.070, +0.114] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.570 [0.514, 0.633] | – | 0.532 [0.429, 0.640] | – | – | +0.038 [-0.069, +0.149] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.563 [0.481, 0.639] | – | 0.482 [0.376, 0.588] | – | – | +0.081 [-0.056, +0.220] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.537 [0.465, 0.609] | – | 0.482 [0.376, 0.588] | – | – | +0.055 [-0.062, +0.169] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.498 [0.428, 0.567] | – | 0.475 [0.382, 0.577] | – | – | +0.023 [-0.080, +0.118] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.442 [0.350, 0.544] | – | 0.504 [0.406, 0.612] | – | – | -0.062 [-0.182, +0.046] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.565 [0.471, 0.663] | – | 0.518 [0.424, 0.626] | – | – | +0.047 [-0.095, +0.180] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.485 [0.392, 0.567] | – | 0.482 [0.375, 0.596] | – | – | +0.003 [-0.135, +0.125] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.496 [0.423, 0.571] | – | 0.468 [0.378, 0.557] | – | – | +0.028 [-0.067, +0.140] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.533 [0.462, 0.611] | – | 0.468 [0.384, 0.553] | – | – | +0.065 [-0.039, +0.172] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.496 [0.414, 0.595] | – | 0.482 [0.376, 0.585] | – | – | +0.014 [-0.084, +0.147] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.491 [0.423, 0.555] | – | 0.468 [0.362, 0.582] | – | – | +0.023 [-0.103, +0.146] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.519 [0.448, 0.575] | – | 0.504 [0.400, 0.608] | – | – | +0.016 [-0.107, +0.126] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.487 [0.393, 0.584] | – | 0.482 [0.384, 0.582] | – | – | +0.004 [-0.089, +0.107] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.513 [0.460, 0.563] | – | 0.507 [0.440, 0.577] | – | – | +0.006 [-0.063, +0.084] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.572 [0.505, 0.630] | – | 0.529 [0.475, 0.569] | – | – | +0.044 [-0.043, +0.134] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.519 [0.457, 0.574] | – | 0.543 [0.483, 0.595] | – | – | -0.024 [-0.088, +0.047] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.524 [0.479, 0.568] | – | 0.543 [0.483, 0.605] | – | – | -0.019 [-0.078, +0.038] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.543 [0.500, 0.586] | – | 0.521 [0.455, 0.587] | – | – | +0.021 [-0.055, +0.094] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.576 [0.522, 0.638] | – | 0.536 [0.471, 0.600] | – | – | +0.041 [-0.037, +0.125] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.561 [0.499, 0.617] | – | 0.525 [0.469, 0.587] | – | – | +0.036 [-0.041, +0.118] | – | – |
| `v.end` | 518 / 362 | 280 | 0.534 [0.478, 0.609] | – | 0.529 [0.475, 0.584] | – | – | +0.006 [-0.073, +0.095] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.487 [0.393, 0.584]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.439, `m2.end` 0.570, `m3.end` 0.485, `m4.end` 0.487. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.508 [0.483, 0.537]; activation minus words +0.006 [-0.048, +0.058]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.543 [0.516, 0.573]; activation minus words +0.014 [-0.028, +0.063]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

