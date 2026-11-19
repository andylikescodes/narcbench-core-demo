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
| `m1.pre` | 257 / 183 | 139 | 0.465 [0.395, 0.539] | – | 0.500 [0.500, 0.500] | – | – | -0.035 [-0.105, +0.039] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.517 [0.446, 0.594] | – | 0.518 [0.481, 0.562] | – | – | -0.001 [-0.091, +0.087] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.432 [0.340, 0.524] | – | 0.540 [0.444, 0.639] | – | – | -0.108 [-0.204, -0.017] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.529 [0.448, 0.607] | – | 0.576 [0.462, 0.678] | – | – | -0.046 [-0.159, +0.070] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.437 [0.356, 0.518] | – | 0.576 [0.468, 0.678] | – | – | -0.138 [-0.229, -0.041] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.553 [0.466, 0.628] | – | 0.554 [0.430, 0.665] | – | – | -0.001 [-0.111, +0.122] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.546 [0.468, 0.609] | – | 0.433 [0.333, 0.532] | – | – | +0.113 [-0.003, +0.215] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.532 [0.464, 0.591] | – | 0.433 [0.333, 0.532] | – | – | +0.099 [+0.007, +0.194] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.536 [0.452, 0.617] | – | 0.504 [0.394, 0.606] | – | – | +0.033 [-0.110, +0.156] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.549 [0.473, 0.616] | – | 0.475 [0.366, 0.590] | – | – | +0.074 [-0.043, +0.185] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.567 [0.473, 0.662] | – | 0.489 [0.394, 0.596] | – | – | +0.078 [-0.049, +0.194] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.539 [0.458, 0.620] | – | 0.475 [0.360, 0.584] | – | – | +0.064 [-0.066, +0.213] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.455 [0.358, 0.556] | – | 0.568 [0.449, 0.681] | – | – | -0.114 [-0.273, +0.058] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.471 [0.373, 0.566] | – | 0.568 [0.449, 0.681] | – | – | -0.098 [-0.259, +0.069] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.653 [0.569, 0.725] | – | 0.576 [0.482, 0.665] | – | – | +0.078 [-0.041, +0.184] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.481 [0.380, 0.574] | – | 0.619 [0.510, 0.717] | – | – | -0.138 [-0.275, -0.007] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.512 [0.443, 0.598] | – | 0.633 [0.514, 0.744] | – | – | -0.121 [-0.263, +0.036] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.547 [0.485, 0.612] | – | 0.604 [0.500, 0.703] | – | – | -0.058 [-0.169, +0.057] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.582 [0.493, 0.666] | – | 0.447 [0.333, 0.556] | – | – | +0.135 [-0.003, +0.281] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.538 [0.438, 0.643] | – | 0.440 [0.331, 0.549] | – | – | +0.098 [-0.040, +0.243] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.478 [0.408, 0.546] | – | 0.475 [0.357, 0.587] | – | – | +0.003 [-0.128, +0.135] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.512 [0.444, 0.588] | – | 0.489 [0.377, 0.590] | – | – | +0.023 [-0.094, +0.158] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.600 [0.530, 0.682] | – | 0.496 [0.378, 0.607] | – | – | +0.104 [-0.019, +0.258] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.511 [0.415, 0.612] | – | 0.496 [0.411, 0.581] | – | – | +0.014 [-0.099, +0.124] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.568 [0.520, 0.617] | – | 0.500 [0.441, 0.558] | – | – | +0.068 [-0.013, +0.155] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.517 [0.456, 0.577] | – | 0.496 [0.440, 0.555] | – | – | +0.021 [-0.046, +0.093] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.521 [0.462, 0.586] | – | 0.482 [0.398, 0.552] | – | – | +0.039 [-0.033, +0.128] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.466 [0.405, 0.532] | – | 0.464 [0.388, 0.533] | – | – | +0.002 [-0.089, +0.100] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.502 [0.454, 0.551] | – | 0.493 [0.419, 0.570] | – | – | +0.009 [-0.062, +0.082] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.511 [0.440, 0.582] | – | 0.479 [0.400, 0.559] | – | – | +0.032 [-0.061, +0.121] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.489 [0.442, 0.533] | – | 0.493 [0.405, 0.579] | – | – | -0.004 [-0.106, +0.093] | – | – |
| `v.end` | 518 / 362 | 280 | 0.453 [0.418, 0.488] | – | 0.486 [0.405, 0.565] | – | – | -0.033 [-0.123, +0.062] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.511 [0.415, 0.612]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.553, `m2.end` 0.539, `m3.end` 0.547, `m4.end` 0.511. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.t0`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.523 [0.497, 0.550]; activation minus words +0.002 [-0.046, +0.061]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.503 [0.479, 0.528]; activation minus words +0.017 [-0.037, +0.074]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

