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
| `m1.pre` | 257 / 183 | 139 | 0.445 [0.383, 0.517] | – | 0.500 [0.500, 0.500] | – | – | -0.055 [-0.117, +0.017] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.525 [0.446, 0.595] | – | 0.511 [0.486, 0.537] | – | – | +0.014 [-0.069, +0.092] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.532 [0.447, 0.615] | – | 0.514 [0.448, 0.574] | – | – | +0.018 [-0.083, +0.122] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.508 [0.432, 0.586] | – | 0.424 [0.310, 0.519] | – | – | +0.083 [-0.018, +0.200] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.475 [0.368, 0.589] | – | 0.403 [0.257, 0.520] | – | – | +0.072 [-0.043, +0.210] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.401 [0.328, 0.458] | – | 0.460 [0.344, 0.570] | – | – | -0.059 [-0.152, +0.032] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.384 [0.305, 0.467] | – | 0.397 [0.314, 0.509] | – | – | -0.013 [-0.115, +0.066] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.477 [0.409, 0.545] | – | 0.383 [0.305, 0.488] | – | – | +0.094 [-0.026, +0.184] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.462 [0.386, 0.537] | – | 0.447 [0.354, 0.562] | – | – | +0.016 [-0.111, +0.117] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.559 [0.470, 0.640] | – | 0.461 [0.368, 0.571] | – | – | +0.098 [-0.045, +0.206] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.484 [0.372, 0.586] | – | 0.496 [0.409, 0.603] | – | – | -0.013 [-0.146, +0.108] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.664 [0.561, 0.759] | – | 0.496 [0.400, 0.617] | – | – | +0.167 [+0.034, +0.280] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.491 [0.416, 0.569] | – | 0.424 [0.333, 0.513] | – | – | +0.066 [-0.045, +0.196] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.396 [0.319, 0.480] | – | 0.417 [0.320, 0.505] | – | – | -0.022 [-0.151, +0.125] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.486 [0.391, 0.575] | – | 0.432 [0.347, 0.512] | – | – | +0.055 [-0.022, +0.147] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.419 [0.308, 0.517] | – | 0.424 [0.325, 0.515] | – | – | -0.006 [-0.157, +0.131] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.430 [0.349, 0.509] | – | 0.417 [0.331, 0.500] | – | – | +0.013 [-0.112, +0.128] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.545 [0.429, 0.667] | – | 0.446 [0.368, 0.518] | – | – | +0.099 [-0.008, +0.222] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.409 [0.316, 0.500] | – | 0.489 [0.425, 0.558] | – | – | -0.081 [-0.191, +0.016] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.440 [0.349, 0.508] | – | 0.489 [0.425, 0.558] | – | – | -0.050 [-0.161, +0.045] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.545 [0.480, 0.605] | – | 0.447 [0.369, 0.548] | – | – | +0.098 [-0.013, +0.185] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.577 [0.492, 0.652] | – | 0.461 [0.387, 0.550] | – | – | +0.116 [-0.010, +0.234] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.542 [0.450, 0.624] | – | 0.454 [0.362, 0.545] | – | – | +0.088 [-0.053, +0.213] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.590 [0.491, 0.680] | – | 0.489 [0.420, 0.566] | – | – | +0.101 [-0.002, +0.192] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.476 [0.423, 0.539] | – | 0.518 [0.466, 0.566] | – | – | -0.041 [-0.110, +0.028] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.474 [0.431, 0.518] | – | 0.496 [0.450, 0.544] | – | – | -0.023 [-0.070, +0.028] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.471 [0.413, 0.523] | – | 0.539 [0.490, 0.593] | – | – | -0.069 [-0.145, +0.002] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.486 [0.428, 0.554] | – | 0.536 [0.479, 0.597] | – | – | -0.049 [-0.123, +0.037] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.422 [0.371, 0.471] | – | 0.518 [0.457, 0.579] | – | – | -0.096 [-0.176, -0.022] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.476 [0.422, 0.522] | – | 0.511 [0.449, 0.571] | – | – | -0.035 [-0.100, +0.035] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.525 [0.485, 0.572] | – | 0.514 [0.462, 0.576] | – | – | +0.011 [-0.049, +0.078] | – | – |
| `v.end` | 518 / 362 | 280 | 0.467 [0.391, 0.540] | – | 0.518 [0.469, 0.574] | – | – | -0.051 [-0.128, +0.024] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.590 [0.491, 0.680]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.401, `m2.end` 0.664, `m3.end` 0.545, `m4.end` 0.590. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.491 [0.468, 0.511]; activation minus words +0.038 [-0.017, +0.079]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.475 [0.443, 0.507]; activation minus words -0.044 [-0.088, +0.003]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

