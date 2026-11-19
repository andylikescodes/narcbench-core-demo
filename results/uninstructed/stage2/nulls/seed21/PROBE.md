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
| `m1.pre` | 257 / 183 | 139 | 0.519 [0.448, 0.594] | – | 0.500 [0.500, 0.500] | – | – | +0.019 [-0.052, +0.094] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.462 [0.396, 0.526] | – | 0.518 [0.468, 0.565] | – | – | -0.056 [-0.139, +0.030] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.414 [0.349, 0.487] | – | 0.475 [0.382, 0.583] | – | – | -0.060 [-0.191, +0.044] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.525 [0.419, 0.622] | – | 0.475 [0.360, 0.606] | – | – | +0.050 [-0.140, +0.231] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.540 [0.452, 0.617] | – | 0.446 [0.368, 0.547] | – | – | +0.094 [-0.059, +0.208] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.492 [0.381, 0.598] | – | 0.439 [0.321, 0.565] | – | – | +0.053 [-0.089, +0.197] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.370 [0.289, 0.455] | – | 0.496 [0.374, 0.612] | – | – | -0.126 [-0.280, +0.021] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.443 [0.337, 0.545] | – | 0.496 [0.376, 0.613] | – | – | -0.054 [-0.208, +0.084] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.465 [0.362, 0.573] | – | 0.504 [0.388, 0.621] | – | – | -0.038 [-0.140, +0.075] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.515 [0.433, 0.617] | – | 0.574 [0.458, 0.696] | – | – | -0.060 [-0.162, +0.054] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.481 [0.412, 0.570] | – | 0.589 [0.477, 0.689] | – | – | -0.108 [-0.243, +0.060] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.528 [0.456, 0.614] | – | 0.560 [0.435, 0.676] | – | – | -0.033 [-0.168, +0.107] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.498 [0.414, 0.590] | – | 0.482 [0.401, 0.570] | – | – | +0.016 [-0.083, +0.126] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.499 [0.423, 0.582] | – | 0.482 [0.401, 0.570] | – | – | +0.017 [-0.055, +0.098] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.469 [0.399, 0.545] | – | 0.460 [0.352, 0.572] | – | – | +0.009 [-0.092, +0.099] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.527 [0.429, 0.617] | – | 0.446 [0.353, 0.538] | – | – | +0.081 [-0.038, +0.204] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.522 [0.454, 0.584] | – | 0.439 [0.357, 0.521] | – | – | +0.083 [-0.032, +0.193] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.517 [0.434, 0.606] | – | 0.453 [0.359, 0.561] | – | – | +0.063 [-0.055, +0.169] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.552 [0.469, 0.643] | – | 0.504 [0.381, 0.639] | – | – | +0.048 [-0.077, +0.155] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.477 [0.390, 0.556] | – | 0.496 [0.366, 0.639] | – | – | -0.020 [-0.147, +0.095] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.465 [0.364, 0.563] | – | 0.532 [0.397, 0.665] | – | – | -0.067 [-0.213, +0.068] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.496 [0.415, 0.577] | – | 0.546 [0.424, 0.671] | – | – | -0.050 [-0.182, +0.082] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.470 [0.371, 0.550] | – | 0.539 [0.431, 0.647] | – | – | -0.070 [-0.218, +0.076] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.529 [0.455, 0.600] | – | 0.560 [0.465, 0.662] | – | – | -0.031 [-0.137, +0.069] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.534 [0.477, 0.593] | – | 0.504 [0.446, 0.565] | – | – | +0.030 [-0.049, +0.113] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.514 [0.465, 0.556] | – | 0.525 [0.451, 0.597] | – | – | -0.011 [-0.088, +0.068] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.518 [0.457, 0.571] | – | 0.482 [0.399, 0.554] | – | – | +0.036 [-0.044, +0.115] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.519 [0.452, 0.577] | – | 0.482 [0.396, 0.559] | – | – | +0.036 [-0.037, +0.116] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.487 [0.441, 0.526] | – | 0.507 [0.418, 0.583] | – | – | -0.020 [-0.112, +0.081] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.471 [0.418, 0.535] | – | 0.507 [0.421, 0.573] | – | – | -0.036 [-0.123, +0.065] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.514 [0.448, 0.579] | – | 0.475 [0.399, 0.538] | – | – | +0.039 [-0.038, +0.115] | – | – |
| `v.end` | 518 / 362 | 280 | 0.499 [0.439, 0.563] | – | 0.479 [0.404, 0.540] | – | – | +0.020 [-0.055, +0.104] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.529 [0.455, 0.600]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.492, `m2.end` 0.528, `m3.end` 0.517, `m4.end` 0.529. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.491 [0.466, 0.513]; activation minus words -0.010 [-0.068, +0.045]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.507 [0.476, 0.535]; activation minus words +0.012 [-0.040, +0.069]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

