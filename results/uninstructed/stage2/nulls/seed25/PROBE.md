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
| `m1.pre` | 257 / 183 | 139 | 0.474 [0.382, 0.562] | – | 0.500 [0.500, 0.500] | – | – | -0.026 [-0.118, +0.062] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.407 [0.343, 0.488] | – | 0.475 [0.435, 0.526] | – | – | -0.068 [-0.132, -0.002] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.489 [0.405, 0.567] | – | 0.450 [0.341, 0.570] | – | – | +0.040 [-0.067, +0.130] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.532 [0.405, 0.668] | – | 0.518 [0.396, 0.626] | – | – | +0.014 [-0.135, +0.185] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.485 [0.400, 0.592] | – | 0.439 [0.349, 0.533] | – | – | +0.046 [-0.071, +0.167] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.502 [0.422, 0.609] | – | 0.439 [0.325, 0.546] | – | – | +0.063 [-0.068, +0.215] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.479 [0.389, 0.571] | – | 0.475 [0.367, 0.594] | – | – | +0.004 [-0.146, +0.127] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.487 [0.408, 0.567] | – | 0.496 [0.393, 0.617] | – | – | -0.010 [-0.146, +0.126] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.477 [0.383, 0.559] | – | 0.496 [0.405, 0.597] | – | – | -0.020 [-0.155, +0.102] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.519 [0.439, 0.620] | – | 0.511 [0.421, 0.611] | – | – | +0.009 [-0.134, +0.143] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.504 [0.412, 0.584] | – | 0.525 [0.416, 0.630] | – | – | -0.021 [-0.163, +0.115] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.424 [0.354, 0.498] | – | 0.525 [0.450, 0.607] | – | – | -0.101 [-0.226, +0.005] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.450 [0.359, 0.542] | – | 0.396 [0.275, 0.512] | – | – | +0.055 [-0.059, +0.182] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.419 [0.328, 0.514] | – | 0.388 [0.273, 0.503] | – | – | +0.030 [-0.105, +0.168] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.514 [0.434, 0.599] | – | 0.403 [0.267, 0.524] | – | – | +0.111 [-0.045, +0.284] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.509 [0.425, 0.602] | – | 0.403 [0.271, 0.522] | – | – | +0.106 [-0.060, +0.285] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.581 [0.474, 0.673] | – | 0.396 [0.281, 0.500] | – | – | +0.186 [+0.082, +0.312] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.529 [0.458, 0.593] | – | 0.432 [0.339, 0.519] | – | – | +0.098 [+0.015, +0.185] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.538 [0.451, 0.626] | – | 0.482 [0.393, 0.579] | – | – | +0.055 [-0.075, +0.172] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.445 [0.342, 0.544] | – | 0.489 [0.391, 0.586] | – | – | -0.044 [-0.185, +0.080] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.502 [0.407, 0.590] | – | 0.504 [0.412, 0.597] | – | – | -0.001 [-0.119, +0.113] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.579 [0.498, 0.663] | – | 0.525 [0.423, 0.632] | – | – | +0.054 [-0.094, +0.196] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.489 [0.394, 0.579] | – | 0.525 [0.437, 0.619] | – | – | -0.035 [-0.147, +0.079] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.521 [0.457, 0.588] | – | 0.511 [0.403, 0.621] | – | – | +0.010 [-0.094, +0.103] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.501 [0.447, 0.548] | – | 0.514 [0.445, 0.567] | – | – | -0.014 [-0.080, +0.069] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.511 [0.454, 0.568] | – | 0.504 [0.433, 0.569] | – | – | +0.008 [-0.063, +0.087] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.486 [0.440, 0.534] | – | 0.504 [0.430, 0.583] | – | – | -0.018 [-0.117, +0.072] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.519 [0.452, 0.573] | – | 0.461 [0.382, 0.540] | – | – | +0.058 [-0.040, +0.144] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.523 [0.479, 0.571] | – | 0.471 [0.406, 0.544] | – | – | +0.051 [-0.044, +0.145] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.556 [0.505, 0.606] | – | 0.496 [0.428, 0.567] | – | – | +0.059 [-0.001, +0.123] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.529 [0.462, 0.592] | – | 0.518 [0.439, 0.593] | – | – | +0.011 [-0.079, +0.091] | – | – |
| `v.end` | 518 / 362 | 280 | 0.504 [0.443, 0.555] | – | 0.518 [0.438, 0.596] | – | – | -0.014 [-0.103, +0.075] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.521 [0.457, 0.588]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.502, `m2.end` 0.424, `m3.end` 0.529, `m4.end` 0.521. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m3.q3`, `m3.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.494 [0.472, 0.517]; activation minus words +0.023 [-0.032, +0.083]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.516 [0.489, 0.539]; activation minus words +0.018 [-0.042, +0.074]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

