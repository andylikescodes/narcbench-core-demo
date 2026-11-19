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
| `m1.pre` | 257 / 183 | 139 | 0.524 [0.460, 0.608] | – | 0.500 [0.500, 0.500] | – | – | +0.024 [-0.040, +0.108] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.486 [0.399, 0.569] | – | 0.453 [0.412, 0.496] | – | – | +0.033 [-0.061, +0.121] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.547 [0.476, 0.608] | – | 0.478 [0.382, 0.574] | – | – | +0.068 [-0.048, +0.182] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.499 [0.422, 0.578] | – | 0.417 [0.297, 0.531] | – | – | +0.082 [-0.068, +0.236] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.659 [0.556, 0.754] | – | 0.504 [0.403, 0.604] | – | – | +0.155 [-0.009, +0.311] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.489 [0.359, 0.610] | – | 0.504 [0.410, 0.583] | – | – | -0.014 [-0.165, +0.127] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.565 [0.478, 0.640] | – | 0.617 [0.536, 0.710] | – | – | -0.052 [-0.170, +0.037] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.597 [0.516, 0.682] | – | 0.617 [0.538, 0.711] | – | – | -0.020 [-0.140, +0.101] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.401 [0.331, 0.480] | – | 0.574 [0.482, 0.662] | – | – | -0.173 [-0.288, -0.055] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.567 [0.479, 0.659] | – | 0.582 [0.495, 0.665] | – | – | -0.014 [-0.163, +0.134] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.523 [0.469, 0.572] | – | 0.539 [0.430, 0.641] | – | – | -0.016 [-0.115, +0.079] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.550 [0.479, 0.628] | – | 0.504 [0.391, 0.615] | – | – | +0.047 [-0.093, +0.193] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.603 [0.503, 0.686] | – | 0.424 [0.313, 0.539] | – | – | +0.178 [+0.034, +0.315] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.508 [0.427, 0.582] | – | 0.417 [0.310, 0.530] | – | – | +0.091 [-0.044, +0.210] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.496 [0.396, 0.609] | – | 0.460 [0.331, 0.587] | – | – | +0.036 [-0.111, +0.193] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.550 [0.479, 0.638] | – | 0.590 [0.462, 0.697] | – | – | -0.040 [-0.184, +0.127] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.522 [0.445, 0.603] | – | 0.504 [0.394, 0.601] | – | – | +0.019 [-0.101, +0.142] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.455 [0.362, 0.558] | – | 0.518 [0.426, 0.605] | – | – | -0.063 [-0.172, +0.059] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.601 [0.540, 0.662] | – | 0.567 [0.482, 0.659] | – | – | +0.034 [-0.066, +0.136] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.529 [0.434, 0.631] | – | 0.560 [0.476, 0.655] | – | – | -0.031 [-0.161, +0.095] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.404 [0.315, 0.514] | – | 0.532 [0.433, 0.641] | – | – | -0.128 [-0.244, +0.003] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.542 [0.475, 0.608] | – | 0.518 [0.419, 0.624] | – | – | +0.024 [-0.093, +0.130] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.488 [0.411, 0.566] | – | 0.454 [0.360, 0.554] | – | – | +0.034 [-0.107, +0.167] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.462 [0.372, 0.563] | – | 0.489 [0.399, 0.590] | – | – | -0.027 [-0.111, +0.064] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.534 [0.477, 0.591] | – | 0.464 [0.420, 0.512] | – | – | +0.070 [-0.000, +0.130] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.481 [0.424, 0.538] | – | 0.457 [0.412, 0.506] | – | – | +0.024 [-0.053, +0.100] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.509 [0.449, 0.571] | – | 0.443 [0.371, 0.510] | – | – | +0.066 [-0.004, +0.136] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.497 [0.420, 0.572] | – | 0.471 [0.413, 0.532] | – | – | +0.026 [-0.047, +0.093] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.516 [0.442, 0.596] | – | 0.464 [0.397, 0.526] | – | – | +0.052 [-0.027, +0.149] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.538 [0.483, 0.597] | – | 0.464 [0.396, 0.527] | – | – | +0.074 [+0.009, +0.136] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.464 [0.390, 0.543] | – | 0.475 [0.401, 0.533] | – | – | -0.011 [-0.075, +0.061] | – | – |
| `v.end` | 518 / 362 | 280 | 0.485 [0.411, 0.560] | – | 0.471 [0.398, 0.535] | – | – | +0.014 [-0.054, +0.084] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.462 [0.372, 0.563]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.489, `m2.end` 0.550, `m3.end` 0.455, `m4.end` 0.462. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m3.pre`, `p3.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.524 [0.501, 0.550]; activation minus words +0.010 [-0.036, +0.060]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.503 [0.458, 0.550]; activation minus words +0.039 [-0.008, +0.086]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

