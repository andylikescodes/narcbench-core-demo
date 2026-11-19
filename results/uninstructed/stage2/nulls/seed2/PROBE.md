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
| `m1.pre` | 257 / 183 | 139 | 0.515 [0.447, 0.590] | – | 0.500 [0.500, 0.500] | – | – | +0.015 [-0.053, +0.090] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.537 [0.467, 0.599] | – | 0.496 [0.457, 0.531] | – | – | +0.041 [-0.033, +0.121] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.547 [0.438, 0.660] | – | 0.608 [0.489, 0.712] | – | – | -0.061 [-0.178, +0.067] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.528 [0.424, 0.617] | – | 0.540 [0.421, 0.654] | – | – | -0.012 [-0.161, +0.132] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.450 [0.364, 0.531] | – | 0.561 [0.447, 0.652] | – | – | -0.111 [-0.244, +0.034] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.391 [0.322, 0.475] | – | 0.511 [0.426, 0.602] | – | – | -0.119 [-0.225, -0.005] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.533 [0.457, 0.592] | – | 0.567 [0.455, 0.689] | – | – | -0.034 [-0.160, +0.080] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.482 [0.391, 0.572] | – | 0.560 [0.452, 0.678] | – | – | -0.078 [-0.207, +0.065] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.440 [0.363, 0.518] | – | 0.567 [0.452, 0.688] | – | – | -0.128 [-0.224, -0.025] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.461 [0.375, 0.544] | – | 0.546 [0.407, 0.694] | – | – | -0.085 [-0.236, +0.067] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.494 [0.394, 0.596] | – | 0.589 [0.492, 0.697] | – | – | -0.095 [-0.239, +0.037] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.577 [0.504, 0.655] | – | 0.511 [0.415, 0.620] | – | – | +0.067 [-0.073, +0.189] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.485 [0.391, 0.588] | – | 0.432 [0.338, 0.526] | – | – | +0.053 [-0.053, +0.186] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.390 [0.303, 0.488] | – | 0.432 [0.338, 0.526] | – | – | -0.042 [-0.146, +0.087] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.407 [0.321, 0.500] | – | 0.439 [0.348, 0.539] | – | – | -0.032 [-0.159, +0.100] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.459 [0.377, 0.532] | – | 0.410 [0.329, 0.507] | – | – | +0.049 [-0.103, +0.168] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.450 [0.363, 0.524] | – | 0.475 [0.399, 0.565] | – | – | -0.024 [-0.125, +0.065] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.360 [0.263, 0.468] | – | 0.439 [0.354, 0.519] | – | – | -0.079 [-0.148, +0.003] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.530 [0.438, 0.606] | – | 0.539 [0.440, 0.653] | – | – | -0.009 [-0.162, +0.116] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.553 [0.492, 0.601] | – | 0.539 [0.440, 0.653] | – | – | +0.014 [-0.115, +0.120] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.494 [0.393, 0.585] | – | 0.582 [0.479, 0.689] | – | – | -0.088 [-0.260, +0.057] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.437 [0.316, 0.559] | – | 0.518 [0.426, 0.622] | – | – | -0.081 [-0.251, +0.074] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.548 [0.482, 0.619] | – | 0.496 [0.395, 0.610] | – | – | +0.051 [-0.063, +0.160] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.489 [0.401, 0.564] | – | 0.518 [0.433, 0.607] | – | – | -0.028 [-0.140, +0.071] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.384 [0.338, 0.436] | – | 0.479 [0.415, 0.541] | – | – | -0.095 [-0.169, -0.019] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.469 [0.407, 0.538] | – | 0.464 [0.399, 0.524] | – | – | +0.005 [-0.068, +0.085] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.469 [0.414, 0.516] | – | 0.507 [0.436, 0.575] | – | – | -0.038 [-0.119, +0.037] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.444 [0.387, 0.500] | – | 0.493 [0.437, 0.541] | – | – | -0.049 [-0.119, +0.027] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.481 [0.430, 0.534] | – | 0.504 [0.447, 0.554] | – | – | -0.023 [-0.092, +0.052] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.498 [0.437, 0.565] | – | 0.514 [0.453, 0.567] | – | – | -0.016 [-0.115, +0.091] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.512 [0.457, 0.568] | – | 0.518 [0.466, 0.566] | – | – | -0.006 [-0.076, +0.066] | – | – |
| `v.end` | 518 / 362 | 280 | 0.528 [0.483, 0.584] | – | 0.496 [0.437, 0.550] | – | – | +0.031 [-0.023, +0.094] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.489 [0.401, 0.564]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.391, `m2.end` 0.577, `m3.end` 0.360, `m4.end` 0.489. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.482 [0.460, 0.500]; activation minus words -0.034 [-0.088, +0.011]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.473 [0.449, 0.499]; activation minus words -0.024 [-0.072, +0.029]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

