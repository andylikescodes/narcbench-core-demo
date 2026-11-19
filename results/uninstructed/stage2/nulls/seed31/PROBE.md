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
| `m1.pre` | 257 / 183 | 139 | 0.537 [0.435, 0.625] | – | 0.500 [0.500, 0.500] | – | – | +0.037 [-0.065, +0.125] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.535 [0.470, 0.605] | – | 0.496 [0.455, 0.537] | – | – | +0.038 [-0.023, +0.105] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.429 [0.365, 0.501] | – | 0.489 [0.387, 0.582] | – | – | -0.060 [-0.168, +0.073] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.600 [0.530, 0.679] | – | 0.489 [0.365, 0.597] | – | – | +0.111 [-0.042, +0.291] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.577 [0.491, 0.668] | – | 0.518 [0.431, 0.606] | – | – | +0.059 [-0.088, +0.209] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.469 [0.389, 0.554] | – | 0.468 [0.393, 0.554] | – | – | +0.001 [-0.098, +0.095] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.569 [0.493, 0.654] | – | 0.539 [0.415, 0.651] | – | – | +0.030 [-0.104, +0.184] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.582 [0.515, 0.650] | – | 0.532 [0.402, 0.648] | – | – | +0.050 [-0.076, +0.187] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.586 [0.526, 0.656] | – | 0.525 [0.412, 0.623] | – | – | +0.061 [-0.051, +0.187] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.545 [0.460, 0.633] | – | 0.582 [0.483, 0.669] | – | – | -0.037 [-0.151, +0.085] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.563 [0.466, 0.644] | – | 0.560 [0.443, 0.652] | – | – | +0.003 [-0.084, +0.090] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.498 [0.416, 0.590] | – | 0.596 [0.489, 0.685] | – | – | -0.098 [-0.194, +0.032] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.576 [0.507, 0.635] | – | 0.489 [0.390, 0.575] | – | – | +0.086 [-0.021, +0.195] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.557 [0.468, 0.649] | – | 0.489 [0.390, 0.578] | – | – | +0.068 [-0.036, +0.175] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.466 [0.381, 0.556] | – | 0.475 [0.385, 0.552] | – | – | -0.009 [-0.109, +0.087] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.453 [0.372, 0.543] | – | 0.482 [0.402, 0.560] | – | – | -0.029 [-0.162, +0.108] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.522 [0.464, 0.581] | – | 0.453 [0.387, 0.518] | – | – | +0.069 [-0.020, +0.163] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.433 [0.352, 0.521] | – | 0.518 [0.443, 0.588] | – | – | -0.085 [-0.186, +0.033] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.511 [0.424, 0.598] | – | 0.596 [0.487, 0.689] | – | – | -0.085 [-0.214, +0.066] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.482 [0.380, 0.576] | – | 0.596 [0.487, 0.689] | – | – | -0.113 [-0.256, +0.009] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.444 [0.330, 0.554] | – | 0.567 [0.460, 0.664] | – | – | -0.123 [-0.227, -0.016] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.567 [0.499, 0.627] | – | 0.574 [0.465, 0.678] | – | – | -0.007 [-0.121, +0.107] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.464 [0.379, 0.556] | – | 0.574 [0.474, 0.673] | – | – | -0.111 [-0.231, +0.018] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.427 [0.354, 0.503] | – | 0.610 [0.481, 0.722] | – | – | -0.183 [-0.312, -0.050] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.479 [0.431, 0.518] | – | 0.493 [0.426, 0.555] | – | – | -0.014 [-0.072, +0.035] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.508 [0.450, 0.566] | – | 0.496 [0.426, 0.561] | – | – | +0.011 [-0.080, +0.100] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.482 [0.431, 0.526] | – | 0.500 [0.444, 0.562] | – | – | -0.018 [-0.100, +0.042] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.535 [0.488, 0.586] | – | 0.514 [0.449, 0.581] | – | – | +0.021 [-0.046, +0.096] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.501 [0.468, 0.536] | – | 0.486 [0.412, 0.553] | – | – | +0.016 [-0.069, +0.100] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.496 [0.436, 0.546] | – | 0.496 [0.405, 0.580] | – | – | -0.001 [-0.094, +0.095] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.506 [0.447, 0.562] | – | 0.543 [0.459, 0.615] | – | – | -0.037 [-0.144, +0.065] | – | – |
| `v.end` | 518 / 362 | 280 | 0.496 [0.421, 0.566] | – | 0.543 [0.469, 0.611] | – | – | -0.047 [-0.139, +0.043] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.427 [0.354, 0.503]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.469, `m2.end` 0.498, `m3.end` 0.433, `m4.end` 0.427. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.516 [0.497, 0.537]; activation minus words -0.014 [-0.062, +0.038]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.500 [0.479, 0.514]; activation minus words -0.009 [-0.066, +0.051]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** layer 21, anchor `m2.pre` (score there 0.589).

