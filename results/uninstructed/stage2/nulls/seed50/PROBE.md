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
| `m1.pre` | 257 / 183 | 139 | 0.457 [0.383, 0.531] | – | 0.500 [0.500, 0.500] | – | – | -0.043 [-0.117, +0.031] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.566 [0.473, 0.652] | – | 0.540 [0.481, 0.598] | – | – | +0.027 [-0.054, +0.107] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.574 [0.496, 0.653] | – | 0.673 [0.586, 0.757] | – | – | -0.099 [-0.223, +0.021] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.459 [0.372, 0.547] | – | 0.504 [0.412, 0.600] | – | – | -0.045 [-0.175, +0.071] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.613 [0.525, 0.690] | – | 0.504 [0.375, 0.641] | – | – | +0.109 [-0.065, +0.247] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.488 [0.412, 0.572] | – | 0.532 [0.415, 0.652] | – | – | -0.045 [-0.158, +0.077] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.482 [0.395, 0.573] | – | 0.553 [0.444, 0.661] | – | – | -0.071 [-0.217, +0.064] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.573 [0.512, 0.644] | – | 0.553 [0.444, 0.661] | – | – | +0.020 [-0.094, +0.146] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.579 [0.500, 0.668] | – | 0.638 [0.538, 0.731] | – | – | -0.060 [-0.165, +0.055] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.501 [0.442, 0.565] | – | 0.624 [0.540, 0.708] | – | – | -0.123 [-0.227, -0.005] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.563 [0.494, 0.631] | – | 0.560 [0.486, 0.636] | – | – | +0.003 [-0.090, +0.099] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.579 [0.503, 0.662] | – | 0.582 [0.513, 0.655] | – | – | -0.003 [-0.098, +0.098] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.388 [0.320, 0.472] | – | 0.554 [0.461, 0.639] | – | – | -0.165 [-0.265, -0.041] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.504 [0.425, 0.575] | – | 0.561 [0.464, 0.650] | – | – | -0.058 [-0.175, +0.065] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.489 [0.400, 0.573] | – | 0.496 [0.404, 0.595] | – | – | -0.007 [-0.163, +0.130] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.485 [0.391, 0.580] | – | 0.532 [0.430, 0.630] | – | – | -0.047 [-0.170, +0.061] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.544 [0.468, 0.632] | – | 0.561 [0.459, 0.665] | – | – | -0.017 [-0.152, +0.118] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.535 [0.431, 0.639] | – | 0.532 [0.431, 0.651] | – | – | +0.003 [-0.175, +0.140] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.504 [0.404, 0.607] | – | 0.504 [0.423, 0.573] | – | – | +0.000 [-0.124, +0.140] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.499 [0.434, 0.574] | – | 0.511 [0.442, 0.574] | – | – | -0.011 [-0.098, +0.092] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.482 [0.396, 0.594] | – | 0.518 [0.457, 0.580] | – | – | -0.035 [-0.130, +0.080] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.549 [0.453, 0.630] | – | 0.489 [0.424, 0.544] | – | – | +0.060 [-0.059, +0.167] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.574 [0.491, 0.650] | – | 0.560 [0.483, 0.651] | – | – | +0.014 [-0.123, +0.135] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.552 [0.462, 0.636] | – | 0.553 [0.458, 0.649] | – | – | -0.001 [-0.128, +0.123] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.500 [0.456, 0.542] | – | 0.504 [0.443, 0.573] | – | – | -0.004 [-0.075, +0.061] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.439 [0.375, 0.505] | – | 0.500 [0.428, 0.574] | – | – | -0.061 [-0.154, +0.023] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.465 [0.415, 0.519] | – | 0.446 [0.377, 0.515] | – | – | +0.019 [-0.043, +0.085] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.475 [0.400, 0.543] | – | 0.461 [0.391, 0.529] | – | – | +0.014 [-0.082, +0.094] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.481 [0.436, 0.524] | – | 0.443 [0.368, 0.515] | – | – | +0.038 [-0.040, +0.125] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.478 [0.431, 0.520] | – | 0.443 [0.361, 0.522] | – | – | +0.035 [-0.044, +0.119] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.468 [0.417, 0.513] | – | 0.457 [0.386, 0.521] | – | – | +0.011 [-0.046, +0.064] | – | – |
| `v.end` | 518 / 362 | 280 | 0.504 [0.451, 0.559] | – | 0.457 [0.382, 0.526] | – | – | +0.047 [-0.046, +0.144] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.552 [0.462, 0.636]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.488, `m2.end` 0.579, `m3.end` 0.535, `m4.end` 0.552. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.522 [0.504, 0.545]; activation minus words -0.025 [-0.064, +0.011]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.476 [0.451, 0.500]; activation minus words +0.012 [-0.037, +0.062]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

