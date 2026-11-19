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
| `m1.pre` | 257 / 183 | 139 | 0.529 [0.452, 0.604] | – | 0.500 [0.500, 0.500] | – | – | +0.029 [-0.048, +0.104] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.535 [0.458, 0.609] | – | 0.518 [0.480, 0.560] | – | – | +0.017 [-0.094, +0.118] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.554 [0.481, 0.637] | – | 0.522 [0.418, 0.634] | – | – | +0.032 [-0.085, +0.143] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.446 [0.372, 0.521] | – | 0.540 [0.415, 0.664] | – | – | -0.094 [-0.237, +0.049] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.450 [0.354, 0.543] | – | 0.489 [0.411, 0.584] | – | – | -0.039 [-0.168, +0.081] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.471 [0.394, 0.535] | – | 0.446 [0.351, 0.566] | – | – | +0.024 [-0.118, +0.138] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.508 [0.431, 0.602] | – | 0.489 [0.398, 0.571] | – | – | +0.018 [-0.078, +0.151] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.491 [0.391, 0.594] | – | 0.489 [0.398, 0.571] | – | – | +0.001 [-0.141, +0.150] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.542 [0.462, 0.620] | – | 0.482 [0.383, 0.573] | – | – | +0.060 [-0.027, +0.164] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.506 [0.421, 0.599] | – | 0.482 [0.393, 0.568] | – | – | +0.024 [-0.099, +0.159] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.521 [0.438, 0.598] | – | 0.532 [0.420, 0.629] | – | – | -0.011 [-0.099, +0.083] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.559 [0.464, 0.646] | – | 0.468 [0.360, 0.564] | – | – | +0.091 [-0.023, +0.221] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.521 [0.437, 0.614] | – | 0.453 [0.354, 0.575] | – | – | +0.068 [-0.021, +0.156] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.548 [0.465, 0.627] | – | 0.453 [0.354, 0.575] | – | – | +0.095 [-0.038, +0.194] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.646 [0.552, 0.735] | – | 0.532 [0.421, 0.661] | – | – | +0.114 [-0.032, +0.250] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.607 [0.525, 0.701] | – | 0.547 [0.430, 0.669] | – | – | +0.060 [-0.077, +0.197] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.525 [0.433, 0.608] | – | 0.532 [0.409, 0.667] | – | – | -0.007 [-0.155, +0.127] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.547 [0.449, 0.655] | – | 0.568 [0.476, 0.677] | – | – | -0.022 [-0.118, +0.064] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.513 [0.427, 0.595] | – | 0.518 [0.420, 0.602] | – | – | -0.004 [-0.086, +0.085] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.502 [0.438, 0.569] | – | 0.518 [0.420, 0.602] | – | – | -0.016 [-0.115, +0.091] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.587 [0.512, 0.662] | – | 0.496 [0.404, 0.578] | – | – | +0.091 [-0.022, +0.215] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.565 [0.460, 0.684] | – | 0.518 [0.412, 0.615] | – | – | +0.047 [-0.087, +0.184] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.594 [0.531, 0.652] | – | 0.532 [0.426, 0.627] | – | – | +0.062 [-0.038, +0.174] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.583 [0.479, 0.687] | – | 0.518 [0.413, 0.611] | – | – | +0.065 [-0.063, +0.209] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.455 [0.408, 0.499] | – | 0.564 [0.476, 0.640] | – | – | -0.109 [-0.189, -0.020] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.499 [0.455, 0.550] | – | 0.586 [0.525, 0.649] | – | – | -0.086 [-0.146, -0.023] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.512 [0.447, 0.581] | – | 0.575 [0.520, 0.634] | – | – | -0.063 [-0.149, +0.010] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.467 [0.423, 0.511] | – | 0.550 [0.500, 0.608] | – | – | -0.083 [-0.150, -0.023] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.496 [0.442, 0.550] | – | 0.571 [0.536, 0.614] | – | – | -0.076 [-0.133, -0.022] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.513 [0.465, 0.561] | – | 0.582 [0.543, 0.625] | – | – | -0.069 [-0.133, -0.015] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.520 [0.470, 0.572] | – | 0.586 [0.545, 0.629] | – | – | -0.066 [-0.128, +0.005] | – | – |
| `v.end` | 518 / 362 | 280 | 0.576 [0.520, 0.622] | – | 0.586 [0.548, 0.626] | – | – | -0.010 [-0.078, +0.045] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.583 [0.479, 0.687]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.471, `m2.end` 0.559, `m3.end` 0.547, `m4.end` 0.583. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: `p3.pre`.
  - In one number, the mean over the 24 room positions: activation 0.535 [0.515, 0.558]; activation minus words +0.029 [-0.019, +0.074]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.505 [0.480, 0.531]; activation minus words -0.070 [-0.109, -0.034]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** layer 20, anchor `m3.t0` (score there 0.597).

