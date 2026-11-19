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
| `m1.pre` | 257 / 183 | 139 | 0.457 [0.386, 0.530] | – | 0.500 [0.500, 0.500] | – | – | -0.043 [-0.114, +0.030] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.517 [0.429, 0.590] | – | 0.532 [0.500, 0.559] | – | – | -0.015 [-0.096, +0.058] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.471 [0.389, 0.546] | – | 0.453 [0.366, 0.537] | – | – | +0.017 [-0.067, +0.113] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.419 [0.332, 0.494] | – | 0.446 [0.364, 0.519] | – | – | -0.027 [-0.127, +0.074] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.537 [0.462, 0.619] | – | 0.518 [0.417, 0.632] | – | – | +0.019 [-0.101, +0.150] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.557 [0.456, 0.652] | – | 0.482 [0.361, 0.601] | – | – | +0.075 [-0.011, +0.167] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.496 [0.431, 0.567] | – | 0.532 [0.446, 0.623] | – | – | -0.035 [-0.131, +0.074] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.472 [0.400, 0.556] | – | 0.539 [0.450, 0.627] | – | – | -0.067 [-0.157, +0.039] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.532 [0.454, 0.605] | – | 0.610 [0.524, 0.689] | – | – | -0.078 [-0.184, +0.033] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.533 [0.428, 0.634] | – | 0.518 [0.429, 0.593] | – | – | +0.016 [-0.074, +0.117] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.464 [0.374, 0.551] | – | 0.532 [0.452, 0.611] | – | – | -0.068 [-0.166, +0.036] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.491 [0.404, 0.595] | – | 0.582 [0.464, 0.693] | – | – | -0.091 [-0.216, +0.046] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.499 [0.416, 0.589] | – | 0.518 [0.394, 0.636] | – | – | -0.019 [-0.146, +0.116] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.588 [0.490, 0.672] | – | 0.518 [0.394, 0.636] | – | – | +0.071 [-0.058, +0.190] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.584 [0.505, 0.668] | – | 0.525 [0.412, 0.622] | – | – | +0.059 [-0.057, +0.188] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.494 [0.411, 0.584] | – | 0.540 [0.407, 0.657] | – | – | -0.046 [-0.134, +0.057] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.496 [0.399, 0.602] | – | 0.511 [0.379, 0.619] | – | – | -0.014 [-0.131, +0.132] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.515 [0.399, 0.655] | – | 0.468 [0.353, 0.580] | – | – | +0.047 [-0.125, +0.248] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.472 [0.394, 0.550] | – | 0.525 [0.417, 0.634] | – | – | -0.052 [-0.163, +0.051] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.530 [0.439, 0.618] | – | 0.532 [0.418, 0.642] | – | – | -0.001 [-0.128, +0.111] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.454 [0.358, 0.542] | – | 0.553 [0.462, 0.659] | – | – | -0.099 [-0.222, +0.014] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.430 [0.332, 0.520] | – | 0.546 [0.452, 0.657] | – | – | -0.116 [-0.278, +0.013] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.441 [0.354, 0.517] | – | 0.489 [0.400, 0.581] | – | – | -0.048 [-0.174, +0.058] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.445 [0.359, 0.521] | – | 0.454 [0.376, 0.543] | – | – | -0.009 [-0.134, +0.094] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.507 [0.442, 0.570] | – | 0.511 [0.429, 0.598] | – | – | -0.004 [-0.099, +0.087] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.483 [0.436, 0.533] | – | 0.521 [0.434, 0.607] | – | – | -0.039 [-0.115, +0.046] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.536 [0.492, 0.578] | – | 0.496 [0.435, 0.556] | – | – | +0.039 [-0.025, +0.099] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.527 [0.438, 0.607] | – | 0.511 [0.441, 0.581] | – | – | +0.016 [-0.058, +0.092] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.514 [0.451, 0.587] | – | 0.525 [0.447, 0.603] | – | – | -0.011 [-0.086, +0.065] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.490 [0.404, 0.571] | – | 0.536 [0.457, 0.613] | – | – | -0.046 [-0.103, +0.008] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.458 [0.414, 0.500] | – | 0.521 [0.439, 0.598] | – | – | -0.064 [-0.149, +0.023] | – | – |
| `v.end` | 518 / 362 | 280 | 0.541 [0.477, 0.606] | – | 0.529 [0.450, 0.609] | – | – | +0.012 [-0.071, +0.077] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.445 [0.359, 0.521]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.557, `m2.end` 0.491, `m3.end` 0.515, `m4.end` 0.445. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.496 [0.464, 0.523]; activation minus words -0.022 [-0.071, +0.030]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.507 [0.467, 0.540]; activation minus words -0.012 [-0.060, +0.036]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

