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
| `m1.pre` | 257 / 183 | 139 | 0.427 [0.372, 0.490] | – | 0.500 [0.500, 0.500] | – | – | -0.073 [-0.128, -0.010] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.504 [0.422, 0.593] | – | 0.482 [0.446, 0.513] | – | – | +0.022 [-0.058, +0.108] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.547 [0.462, 0.640] | – | 0.568 [0.433, 0.709] | – | – | -0.022 [-0.149, +0.117] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.557 [0.473, 0.647] | – | 0.626 [0.482, 0.753] | – | – | -0.069 [-0.225, +0.113] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.462 [0.360, 0.556] | – | 0.626 [0.526, 0.716] | – | – | -0.164 [-0.283, -0.059] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.476 [0.379, 0.565] | – | 0.612 [0.488, 0.709] | – | – | -0.135 [-0.235, -0.033] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.506 [0.421, 0.572] | – | 0.468 [0.358, 0.583] | – | – | +0.038 [-0.132, +0.191] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.481 [0.386, 0.561] | – | 0.482 [0.382, 0.591] | – | – | -0.001 [-0.153, +0.150] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.552 [0.457, 0.638] | – | 0.468 [0.376, 0.569] | – | – | +0.084 [-0.043, +0.206] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.499 [0.406, 0.589] | – | 0.482 [0.377, 0.598] | – | – | +0.017 [-0.110, +0.148] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.394 [0.310, 0.479] | – | 0.475 [0.386, 0.564] | – | – | -0.081 [-0.223, +0.054] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.494 [0.398, 0.585] | – | 0.496 [0.414, 0.585] | – | – | -0.003 [-0.154, +0.143] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.519 [0.432, 0.599] | – | 0.561 [0.419, 0.693] | – | – | -0.042 [-0.137, +0.077] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.481 [0.392, 0.564] | – | 0.568 [0.427, 0.696] | – | – | -0.088 [-0.190, +0.038] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.563 [0.458, 0.652] | – | 0.568 [0.441, 0.683] | – | – | -0.006 [-0.104, +0.115] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.485 [0.383, 0.580] | – | 0.561 [0.422, 0.684] | – | – | -0.076 [-0.197, +0.060] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.529 [0.437, 0.617] | – | 0.554 [0.418, 0.673] | – | – | -0.024 [-0.121, +0.104] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.583 [0.454, 0.698] | – | 0.590 [0.462, 0.712] | – | – | -0.007 [-0.107, +0.104] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.478 [0.406, 0.561] | – | 0.496 [0.430, 0.563] | – | – | -0.018 [-0.110, +0.066] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.509 [0.425, 0.588] | – | 0.482 [0.413, 0.557] | – | – | +0.027 [-0.080, +0.122] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.462 [0.369, 0.547] | – | 0.511 [0.427, 0.603] | – | – | -0.048 [-0.159, +0.052] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.472 [0.377, 0.584] | – | 0.518 [0.453, 0.578] | – | – | -0.045 [-0.156, +0.073] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.599 [0.546, 0.647] | – | 0.482 [0.417, 0.549] | – | – | +0.116 [+0.050, +0.182] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.543 [0.440, 0.641] | – | 0.482 [0.404, 0.563] | – | – | +0.061 [-0.058, +0.183] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.437 [0.372, 0.503] | – | 0.554 [0.482, 0.625] | – | – | -0.116 [-0.178, -0.046] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.542 [0.468, 0.609] | – | 0.557 [0.482, 0.628] | – | – | -0.015 [-0.108, +0.063] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.508 [0.459, 0.557] | – | 0.539 [0.467, 0.600] | – | – | -0.031 [-0.104, +0.036] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.562 [0.507, 0.615] | – | 0.546 [0.478, 0.618] | – | – | +0.016 [-0.064, +0.094] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.496 [0.436, 0.552] | – | 0.539 [0.474, 0.605] | – | – | -0.044 [-0.126, +0.028] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.513 [0.450, 0.577] | – | 0.532 [0.466, 0.598] | – | – | -0.019 [-0.117, +0.068] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.557 [0.486, 0.621] | – | 0.496 [0.419, 0.580] | – | – | +0.061 [-0.045, +0.145] | – | – |
| `v.end` | 518 / 362 | 280 | 0.548 [0.487, 0.608] | – | 0.493 [0.429, 0.560] | – | – | +0.055 [-0.029, +0.131] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.543 [0.440, 0.641]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.476, `m2.end` 0.494, `m3.end` 0.583, `m4.end` 0.543. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.q3`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.505 [0.464, 0.543]; activation minus words -0.022 [-0.066, +0.020]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.520 [0.489, 0.554]; activation minus words -0.012 [-0.066, +0.033]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

