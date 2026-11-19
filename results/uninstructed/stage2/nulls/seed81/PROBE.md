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
| `m1.pre` | 257 / 183 | 139 | 0.495 [0.414, 0.594] | – | 0.500 [0.500, 0.500] | – | – | -0.005 [-0.086, +0.094] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.447 [0.351, 0.547] | – | 0.489 [0.445, 0.526] | – | – | -0.042 [-0.143, +0.056] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.560 [0.450, 0.658] | – | 0.511 [0.404, 0.610] | – | – | +0.049 [-0.089, +0.191] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.482 [0.387, 0.579] | – | 0.532 [0.431, 0.634] | – | – | -0.050 [-0.179, +0.072] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.453 [0.372, 0.539] | – | 0.554 [0.462, 0.656] | – | – | -0.101 [-0.203, -0.009] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.429 [0.334, 0.523] | – | 0.554 [0.465, 0.641] | – | – | -0.125 [-0.236, -0.026] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.509 [0.415, 0.594] | – | 0.567 [0.466, 0.651] | – | – | -0.058 [-0.173, +0.071] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.504 [0.390, 0.603] | – | 0.574 [0.477, 0.658] | – | – | -0.071 [-0.194, +0.062] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.559 [0.497, 0.630] | – | 0.589 [0.453, 0.694] | – | – | -0.030 [-0.168, +0.138] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.525 [0.447, 0.596] | – | 0.582 [0.458, 0.680] | – | – | -0.057 [-0.178, +0.083] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.545 [0.467, 0.624] | – | 0.532 [0.421, 0.641] | – | – | +0.013 [-0.112, +0.123] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.440 [0.324, 0.560] | – | 0.546 [0.439, 0.651] | – | – | -0.106 [-0.223, +0.017] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.581 [0.485, 0.666] | – | 0.540 [0.431, 0.661] | – | – | +0.042 [-0.086, +0.147] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.601 [0.525, 0.672] | – | 0.554 [0.444, 0.669] | – | – | +0.047 [-0.090, +0.169] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.458 [0.368, 0.548] | – | 0.561 [0.435, 0.678] | – | – | -0.104 [-0.253, +0.058] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.440 [0.377, 0.521] | – | 0.540 [0.429, 0.652] | – | – | -0.099 [-0.193, +0.000] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.485 [0.384, 0.589] | – | 0.525 [0.422, 0.621] | – | – | -0.040 [-0.195, +0.094] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.495 [0.421, 0.582] | – | 0.525 [0.427, 0.625] | – | – | -0.030 [-0.144, +0.094] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.491 [0.405, 0.569] | – | 0.589 [0.477, 0.694] | – | – | -0.098 [-0.192, +0.002] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.562 [0.448, 0.656] | – | 0.589 [0.477, 0.694] | – | – | -0.027 [-0.140, +0.077] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.519 [0.443, 0.591] | – | 0.553 [0.439, 0.654] | – | – | -0.034 [-0.135, +0.072] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.494 [0.415, 0.574] | – | 0.560 [0.440, 0.669] | – | – | -0.067 [-0.178, +0.063] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.501 [0.432, 0.571] | – | 0.553 [0.424, 0.669] | – | – | -0.052 [-0.161, +0.075] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.406 [0.324, 0.482] | – | 0.518 [0.396, 0.636] | – | – | -0.112 [-0.231, +0.012] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.487 [0.451, 0.522] | – | 0.493 [0.412, 0.585] | – | – | -0.006 [-0.106, +0.083] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.517 [0.433, 0.591] | – | 0.489 [0.413, 0.574] | – | – | +0.028 [-0.050, +0.092] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.538 [0.482, 0.582] | – | 0.482 [0.423, 0.549] | – | – | +0.056 [-0.040, +0.132] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.526 [0.473, 0.574] | – | 0.486 [0.425, 0.546] | – | – | +0.040 [-0.040, +0.115] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.511 [0.459, 0.562] | – | 0.496 [0.428, 0.563] | – | – | +0.015 [-0.073, +0.104] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.474 [0.419, 0.520] | – | 0.486 [0.415, 0.555] | – | – | -0.012 [-0.111, +0.073] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.439 [0.390, 0.486] | – | 0.482 [0.424, 0.537] | – | – | -0.044 [-0.121, +0.031] | – | – |
| `v.end` | 518 / 362 | 280 | 0.471 [0.427, 0.518] | – | 0.475 [0.422, 0.522] | – | – | -0.004 [-0.057, +0.059] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.406 [0.324, 0.482]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.429, `m2.end` 0.440, `m3.end` 0.495, `m4.end` 0.406. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.499 [0.472, 0.526]; activation minus words -0.048 [-0.095, +0.005]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.495 [0.467, 0.521]; activation minus words +0.009 [-0.054, +0.065]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

