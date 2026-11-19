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
| `m1.pre` | 257 / 183 | 139 | 0.482 [0.410, 0.563] | – | 0.500 [0.500, 0.500] | – | – | -0.018 [-0.090, +0.063] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.446 [0.372, 0.523] | – | 0.482 [0.444, 0.518] | – | – | -0.036 [-0.123, +0.055] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.540 [0.453, 0.629] | – | 0.489 [0.381, 0.612] | – | – | +0.050 [-0.053, +0.151] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.443 [0.345, 0.550] | – | 0.410 [0.306, 0.514] | – | – | +0.033 [-0.072, +0.142] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.505 [0.408, 0.603] | – | 0.381 [0.300, 0.471] | – | – | +0.124 [-0.007, +0.258] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.495 [0.398, 0.602] | – | 0.417 [0.348, 0.492] | – | – | +0.078 [-0.042, +0.198] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.587 [0.480, 0.678] | – | 0.560 [0.408, 0.701] | – | – | +0.027 [-0.109, +0.160] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.494 [0.404, 0.572] | – | 0.560 [0.410, 0.696] | – | – | -0.067 [-0.232, +0.083] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.587 [0.495, 0.681] | – | 0.574 [0.453, 0.683] | – | – | +0.013 [-0.077, +0.119] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.546 [0.461, 0.630] | – | 0.574 [0.444, 0.681] | – | – | -0.028 [-0.134, +0.094] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.430 [0.338, 0.522] | – | 0.560 [0.407, 0.693] | – | – | -0.130 [-0.320, +0.056] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.549 [0.463, 0.631] | – | 0.539 [0.385, 0.663] | – | – | +0.010 [-0.119, +0.159] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.452 [0.394, 0.521] | – | 0.424 [0.329, 0.522] | – | – | +0.027 [-0.077, +0.144] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.524 [0.435, 0.610] | – | 0.417 [0.326, 0.511] | – | – | +0.106 [-0.017, +0.237] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.505 [0.384, 0.613] | – | 0.424 [0.326, 0.516] | – | – | +0.081 [-0.050, +0.215] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.525 [0.436, 0.609] | – | 0.417 [0.321, 0.498] | – | – | +0.108 [-0.023, +0.247] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.429 [0.337, 0.509] | – | 0.367 [0.271, 0.458] | – | – | +0.062 [-0.038, +0.166] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.463 [0.392, 0.532] | – | 0.460 [0.357, 0.556] | – | – | +0.003 [-0.118, +0.133] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.516 [0.435, 0.589] | – | 0.504 [0.410, 0.589] | – | – | +0.013 [-0.082, +0.115] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.545 [0.451, 0.635] | – | 0.504 [0.402, 0.597] | – | – | +0.041 [-0.106, +0.197] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.433 [0.368, 0.510] | – | 0.546 [0.447, 0.630] | – | – | -0.113 [-0.227, +0.027] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.506 [0.424, 0.593] | – | 0.567 [0.451, 0.660] | – | – | -0.061 [-0.197, +0.098] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.443 [0.363, 0.517] | – | 0.596 [0.459, 0.710] | – | – | -0.153 [-0.294, -0.004] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.533 [0.462, 0.601] | – | 0.560 [0.450, 0.662] | – | – | -0.027 [-0.148, +0.107] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.521 [0.454, 0.585] | – | 0.521 [0.464, 0.575] | – | – | -0.001 [-0.080, +0.074] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.496 [0.437, 0.561] | – | 0.539 [0.475, 0.603] | – | – | -0.043 [-0.112, +0.036] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.517 [0.447, 0.572] | – | 0.586 [0.522, 0.643] | – | – | -0.069 [-0.156, +0.015] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.479 [0.431, 0.529] | – | 0.604 [0.543, 0.655] | – | – | -0.124 [-0.180, -0.061] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.474 [0.429, 0.517] | – | 0.568 [0.512, 0.618] | – | – | -0.094 [-0.150, -0.026] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.483 [0.432, 0.538] | – | 0.568 [0.510, 0.621] | – | – | -0.085 [-0.141, -0.025] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.481 [0.419, 0.551] | – | 0.557 [0.495, 0.619] | – | – | -0.076 [-0.140, -0.012] | – | – |
| `v.end` | 518 / 362 | 280 | 0.484 [0.426, 0.538] | – | 0.568 [0.502, 0.629] | – | – | -0.084 [-0.168, -0.002] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.533 [0.462, 0.601]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.495, `m2.end` 0.549, `m3.end` 0.463, `m4.end` 0.533. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: `p2.pre`.
  - In one number, the mean over the 24 room positions: activation 0.499 [0.475, 0.525]; activation minus words +0.006 [-0.045, +0.062]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.492 [0.459, 0.524]; activation minus words -0.072 [-0.107, -0.036]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

