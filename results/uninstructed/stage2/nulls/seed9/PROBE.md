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
| `m1.pre` | 257 / 183 | 139 | 0.548 [0.473, 0.629] | – | 0.500 [0.500, 0.500] | – | – | +0.048 [-0.027, +0.129] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.418 [0.343, 0.493] | – | 0.482 [0.439, 0.519] | – | – | -0.064 [-0.148, +0.015] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.586 [0.519, 0.656] | – | 0.482 [0.405, 0.587] | – | – | +0.104 [-0.011, +0.197] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.521 [0.428, 0.605] | – | 0.597 [0.515, 0.685] | – | – | -0.076 [-0.221, +0.055] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.404 [0.331, 0.469] | – | 0.554 [0.476, 0.640] | – | – | -0.150 [-0.262, -0.051] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.508 [0.414, 0.609] | – | 0.554 [0.484, 0.643] | – | – | -0.046 [-0.153, +0.052] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.530 [0.427, 0.632] | – | 0.539 [0.439, 0.629] | – | – | -0.009 [-0.145, +0.140] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.416 [0.354, 0.481] | – | 0.546 [0.449, 0.636] | – | – | -0.130 [-0.218, -0.029] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.495 [0.402, 0.589] | – | 0.468 [0.381, 0.560] | – | – | +0.027 [-0.097, +0.143] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.516 [0.417, 0.605] | – | 0.440 [0.352, 0.530] | – | – | +0.077 [-0.046, +0.201] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.481 [0.394, 0.551] | – | 0.440 [0.326, 0.559] | – | – | +0.041 [-0.110, +0.183] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.384 [0.296, 0.472] | – | 0.376 [0.257, 0.491] | – | – | +0.009 [-0.132, +0.151] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.534 [0.483, 0.583] | – | 0.540 [0.456, 0.630] | – | – | -0.006 [-0.112, +0.094] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.541 [0.464, 0.640] | – | 0.532 [0.452, 0.621] | – | – | +0.009 [-0.068, +0.106] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.545 [0.465, 0.624] | – | 0.554 [0.479, 0.635] | – | – | -0.009 [-0.125, +0.101] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.417 [0.329, 0.508] | – | 0.540 [0.450, 0.635] | – | – | -0.122 [-0.241, -0.012] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.466 [0.379, 0.564] | – | 0.540 [0.461, 0.630] | – | – | -0.073 [-0.208, +0.058] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.494 [0.408, 0.575] | – | 0.561 [0.476, 0.646] | – | – | -0.068 [-0.191, +0.042] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.434 [0.360, 0.509] | – | 0.369 [0.247, 0.482] | – | – | +0.065 [-0.052, +0.177] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.410 [0.320, 0.495] | – | 0.369 [0.247, 0.482] | – | – | +0.041 [-0.055, +0.140] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.384 [0.299, 0.455] | – | 0.369 [0.248, 0.483] | – | – | +0.016 [-0.139, +0.171] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.444 [0.380, 0.526] | – | 0.369 [0.252, 0.471] | – | – | +0.075 [-0.063, +0.235] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.441 [0.380, 0.502] | – | 0.340 [0.215, 0.455] | – | – | +0.101 [-0.007, +0.227] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.516 [0.398, 0.614] | – | 0.326 [0.195, 0.453] | – | – | +0.190 [+0.084, +0.291] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.518 [0.445, 0.581] | – | 0.471 [0.401, 0.553] | – | – | +0.046 [-0.054, +0.139] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.507 [0.445, 0.574] | – | 0.436 [0.350, 0.530] | – | – | +0.071 [-0.029, +0.171] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.499 [0.453, 0.545] | – | 0.464 [0.389, 0.541] | – | – | +0.034 [-0.031, +0.102] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.534 [0.466, 0.596] | – | 0.450 [0.373, 0.530] | – | – | +0.084 [-0.019, +0.185] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.484 [0.448, 0.529] | – | 0.454 [0.374, 0.543] | – | – | +0.030 [-0.076, +0.129] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.469 [0.402, 0.542] | – | 0.454 [0.374, 0.541] | – | – | +0.016 [-0.068, +0.108] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.475 [0.410, 0.535] | – | 0.446 [0.368, 0.529] | – | – | +0.029 [-0.066, +0.118] | – | – |
| `v.end` | 518 / 362 | 280 | 0.511 [0.449, 0.568] | – | 0.457 [0.391, 0.528] | – | – | +0.054 [-0.035, +0.145] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.516 [0.398, 0.614]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.508, `m2.end` 0.384, `m3.end` 0.494, `m4.end` 0.516. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.476 [0.451, 0.499]; activation minus words +0.002 [-0.040, +0.045]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.500 [0.467, 0.529]; activation minus words +0.046 [-0.024, +0.111]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

