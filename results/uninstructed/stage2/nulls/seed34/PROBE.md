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
| `m1.pre` | 257 / 183 | 139 | 0.460 [0.380, 0.532] | – | 0.500 [0.500, 0.500] | – | – | -0.040 [-0.120, +0.032] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.495 [0.415, 0.568] | – | 0.496 [0.468, 0.520] | – | – | -0.001 [-0.077, +0.071] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.534 [0.430, 0.623] | – | 0.536 [0.435, 0.630] | – | – | -0.002 [-0.167, +0.154] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.494 [0.405, 0.574] | – | 0.518 [0.415, 0.625] | – | – | -0.024 [-0.169, +0.086] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.478 [0.379, 0.584] | – | 0.640 [0.512, 0.760] | – | – | -0.163 [-0.311, -0.016] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.462 [0.366, 0.551] | – | 0.540 [0.430, 0.650] | – | – | -0.078 [-0.215, +0.035] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.505 [0.404, 0.590] | – | 0.468 [0.360, 0.566] | – | – | +0.037 [-0.108, +0.171] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.570 [0.497, 0.638] | – | 0.461 [0.352, 0.559] | – | – | +0.109 [+0.007, +0.223] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.474 [0.395, 0.548] | – | 0.518 [0.406, 0.621] | – | – | -0.044 [-0.142, +0.051] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.417 [0.349, 0.492] | – | 0.468 [0.360, 0.563] | – | – | -0.051 [-0.172, +0.103] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.423 [0.331, 0.537] | – | 0.447 [0.331, 0.553] | – | – | -0.024 [-0.164, +0.132] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.407 [0.325, 0.490] | – | 0.355 [0.259, 0.450] | – | – | +0.052 [-0.046, +0.149] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.541 [0.445, 0.626] | – | 0.604 [0.479, 0.729] | – | – | -0.063 [-0.224, +0.068] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.498 [0.411, 0.573] | – | 0.604 [0.479, 0.729] | – | – | -0.106 [-0.252, +0.017] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.447 [0.363, 0.520] | – | 0.619 [0.500, 0.745] | – | – | -0.171 [-0.349, -0.017] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.558 [0.464, 0.656] | – | 0.655 [0.528, 0.781] | – | – | -0.096 [-0.258, +0.032] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.613 [0.515, 0.690] | – | 0.633 [0.519, 0.748] | – | – | -0.020 [-0.202, +0.140] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.481 [0.363, 0.576] | – | 0.597 [0.460, 0.731] | – | – | -0.117 [-0.283, +0.027] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.611 [0.524, 0.696] | – | 0.418 [0.321, 0.513] | – | – | +0.193 [+0.059, +0.341] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.569 [0.500, 0.642] | – | 0.411 [0.315, 0.508] | – | – | +0.157 [+0.052, +0.277] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.508 [0.414, 0.598] | – | 0.433 [0.329, 0.534] | – | – | +0.075 [-0.062, +0.217] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.498 [0.419, 0.580] | – | 0.440 [0.322, 0.547] | – | – | +0.058 [-0.097, +0.218] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.548 [0.441, 0.665] | – | 0.461 [0.375, 0.540] | – | – | +0.087 [-0.059, +0.240] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.505 [0.409, 0.615] | – | 0.504 [0.394, 0.610] | – | – | +0.001 [-0.099, +0.106] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.534 [0.476, 0.589] | – | 0.504 [0.437, 0.569] | – | – | +0.031 [-0.052, +0.115] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.460 [0.405, 0.510] | – | 0.525 [0.462, 0.591] | – | – | -0.065 [-0.127, +0.001] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.458 [0.412, 0.502] | – | 0.475 [0.412, 0.542] | – | – | -0.017 [-0.108, +0.065] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.509 [0.460, 0.557] | – | 0.482 [0.418, 0.554] | – | – | +0.026 [-0.063, +0.108] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.471 [0.415, 0.536] | – | 0.482 [0.401, 0.570] | – | – | -0.011 [-0.080, +0.061] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.476 [0.430, 0.534] | – | 0.446 [0.380, 0.522] | – | – | +0.029 [-0.051, +0.119] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.512 [0.458, 0.564] | – | 0.464 [0.399, 0.535] | – | – | +0.048 [-0.044, +0.139] | – | – |
| `v.end` | 518 / 362 | 280 | 0.521 [0.475, 0.558] | – | 0.446 [0.373, 0.527] | – | – | +0.074 [-0.026, +0.159] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.505 [0.409, 0.615]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.462, `m2.end` 0.407, `m3.end` 0.481, `m4.end` 0.505. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.t0`, `m4.pre`, `m4.t0`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.504 [0.479, 0.524]; activation minus words -0.010 [-0.067, +0.043]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.493 [0.474, 0.512]; activation minus words +0.014 [-0.049, +0.071]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

