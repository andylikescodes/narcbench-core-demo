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
| `m1.pre` | 257 / 183 | 139 | 0.542 [0.460, 0.612] | – | 0.500 [0.500, 0.500] | – | – | +0.042 [-0.040, +0.112] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.419 [0.324, 0.518] | – | 0.532 [0.486, 0.581] | – | – | -0.113 [-0.205, -0.008] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.504 [0.391, 0.638] | – | 0.460 [0.361, 0.561] | – | – | +0.043 [-0.063, +0.172] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.496 [0.402, 0.595] | – | 0.396 [0.313, 0.478] | – | – | +0.101 [-0.035, +0.231] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.495 [0.397, 0.584] | – | 0.432 [0.355, 0.514] | – | – | +0.063 [-0.046, +0.167] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.495 [0.423, 0.581] | – | 0.410 [0.321, 0.515] | – | – | +0.085 [-0.008, +0.177] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.533 [0.455, 0.608] | – | 0.496 [0.373, 0.607] | – | – | +0.037 [-0.094, +0.182] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.555 [0.490, 0.620] | – | 0.504 [0.385, 0.615] | – | – | +0.051 [-0.089, +0.210] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.576 [0.486, 0.662] | – | 0.482 [0.370, 0.582] | – | – | +0.094 [-0.049, +0.254] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.506 [0.448, 0.564] | – | 0.475 [0.375, 0.565] | – | – | +0.031 [-0.093, +0.170] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.452 [0.355, 0.545] | – | 0.504 [0.410, 0.587] | – | – | -0.051 [-0.171, +0.065] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.512 [0.441, 0.585] | – | 0.511 [0.405, 0.600] | – | – | +0.001 [-0.110, +0.116] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.459 [0.394, 0.535] | – | 0.388 [0.300, 0.509] | – | – | +0.071 [-0.053, +0.176] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.468 [0.396, 0.543] | – | 0.396 [0.295, 0.528] | – | – | +0.072 [-0.084, +0.207] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.406 [0.307, 0.502] | – | 0.410 [0.300, 0.542] | – | – | -0.004 [-0.194, +0.161] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.443 [0.379, 0.506] | – | 0.432 [0.338, 0.558] | – | – | +0.012 [-0.121, +0.122] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.384 [0.298, 0.484] | – | 0.360 [0.273, 0.468] | – | – | +0.024 [-0.073, +0.104] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.481 [0.401, 0.563] | – | 0.388 [0.301, 0.490] | – | – | +0.092 [-0.016, +0.188] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.471 [0.368, 0.569] | – | 0.518 [0.431, 0.584] | – | – | -0.047 [-0.160, +0.096] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.471 [0.382, 0.559] | – | 0.511 [0.425, 0.576] | – | – | -0.040 [-0.139, +0.074] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.506 [0.434, 0.570] | – | 0.532 [0.444, 0.599] | – | – | -0.026 [-0.115, +0.075] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.501 [0.412, 0.586] | – | 0.504 [0.406, 0.570] | – | – | -0.003 [-0.119, +0.140] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.499 [0.412, 0.567] | – | 0.518 [0.415, 0.595] | – | – | -0.018 [-0.116, +0.095] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.462 [0.405, 0.520] | – | 0.496 [0.400, 0.574] | – | – | -0.034 [-0.132, +0.078] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.431 [0.368, 0.496] | – | 0.454 [0.389, 0.515] | – | – | -0.022 [-0.086, +0.044] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.534 [0.476, 0.592] | – | 0.511 [0.449, 0.570] | – | – | +0.024 [-0.048, +0.092] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.438 [0.386, 0.506] | – | 0.475 [0.408, 0.544] | – | – | -0.037 [-0.124, +0.065] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.519 [0.461, 0.578] | – | 0.482 [0.424, 0.544] | – | – | +0.036 [-0.038, +0.109] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.506 [0.448, 0.567] | – | 0.468 [0.412, 0.535] | – | – | +0.038 [-0.022, +0.090] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.555 [0.498, 0.607] | – | 0.479 [0.422, 0.545] | – | – | +0.076 [-0.009, +0.151] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.576 [0.539, 0.607] | – | 0.496 [0.433, 0.563] | – | – | +0.080 [+0.015, +0.134] | – | – |
| `v.end` | 518 / 362 | 280 | 0.461 [0.400, 0.525] | – | 0.507 [0.442, 0.585] | – | – | -0.046 [-0.118, +0.020] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.462 [0.405, 0.520]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.495, `m2.end` 0.512, `m3.end` 0.481, `m4.end` 0.462. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `v.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.485 [0.466, 0.507]; activation minus words +0.020 [-0.020, +0.065]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.502 [0.472, 0.533]; activation minus words +0.019 [-0.029, +0.064]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

