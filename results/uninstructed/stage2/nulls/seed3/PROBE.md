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
| `m1.pre` | 257 / 183 | 139 | 0.428 [0.360, 0.501] | – | 0.500 [0.500, 0.500] | – | – | -0.072 [-0.140, +0.001] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.363 [0.271, 0.445] | – | 0.446 [0.414, 0.479] | – | – | -0.083 [-0.166, +0.001] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.458 [0.392, 0.514] | – | 0.565 [0.478, 0.648] | – | – | -0.107 [-0.192, -0.011] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.529 [0.470, 0.587] | – | 0.604 [0.526, 0.692] | – | – | -0.075 [-0.195, +0.032] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.439 [0.362, 0.525] | – | 0.626 [0.550, 0.713] | – | – | -0.187 [-0.303, -0.073] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.400 [0.325, 0.480] | – | 0.619 [0.540, 0.693] | – | – | -0.219 [-0.321, -0.120] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.458 [0.390, 0.529] | – | 0.411 [0.339, 0.491] | – | – | +0.047 [-0.041, +0.133] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.478 [0.366, 0.579] | – | 0.404 [0.336, 0.481] | – | – | +0.074 [-0.023, +0.164] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.553 [0.477, 0.648] | – | 0.404 [0.336, 0.495] | – | – | +0.149 [+0.061, +0.241] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.390 [0.295, 0.493] | – | 0.404 [0.323, 0.497] | – | – | -0.014 [-0.133, +0.100] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.433 [0.357, 0.498] | – | 0.376 [0.305, 0.472] | – | – | +0.057 [-0.069, +0.158] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.538 [0.447, 0.626] | – | 0.390 [0.290, 0.508] | – | – | +0.148 [+0.031, +0.241] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.496 [0.430, 0.561] | – | 0.489 [0.408, 0.569] | – | – | +0.007 [-0.090, +0.106] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.433 [0.365, 0.522] | – | 0.482 [0.398, 0.566] | – | – | -0.049 [-0.141, +0.070] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.521 [0.447, 0.606] | – | 0.504 [0.394, 0.610] | – | – | +0.017 [-0.103, +0.157] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.525 [0.436, 0.610] | – | 0.511 [0.417, 0.597] | – | – | +0.014 [-0.085, +0.110] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.509 [0.413, 0.611] | – | 0.568 [0.484, 0.654] | – | – | -0.059 [-0.191, +0.074] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.445 [0.375, 0.509] | – | 0.597 [0.511, 0.672] | – | – | -0.153 [-0.245, -0.076] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.529 [0.449, 0.621] | – | 0.411 [0.287, 0.544] | – | – | +0.118 [+0.011, +0.221] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.536 [0.427, 0.650] | – | 0.426 [0.303, 0.552] | – | – | +0.111 [-0.028, +0.250] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.543 [0.449, 0.633] | – | 0.397 [0.282, 0.522] | – | – | +0.146 [+0.013, +0.267] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.543 [0.418, 0.642] | – | 0.418 [0.328, 0.526] | – | – | +0.125 [-0.070, +0.275] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.427 [0.351, 0.490] | – | 0.404 [0.308, 0.500] | – | – | +0.023 [-0.126, +0.135] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.528 [0.436, 0.599] | – | 0.418 [0.328, 0.512] | – | – | +0.109 [-0.010, +0.203] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.481 [0.422, 0.550] | – | 0.529 [0.468, 0.590] | – | – | -0.048 [-0.129, +0.038] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.493 [0.430, 0.564] | – | 0.539 [0.466, 0.620] | – | – | -0.046 [-0.150, +0.056] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.484 [0.430, 0.541] | – | 0.493 [0.425, 0.561] | – | – | -0.009 [-0.089, +0.066] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.486 [0.419, 0.551] | – | 0.493 [0.429, 0.561] | – | – | -0.006 [-0.090, +0.075] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.506 [0.472, 0.544] | – | 0.504 [0.445, 0.575] | – | – | +0.003 [-0.071, +0.062] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.459 [0.405, 0.513] | – | 0.464 [0.409, 0.535] | – | – | -0.005 [-0.081, +0.052] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.466 [0.408, 0.522] | – | 0.475 [0.415, 0.541] | – | – | -0.009 [-0.082, +0.052] | – | – |
| `v.end` | 518 / 362 | 280 | 0.470 [0.386, 0.551] | – | 0.486 [0.418, 0.559] | – | – | -0.016 [-0.105, +0.069] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.528 [0.436, 0.599]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.400, `m2.end` 0.538, `m3.end` 0.445, `m4.end` 0.528. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.q1`, `m2.end`, `m4.pre`, `m4.q1`. The bag of words' own onset: `m1.q2`.
  - In one number, the mean over the 24 room positions: activation 0.479 [0.465, 0.496]; activation minus words +0.005 [-0.044, +0.051]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.481 [0.450, 0.509]; activation minus words -0.017 [-0.066, +0.026]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

