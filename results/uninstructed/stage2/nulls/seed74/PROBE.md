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
| `m1.pre` | 257 / 183 | 139 | 0.380 [0.298, 0.465] | – | 0.500 [0.500, 0.500] | – | – | -0.120 [-0.202, -0.035] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.507 [0.423, 0.593] | – | 0.525 [0.496, 0.552] | – | – | -0.018 [-0.106, +0.074] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.511 [0.440, 0.574] | – | 0.525 [0.436, 0.604] | – | – | -0.014 [-0.106, +0.096] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.494 [0.420, 0.564] | – | 0.547 [0.461, 0.625] | – | – | -0.053 [-0.148, +0.045] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.580 [0.488, 0.678] | – | 0.604 [0.497, 0.702] | – | – | -0.024 [-0.175, +0.144] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.522 [0.434, 0.623] | – | 0.583 [0.484, 0.669] | – | – | -0.060 [-0.189, +0.076] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.505 [0.421, 0.595] | – | 0.504 [0.406, 0.613] | – | – | +0.001 [-0.077, +0.086] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.461 [0.376, 0.551] | – | 0.511 [0.418, 0.615] | – | – | -0.050 [-0.152, +0.049] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.416 [0.333, 0.498] | – | 0.489 [0.373, 0.611] | – | – | -0.074 [-0.175, +0.006] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.437 [0.352, 0.513] | – | 0.496 [0.406, 0.595] | – | – | -0.060 [-0.171, +0.044] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.383 [0.296, 0.479] | – | 0.504 [0.396, 0.626] | – | – | -0.121 [-0.240, +0.012] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.440 [0.334, 0.547] | – | 0.496 [0.396, 0.614] | – | – | -0.057 [-0.178, +0.066] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.604 [0.537, 0.667] | – | 0.583 [0.457, 0.687] | – | – | +0.022 [-0.105, +0.169] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.588 [0.503, 0.662] | – | 0.590 [0.477, 0.690] | – | – | -0.001 [-0.123, +0.125] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.540 [0.448, 0.641] | – | 0.597 [0.470, 0.713] | – | – | -0.058 [-0.234, +0.140] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.435 [0.350, 0.541] | – | 0.554 [0.438, 0.652] | – | – | -0.119 [-0.267, +0.068] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.478 [0.368, 0.600] | – | 0.547 [0.429, 0.640] | – | – | -0.069 [-0.215, +0.100] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.529 [0.460, 0.591] | – | 0.568 [0.441, 0.684] | – | – | -0.039 [-0.148, +0.087] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.445 [0.364, 0.541] | – | 0.468 [0.364, 0.598] | – | – | -0.023 [-0.163, +0.094] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.470 [0.398, 0.561] | – | 0.475 [0.372, 0.600] | – | – | -0.006 [-0.137, +0.098] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.355 [0.267, 0.454] | – | 0.504 [0.406, 0.621] | – | – | -0.149 [-0.284, -0.025] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.433 [0.350, 0.528] | – | 0.511 [0.408, 0.619] | – | – | -0.078 [-0.174, +0.010] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.467 [0.404, 0.545] | – | 0.489 [0.385, 0.598] | – | – | -0.023 [-0.125, +0.083] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.441 [0.343, 0.548] | – | 0.539 [0.424, 0.660] | – | – | -0.098 [-0.202, +0.013] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.468 [0.423, 0.514] | – | 0.496 [0.423, 0.570] | – | – | -0.029 [-0.120, +0.070] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.441 [0.376, 0.502] | – | 0.475 [0.407, 0.549] | – | – | -0.034 [-0.109, +0.034] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.422 [0.365, 0.483] | – | 0.486 [0.414, 0.553] | – | – | -0.064 [-0.128, +0.006] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.471 [0.408, 0.536] | – | 0.479 [0.408, 0.549] | – | – | -0.007 [-0.101, +0.095] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.461 [0.407, 0.511] | – | 0.439 [0.363, 0.516] | – | – | +0.021 [-0.060, +0.094] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.432 [0.363, 0.514] | – | 0.429 [0.349, 0.507] | – | – | +0.004 [-0.069, +0.082] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.443 [0.393, 0.495] | – | 0.450 [0.369, 0.529] | – | – | -0.007 [-0.067, +0.050] | – | – |
| `v.end` | 518 / 362 | 280 | 0.494 [0.441, 0.547] | – | 0.432 [0.359, 0.502] | – | – | +0.061 [-0.019, +0.147] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.441 [0.343, 0.548]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.522, `m2.end` 0.440, `m3.end` 0.529, `m4.end` 0.441. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.476 [0.449, 0.507]; activation minus words -0.054 [-0.105, +0.003]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.454 [0.424, 0.484]; activation minus words -0.007 [-0.057, +0.044]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

