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
| `m1.pre` | 257 / 183 | 139 | 0.413 [0.340, 0.488] | – | 0.500 [0.500, 0.500] | – | – | -0.087 [-0.160, -0.012] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.432 [0.379, 0.495] | – | 0.460 [0.426, 0.500] | – | – | -0.029 [-0.087, +0.031] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.545 [0.458, 0.613] | – | 0.450 [0.339, 0.573] | – | – | +0.096 [-0.057, +0.237] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.426 [0.323, 0.529] | – | 0.489 [0.375, 0.614] | – | – | -0.063 [-0.226, +0.104] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.525 [0.454, 0.603] | – | 0.525 [0.435, 0.616] | – | – | +0.000 [-0.089, +0.080] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.542 [0.466, 0.616] | – | 0.475 [0.402, 0.547] | – | – | +0.068 [-0.050, +0.185] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.481 [0.416, 0.543] | – | 0.326 [0.236, 0.399] | – | – | +0.155 [+0.088, +0.233] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.494 [0.417, 0.562] | – | 0.326 [0.236, 0.399] | – | – | +0.167 [+0.088, +0.255] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.513 [0.415, 0.622] | – | 0.333 [0.248, 0.424] | – | – | +0.180 [+0.044, +0.317] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.536 [0.463, 0.611] | – | 0.426 [0.339, 0.513] | – | – | +0.111 [-0.010, +0.234] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.508 [0.417, 0.595] | – | 0.411 [0.331, 0.489] | – | – | +0.096 [+0.009, +0.196] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.528 [0.444, 0.614] | – | 0.475 [0.379, 0.560] | – | – | +0.052 [-0.041, +0.150] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.567 [0.485, 0.649] | – | 0.475 [0.394, 0.565] | – | – | +0.092 [-0.023, +0.208] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.583 [0.504, 0.666] | – | 0.468 [0.391, 0.552] | – | – | +0.115 [+0.026, +0.210] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.492 [0.427, 0.556] | – | 0.518 [0.435, 0.603] | – | – | -0.026 [-0.142, +0.078] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.519 [0.428, 0.625] | – | 0.468 [0.389, 0.549] | – | – | +0.052 [-0.059, +0.169] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.610 [0.545, 0.669] | – | 0.482 [0.407, 0.568] | – | – | +0.128 [+0.017, +0.219] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.445 [0.369, 0.524] | – | 0.439 [0.347, 0.535] | – | – | +0.006 [-0.105, +0.116] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.457 [0.349, 0.556] | – | 0.511 [0.374, 0.624] | – | – | -0.054 [-0.173, +0.058] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.484 [0.400, 0.558] | – | 0.496 [0.363, 0.607] | – | – | -0.013 [-0.128, +0.110] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.454 [0.377, 0.540] | – | 0.511 [0.381, 0.612] | – | – | -0.057 [-0.185, +0.104] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.515 [0.442, 0.580] | – | 0.482 [0.394, 0.574] | – | – | +0.033 [-0.077, +0.131] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.504 [0.432, 0.598] | – | 0.475 [0.365, 0.583] | – | – | +0.028 [-0.102, +0.175] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.448 [0.369, 0.534] | – | 0.496 [0.386, 0.601] | – | – | -0.048 [-0.142, +0.055] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.534 [0.476, 0.586] | – | 0.457 [0.368, 0.561] | – | – | +0.077 [-0.052, +0.177] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.560 [0.492, 0.628] | – | 0.489 [0.407, 0.583] | – | – | +0.071 [-0.028, +0.178] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.484 [0.424, 0.551] | – | 0.475 [0.412, 0.547] | – | – | +0.009 [-0.070, +0.081] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.541 [0.477, 0.610] | – | 0.471 [0.411, 0.541] | – | – | +0.070 [-0.004, +0.147] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.502 [0.444, 0.561] | – | 0.468 [0.403, 0.539] | – | – | +0.034 [-0.061, +0.126] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.511 [0.447, 0.574] | – | 0.450 [0.386, 0.525] | – | – | +0.061 [-0.029, +0.152] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.486 [0.423, 0.553] | – | 0.471 [0.407, 0.545] | – | – | +0.014 [-0.066, +0.095] | – | – |
| `v.end` | 518 / 362 | 280 | 0.508 [0.433, 0.580] | – | 0.461 [0.392, 0.537] | – | – | +0.047 [-0.042, +0.137] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.448 [0.369, 0.534]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.542, `m2.end` 0.528, `m3.end` 0.445, `m4.end` 0.448. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.pre`, `m2.t0`, `m2.q1`, `m2.q3`, `m3.t0`, `m3.q3`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.501 [0.477, 0.527]; activation minus words +0.042 [+0.005, +0.085]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.516 [0.477, 0.559]; activation minus words +0.048 [-0.022, +0.115]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

