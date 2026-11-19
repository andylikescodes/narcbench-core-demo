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
| `m1.pre` | 257 / 183 | 139 | 0.501 [0.438, 0.561] | – | 0.500 [0.500, 0.500] | – | – | +0.001 [-0.062, +0.061] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.527 [0.459, 0.589] | – | 0.446 [0.383, 0.500] | – | – | +0.081 [-0.006, +0.164] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.446 [0.358, 0.524] | – | 0.496 [0.414, 0.569] | – | – | -0.050 [-0.152, +0.039] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.450 [0.354, 0.558] | – | 0.568 [0.488, 0.657] | – | – | -0.118 [-0.227, -0.018] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.466 [0.391, 0.549] | – | 0.568 [0.473, 0.653] | – | – | -0.102 [-0.211, +0.025] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.479 [0.380, 0.581] | – | 0.468 [0.386, 0.551] | – | – | +0.012 [-0.087, +0.110] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.521 [0.440, 0.606] | – | 0.411 [0.326, 0.507] | – | – | +0.109 [+0.007, +0.219] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.472 [0.386, 0.563] | – | 0.404 [0.317, 0.496] | – | – | +0.068 [-0.042, +0.176] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.506 [0.424, 0.574] | – | 0.496 [0.386, 0.585] | – | – | +0.010 [-0.077, +0.105] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.562 [0.473, 0.652] | – | 0.489 [0.370, 0.603] | – | – | +0.072 [-0.064, +0.221] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.536 [0.441, 0.634] | – | 0.433 [0.323, 0.542] | – | – | +0.104 [-0.045, +0.274] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.573 [0.476, 0.661] | – | 0.397 [0.291, 0.512] | – | – | +0.176 [+0.047, +0.295] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.573 [0.511, 0.633] | – | 0.424 [0.345, 0.512] | – | – | +0.148 [+0.053, +0.242] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.481 [0.413, 0.565] | – | 0.417 [0.339, 0.505] | – | – | +0.063 [-0.025, +0.156] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.573 [0.486, 0.657] | – | 0.439 [0.356, 0.524] | – | – | +0.134 [+0.041, +0.219] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.555 [0.468, 0.641] | – | 0.460 [0.373, 0.545] | – | – | +0.095 [+0.004, +0.181] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.435 [0.338, 0.541] | – | 0.475 [0.387, 0.565] | – | – | -0.040 [-0.147, +0.067] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.447 [0.357, 0.539] | – | 0.518 [0.430, 0.607] | – | – | -0.071 [-0.157, +0.012] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.536 [0.453, 0.614] | – | 0.340 [0.246, 0.435] | – | – | +0.196 [+0.090, +0.299] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.504 [0.407, 0.601] | – | 0.326 [0.229, 0.422] | – | – | +0.177 [+0.058, +0.309] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.390 [0.313, 0.469] | – | 0.418 [0.325, 0.515] | – | – | -0.028 [-0.146, +0.086] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.591 [0.512, 0.661] | – | 0.440 [0.342, 0.545] | – | – | +0.152 [+0.010, +0.272] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.518 [0.424, 0.621] | – | 0.397 [0.312, 0.492] | – | – | +0.121 [+0.021, +0.242] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.539 [0.446, 0.639] | – | 0.418 [0.318, 0.516] | – | – | +0.121 [+0.032, +0.221] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.513 [0.454, 0.575] | – | 0.457 [0.383, 0.538] | – | – | +0.056 [-0.021, +0.130] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.448 [0.384, 0.519] | – | 0.461 [0.378, 0.562] | – | – | -0.013 [-0.097, +0.070] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.477 [0.426, 0.539] | – | 0.446 [0.389, 0.520] | – | – | +0.031 [-0.046, +0.099] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.494 [0.424, 0.570] | – | 0.479 [0.417, 0.553] | – | – | +0.016 [-0.066, +0.095] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.531 [0.487, 0.579] | – | 0.507 [0.422, 0.612] | – | – | +0.024 [-0.088, +0.118] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.555 [0.505, 0.605] | – | 0.500 [0.416, 0.598] | – | – | +0.055 [-0.024, +0.120] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.547 [0.491, 0.603] | – | 0.504 [0.418, 0.602] | – | – | +0.044 [-0.041, +0.124] | – | – |
| `v.end` | 518 / 362 | 280 | 0.476 [0.418, 0.539] | – | 0.500 [0.417, 0.594] | – | – | -0.024 [-0.097, +0.038] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.539 [0.446, 0.639]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.479, `m2.end` 0.573, `m3.end` 0.447, `m4.end` 0.539. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.pre`, `m2.end`, `m3.pre`, `m3.q1`, `m3.q2`, `m4.pre`, `m4.t0`, `m4.q2`, `m4.q3`, `m4.end`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.508 [0.478, 0.538]; activation minus words +0.060 [+0.013, +0.108]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.505 [0.472, 0.544]; activation minus words +0.024 [-0.038, +0.076]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

