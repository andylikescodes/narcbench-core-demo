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
| `m1.pre` | 257 / 183 | 139 | 0.402 [0.303, 0.515] | – | 0.500 [0.500, 0.500] | – | – | -0.098 [-0.197, +0.015] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.412 [0.337, 0.496] | – | 0.446 [0.406, 0.488] | – | – | -0.034 [-0.102, +0.042] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.472 [0.386, 0.571] | – | 0.554 [0.444, 0.670] | – | – | -0.082 [-0.237, +0.049] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.463 [0.356, 0.566] | – | 0.504 [0.419, 0.595] | – | – | -0.040 [-0.171, +0.083] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.482 [0.370, 0.597] | – | 0.482 [0.389, 0.570] | – | – | +0.000 [-0.137, +0.157] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.380 [0.294, 0.476] | – | 0.432 [0.331, 0.542] | – | – | -0.052 [-0.180, +0.080] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.519 [0.413, 0.626] | – | 0.433 [0.317, 0.559] | – | – | +0.087 [-0.059, +0.216] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.539 [0.470, 0.602] | – | 0.447 [0.331, 0.573] | – | – | +0.092 [-0.068, +0.234] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.464 [0.362, 0.570] | – | 0.426 [0.302, 0.571] | – | – | +0.038 [-0.116, +0.184] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.475 [0.377, 0.554] | – | 0.454 [0.336, 0.593] | – | – | +0.021 [-0.153, +0.161] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.601 [0.535, 0.674] | – | 0.440 [0.316, 0.553] | – | – | +0.162 [+0.067, +0.257] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.434 [0.365, 0.506] | – | 0.418 [0.333, 0.496] | – | – | +0.016 [-0.047, +0.089] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.420 [0.344, 0.503] | – | 0.453 [0.368, 0.552] | – | – | -0.033 [-0.170, +0.098] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.391 [0.338, 0.442] | – | 0.460 [0.373, 0.559] | – | – | -0.069 [-0.194, +0.039] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.542 [0.456, 0.625] | – | 0.496 [0.407, 0.582] | – | – | +0.046 [-0.074, +0.166] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.518 [0.448, 0.584] | – | 0.453 [0.373, 0.533] | – | – | +0.065 [-0.036, +0.163] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.476 [0.377, 0.574] | – | 0.460 [0.376, 0.549] | – | – | +0.016 [-0.100, +0.127] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.472 [0.378, 0.573] | – | 0.468 [0.382, 0.549] | – | – | +0.004 [-0.105, +0.118] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.530 [0.452, 0.601] | – | 0.475 [0.341, 0.615] | – | – | +0.055 [-0.116, +0.204] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.587 [0.505, 0.673] | – | 0.468 [0.340, 0.599] | – | – | +0.119 [-0.036, +0.275] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.522 [0.438, 0.600] | – | 0.482 [0.391, 0.582] | – | – | +0.040 [-0.077, +0.146] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.492 [0.404, 0.593] | – | 0.468 [0.366, 0.571] | – | – | +0.024 [-0.117, +0.171] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.594 [0.488, 0.690] | – | 0.404 [0.292, 0.508] | – | – | +0.190 [+0.032, +0.362] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.495 [0.431, 0.561] | – | 0.440 [0.316, 0.560] | – | – | +0.055 [-0.085, +0.186] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.484 [0.439, 0.534] | – | 0.486 [0.425, 0.555] | – | – | -0.002 [-0.077, +0.076] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.499 [0.451, 0.547] | – | 0.482 [0.423, 0.545] | – | – | +0.017 [-0.063, +0.092] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.524 [0.460, 0.587] | – | 0.504 [0.408, 0.606] | – | – | +0.020 [-0.100, +0.144] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.541 [0.479, 0.621] | – | 0.514 [0.415, 0.623] | – | – | +0.027 [-0.077, +0.125] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.539 [0.481, 0.599] | – | 0.500 [0.416, 0.592] | – | – | +0.039 [-0.045, +0.135] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.571 [0.497, 0.644] | – | 0.493 [0.412, 0.585] | – | – | +0.078 [-0.002, +0.152] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.506 [0.450, 0.569] | – | 0.489 [0.399, 0.593] | – | – | +0.017 [-0.102, +0.126] | – | – |
| `v.end` | 518 / 362 | 280 | 0.503 [0.440, 0.566] | – | 0.504 [0.413, 0.612] | – | – | -0.001 [-0.084, +0.078] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.495 [0.431, 0.561]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.380, `m2.end` 0.434, `m3.end` 0.472, `m4.end` 0.495. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.q3`, `m4.q3`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.487 [0.467, 0.507]; activation minus words +0.026 [-0.029, +0.079]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.521 [0.487, 0.563]; activation minus words +0.024 [-0.048, +0.096]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

