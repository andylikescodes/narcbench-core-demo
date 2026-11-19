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
| `m1.pre` | 257 / 183 | 139 | 0.555 [0.492, 0.617] | – | 0.500 [0.500, 0.500] | – | – | +0.055 [-0.008, +0.117] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.531 [0.411, 0.642] | – | 0.547 [0.494, 0.590] | – | – | -0.016 [-0.126, +0.085] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.446 [0.349, 0.545] | – | 0.446 [0.333, 0.561] | – | – | +0.000 [-0.120, +0.128] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.517 [0.411, 0.616] | – | 0.482 [0.389, 0.575] | – | – | +0.035 [-0.095, +0.135] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.508 [0.434, 0.583] | – | 0.446 [0.345, 0.537] | – | – | +0.062 [-0.035, +0.165] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.361 [0.316, 0.408] | – | 0.446 [0.358, 0.542] | – | – | -0.085 [-0.185, +0.011] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.491 [0.414, 0.566] | – | 0.511 [0.415, 0.616] | – | – | -0.020 [-0.112, +0.066] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.491 [0.426, 0.553] | – | 0.511 [0.413, 0.614] | – | – | -0.020 [-0.136, +0.093] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.511 [0.443, 0.571] | – | 0.482 [0.359, 0.599] | – | – | +0.028 [-0.118, +0.183] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.445 [0.351, 0.536] | – | 0.447 [0.367, 0.531] | – | – | -0.001 [-0.106, +0.103] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.414 [0.348, 0.474] | – | 0.397 [0.326, 0.475] | – | – | +0.017 [-0.093, +0.115] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.526 [0.457, 0.589] | – | 0.447 [0.349, 0.548] | – | – | +0.079 [-0.024, +0.188] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.390 [0.319, 0.482] | – | 0.410 [0.328, 0.483] | – | – | -0.020 [-0.138, +0.121] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.455 [0.392, 0.521] | – | 0.410 [0.328, 0.483] | – | – | +0.045 [-0.073, +0.171] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.499 [0.418, 0.577] | – | 0.424 [0.339, 0.513] | – | – | +0.075 [-0.032, +0.181] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.519 [0.447, 0.584] | – | 0.396 [0.292, 0.500] | – | – | +0.124 [-0.009, +0.245] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.492 [0.412, 0.569] | – | 0.417 [0.330, 0.507] | – | – | +0.075 [-0.008, +0.157] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.401 [0.346, 0.464] | – | 0.396 [0.310, 0.470] | – | – | +0.006 [-0.083, +0.109] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.583 [0.496, 0.664] | – | 0.468 [0.396, 0.560] | – | – | +0.115 [+0.016, +0.193] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.552 [0.450, 0.643] | – | 0.468 [0.396, 0.560] | – | – | +0.084 [-0.036, +0.183] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.494 [0.395, 0.600] | – | 0.504 [0.427, 0.588] | – | – | -0.010 [-0.148, +0.119] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.583 [0.512, 0.659] | – | 0.518 [0.435, 0.605] | – | – | +0.065 [-0.031, +0.163] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.560 [0.471, 0.645] | – | 0.560 [0.476, 0.649] | – | – | +0.000 [-0.109, +0.114] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.567 [0.474, 0.655] | – | 0.560 [0.462, 0.650] | – | – | +0.007 [-0.061, +0.079] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.516 [0.460, 0.569] | – | 0.500 [0.431, 0.560] | – | – | +0.016 [-0.068, +0.102] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.498 [0.433, 0.567] | – | 0.479 [0.394, 0.554] | – | – | +0.019 [-0.067, +0.120] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.512 [0.451, 0.565] | – | 0.482 [0.402, 0.560] | – | – | +0.030 [-0.094, +0.138] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.539 [0.472, 0.607] | – | 0.486 [0.404, 0.560] | – | – | +0.053 [-0.036, +0.142] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.451 [0.403, 0.498] | – | 0.514 [0.435, 0.596] | – | – | -0.063 [-0.158, +0.016] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.472 [0.415, 0.528] | – | 0.525 [0.444, 0.606] | – | – | -0.053 [-0.146, +0.039] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.511 [0.465, 0.555] | – | 0.529 [0.452, 0.608] | – | – | -0.018 [-0.094, +0.071] | – | – |
| `v.end` | 518 / 362 | 280 | 0.486 [0.441, 0.545] | – | 0.500 [0.418, 0.584] | – | – | -0.014 [-0.107, +0.092] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.567 [0.474, 0.655]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.361, `m2.end` 0.526, `m3.end` 0.401, `m4.end` 0.567. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.495 [0.472, 0.518]; activation minus words +0.029 [-0.012, +0.067]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.498 [0.473, 0.527]; activation minus words -0.004 [-0.080, +0.068]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

