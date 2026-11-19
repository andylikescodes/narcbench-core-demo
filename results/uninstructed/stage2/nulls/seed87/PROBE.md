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
| `m1.pre` | 257 / 183 | 139 | 0.448 [0.377, 0.517] | – | 0.500 [0.500, 0.500] | – | – | -0.052 [-0.123, +0.017] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.499 [0.435, 0.576] | – | 0.482 [0.439, 0.522] | – | – | +0.017 [-0.071, +0.112] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.534 [0.442, 0.613] | – | 0.457 [0.361, 0.552] | – | – | +0.077 [-0.042, +0.191] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.506 [0.429, 0.588] | – | 0.475 [0.389, 0.566] | – | – | +0.032 [-0.090, +0.143] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.527 [0.441, 0.602] | – | 0.532 [0.453, 0.632] | – | – | -0.006 [-0.121, +0.088] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.468 [0.382, 0.561] | – | 0.540 [0.458, 0.618] | – | – | -0.072 [-0.202, +0.073] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.533 [0.419, 0.661] | – | 0.518 [0.359, 0.665] | – | – | +0.016 [-0.126, +0.155] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.496 [0.414, 0.578] | – | 0.518 [0.359, 0.665] | – | – | -0.021 [-0.156, +0.105] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.591 [0.487, 0.690] | – | 0.482 [0.342, 0.610] | – | – | +0.109 [-0.002, +0.220] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.462 [0.387, 0.546] | – | 0.454 [0.308, 0.597] | – | – | +0.009 [-0.128, +0.159] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.492 [0.418, 0.571] | – | 0.426 [0.307, 0.551] | – | – | +0.067 [-0.054, +0.195] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.591 [0.530, 0.650] | – | 0.376 [0.254, 0.514] | – | – | +0.216 [+0.085, +0.336] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.529 [0.444, 0.613] | – | 0.518 [0.446, 0.578] | – | – | +0.012 [-0.072, +0.114] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.580 [0.481, 0.670] | – | 0.518 [0.446, 0.578] | – | – | +0.062 [-0.042, +0.166] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.424 [0.338, 0.503] | – | 0.532 [0.446, 0.603] | – | – | -0.108 [-0.198, -0.024] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.422 [0.343, 0.498] | – | 0.504 [0.423, 0.577] | – | – | -0.082 [-0.204, +0.051] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.486 [0.406, 0.559] | – | 0.489 [0.402, 0.567] | – | – | -0.003 [-0.107, +0.108] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.538 [0.451, 0.615] | – | 0.453 [0.359, 0.540] | – | – | +0.085 [-0.019, +0.198] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.516 [0.401, 0.629] | – | 0.376 [0.222, 0.533] | – | – | +0.140 [-0.028, +0.298] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.441 [0.357, 0.523] | – | 0.376 [0.222, 0.533] | – | – | +0.065 [-0.124, +0.222] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.607 [0.536, 0.672] | – | 0.397 [0.246, 0.549] | – | – | +0.210 [+0.064, +0.352] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.487 [0.419, 0.563] | – | 0.355 [0.216, 0.492] | – | – | +0.132 [-0.006, +0.285] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.474 [0.398, 0.547] | – | 0.404 [0.268, 0.538] | – | – | +0.070 [-0.049, +0.191] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.468 [0.370, 0.564] | – | 0.440 [0.306, 0.572] | – | – | +0.028 [-0.119, +0.157] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.500 [0.429, 0.555] | – | 0.436 [0.363, 0.500] | – | – | +0.064 [-0.021, +0.138] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.533 [0.452, 0.608] | – | 0.436 [0.348, 0.525] | – | – | +0.097 [-0.015, +0.210] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.524 [0.471, 0.571] | – | 0.479 [0.417, 0.548] | – | – | +0.045 [-0.040, +0.115] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.531 [0.489, 0.579] | – | 0.486 [0.427, 0.549] | – | – | +0.045 [-0.029, +0.113] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.532 [0.478, 0.590] | – | 0.511 [0.453, 0.568] | – | – | +0.021 [-0.049, +0.100] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.494 [0.451, 0.552] | – | 0.475 [0.407, 0.538] | – | – | +0.019 [-0.053, +0.107] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.428 [0.366, 0.501] | – | 0.454 [0.392, 0.518] | – | – | -0.026 [-0.113, +0.075] | – | – |
| `v.end` | 518 / 362 | 280 | 0.515 [0.445, 0.581] | – | 0.461 [0.392, 0.530] | – | – | +0.054 [-0.031, +0.141] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.468 [0.370, 0.564]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.468, `m2.end` 0.591, `m3.end` 0.538, `m4.end` 0.468. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.end`, `m4.q1`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.505 [0.477, 0.533]; activation minus words +0.042 [-0.013, +0.096]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.507 [0.470, 0.546]; activation minus words +0.040 [-0.020, +0.101]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

