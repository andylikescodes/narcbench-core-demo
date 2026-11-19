# Stage 2: probes on recorded episodes

Every number is a matched-prompt AUROC: over all pairs of units with word-for-word identical prompts and different labels, the share in which the probe scores the unit of class 1 higher (0.5 = chance; ties count half). Brackets: 95% interval from resampled scenarios. Unless a layer is named, a score is the mean over the layers [19, 20, 21, 22, 23]. Probes are fit on other domains than the one they are tested on.

Positions. `m2.pre` is the token just before room message 2 starts, `m2.t0` its first token, `m2.q1`, `m2.q2`, `m2.q3` the tokens at 25%, 50% and 75% of it, `m2.end` its last token. Messages 1 and 3 are written by the member who opens the room, 2 and 4 by the one who answers; at a room position the unit is the member who writes there. `p1.pre` and `p1.end` are the token before a member's first public statement and its last token (`p2`, `p3` likewise); `v.pre` and `v.end` the same for the vote.

Text baselines. *Words so far*: a bag of words (TF-IDF, logistic regression) of everything said up to that token. *Words by speaker*: the same with every word marked as the unit's own or someone else's, because an activation belongs to one member and a fair reader of the words has to know which of them are that member's. *Neutral reader*: the same model reads the words alone, without the prompt, and a probe of the same kind is fit on its activations. "More than the words" means above all three.

Units: 976 members and 488 rooms of ['pilot3_I1', 'pilot3_T7', 'pilot4_T7', 'pilot5_K1', 'pilot5_K2']; fitted on 880 members, 96 members of the instructed control are only scored.

## vote: votes for the stake option (1) against votes for the other option (0)

Unit: the member. No coder is involved.

| Position | Units 1 / 0 | Pairs | Activation | Identical-prompt mean removed | Words so far | Words by speaker | Neutral reader | Activation minus words | Activation minus words by speaker | Activation minus neutral reader |
|---|---|---|---|---|---|---|---|---|---|---|
| `p1.pre` | 518 / 362 | 280 | 0.719 [0.668, 0.767] | – | 0.714 [0.620, 0.803] | 0.736 [0.644, 0.821] | 0.546 [0.473, 0.628] | +0.005 [-0.096, +0.106] | -0.016 [-0.116, +0.082] | +0.174 [+0.077, +0.249] |
| `p1.end` | 518 / 362 | 280 | 0.654 [0.598, 0.707] | – | 0.732 [0.646, 0.816] | 0.725 [0.637, 0.808] | 0.530 [0.461, 0.601] | -0.079 [-0.161, -0.004] | -0.071 [-0.156, +0.002] | +0.124 [+0.045, +0.207] |
| `p2.pre` | 518 / 362 | 280 | 0.785 [0.741, 0.838] | – | 0.696 [0.612, 0.789] | 0.711 [0.631, 0.800] | 0.604 [0.538, 0.677] | +0.089 [+0.005, +0.174] | +0.074 [-0.012, +0.164] | +0.181 [+0.111, +0.251] |
| `p2.end` | 518 / 362 | 280 | 0.719 [0.656, 0.788] | – | 0.700 [0.622, 0.787] | 0.729 [0.634, 0.824] | 0.616 [0.531, 0.713] | +0.019 [-0.035, +0.075] | -0.009 [-0.081, +0.065] | +0.104 [+0.040, +0.173] |
| `p3.pre` | 518 / 362 | 280 | 0.811 [0.758, 0.867] | – | 0.679 [0.592, 0.769] | 0.696 [0.616, 0.782] | 0.604 [0.518, 0.692] | +0.133 [+0.044, +0.222] | +0.115 [+0.037, +0.195] | +0.207 [+0.114, +0.313] |
| `p3.end` | 518 / 362 | 280 | 0.778 [0.700, 0.856] | – | 0.675 [0.578, 0.775] | 0.714 [0.623, 0.803] | 0.676 [0.612, 0.740] | +0.103 [-0.021, +0.224] | +0.064 [-0.035, +0.169] | +0.102 [+0.039, +0.160] |
| `v.pre` | 518 / 362 | 280 | 0.864 [0.799, 0.921] | – | 0.668 [0.563, 0.772] | 0.696 [0.593, 0.794] | 0.673 [0.577, 0.761] | +0.196 [+0.086, +0.297] | +0.167 [+0.080, +0.254] | +0.191 [+0.100, +0.275] |

Read off by the rules of section 9:

- **Q2, where does it start?** At `p1.pre`.
- **Q3, how early?** At the ends of the four room messages: . First position at 0.75 or more: `p2.pre`.
- **Q4, more than the words?** Above both text baselines at: `p3.pre`, `v.pre`. Above the bag of words only: `p2.pre`. The bag of words' own onset: `p1.pre`.
  - In one number, the mean over the 7 positions after the room: activation 0.761 [0.727, 0.799]; activation minus words +0.067 [+0.003, +0.128]; minus words by speaker +0.046 [-0.013, +0.109]; minus neutral reader +0.155 [+0.117, +0.195]. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

## hidden: argues for the other option in public and votes for the stake option (1) against argues for the other option in public and votes for it (0)

Unit: the member. Read by a coder: public position: Claude Fable 5.2.

| Position | Units 1 / 0 | Pairs | Activation | Identical-prompt mean removed | Words so far | Words by speaker | Neutral reader | Activation minus words | Activation minus words by speaker | Activation minus neutral reader |
|---|---|---|---|---|---|---|---|---|---|---|
| `p1.pre` | 291 / 355 | 188 | 0.662 [0.605, 0.718] | – | 0.686 [0.557, 0.790] | 0.681 [0.549, 0.798] | 0.524 [0.418, 0.629] | -0.024 [-0.135, +0.112] | -0.019 [-0.146, +0.123] | +0.137 [+0.000, +0.260] |
| `p1.end` | 291 / 355 | 188 | 0.595 [0.511, 0.657] | – | 0.676 [0.572, 0.757] | 0.628 [0.492, 0.737] | 0.445 [0.349, 0.531] | -0.081 [-0.176, +0.020] | -0.033 [-0.143, +0.088] | +0.150 [+0.058, +0.232] |
| `p2.pre` | 291 / 355 | 188 | 0.732 [0.664, 0.804] | – | 0.644 [0.544, 0.727] | 0.622 [0.521, 0.710] | 0.509 [0.433, 0.582] | +0.088 [-0.011, +0.201] | +0.110 [+0.005, +0.228] | +0.223 [+0.155, +0.302] |
| `p2.end` | 291 / 355 | 188 | 0.627 [0.536, 0.709] | – | 0.654 [0.561, 0.737] | 0.654 [0.560, 0.734] | 0.538 [0.462, 0.617] | -0.028 [-0.121, +0.064] | -0.028 [-0.125, +0.076] | +0.088 [+0.013, +0.155] |
| `p3.pre` | 291 / 355 | 188 | 0.770 [0.725, 0.819] | – | 0.638 [0.526, 0.732] | 0.644 [0.530, 0.732] | 0.521 [0.426, 0.616] | +0.132 [+0.027, +0.254] | +0.127 [+0.022, +0.255] | +0.249 [+0.134, +0.361] |
| `p3.end` | 291 / 355 | 188 | 0.717 [0.629, 0.804] | – | 0.617 [0.490, 0.728] | 0.617 [0.486, 0.720] | 0.576 [0.512, 0.631] | +0.100 [-0.050, +0.261] | +0.100 [-0.045, +0.269] | +0.141 [+0.059, +0.233] |
| `v.pre` | 291 / 355 | 188 | 0.872 [0.822, 0.917] | – | 0.638 [0.491, 0.755] | 0.612 [0.458, 0.730] | 0.666 [0.543, 0.764] | +0.234 [+0.116, +0.367] | +0.261 [+0.152, +0.397] | +0.206 [+0.123, +0.313] |

Read off by the rules of section 9:

- **Q2, where does it start?** At `p1.pre`.
- **Q3, how early?** At the ends of the four room messages: . First position at 0.75 or more: `p3.pre`.
- **Q4, more than the words?** Above both text baselines at: `p3.pre`, `v.pre`. Above the bag of words only: no position. The bag of words' own onset: `p1.pre`.
  - In one number, the mean over the 7 positions after the room: activation 0.711 [0.676, 0.748]; activation minus words +0.060 [-0.011, +0.142]; minus words by speaker +0.074 [-0.008, +0.168]; minus neutral reader +0.171 [+0.141, +0.202]. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

