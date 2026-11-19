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
| `m1.pre` | 257 / 183 | 139 | 0.483 [0.414, 0.539] | – | 0.500 [0.500, 0.500] | – | – | -0.017 [-0.086, +0.039] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.506 [0.418, 0.576] | – | 0.518 [0.480, 0.565] | – | – | -0.012 [-0.117, +0.078] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.494 [0.371, 0.593] | – | 0.486 [0.398, 0.573] | – | – | +0.008 [-0.101, +0.092] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.460 [0.358, 0.549] | – | 0.396 [0.306, 0.507] | – | – | +0.065 [-0.073, +0.178] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.548 [0.473, 0.628] | – | 0.410 [0.302, 0.512] | – | – | +0.138 [+0.014, +0.252] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.505 [0.423, 0.595] | – | 0.345 [0.226, 0.458] | – | – | +0.160 [+0.059, +0.271] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.567 [0.487, 0.638] | – | 0.489 [0.369, 0.581] | – | – | +0.078 [-0.034, +0.190] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.528 [0.421, 0.621] | – | 0.489 [0.370, 0.581] | – | – | +0.038 [-0.064, +0.148] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.485 [0.383, 0.592] | – | 0.426 [0.331, 0.509] | – | – | +0.060 [-0.068, +0.199] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.570 [0.480, 0.657] | – | 0.411 [0.305, 0.504] | – | – | +0.159 [+0.033, +0.279] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.452 [0.361, 0.539] | – | 0.475 [0.346, 0.584] | – | – | -0.023 [-0.143, +0.119] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.557 [0.464, 0.653] | – | 0.418 [0.300, 0.518] | – | – | +0.139 [+0.014, +0.275] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.433 [0.353, 0.504] | – | 0.446 [0.326, 0.563] | – | – | -0.013 [-0.134, +0.101] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.528 [0.461, 0.599] | – | 0.446 [0.326, 0.563] | – | – | +0.082 [-0.035, +0.208] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.529 [0.454, 0.592] | – | 0.403 [0.295, 0.518] | – | – | +0.127 [+0.001, +0.248] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.332 [0.259, 0.408] | – | 0.446 [0.339, 0.552] | – | – | -0.114 [-0.250, +0.024] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.452 [0.351, 0.567] | – | 0.410 [0.333, 0.504] | – | – | +0.042 [-0.117, +0.186] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.515 [0.427, 0.629] | – | 0.439 [0.371, 0.523] | – | – | +0.076 [-0.044, +0.195] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.569 [0.490, 0.655] | – | 0.454 [0.307, 0.577] | – | – | +0.115 [-0.019, +0.275] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.573 [0.479, 0.665] | – | 0.454 [0.307, 0.577] | – | – | +0.119 [-0.044, +0.308] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.508 [0.417, 0.599] | – | 0.525 [0.394, 0.634] | – | – | -0.017 [-0.159, +0.154] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.548 [0.468, 0.625] | – | 0.511 [0.381, 0.613] | – | – | +0.037 [-0.095, +0.170] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.509 [0.406, 0.604] | – | 0.454 [0.299, 0.585] | – | – | +0.055 [-0.123, +0.243] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.504 [0.418, 0.582] | – | 0.489 [0.355, 0.610] | – | – | +0.014 [-0.113, +0.145] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.538 [0.497, 0.580] | – | 0.507 [0.447, 0.563] | – | – | +0.031 [-0.050, +0.117] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.555 [0.509, 0.606] | – | 0.493 [0.429, 0.552] | – | – | +0.062 [-0.001, +0.131] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.502 [0.430, 0.587] | – | 0.500 [0.446, 0.567] | – | – | +0.002 [-0.090, +0.097] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.420 [0.374, 0.468] | – | 0.496 [0.435, 0.561] | – | – | -0.076 [-0.148, -0.010] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.542 [0.486, 0.602] | – | 0.518 [0.466, 0.584] | – | – | +0.024 [-0.053, +0.100] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.490 [0.435, 0.547] | – | 0.507 [0.449, 0.572] | – | – | -0.017 [-0.109, +0.063] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.461 [0.387, 0.542] | – | 0.525 [0.471, 0.586] | – | – | -0.064 [-0.155, +0.031] | – | – |
| `v.end` | 518 / 362 | 280 | 0.511 [0.435, 0.580] | – | 0.518 [0.464, 0.581] | – | – | -0.007 [-0.091, +0.069] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.504 [0.418, 0.582]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.505, `m2.end` 0.557, `m3.end` 0.515, `m4.end` 0.504. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.q3`, `m1.end`, `m2.q2`, `m2.end`, `m3.q1`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.507 [0.483, 0.528]; activation minus words +0.055 [+0.009, +0.111]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.502 [0.471, 0.538]; activation minus words -0.006 [-0.062, +0.045]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

