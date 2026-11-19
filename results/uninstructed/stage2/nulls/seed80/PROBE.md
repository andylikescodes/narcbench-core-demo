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
| `m1.pre` | 257 / 183 | 139 | 0.473 [0.411, 0.532] | – | 0.500 [0.500, 0.500] | – | – | -0.027 [-0.089, +0.032] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.586 [0.521, 0.651] | – | 0.496 [0.464, 0.529] | – | – | +0.090 [+0.006, +0.165] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.537 [0.457, 0.611] | – | 0.568 [0.471, 0.649] | – | – | -0.032 [-0.133, +0.093] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.466 [0.400, 0.538] | – | 0.640 [0.553, 0.724] | – | – | -0.174 [-0.283, -0.059] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.481 [0.405, 0.570] | – | 0.576 [0.496, 0.653] | – | – | -0.095 [-0.192, +0.024] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.512 [0.431, 0.600] | – | 0.540 [0.457, 0.637] | – | – | -0.027 [-0.152, +0.081] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.566 [0.467, 0.648] | – | 0.567 [0.468, 0.670] | – | – | -0.001 [-0.154, +0.147] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.521 [0.424, 0.600] | – | 0.560 [0.462, 0.664] | – | – | -0.040 [-0.168, +0.052] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.454 [0.379, 0.525] | – | 0.617 [0.528, 0.700] | – | – | -0.163 [-0.268, -0.068] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.519 [0.415, 0.618] | – | 0.582 [0.489, 0.672] | – | – | -0.062 [-0.222, +0.094] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.435 [0.360, 0.513] | – | 0.532 [0.441, 0.639] | – | – | -0.096 [-0.233, +0.033] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.377 [0.283, 0.471] | – | 0.482 [0.372, 0.593] | – | – | -0.105 [-0.264, +0.051] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.604 [0.527, 0.695] | – | 0.489 [0.404, 0.568] | – | – | +0.115 [+0.005, +0.250] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.563 [0.484, 0.642] | – | 0.489 [0.404, 0.568] | – | – | +0.073 [-0.023, +0.183] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.469 [0.367, 0.570] | – | 0.460 [0.354, 0.552] | – | – | +0.009 [-0.123, +0.136] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.458 [0.374, 0.546] | – | 0.511 [0.413, 0.604] | – | – | -0.053 [-0.193, +0.092] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.504 [0.409, 0.587] | – | 0.489 [0.393, 0.588] | – | – | +0.014 [-0.125, +0.133] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.403 [0.337, 0.475] | – | 0.504 [0.419, 0.594] | – | – | -0.101 [-0.198, +0.006] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.477 [0.397, 0.559] | – | 0.426 [0.325, 0.526] | – | – | +0.051 [-0.055, +0.161] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.404 [0.329, 0.483] | – | 0.418 [0.319, 0.522] | – | – | -0.014 [-0.131, +0.113] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.484 [0.383, 0.587] | – | 0.426 [0.323, 0.536] | – | – | +0.058 [-0.081, +0.199] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.435 [0.331, 0.536] | – | 0.454 [0.341, 0.577] | – | – | -0.018 [-0.167, +0.143] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.467 [0.381, 0.563] | – | 0.433 [0.328, 0.535] | – | – | +0.034 [-0.119, +0.189] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.444 [0.359, 0.528] | – | 0.440 [0.333, 0.544] | – | – | +0.004 [-0.142, +0.146] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.464 [0.404, 0.525] | – | 0.482 [0.430, 0.539] | – | – | -0.018 [-0.096, +0.053] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.534 [0.472, 0.594] | – | 0.411 [0.361, 0.461] | – | – | +0.124 [+0.047, +0.195] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.578 [0.532, 0.625] | – | 0.436 [0.377, 0.485] | – | – | +0.142 [+0.063, +0.231] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.492 [0.413, 0.556] | – | 0.429 [0.366, 0.483] | – | – | +0.064 [-0.017, +0.140] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.583 [0.536, 0.629] | – | 0.450 [0.384, 0.502] | – | – | +0.133 [+0.077, +0.198] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.455 [0.384, 0.530] | – | 0.454 [0.395, 0.507] | – | – | +0.001 [-0.079, +0.086] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.494 [0.444, 0.548] | – | 0.436 [0.383, 0.490] | – | – | +0.059 [-0.017, +0.132] | – | – |
| `v.end` | 518 / 362 | 280 | 0.454 [0.397, 0.514] | – | 0.436 [0.377, 0.491] | – | – | +0.018 [-0.077, +0.120] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.444 [0.359, 0.528]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.512, `m2.end` 0.377, `m3.end` 0.403, `m4.end` 0.444. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.t0`, `m3.pre`, `p1.end`, `p2.pre`, `p3.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.485 [0.463, 0.509]; activation minus words -0.023 [-0.072, +0.030]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.507 [0.481, 0.536]; activation minus words +0.065 [+0.019, +0.115]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

