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
| `m1.pre` | 257 / 183 | 139 | 0.488 [0.415, 0.567] | – | 0.500 [0.500, 0.500] | – | – | -0.012 [-0.085, +0.067] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.452 [0.366, 0.539] | – | 0.504 [0.461, 0.550] | – | – | -0.052 [-0.112, +0.007] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.550 [0.446, 0.639] | – | 0.489 [0.371, 0.608] | – | – | +0.060 [-0.049, +0.166] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.512 [0.433, 0.601] | – | 0.453 [0.333, 0.567] | – | – | +0.059 [-0.049, +0.187] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.452 [0.382, 0.509] | – | 0.547 [0.444, 0.651] | – | – | -0.095 [-0.224, +0.010] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.427 [0.339, 0.531] | – | 0.525 [0.440, 0.618] | – | – | -0.098 [-0.213, +0.008] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.445 [0.361, 0.541] | – | 0.504 [0.386, 0.632] | – | – | -0.058 [-0.202, +0.087] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.496 [0.395, 0.590] | – | 0.496 [0.378, 0.620] | – | – | -0.000 [-0.123, +0.102] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.552 [0.458, 0.644] | – | 0.475 [0.367, 0.600] | – | – | +0.077 [-0.097, +0.230] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.482 [0.408, 0.551] | – | 0.532 [0.432, 0.646] | – | – | -0.050 [-0.179, +0.068] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.472 [0.375, 0.565] | – | 0.511 [0.395, 0.613] | – | – | -0.038 [-0.152, +0.073] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.479 [0.403, 0.568] | – | 0.525 [0.442, 0.612] | – | – | -0.045 [-0.168, +0.082] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.413 [0.321, 0.516] | – | 0.525 [0.412, 0.639] | – | – | -0.112 [-0.241, +0.020] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.430 [0.353, 0.514] | – | 0.532 [0.413, 0.646] | – | – | -0.102 [-0.216, +0.030] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.479 [0.380, 0.563] | – | 0.525 [0.420, 0.641] | – | – | -0.046 [-0.194, +0.070] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.565 [0.474, 0.660] | – | 0.482 [0.377, 0.611] | – | – | +0.083 [-0.048, +0.210] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.554 [0.463, 0.646] | – | 0.489 [0.388, 0.602] | – | – | +0.065 [-0.020, +0.149] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.499 [0.384, 0.618] | – | 0.496 [0.385, 0.620] | – | – | +0.003 [-0.118, +0.130] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.515 [0.431, 0.596] | – | 0.603 [0.527, 0.683] | – | – | -0.088 [-0.213, +0.038] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.521 [0.448, 0.586] | – | 0.610 [0.536, 0.690] | – | – | -0.089 [-0.184, -0.002] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.508 [0.424, 0.586] | – | 0.596 [0.503, 0.685] | – | – | -0.088 [-0.186, +0.017] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.394 [0.307, 0.483] | – | 0.546 [0.456, 0.643] | – | – | -0.152 [-0.306, -0.003] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.494 [0.413, 0.579] | – | 0.525 [0.450, 0.606] | – | – | -0.031 [-0.139, +0.071] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.430 [0.358, 0.500] | – | 0.546 [0.462, 0.638] | – | – | -0.116 [-0.228, -0.023] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.481 [0.435, 0.524] | – | 0.518 [0.460, 0.579] | – | – | -0.037 [-0.112, +0.034] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.468 [0.417, 0.516] | – | 0.493 [0.436, 0.556] | – | – | -0.025 [-0.101, +0.050] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.504 [0.455, 0.562] | – | 0.504 [0.442, 0.565] | – | – | +0.001 [-0.080, +0.088] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.436 [0.385, 0.490] | – | 0.468 [0.411, 0.525] | – | – | -0.031 [-0.095, +0.034] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.489 [0.427, 0.549] | – | 0.486 [0.417, 0.557] | – | – | +0.003 [-0.093, +0.094] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.525 [0.466, 0.577] | – | 0.461 [0.397, 0.531] | – | – | +0.064 [-0.038, +0.153] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.484 [0.429, 0.531] | – | 0.457 [0.394, 0.524] | – | – | +0.027 [-0.047, +0.097] | – | – |
| `v.end` | 518 / 362 | 280 | 0.511 [0.440, 0.580] | – | 0.468 [0.398, 0.536] | – | – | +0.043 [-0.031, +0.122] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.430 [0.358, 0.500]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.427, `m2.end` 0.479, `m3.end` 0.499, `m4.end` 0.430. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: `m4.pre`.
  - In one number, the mean over the 24 room positions: activation 0.484 [0.455, 0.513]; activation minus words -0.039 [-0.085, +0.006]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.487 [0.467, 0.510]; activation minus words +0.006 [-0.052, +0.059]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

