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
| `m1.pre` | 257 / 183 | 139 | 0.442 [0.398, 0.488] | – | 0.500 [0.500, 0.500] | – | – | -0.058 [-0.102, -0.012] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.470 [0.419, 0.519] | – | 0.504 [0.461, 0.538] | – | – | -0.034 [-0.096, +0.040] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.489 [0.377, 0.592] | – | 0.540 [0.423, 0.659] | – | – | -0.050 [-0.167, +0.045] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.512 [0.442, 0.591] | – | 0.453 [0.366, 0.557] | – | – | +0.059 [-0.075, +0.182] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.501 [0.430, 0.578] | – | 0.554 [0.442, 0.667] | – | – | -0.053 [-0.165, +0.067] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.563 [0.483, 0.631] | – | 0.518 [0.413, 0.633] | – | – | +0.045 [-0.076, +0.159] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.423 [0.323, 0.516] | – | 0.468 [0.376, 0.545] | – | – | -0.045 [-0.181, +0.077] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.407 [0.302, 0.506] | – | 0.468 [0.376, 0.545] | – | – | -0.061 [-0.207, +0.071] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.460 [0.365, 0.547] | – | 0.468 [0.370, 0.557] | – | – | -0.009 [-0.150, +0.133] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.492 [0.400, 0.578] | – | 0.532 [0.421, 0.633] | – | – | -0.040 [-0.163, +0.074] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.529 [0.455, 0.606] | – | 0.482 [0.396, 0.569] | – | – | +0.047 [-0.065, +0.158] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.455 [0.372, 0.541] | – | 0.496 [0.407, 0.587] | – | – | -0.041 [-0.176, +0.086] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.597 [0.523, 0.670] | – | 0.576 [0.470, 0.678] | – | – | +0.022 [-0.103, +0.147] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.567 [0.463, 0.664] | – | 0.576 [0.470, 0.678] | – | – | -0.009 [-0.151, +0.137] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.571 [0.479, 0.661] | – | 0.547 [0.439, 0.657] | – | – | +0.024 [-0.086, +0.140] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.492 [0.393, 0.579] | – | 0.525 [0.440, 0.614] | – | – | -0.033 [-0.125, +0.046] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.483 [0.364, 0.591] | – | 0.518 [0.415, 0.625] | – | – | -0.035 [-0.166, +0.118] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.535 [0.433, 0.643] | – | 0.518 [0.411, 0.627] | – | – | +0.017 [-0.110, +0.143] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.468 [0.404, 0.538] | – | 0.475 [0.383, 0.562] | – | – | -0.007 [-0.105, +0.102] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.460 [0.389, 0.531] | – | 0.461 [0.370, 0.545] | – | – | -0.001 [-0.095, +0.087] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.407 [0.327, 0.482] | – | 0.433 [0.363, 0.502] | – | – | -0.026 [-0.128, +0.074] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.515 [0.411, 0.635] | – | 0.433 [0.354, 0.514] | – | – | +0.082 [+0.004, +0.196] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.504 [0.429, 0.596] | – | 0.454 [0.374, 0.546] | – | – | +0.050 [-0.039, +0.145] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.525 [0.460, 0.598] | – | 0.461 [0.379, 0.546] | – | – | +0.064 [-0.023, +0.143] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.577 [0.504, 0.635] | – | 0.493 [0.397, 0.581] | – | – | +0.084 [-0.011, +0.180] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.519 [0.441, 0.585] | – | 0.493 [0.387, 0.591] | – | – | +0.026 [-0.095, +0.145] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.529 [0.474, 0.580] | – | 0.446 [0.350, 0.541] | – | – | +0.082 [-0.027, +0.195] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.533 [0.488, 0.570] | – | 0.450 [0.358, 0.540] | – | – | +0.083 [-0.005, +0.172] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.514 [0.479, 0.546] | – | 0.479 [0.377, 0.580] | – | – | +0.036 [-0.071, +0.151] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.522 [0.464, 0.569] | – | 0.475 [0.364, 0.580] | – | – | +0.047 [-0.065, +0.159] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.495 [0.435, 0.559] | – | 0.496 [0.392, 0.590] | – | – | -0.001 [-0.088, +0.085] | – | – |
| `v.end` | 518 / 362 | 280 | 0.490 [0.421, 0.567] | – | 0.489 [0.393, 0.576] | – | – | +0.001 [-0.104, +0.108] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.525 [0.460, 0.598]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.563, `m2.end` 0.455, `m3.end` 0.535, `m4.end` 0.525. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.q2`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.494 [0.475, 0.515]; activation minus words -0.004 [-0.055, +0.043]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.522 [0.491, 0.549]; activation minus words +0.045 [-0.039, +0.134]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

