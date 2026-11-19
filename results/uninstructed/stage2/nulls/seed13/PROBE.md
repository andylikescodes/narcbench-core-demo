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
| `m1.pre` | 257 / 183 | 139 | 0.582 [0.498, 0.660] | – | 0.500 [0.500, 0.500] | – | – | +0.082 [-0.002, +0.160] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.561 [0.509, 0.611] | – | 0.504 [0.478, 0.529] | – | – | +0.058 [-0.002, +0.117] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.518 [0.446, 0.592] | – | 0.529 [0.415, 0.628] | – | – | -0.011 [-0.115, +0.117] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.505 [0.417, 0.597] | – | 0.532 [0.431, 0.621] | – | – | -0.027 [-0.145, +0.108] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.494 [0.422, 0.566] | – | 0.576 [0.465, 0.689] | – | – | -0.082 [-0.193, +0.029] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.481 [0.399, 0.552] | – | 0.504 [0.415, 0.606] | – | – | -0.023 [-0.128, +0.066] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.536 [0.455, 0.612] | – | 0.447 [0.373, 0.523] | – | – | +0.089 [-0.023, +0.184] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.502 [0.426, 0.584] | – | 0.468 [0.391, 0.548] | – | – | +0.034 [-0.056, +0.136] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.523 [0.426, 0.615] | – | 0.475 [0.386, 0.554] | – | – | +0.048 [-0.050, +0.162] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.445 [0.382, 0.518] | – | 0.489 [0.419, 0.556] | – | – | -0.044 [-0.150, +0.081] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.489 [0.421, 0.568] | – | 0.454 [0.381, 0.534] | – | – | +0.035 [-0.054, +0.131] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.589 [0.486, 0.690] | – | 0.539 [0.469, 0.606] | – | – | +0.050 [-0.047, +0.147] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.433 [0.354, 0.524] | – | 0.590 [0.491, 0.696] | – | – | -0.157 [-0.268, -0.047] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.427 [0.323, 0.539] | – | 0.590 [0.491, 0.696] | – | – | -0.163 [-0.269, -0.063] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.472 [0.386, 0.562] | – | 0.554 [0.472, 0.651] | – | – | -0.082 [-0.183, +0.003] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.542 [0.440, 0.634] | – | 0.504 [0.419, 0.599] | – | – | +0.039 [-0.114, +0.186] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.519 [0.430, 0.608] | – | 0.511 [0.423, 0.607] | – | – | +0.009 [-0.121, +0.128] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.445 [0.369, 0.531] | – | 0.511 [0.410, 0.622] | – | – | -0.066 [-0.195, +0.061] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.518 [0.428, 0.601] | – | 0.525 [0.434, 0.600] | – | – | -0.007 [-0.112, +0.094] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.470 [0.382, 0.557] | – | 0.525 [0.434, 0.600] | – | – | -0.055 [-0.164, +0.056] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.570 [0.480, 0.648] | – | 0.546 [0.441, 0.636] | – | – | +0.024 [-0.084, +0.156] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.465 [0.392, 0.543] | – | 0.567 [0.458, 0.656] | – | – | -0.102 [-0.252, +0.065] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.444 [0.332, 0.546] | – | 0.525 [0.417, 0.610] | – | – | -0.081 [-0.228, +0.065] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.489 [0.413, 0.572] | – | 0.511 [0.403, 0.603] | – | – | -0.021 [-0.164, +0.138] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.566 [0.517, 0.630] | – | 0.539 [0.450, 0.608] | – | – | +0.027 [-0.054, +0.142] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.486 [0.435, 0.535] | – | 0.589 [0.502, 0.658] | – | – | -0.103 [-0.184, -0.007] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.481 [0.429, 0.533] | – | 0.532 [0.421, 0.611] | – | – | -0.051 [-0.125, +0.049] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.495 [0.438, 0.553] | – | 0.529 [0.436, 0.596] | – | – | -0.034 [-0.117, +0.070] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.542 [0.509, 0.579] | – | 0.496 [0.404, 0.572] | – | – | +0.046 [-0.038, +0.149] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.551 [0.502, 0.601] | – | 0.489 [0.391, 0.565] | – | – | +0.061 [-0.020, +0.177] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.416 [0.370, 0.460] | – | 0.496 [0.399, 0.573] | – | – | -0.080 [-0.172, +0.028] | – | – |
| `v.end` | 518 / 362 | 280 | 0.525 [0.467, 0.578] | – | 0.489 [0.402, 0.557] | – | – | +0.036 [-0.033, +0.126] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.489 [0.413, 0.572]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.481, `m2.end` 0.589, `m3.end` 0.445, `m4.end` 0.489. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: no position. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.501 [0.476, 0.529]; activation minus words -0.019 [-0.056, +0.017]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.508 [0.485, 0.533]; activation minus words -0.012 [-0.072, +0.079]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

