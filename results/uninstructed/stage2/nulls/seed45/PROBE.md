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
| `m1.pre` | 257 / 183 | 139 | 0.473 [0.410, 0.540] | – | 0.500 [0.500, 0.500] | – | – | -0.027 [-0.090, +0.040] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.461 [0.400, 0.518] | – | 0.504 [0.464, 0.539] | – | – | -0.042 [-0.100, +0.018] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.505 [0.423, 0.603] | – | 0.558 [0.454, 0.661] | – | – | -0.053 [-0.176, +0.080] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.574 [0.488, 0.662] | – | 0.597 [0.476, 0.697] | – | – | -0.023 [-0.138, +0.122] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.511 [0.424, 0.599] | – | 0.583 [0.462, 0.697] | – | – | -0.072 [-0.154, +0.012] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.436 [0.341, 0.519] | – | 0.446 [0.331, 0.574] | – | – | -0.010 [-0.155, +0.115] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.454 [0.366, 0.548] | – | 0.539 [0.414, 0.658] | – | – | -0.085 [-0.212, +0.049] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.445 [0.348, 0.541] | – | 0.539 [0.414, 0.658] | – | – | -0.094 [-0.213, +0.004] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.570 [0.502, 0.651] | – | 0.511 [0.384, 0.628] | – | – | +0.060 [-0.039, +0.180] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.447 [0.347, 0.543] | – | 0.426 [0.315, 0.532] | – | – | +0.021 [-0.103, +0.139] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.363 [0.297, 0.433] | – | 0.369 [0.279, 0.459] | – | – | -0.006 [-0.125, +0.115] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.496 [0.395, 0.611] | – | 0.333 [0.237, 0.434] | – | – | +0.163 [+0.032, +0.310] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.512 [0.424, 0.596] | – | 0.496 [0.392, 0.600] | – | – | +0.016 [-0.066, +0.099] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.463 [0.367, 0.544] | – | 0.496 [0.392, 0.600] | – | – | -0.033 [-0.163, +0.069] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.535 [0.438, 0.623] | – | 0.540 [0.428, 0.641] | – | – | -0.004 [-0.107, +0.086] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.465 [0.387, 0.549] | – | 0.583 [0.476, 0.683] | – | – | -0.118 [-0.233, +0.004] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.568 [0.481, 0.640] | – | 0.568 [0.463, 0.672] | – | – | +0.000 [-0.093, +0.095] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.547 [0.451, 0.627] | – | 0.604 [0.500, 0.702] | – | – | -0.058 [-0.152, +0.043] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.467 [0.383, 0.545] | – | 0.404 [0.306, 0.494] | – | – | +0.062 [-0.044, +0.168] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.472 [0.390, 0.542] | – | 0.404 [0.306, 0.494] | – | – | +0.068 [-0.039, +0.175] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.538 [0.441, 0.630] | – | 0.404 [0.319, 0.484] | – | – | +0.133 [+0.011, +0.271] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.437 [0.372, 0.519] | – | 0.397 [0.294, 0.490] | – | – | +0.040 [-0.077, +0.182] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.437 [0.350, 0.514] | – | 0.404 [0.302, 0.504] | – | – | +0.033 [-0.094, +0.162] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.394 [0.330, 0.475] | – | 0.369 [0.264, 0.465] | – | – | +0.026 [-0.108, +0.171] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.514 [0.475, 0.558] | – | 0.457 [0.395, 0.509] | – | – | +0.057 [-0.007, +0.127] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.508 [0.466, 0.552] | – | 0.475 [0.412, 0.530] | – | – | +0.033 [-0.042, +0.111] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.438 [0.374, 0.510] | – | 0.479 [0.400, 0.557] | – | – | -0.041 [-0.116, +0.029] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.480 [0.431, 0.535] | – | 0.475 [0.399, 0.551] | – | – | +0.005 [-0.073, +0.077] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.519 [0.479, 0.563] | – | 0.489 [0.408, 0.558] | – | – | +0.030 [-0.051, +0.126] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.442 [0.378, 0.507] | – | 0.457 [0.384, 0.522] | – | – | -0.015 [-0.088, +0.074] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.478 [0.425, 0.520] | – | 0.471 [0.395, 0.541] | – | – | +0.006 [-0.061, +0.072] | – | – |
| `v.end` | 518 / 362 | 280 | 0.517 [0.465, 0.563] | – | 0.475 [0.405, 0.535] | – | – | +0.042 [-0.004, +0.088] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.394 [0.330, 0.475]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.436, `m2.end` 0.496, `m3.end` 0.547, `m4.end` 0.394. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.end`, `m4.q1`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.482 [0.455, 0.507]; activation minus words -0.000 [-0.034, +0.039]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.487 [0.464, 0.512]; activation minus words +0.015 [-0.030, +0.069]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

