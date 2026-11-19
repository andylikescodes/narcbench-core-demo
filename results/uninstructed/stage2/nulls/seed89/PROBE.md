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
| `m1.pre` | 257 / 183 | 139 | 0.525 [0.435, 0.611] | – | 0.500 [0.500, 0.500] | – | – | +0.025 [-0.065, +0.111] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.426 [0.352, 0.505] | – | 0.511 [0.466, 0.553] | – | – | -0.085 [-0.145, -0.020] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.550 [0.492, 0.605] | – | 0.403 [0.296, 0.500] | – | – | +0.147 [+0.033, +0.278] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.423 [0.355, 0.500] | – | 0.489 [0.390, 0.574] | – | – | -0.066 [-0.176, +0.078] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.532 [0.443, 0.621] | – | 0.561 [0.460, 0.667] | – | – | -0.029 [-0.134, +0.091] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.488 [0.405, 0.581] | – | 0.525 [0.421, 0.632] | – | – | -0.037 [-0.113, +0.043] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.519 [0.441, 0.591] | – | 0.539 [0.429, 0.641] | – | – | -0.020 [-0.125, +0.086] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.557 [0.481, 0.637] | – | 0.539 [0.429, 0.641] | – | – | +0.018 [-0.108, +0.143] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.454 [0.373, 0.521] | – | 0.482 [0.388, 0.570] | – | – | -0.028 [-0.133, +0.069] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.569 [0.451, 0.689] | – | 0.525 [0.419, 0.622] | – | – | +0.044 [-0.099, +0.187] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.491 [0.413, 0.571] | – | 0.574 [0.474, 0.667] | – | – | -0.084 [-0.154, +0.003] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.499 [0.402, 0.603] | – | 0.496 [0.362, 0.611] | – | – | +0.003 [-0.159, +0.165] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.445 [0.341, 0.535] | – | 0.590 [0.485, 0.697] | – | – | -0.145 [-0.299, -0.013] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.409 [0.332, 0.487] | – | 0.590 [0.485, 0.697] | – | – | -0.181 [-0.286, -0.080] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.471 [0.394, 0.553] | – | 0.561 [0.446, 0.667] | – | – | -0.091 [-0.217, +0.060] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.534 [0.441, 0.606] | – | 0.525 [0.417, 0.634] | – | – | +0.009 [-0.144, +0.147] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.475 [0.396, 0.545] | – | 0.576 [0.482, 0.667] | – | – | -0.101 [-0.215, +0.012] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.483 [0.413, 0.552] | – | 0.489 [0.384, 0.597] | – | – | -0.006 [-0.117, +0.106] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.391 [0.305, 0.494] | – | 0.461 [0.343, 0.562] | – | – | -0.070 [-0.212, +0.088] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.435 [0.366, 0.506] | – | 0.461 [0.343, 0.562] | – | – | -0.026 [-0.143, +0.111] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.452 [0.384, 0.526] | – | 0.496 [0.388, 0.593] | – | – | -0.044 [-0.165, +0.093] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.565 [0.486, 0.638] | – | 0.496 [0.372, 0.600] | – | – | +0.068 [-0.024, +0.179] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.448 [0.362, 0.538] | – | 0.525 [0.411, 0.625] | – | – | -0.077 [-0.183, +0.053] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.396 [0.322, 0.482] | – | 0.518 [0.401, 0.631] | – | – | -0.122 [-0.240, +0.010] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.435 [0.392, 0.480] | – | 0.479 [0.395, 0.555] | – | – | -0.044 [-0.141, +0.065] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.487 [0.417, 0.554] | – | 0.475 [0.392, 0.549] | – | – | +0.012 [-0.077, +0.105] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.446 [0.400, 0.483] | – | 0.407 [0.344, 0.465] | – | – | +0.039 [-0.029, +0.108] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.436 [0.379, 0.509] | – | 0.446 [0.387, 0.502] | – | – | -0.010 [-0.090, +0.085] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.445 [0.402, 0.492] | – | 0.464 [0.387, 0.542] | – | – | -0.019 [-0.113, +0.083] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.492 [0.445, 0.544] | – | 0.461 [0.386, 0.538] | – | – | +0.031 [-0.033, +0.106] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.433 [0.385, 0.483] | – | 0.475 [0.396, 0.551] | – | – | -0.042 [-0.127, +0.048] | – | – |
| `v.end` | 518 / 362 | 280 | 0.545 [0.488, 0.613] | – | 0.482 [0.407, 0.556] | – | – | +0.063 [-0.011, +0.146] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.396 [0.322, 0.482]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.488, `m2.end` 0.499, `m3.end` 0.483, `m4.end` 0.396. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m1.q1`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.481 [0.458, 0.503]; activation minus words -0.037 [-0.089, +0.023]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.465 [0.443, 0.491]; activation minus words +0.004 [-0.063, +0.077]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

