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
| `m1.pre` | 257 / 183 | 139 | 0.429 [0.364, 0.495] | – | 0.500 [0.500, 0.500] | – | – | -0.071 [-0.136, -0.005] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.517 [0.438, 0.599] | – | 0.511 [0.468, 0.554] | – | – | +0.006 [-0.078, +0.084] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.571 [0.506, 0.644] | – | 0.543 [0.440, 0.636] | – | – | +0.028 [-0.056, +0.135] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.466 [0.350, 0.556] | – | 0.561 [0.452, 0.657] | – | – | -0.095 [-0.223, +0.037] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.563 [0.469, 0.647] | – | 0.482 [0.362, 0.589] | – | – | +0.081 [-0.065, +0.221] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.506 [0.414, 0.605] | – | 0.504 [0.365, 0.628] | – | – | +0.003 [-0.120, +0.158] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.522 [0.429, 0.623] | – | 0.504 [0.415, 0.596] | – | – | +0.018 [-0.093, +0.146] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.454 [0.343, 0.572] | – | 0.511 [0.421, 0.606] | – | – | -0.057 [-0.172, +0.070] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.550 [0.460, 0.639] | – | 0.496 [0.403, 0.589] | – | – | +0.054 [-0.037, +0.149] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.502 [0.428, 0.580] | – | 0.475 [0.387, 0.568] | – | – | +0.027 [-0.077, +0.129] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.492 [0.397, 0.581] | – | 0.539 [0.440, 0.631] | – | – | -0.047 [-0.193, +0.094] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.542 [0.459, 0.616] | – | 0.518 [0.441, 0.591] | – | – | +0.024 [-0.101, +0.141] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.547 [0.471, 0.628] | – | 0.561 [0.417, 0.688] | – | – | -0.014 [-0.156, +0.139] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.557 [0.473, 0.630] | – | 0.554 [0.417, 0.675] | – | – | +0.003 [-0.116, +0.137] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.462 [0.380, 0.552] | – | 0.576 [0.429, 0.703] | – | – | -0.114 [-0.242, +0.039] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.581 [0.479, 0.677] | – | 0.547 [0.400, 0.677] | – | – | +0.035 [-0.109, +0.186] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.386 [0.311, 0.470] | – | 0.561 [0.436, 0.675] | – | – | -0.176 [-0.300, -0.045] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.496 [0.422, 0.572] | – | 0.619 [0.461, 0.759] | – | – | -0.122 [-0.242, +0.011] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.555 [0.460, 0.632] | – | 0.454 [0.355, 0.547] | – | – | +0.101 [-0.009, +0.193] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.599 [0.504, 0.686] | – | 0.454 [0.355, 0.547] | – | – | +0.145 [+0.025, +0.261] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.489 [0.384, 0.586] | – | 0.468 [0.381, 0.558] | – | – | +0.021 [-0.101, +0.148] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.458 [0.354, 0.556] | – | 0.447 [0.368, 0.528] | – | – | +0.011 [-0.103, +0.116] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.499 [0.408, 0.584] | – | 0.404 [0.327, 0.490] | – | – | +0.095 [-0.031, +0.223] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.512 [0.410, 0.614] | – | 0.461 [0.357, 0.570] | – | – | +0.051 [-0.079, +0.192] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.594 [0.536, 0.654] | – | 0.475 [0.394, 0.556] | – | – | +0.119 [+0.025, +0.218] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.475 [0.423, 0.535] | – | 0.489 [0.391, 0.581] | – | – | -0.014 [-0.120, +0.112] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.513 [0.454, 0.574] | – | 0.486 [0.409, 0.552] | – | – | +0.027 [-0.061, +0.120] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.562 [0.518, 0.605] | – | 0.489 [0.419, 0.558] | – | – | +0.073 [-0.008, +0.151] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.519 [0.479, 0.559] | – | 0.493 [0.438, 0.542] | – | – | +0.026 [-0.047, +0.106] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.517 [0.456, 0.588] | – | 0.486 [0.433, 0.536] | – | – | +0.031 [-0.054, +0.122] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.519 [0.486, 0.555] | – | 0.500 [0.442, 0.548] | – | – | +0.019 [-0.035, +0.087] | – | – |
| `v.end` | 518 / 362 | 280 | 0.481 [0.421, 0.547] | – | 0.493 [0.440, 0.541] | – | – | -0.011 [-0.086, +0.076] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.512 [0.410, 0.614]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.506, `m2.end` 0.542, `m3.end` 0.496, `m4.end` 0.512. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m4.t0`, `p1.pre`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.511 [0.481, 0.544]; activation minus words +0.000 [-0.039, +0.046]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.523 [0.497, 0.555]; activation minus words +0.034 [-0.023, +0.101]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

