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
| `m1.pre` | 257 / 183 | 139 | 0.472 [0.384, 0.568] | – | 0.500 [0.500, 0.500] | – | – | -0.028 [-0.116, +0.068] | – | – |
| `m1.t0` | 257 / 183 | 139 | 0.460 [0.391, 0.537] | – | 0.504 [0.477, 0.527] | – | – | -0.043 [-0.114, +0.032] | – | – |
| `m1.q1` | 257 / 183 | 139 | 0.446 [0.372, 0.523] | – | 0.399 [0.296, 0.507] | – | – | +0.047 [-0.092, +0.182] | – | – |
| `m1.q2` | 257 / 183 | 139 | 0.429 [0.333, 0.512] | – | 0.525 [0.444, 0.612] | – | – | -0.096 [-0.233, +0.023] | – | – |
| `m1.q3` | 257 / 183 | 139 | 0.466 [0.388, 0.552] | – | 0.482 [0.395, 0.569] | – | – | -0.016 [-0.132, +0.111] | – | – |
| `m1.end` | 257 / 183 | 139 | 0.506 [0.448, 0.572] | – | 0.504 [0.406, 0.606] | – | – | +0.003 [-0.116, +0.116] | – | – |
| `m2.pre` | 261 / 179 | 141 | 0.521 [0.436, 0.614] | – | 0.411 [0.311, 0.509] | – | – | +0.109 [+0.003, +0.240] | – | – |
| `m2.t0` | 261 / 179 | 141 | 0.536 [0.438, 0.624] | – | 0.411 [0.303, 0.519] | – | – | +0.125 [+0.055, +0.190] | – | – |
| `m2.q1` | 261 / 179 | 141 | 0.494 [0.413, 0.587] | – | 0.397 [0.296, 0.500] | – | – | +0.096 [+0.000, +0.186] | – | – |
| `m2.q2` | 261 / 179 | 141 | 0.538 [0.420, 0.662] | – | 0.404 [0.281, 0.528] | – | – | +0.133 [+0.036, +0.248] | – | – |
| `m2.q3` | 261 / 179 | 141 | 0.616 [0.525, 0.693] | – | 0.397 [0.289, 0.507] | – | – | +0.218 [+0.102, +0.342] | – | – |
| `m2.end` | 261 / 179 | 141 | 0.479 [0.402, 0.562] | – | 0.440 [0.304, 0.573] | – | – | +0.040 [-0.123, +0.212] | – | – |
| `m3.pre` | 257 / 183 | 139 | 0.524 [0.451, 0.597] | – | 0.504 [0.421, 0.587] | – | – | +0.020 [-0.076, +0.104] | – | – |
| `m3.t0` | 257 / 183 | 139 | 0.528 [0.434, 0.613] | – | 0.504 [0.421, 0.587] | – | – | +0.024 [-0.054, +0.101] | – | – |
| `m3.q1` | 257 / 183 | 139 | 0.518 [0.455, 0.586] | – | 0.518 [0.429, 0.600] | – | – | +0.000 [-0.100, +0.113] | – | – |
| `m3.q2` | 257 / 183 | 139 | 0.482 [0.402, 0.575] | – | 0.518 [0.414, 0.616] | – | – | -0.036 [-0.150, +0.069] | – | – |
| `m3.q3` | 257 / 183 | 139 | 0.387 [0.322, 0.448] | – | 0.511 [0.419, 0.614] | – | – | -0.124 [-0.230, -0.028] | – | – |
| `m3.end` | 257 / 183 | 139 | 0.430 [0.355, 0.520] | – | 0.475 [0.376, 0.574] | – | – | -0.045 [-0.159, +0.092] | – | – |
| `m4.pre` | 261 / 179 | 141 | 0.413 [0.335, 0.506] | – | 0.411 [0.302, 0.540] | – | – | +0.001 [-0.139, +0.134] | – | – |
| `m4.t0` | 261 / 179 | 141 | 0.438 [0.354, 0.532] | – | 0.411 [0.295, 0.544] | – | – | +0.027 [-0.133, +0.177] | – | – |
| `m4.q1` | 261 / 179 | 141 | 0.526 [0.449, 0.591] | – | 0.447 [0.338, 0.574] | – | – | +0.079 [-0.086, +0.225] | – | – |
| `m4.q2` | 261 / 179 | 141 | 0.433 [0.369, 0.487] | – | 0.461 [0.354, 0.589] | – | – | -0.028 [-0.174, +0.106] | – | – |
| `m4.q3` | 261 / 179 | 141 | 0.502 [0.421, 0.579] | – | 0.433 [0.327, 0.537] | – | – | +0.070 [-0.057, +0.178] | – | – |
| `m4.end` | 261 / 179 | 141 | 0.475 [0.388, 0.551] | – | 0.440 [0.324, 0.550] | – | – | +0.035 [-0.059, +0.113] | – | – |
| `p1.pre` | 518 / 362 | 280 | 0.499 [0.441, 0.558] | – | 0.514 [0.440, 0.587] | – | – | -0.016 [-0.107, +0.079] | – | – |
| `p1.end` | 518 / 362 | 280 | 0.490 [0.439, 0.539] | – | 0.496 [0.421, 0.567] | – | – | -0.006 [-0.077, +0.063] | – | – |
| `p2.pre` | 518 / 362 | 280 | 0.528 [0.481, 0.572] | – | 0.475 [0.384, 0.562] | – | – | +0.053 [-0.033, +0.142] | – | – |
| `p2.end` | 518 / 362 | 280 | 0.451 [0.388, 0.507] | – | 0.464 [0.372, 0.544] | – | – | -0.014 [-0.108, +0.083] | – | – |
| `p3.pre` | 518 / 362 | 280 | 0.506 [0.450, 0.558] | – | 0.493 [0.402, 0.573] | – | – | +0.013 [-0.080, +0.111] | – | – |
| `p3.end` | 518 / 362 | 280 | 0.450 [0.391, 0.506] | – | 0.500 [0.409, 0.580] | – | – | -0.050 [-0.145, +0.061] | – | – |
| `v.pre` | 518 / 362 | 280 | 0.461 [0.409, 0.513] | – | 0.489 [0.407, 0.558] | – | – | -0.028 [-0.098, +0.049] | – | – |
| `v.end` | 518 / 362 | 280 | 0.467 [0.419, 0.527] | – | 0.482 [0.388, 0.562] | – | – | -0.015 [-0.106, +0.102] | – | – |

Read off by the rules of section 9:

- **Q1, a signal at all?** At the last token of the room (`m4.end`): 0.475 [0.388, 0.551]. No: the interval includes 0.5.
- **Q2, where does it start?** Nowhere: the lower bound never stays above 0.5 for three positions in a row.
- **Q3, how early?** At the ends of the four room messages: `m1.end` 0.506, `m2.end` 0.479, `m3.end` 0.430, `m4.end` 0.475. First position at 0.75 or more: none.
- **Q4, more than the words?** Above both text baselines at: no position. Above the bag of words only: `m2.pre`, `m2.t0`, `m2.q1`, `m2.q2`, `m2.q3`. The bag of words' own onset: none.
  - In one number, the mean over the 24 room positions: activation 0.484 [0.461, 0.509]; activation minus words +0.026 [-0.029, +0.087]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
  - In one number, the mean over the 8 positions after the room: activation 0.481 [0.457, 0.502]; activation minus words -0.008 [-0.074, +0.064]; minus words by speaker –; minus neutral reader –. Not more than the words in this phase.
- **Layer and anchor for the causal phase (15.2):** no onset inside the room at any layer: the room interventions of section 15 have no anchor

