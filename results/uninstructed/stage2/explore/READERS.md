# Stage 2, exploratory: the probe against stronger readers of the words

Added after the probe results were seen; it is not among the rules fixed in advance. Written by `scripts/uninstructed_stage2_readers.py`; every number is in `readers.json`.

Every number is a matched-prompt AUROC (0.5 = chance) on the pairs of the sweep, with a 95% interval from 2,000 resamples of the scenarios; a difference is paired. *Activation*: the probe on the member's activation, mean of the five layers. The readers see words only: Claude Fable 5.2's codes were made without the votes, and a small logistic regression is fit on them on the other domains.

## vote: members who vote for the stake option against members who do not

518 against 362 members, 280 pairs in 34 scenarios.

- Claude's reading of the room: 0.729 [0.625, 0.812]
- Claude's reading of the room and of the public position: 0.791 [0.710, 0.860]
- One feature: the room read as an agreement on the stake option: 0.691 [0.603, 0.773]

| Position | Activation | Activation minus Claude's reading of the room | Activation minus Claude's reading of the room and of the public position | Activation minus the one feature | Activation minus the neutral reader, joined | Neutral reader, joined |
|---|---|---|---|---|---|---|
| `p1.pre` | 0.719 [0.665, 0.767] | -0.010 [-0.085, +0.073] | -0.072 [-0.140, +0.004] | +0.028 [-0.051, +0.112] | +0.080 [+0.004, +0.149] | 0.639 [0.563, 0.717] |
| `p1.end` | 0.656 [0.595, 0.706] | -0.073 [-0.134, -0.003] | -0.135 [-0.185, -0.078] | -0.035 [-0.086, +0.020] | +0.042 [-0.027, +0.100] | 0.614 [0.543, 0.672] |
| `p2.pre` | 0.788 [0.741, 0.840] | +0.059 [-0.030, +0.164] | -0.003 [-0.081, +0.081] | +0.097 [-0.003, +0.203] | +0.203 [+0.128, +0.276] | 0.585 [0.520, 0.666] |
| `p2.end` | 0.722 [0.658, 0.785] | -0.006 [-0.096, +0.098] | -0.069 [-0.136, +0.004] | +0.031 [-0.068, +0.136] | +0.048 [-0.034, +0.122] | 0.674 [0.599, 0.750] |
| `p3.pre` | 0.809 [0.761, 0.864] | +0.081 [-0.005, +0.187] | +0.018 [-0.052, +0.105] | +0.118 [+0.031, +0.226] | +0.140 [+0.051, +0.227] | 0.669 [0.592, 0.751] |
| `p3.end` | 0.779 [0.697, 0.855] | +0.050 [-0.055, +0.175] | -0.012 [-0.092, +0.077] | +0.088 [-0.024, +0.215] | +0.082 [-0.010, +0.187] | 0.696 [0.616, 0.760] |
| `v.pre` | 0.864 [0.799, 0.921] | +0.135 [+0.052, +0.236] | +0.073 [-0.008, +0.159] | +0.173 [+0.087, +0.281] | +0.190 [+0.116, +0.265] | 0.674 [0.575, 0.770] |
| mean of the seven | 0.762 [0.728, 0.799] | +0.034 [-0.036, +0.122] | -0.029 [-0.082, +0.035] | +0.071 [-0.004, +0.160] | +0.112 [+0.063, +0.164] | 0.650 [0.588, 0.710] |

## hidden: members who argue for the other option in public: those who vote for the stake option all the same against those who do not

291 against 355 members, 188 pairs in 27 scenarios.

- Claude's reading of the room: 0.715 [0.592, 0.808]
- One feature: the room read as an agreement on the stake option: 0.673 [0.557, 0.766]

| Position | Activation | Activation minus Claude's reading of the room | Activation minus the one feature | Activation minus the neutral reader, joined | Neutral reader, joined |
|---|---|---|---|---|---|
| `p1.pre` | 0.656 [0.600, 0.713] | -0.059 [-0.150, +0.062] | -0.016 [-0.116, +0.113] | +0.067 [-0.013, +0.139] | 0.589 [0.515, 0.674] |
| `p1.end` | 0.593 [0.514, 0.653] | -0.123 [-0.209, -0.015] | -0.080 [-0.161, +0.021] | +0.055 [-0.047, +0.138] | 0.537 [0.451, 0.615] |
| `p2.pre` | 0.732 [0.669, 0.802] | +0.016 [-0.099, +0.160] | +0.059 [-0.070, +0.216] | +0.197 [+0.101, +0.301] | 0.535 [0.432, 0.643] |
| `p2.end` | 0.622 [0.534, 0.706] | -0.093 [-0.194, +0.018] | -0.051 [-0.173, +0.080] | +0.032 [-0.091, +0.128] | 0.590 [0.503, 0.690] |
| `p3.pre` | 0.770 [0.727, 0.821] | +0.055 [-0.050, +0.199] | +0.097 [-0.014, +0.245] | +0.147 [+0.037, +0.269] | 0.623 [0.523, 0.722] |
| `p3.end` | 0.714 [0.627, 0.800] | -0.002 [-0.114, +0.132] | +0.041 [-0.092, +0.207] | +0.067 [-0.022, +0.169] | 0.647 [0.562, 0.717] |
| `v.pre` | 0.873 [0.820, 0.917] | +0.158 [+0.060, +0.280] | +0.201 [+0.102, +0.326] | +0.233 [+0.135, +0.334] | 0.640 [0.531, 0.733] |
| mean of the seven | 0.709 [0.675, 0.742] | -0.007 [-0.091, +0.101] | +0.036 [-0.063, +0.161] | +0.114 [+0.050, +0.180] | 0.595 [0.521, 0.665] |

Reading: where a difference's interval includes zero, the probe is not told apart from that reader of the words. Claude's reading of the public position covers all three statements, so only at `v.pre` has it read no further than the activation.

