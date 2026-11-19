# Stage 2: checks of the probe results on the served tables

Written by `scripts/uninstructed_stage2_recheck.py`, which shares no code with the sweep.

## 1. Identical prompts

Layer 22, the token before a member's first room message. Relative difference between two members of a matched set:

- members who open the room (same text up to that token): 316 comparisons, median 0.79%, largest 1.11%, exactly equal in 19;
- members who answer (the first message differs): median 13.3%.

So identical prompts give the same activation up to the rounding of the GPU pass, and a score at `m1.pre` scatters around 0.5 instead of being exactly 0.5 as on the CPU test.

## 2. Cells of the sweep, computed again

| Contrast | Position | Layer | Units 1 / 0 | Pairs | Here | The sweep | Difference |
|---|---|---|---|---|---|---|---|
| vote | `v.pre` | 22 | 518 / 362 | 280 | 0.900 | 0.896 | +0.004 |
| vote | `p3.pre` | 22 | 518 / 362 | 280 | 0.839 | 0.843 | -0.004 |
| vote | `m4.end` | 22 | 261 / 179 | 141 | 0.645 | 0.638 | +0.007 |
| vote | `m2.end` | 22 | 261 / 179 | 141 | 0.681 | 0.688 | -0.007 |
| vote | `m1.q3` | 19 | 257 / 183 | 139 | 0.734 | 0.734 | +0.000 |
| vote | `m4.pre` | 21 | 261 / 179 | 141 | 0.759 | 0.759 | +0.000 |
| vote | `m1.pre` | 22 | 257 / 183 | 139 | 0.518 | 0.514 | +0.004 |
| hidden | `v.pre` | 22 | 291 / 355 | 188 | 0.926 | 0.920 | +0.005 |
| hidden | `p3.pre` | 22 | 291 / 355 | 188 | 0.819 | 0.819 | +0.000 |
| hidden | `m4.end` | 22 | 144 / 177 | 93 | 0.570 | 0.570 | +0.000 |

Largest difference: 0.007. The units and the pairs are the same in every cell.

## 3. The size of the J-space part

Layer 22.

| Position | Norm of the activation (median) | Norm of its J-space part (median) | Share of the squared norm (mean) |
|---|---|---|---|
| `o1.pre` | 323 | 31 | 1.0% |
| `o1.q2` | 301 | 29 | 0.9% |
| `o2.end` | 308 | 32 | 1.1% |
| `p1.pre` | 357 | 27 | 0.6% |
| `p3.end` | 282 | 35 | 1.5% |
| `v.pre` | 289 | 22 | 0.6% |
| `v.end` | 293 | 36 | 1.5% |

