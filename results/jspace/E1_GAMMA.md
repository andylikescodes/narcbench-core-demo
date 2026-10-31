# E1 (exploratory, post hoc): J-frames from the final-norm-scaled unembedding

Written 31 October 2026, after the pre-registered results in `RESULTS.md` were fixed. Not pre-registered; run because the mid-depth top-25 frames of the plain W_U dictionary turned out to be dominated by a fixed set of unembedding rows aligned with the shared mean direction of the residual stream (junk tokens on Qwen3 and DeepSeek, formatting tokens on GPT-OSS). The logit lens proper reads the residual through the final RMSNorm, so the directions it actually scores are the columns of W_U scaled by the norm's learned weight γ. E1 repeats the whole pipeline with that dictionary (`--unembed-scale final_norm_weight_fp32.npy`; γ fetched by range request from each model's last shard). Outputs: `results/jspace/<model>_gamma/`, tables `CROSS_MODEL_TABLES_E1_gamma.md`, figures `figures_e1/`.

## Core → Transfer at k = 25, band mean (A2)

| model | full | J-only, γ frames (p) | J-only, plain frames | complement, γ | random 25-dim subspace | another activation's γ frame | complement − J (γ) | J − random (γ) |
|---|---:|---|---:|---:|---:|---:|---|---|
| Qwen3-32B | 0.773 | 0.542 [0.495, 0.593] (0.07) | 0.513 | 0.749 [0.667, 0.824] | 0.508 | 0.542 | +0.206 [+0.109, +0.293] | +0.034 [−0.033, +0.108] |
| DeepSeek-R1-32B | 0.733 | 0.496 [0.435, 0.556] (0.56) | 0.500 | 0.734 [0.675, 0.793] | 0.550 | 0.546 | +0.237 [+0.165, +0.310] | −0.054 [−0.111, +0.003] |
| GPT-OSS-20B | 0.748 | 0.615 [0.501, 0.735] (0.05) | 0.568 | 0.715 [0.592, 0.819] | 0.611 | 0.517 | +0.099 [−0.030, +0.230] | +0.004 [−0.125, +0.134] |

Share of the activation's squared norm inside the γ frame at k = 25: 0.019, 0.032, 0.041 (plain: 0.032, 0.035, 0.061).

## The other settings at k = 25 (γ frames; plain in brackets)

| model | Core LDO J-only | within-Transfer LOFO J-only (p) | Core disc → Core final J-only | Core disc → Transfer disc J-only |
|---|---:|---|---:|---:|
| Qwen3-32B | 0.996 (0.991) | 0.633 [0.573, 0.685] (0.000) (0.600) | 0.663 (0.700) | — |
| DeepSeek-R1-32B | 0.952 (0.906) | 0.538 [0.481, 0.591] (0.11) (0.539) | 0.600 (0.579) | 0.511 (0.560) |
| GPT-OSS-20B | 0.802 (0.784) | 0.614 [0.542, 0.685] (0.009) (0.556) | 0.694 (0.507) | 0.519 (0.544) |

## Frames

The γ dictionary changes which rows dominate the frames but not their character. Qwen3's most frequent frame tokens remain fragments and foreign-script tokens (`="\`, `vücud`, `prehensive`, `coder`, `okemon`); DeepSeek's remain CJK and Cyrillic fragments (`ncmp`, `lük`, `こんにちは`, `此前`); GPT-OSS's become discourse markers (`Now`, `This`, `Conclusion`, `Furthermore`, `Additionally`, `---`). Colluder and honest frames still share most of their vocabulary (Jaccard of the top 100: 0.55, 0.75, 0.78). The full probe direction's median squared share inside the γ frames is 0.005, 0.004, 0.007 against 0.005, 0.005, 0.009 for a random direction.

## Reading

Scaling the dictionary by the final norm moves J-only on Transfer by +0.03, −0.00 and +0.05, leaves it within its interval of the random-subspace control on every model, and leaves the complement at or within 0.03 of the full residual. Two pre-registered conditions become marginal at this sample size: on GPT-OSS (26 Transfer episodes) the complement-minus-J interval now includes zero (+0.099 [−0.030, +0.230]); on Qwen3 the complement is 0.024 below full with an interval that excludes zero. Neither changes the substantive reading: with the dictionary the logit lens actually uses, the Core-fit J-space probe still does not transfer, the within-Transfer J-space probe still holds no more than a random subspace (0.63, 0.54, 0.61 against 0.56, 0.56, 0.67), and the complement still carries the signal. E1 is a robustness check on the frame definition, and the pre-registered readings in `RESULTS.md` stand.
