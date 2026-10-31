# Can J-space tell the colluders apart? Results across four models

Written 31 October 2026. Design (fixed before these numbers existed): `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md`. Every number below is in `results/jspace/<model>/summary.json` and the generated `CROSS_MODEL_TABLES.md`; the per-model `RESULTS.md` files carry the full tables. Figures: `figures/`. The Gemma-2-9B numbers are the frozen 1 October ones with Hanley–McNeil intervals (`gemma2_9b_frozen/FROZEN_STATS.md`); the other three models are the paper's own activations (`aaronrose227/narcbench`), analysed on a CPU with the repository's probe code.

## The answer in one paragraph

J-space, defined as the span of the 25 unembedding directions an activation is most aligned with, can separate colluders from honest agents only where everything else can too. In distribution on Core, the J-only probe reaches 0.991 on Qwen3-32B and 0.989 on Gemma-2-9B, but so does a random 25-dimensional subspace (0.988 on Qwen3): the Core signal is strong enough that any 25 directions carry it. Off distribution it is absent from J-space in every way the design could ask. Fit on Core and evaluated on Transfer, the J-only probe sits at 0.513, 0.500 and 0.568 on Qwen3, DeepSeek-R1 and GPT-OSS (permutation p = 0.31, 0.52, 0.13), indistinguishable from a random subspace of the same size and from another activation's frame, while the complement of J-space equals the full residual to within a point or two (0.775 vs 0.773, 0.729 vs 0.733, 0.725 vs 0.748). Fit *within* Transfer with a family held out, J-space still only reaches 0.54 to 0.60 against 0.80 to 0.90 for its complement, and no more than a random subspace. The phase change alone (Core discussion turns to Core final turns, same families) already removes most of what J-space had, and on DeepSeek the family change alone does the same. The reason is geometric: at these depths the top-25 frames are nearly the same fixed set of unembedding rows for every turn, those rows are formatting or junk tokens shared by both roles, and the probe direction has no more weight inside them than a random direction does. The frozen Gemma claim ("J-localization does not transfer") therefore stands, now with controls, uncertainty, and three more models; what it means is narrower than "verbalizable features miss collusion": the W_U top-k frame at mid-depth is not where the transferable role signal lives, and in-distribution J separability is not evidence of anything J-specific.

## 1. What was measured

Twelve arms per activation (design section 2): the full residual; J-only and its complement at k = 8, 16, 25, 50, 100; and the controls that the frozen result lacked: a Haar-random fixed k-dimensional subspace and its complement, the projection onto *another* activation's frame (shuffled), onto k random unembedding rows, onto the top-k principal components of the Core activations (unsupervised), and onto the top-k principal components of the Core colluder-minus-honest episode differences (supervised). The paper's probe pipeline is rerun from scratch on each arm. Five analyses: A1 Core leave-one-domain-out; A2 Core discussion → Transfer final (the frozen protocol); A3 within-Transfer leave-one-family-out; A4 the phase-versus-family split; A5 geometry. Inference is a paired bootstrap over episodes and a within-family permutation test (B = 2000 each).

| | Qwen3-32B-AWQ | DeepSeek-R1-Distill-Qwen-32B | GPT-OSS-20B | Gemma-2-9B-it (frozen) |
|---|---|---|---|---|
| layers | 26–30 | 26–30 | 10–14 | 19–23 |
| Core episodes (domains) | 62 (14) | 56 (14) | 36 (13) | 100 (17) |
| Transfer episodes (families) | 72 (6), finals only | 70 (6), finals + discussion | 26 (3), finals + discussion | 72 (6) |
| full-activation Core LDO primary | 1.000 | 1.000 | 1.000 | 1.000 |
| full-activation Core → Transfer | 0.773 [0.692, 0.847] | 0.733 [0.669, 0.794] | 0.748 [0.644, 0.838] | 0.902 [0.829, 0.975] |

The paper's in-distribution baseline (1.00 for every model) reproduces exactly. Transfer is harder for the official models than for the Gemma replication; the J-space question is asked relative to each model's own full-residual ceiling.

## 2. Answers to the three questions

### Q1. In distribution: yes, but so does any subspace of that size

Core leave-one-domain-out, k = 25, band mean:

| model | full | J-only | complement | random subspace | PCA | shuffled frame |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-32B | 1.000 | 0.991 | 1.000 | 0.988 | 1.000 | 0.580 |
| DeepSeek-R1-32B | 1.000 | 0.906 | 1.000 | 0.974 | 0.990 | 0.779 |
| GPT-OSS-20B | 1.000 | 0.784 | 1.000 | 0.800 | 0.998 | 0.671 |
| Gemma-2-9B | 1.000 | 0.989 | 1.000 | — | — | — |

Reading R1 fires on Qwen3 and Gemma (J within 0.05 of full) and not on DeepSeek or GPT-OSS. In no model does J-space beat a random subspace of 25 dimensions in distribution; on DeepSeek and GPT-OSS it is below one. The fold-paired difference full minus J is +0.02 (5 of 14 folds, sign test p = 0.06) on Qwen3, +0.09 on DeepSeek and +0.13 on GPT-OSS. So the first half of the frozen Gemma statement ("J-space can separate colluders in distribution") is true and uninformative: it is a statement about how easy Core is, not about J-space.

### Q2. Core → Transfer: the transferable signal is not in J-space, on all four models

k = 25, band mean, 95% bootstrap intervals over episodes, permutation p for "above chance":

| model | full | J-only | p(J) | complement of J | random 25-dim subspace | p(random) | another activation's frame | PCA k = 25 |
|---|---|---|---:|---|---|---:|---|---|
| Qwen3-32B | 0.773 [0.692, 0.847] | **0.513** [0.459, 0.564] | 0.31 | **0.775** [0.696, 0.846] | 0.508 [0.446, 0.565] | 0.40 | 0.506 [0.461, 0.550] | 0.784 [0.704, 0.857] |
| DeepSeek-R1-32B | 0.733 [0.669, 0.794] | **0.500** [0.435, 0.566] | 0.52 | **0.729** [0.666, 0.792] | 0.550 [0.492, 0.604] | 0.06 | 0.513 [0.465, 0.562] | 0.680 [0.617, 0.741] |
| GPT-OSS-20B | 0.748 [0.644, 0.838] | **0.568** [0.477, 0.660] | 0.13 | **0.725** [0.621, 0.818] | 0.611 [0.491, 0.725] | 0.05 | 0.563 [0.493, 0.633] | 0.685 [0.604, 0.760] |
| Gemma-2-9B (frozen) | 0.902 [0.829, 0.975] | **0.602** [0.471, 0.733] | 0.13 (two-sided) | **0.898** [0.823, 0.972] | — | — | — | — |

Paired differences on the same episodes:

| model | complement − J | J − random subspace | J − shuffled frame | complement − full | complement of random − full |
|---|---|---|---|---|---|
| Qwen3-32B | +0.262 [+0.172, +0.345] | +0.005 [−0.053, +0.066] | +0.007 [−0.067, +0.077] | +0.002 [−0.019, +0.022] | +0.006 [+0.001, +0.011] |
| DeepSeek-R1-32B | +0.229 [+0.161, +0.292] | −0.050 [−0.120, +0.013] | −0.013 [−0.083, +0.053] | −0.004 [−0.017, +0.009] | +0.000 [−0.002, +0.003] |
| GPT-OSS-20B | +0.157 [+0.073, +0.238] | −0.043 [−0.169, +0.090] | +0.005 [−0.094, +0.115] | −0.022 [−0.043, +0.000] | +0.004 [−0.004, +0.011] |
| Gemma-2-9B (frozen, independent-arms approx.) | +0.296 [+0.145, +0.446] | — | — | −0.005 [−0.109, +0.100] | — |

All three pre-registered conditions of reading R2 hold on every official model: (i) the complement beats J-only by at least 0.15 with an interval that excludes zero; (ii) J-only is not above a random subspace of the same dimension (nor above another activation's frame, nor above random unembedding rows); (iii) the complement equals the full residual. Reading R3 also holds: the complement of a *random* 25-dimensional subspace equals the full residual on every model (the Qwen3 difference of +0.006 is why the equality rule carries a ±0.02 tolerance, recorded as an amendment in the design). So the half of the frozen result that reads "the complement carries the signal" was never evidence about J-space: removing 25 of 5,120 dimensions changes nothing, whichever 25 they are. The evidence is entirely in the J-only arm, which is at chance, and the controls now show that it is at chance *because* it is a 25-dimensional subspace unrelated to the signal, not because something sayable was removed.

The frame size does not matter: J-only stays between 0.50 and 0.63 for k from 8 to 100 on every model (table 4 of `CROSS_MODEL_TABLES.md`), while the share of the activation's squared norm inside the frame rises from 1% to 7%.

### Q3a. Within Transfer: J-space holds at most what a random subspace holds

Leave-one-family-out inside Transfer, k = 25, pooled held-out AUROC:

| model | full | J-only | p(J) | complement | random subspace | p(random) | J − random |
|---|---:|---|---:|---:|---|---:|---|
| Qwen3-32B | 0.924 | 0.600 [0.545, 0.656] | 0.001 | 0.904 | 0.555 [0.506, 0.601] | 0.02 | +0.046 [−0.012, +0.105] |
| DeepSeek-R1-32B | 0.827 | 0.539 [0.497, 0.581] | 0.04 | 0.800 | 0.556 [0.499, 0.612] | 0.04 | −0.017 [−0.079, +0.040] |
| GPT-OSS-20B | 0.777 | 0.556 [0.480, 0.628] | 0.09 | 0.770 | 0.672 [0.587, 0.754] | 0.001 | −0.116 [−0.230, +0.001] |

This is the analysis the frozen result could not distinguish. The answer is the second of the two readings in design item 4, softened: on GPT-OSS the within-Transfer J probe is at its permutation null; on Qwen3 and DeepSeek it is a little above chance (0.60 and 0.54) but no better than a random 25-dimensional subspace, and 0.26 to 0.30 below its complement. There is no role information in J-space on Transfer finals that a Core-fit direction merely failed to pick up; there is essentially none to pick up.

### Q3b. Phase versus family: either shift alone removes it

k = 25, band mean (full / J-only / complement):

| model | Core disc → Core disc (A1) | Core disc → Core final (phase changes) | Core disc → Transfer disc (family changes) | Core disc → Transfer final (both) |
|---|---|---|---|---|
| Qwen3-32B | 1.000 / 0.991 / 1.000 | 1.000 / 0.700 / 1.000 | no Transfer discussion turns in the cache | 0.773 / 0.513 / 0.775 |
| DeepSeek-R1-32B | 1.000 / 0.906 / 1.000 | 0.999 / 0.579 / 0.997 | 0.942 / 0.560 / 0.939 | 0.733 / 0.500 / 0.729 |
| GPT-OSS-20B | 1.000 / 0.784 / 1.000 | 0.924 / 0.507 / 0.923 | 0.606 / 0.544 / 0.550 (few turns) | 0.748 / 0.568 / 0.725 |

Keeping the families and changing only the turn read (discussion to final) takes J-only from 0.99 to 0.70 on Qwen3 and from 0.91 to 0.58 on DeepSeek while the full residual stays at 1.00. Keeping the phase and changing only the family (DeepSeek, which has Transfer discussion turns) takes it from 0.91 to 0.56 while the full residual stays at 0.94. The pre-registered rule, which compares only the phase drop with the total, says "mostly phase" on all three; the DeepSeek row shows that the family change alone is just as destructive. In-distribution J separability does not survive either shift.

## 3. Why: what the frames contain

At the read position (the token after the agent's closing `<end_of_turn>`), the last-token residuals of all turns point almost the same way: cosine with the mean activation 0.97 (Qwen3), 0.98 (DeepSeek), 0.98 (GPT-OSS); in Qwen3 45% of the squared norm sits in five coordinates. The top-25 unembedding directions are therefore nearly the same for every turn. Over the Core discussion turns, the 25 most frequent frame tokens fill 62% (Qwen3), 80% (DeepSeek) and 90% (GPT-OSS) of all frame slots, only 368, 84 and 68 distinct tokens appear in any frame at all, and two random turns share 32%, 58% and 73% of their frames (Jaccard). The tokens are `This`, `4`, `3`, `In`, `-`, `\n\n` on GPT-OSS (formatting and numbering), and undertrained glyphs and fragments on Qwen3 (`="\`, `prehensive`, `silently`, `浙江省`) and DeepSeek (`딮`, `텦`, `ᐕ`). Colluder and honest turns share these vocabularies (Jaccard of the top 100: 0.56, 0.82, 0.79).

Against that frame, the full probe direction (expressed in raw residual coordinates) has a median squared share of 0.006, 0.005 and 0.005 inside each activation's own frame, against 0.005, 0.007 and 0.011 for a random direction and k/d of 0.005 to 0.009: the role direction is no more "sayable" than a random one. The direction the complement arm learns has |cos| 0.93 to 0.97 with the full probe direction; the direction the J-only arm learns has 0.12, 0.02 and 0.002. Where the transferable signal does live is visible in the PCA arms: on Qwen3 the top 16 principal components of the Core activations transfer at 0.798, above the full residual's 0.773, and removing the top 25 components drops transfer to 0.478; on DeepSeek the top 100 components reach the full residual (0.736 vs 0.733) and their removal drops it to 0.515. The Transfer-stable role signal is in the high-variance directions of the residual stream, which are not the directions the unembedding reads out at this depth.

## 4. What this changes in the lane's claims

- "J-localization does not transfer" (1 October, Gemma, five numbers) is now a controlled result on four models: J-only at chance on Transfer with intervals, equal to dimension-matched and frame-matched controls, complement equal to full. The locked safety wording ("mostly non-J coordination signature") can stand.
- Two things the frozen result implied should be dropped. First, "complement ≈ full" is not evidence about J-space (R3). Second, Core in-distribution J separability (0.989 on Gemma) is not evidence that collusion is "in J-space" in any specific sense: a random 25-dimensional subspace does as well.
- The mechanism is narrower than a claim about verbalizable features. What was tested is the logit-lens stand-in for the J-lens (unit unembedding rows, top-k by alignment, at mid-depth, without the final norm), and at these depths that frame is dominated by a fixed set of junk or formatting directions. Whether a true J-lens dictionary (Jacobian frames, which on Gemma gave 0.549, the same answer) or a final-norm-corrected frame changes the picture is the exploratory check in section 6.

## 5. Limitations

- The W_U frame is a proxy for the J-lens; only Gemma has Jacobian frames (rank-64 sketches), and those agreed with the proxy. The official models have none.
- GPT-OSS has 26 Transfer episodes in 3 families; its intervals are wide and its "random subspace" arm happens to sit above J (0.611 vs 0.568, not significant). DeepSeek lost 2 Transfer episodes and 56 Core private rows to zero vectors in the published cache.
- The bootstrap resamples evaluation episodes with the Core-fit direction held fixed; variability from refitting the direction is not included (the fold-paired A1 numbers carry it for Core).
- The random fixed subspace is one draw per layer (seed 42); the shuffled-frame and random-row controls draw per activation.
- A4b exists only for DeepSeek and (thinly) GPT-OSS; the Qwen3 cache holds Transfer finals only. Gemma was not rerun: its caches are not in git; the notebook's section 7b runs this script on a re-extraction.
- The design's reading rules are coarse (fixed thresholds); the tables are the result, the readings are a summary.

## 6. Exploratory, post hoc: frames from the normed unembedding (E1)

Not pre-registered. Because the mid-depth frames above are dominated by rows that align with the shared mean direction of the residual stream, a second frame definition was run after the main results were in: the same procedure with the unembedding columns scaled by the final RMSNorm weight γ, i.e. the dictionary the logit lens actually reads (W_U diag γ), for all three official models (`results/jspace/<model>_gamma/`, written up in `E1_GAMMA.md`). It does not rescue J-space: J-only on Transfer moves to 0.542, 0.496 and 0.615 (from 0.513, 0.500, 0.568), still within its interval of the random-subspace control on every model (J − random +0.034 [−0.033, +0.108], −0.054 [−0.111, +0.003], +0.004 [−0.125, +0.134]); within Transfer it reaches 0.633, 0.538 and 0.614 against 0.915, 0.768 and 0.757 for the complement; the frames remain the same fixed junk or discourse-marker directions shared by both roles. On GPT-OSS (26 episodes) the complement-minus-J interval now includes zero (+0.099 [−0.030, +0.230]) and on Qwen3 the complement sits 0.024 below full; neither changes the reading. Everything above was fixed before E1 was run.

## 7. Files

| | |
|---|---|
| design | `docs/jspace/JSPACE_SEPARABILITY_DESIGN.md` |
| per model | `results/jspace/{qwen3_32b,deepseek_r1_32b,gpt_oss_20b}/{summary.json,RESULTS.md,episode_scores.json}` |
| frozen Gemma | `results/jspace/gemma2_9b_frozen/FROZEN_STATS.md` |
| cross-model tables | `results/jspace/CROSS_MODEL_TABLES.md` (`scripts/jspace_cross_model.py`) |
| figures | `results/jspace/figures/fig_k25_cross_model.png`, `fig_three_settings.png`, `fig_k_curves_<model>.png` |
| scripts | `scripts/jspace_separability.py`, `jspace_offline_stats.py`, `jspace_figures.py`, `download_narcbench_activations.py`, `fetch_unembed_rows.py`; tests in `tests/test_jspace_separability.py` |
