# Can J-space tell the colluders apart? A pre-registered separability analysis

Written 31 October 2026, before any J-space number was computed on the official NARCBench activations. The Gemma-2-9B numbers quoted as background were already in this repository (`results/transfer/TRANSFER_FULL_HARD_METRICS.md`, `docs/NEXT-SCIENCE.md`). Everything else in this document is a plan. The script that executes it is `scripts/jspace_separability.py`; the Gemma frozen-number statistics are `scripts/jspace_offline_stats.py`.

## 1. What is being asked

The weekend lane found, on Gemma-2-9B-it, that a probe trained inside J-space separates colluders from honest agents in-distribution (Core, leave-one-domain-out, W_U top-25 frames: primary 0.989) but does not transfer (Core discussion to Transfer final: 0.602 for W_U frames, 0.549 for Jacobian frames) while the complement does (0.898 / 0.891, against 0.902 for the full residual). Those are five numbers from one model with no uncertainty, no dimension-matched control, and no test of whether J-space on Transfer itself holds role information. "J-space does not carry the transferable signal" was stated from them. This analysis asks the question properly, in three parts.

**Q1, in distribution.** Does a probe restricted to the J-space of each activation separate colluder turns from honest turns, and collusion episodes from control episodes, when train and test share the Core distribution?

**Q2, transfer.** Does the J-space probe fit on Core carry over to the Transfer families, and is its failure specific to J-space or shared by any subspace of the same size?

**Q3, what fails.** If the Core-fit J probe does not transfer, is that because J-space on Transfer activations holds no role information (a probe fit within Transfer also fails), or because the J-space direction learned on Core is the wrong one for Transfer (a probe fit within Transfer succeeds)? And how much of the drop is the phase change (discussion turns to final turns) rather than the family change?

## 2. Definitions

Residual activation h in R^d at layer l, read at the last token of an agent turn with the agent's own response included ("gen" reading, the only reading any recorded number uses). W_U in R^{V x d} is the unembedding matrix; W-hat_i is row i normalized to unit length.

**J-frame of size k.** F_k(h) = span of the k rows W-hat_i with the largest scores s_i = <W-hat_i, h>. Q_k(h) is an orthonormal basis of that span (QR). The frame is chosen per activation, so it follows what that position is about to say.

**J-only and complement.** h_J = Q Q^T h and h_C = h - h_J. They are orthogonal and sum to h. FVE_J = |h_J|^2 / |h|^2 is the share of the activation's squared norm inside its frame.

**Arms.** An arm is a map from activations to vectors in R^d. The probe pipeline (below) is run from scratch on each arm's vectors; the arms never share a fitted direction.

| Arm | Vectors | What it tests |
|---|---|---|
| full | h | the paper's probe |
| J_k | h_J with each activation's own top-k frame | the sayable part |
| C_k | h_C, same frames | everything else |
| randsub_k | P h for one Haar-random k-dimensional subspace P, shared by every activation | a k-dimensional subspace with no relation to the data |
| C_randsub_k | h - P h | removal of an arbitrary k-dimensional subspace |
| shuffled_k | projection of h onto the top-k frame of a different, randomly chosen activation from the same split | a genuine J-frame, but not this activation's |
| C_shuffled_k | h minus that projection | |
| randrows_k | projection onto k uniformly random unembedding rows, drawn per activation | a frame built from W_U but with no alignment to h |
| C_randrows_k | the complement of that | |
| pca_k | projection onto the top-k principal components of the Core discussion activations (unsupervised, shared) | a data-aligned k-dimensional subspace that is not vocabulary-based |
| C_pca_k | the complement of that | |
| contrast_k | projection onto the top-k principal components of the Core per-episode colluder-minus-honest mean differences (supervised, shared) | how few supervised dimensions suffice |

Frame sizes k in {8, 16, 25, 50, 100}. The W_U frame is the logit-lens stand-in for the J-lens dictionary (Gurnee and Lindsey, arXiv:2607.15495); true Jacobian frames exist only for Gemma (rank-64 sketches at L19/21/23) and are reported there as a secondary variant, not for the official models.

**Probe pipeline** (unchanged from `scripts/train_collusion_probe.py` and `scripts/eval_transfer_from_core.py`, which mirror `probes/reproduce.py` of the NARCBench repository). Role direction w: balanced logistic regression (C = 1) on standardized public discussion turns of collusion episodes, colluder = 1, honest = 0. Agent score: w applied to the agent's mean discussion vector (Core) or final-turn vector (Transfer). Episode detectors from the four sorted scores: Peak, Split, and an Asymmetry logistic regression (C = 0.01) on [variance, top minus bottom, split, mean] fit on training episodes. Episode AUROC is collusion versus control. Primary = mean of the three. Band mean = mean over the cached layers.

## 3. Data

| Lane | Model | Core | Transfer | Layers | W_U | Where it runs |
|---|---|---|---|---|---|---|
| Official | Qwen3-32B-AWQ | 31 + 31 episodes, 17 domains | 36 + 36, 6 families, final turns only | 26 to 30 | `Qwen/Qwen3-32B-AWQ` lm_head | this machine |
| Official | DeepSeek-R1-Distill-Qwen-32B | 28 + 28 | 36 + 36, final and discussion turns | 26 to 30 | its lm_head | this machine |
| Official | GPT-OSS-20B | 18 + 18, 13 domains | 13 + 13, 3 families | 10 to 14 | its lm_head | this machine |
| Replication | Gemma-2-9B-it | 50 + 50, 17 domains | 36 + 36, 6 families | 19 to 23 | tied embed_tokens | activations are not in git; the notebook or a pod re-extracts them |

Official activations are `aaronrose227/narcbench` on Hugging Face (`activations/<model>/<tier>/activations_gen.npz` + `metadata_gen.json`, float32, rows aligned, the paper's own extraction). Llama-3.1-70B is skipped: its unembedding is gated. Metadata rows with `channel == "thinking"` or `"private"` never enter a fit or an evaluation. Zero rows (turns without a recorded response) are dropped from fits.

## 4. Pre-registered analyses

**A1 (Q1). Core leave-one-domain-out**, every arm, k = 25 at every cached layer; the full k sweep at the centre layer. Report per-fold AUROCs so arms can be compared fold by fold on identical folds.

**A2 (Q2). Core discussion to Transfer final**, every arm, every k, every layer. Primary outcome: the band-mean primary AUROC.

**A3 (Q3a). Within-Transfer leave-one-family-out**, every arm, every k, every layer: fit the role direction on Transfer final turns of the five other families (collusion episodes only), score the held-out family. This is a Transfer in-distribution probe, so it tells whether J-space on Transfer activations holds role information at all.

**A4 (Q3b). Phase versus family decomposition.** (a) Core discussion to Core final, leave-one-domain-out: the phase changes, the family does not. (b) Core discussion to Transfer discussion where Transfer discussion turns exist (DeepSeek, GPT-OSS; not Qwen3, whose Transfer cache holds finals only; Gemma has them in the `RUNPOD` cache): the family changes, the phase does not. Arms full, J_25, C_25, randsub_25, pca_25.

**A5. Geometry.** FVE_J per arm and k; the share |Q^T w_full|^2 of the full probe direction (expressed in raw residual coordinates) inside each activation's frame against k/d; |cos| between the full direction and the J-only and complement directions; the most frequent frame tokens of colluder and honest turns.

## 5. Inference

**Transfer (A2, A3).** The unit is the episode. Paired bootstrap: resample episodes with replacement, stratified by family and mode, B = 2000; recompute every arm's Peak, Split and Asymmetry AUROC on the same resample; report the 95 percent percentile interval of each arm's primary and of the paired differences C_k minus J_k, J_k minus randsub_k, J_k minus shuffled_k, J_k minus pca_k, C_k minus full. Chance: a permutation test that permutes the episode labels within family (B = 2000) gives a p-value for each arm's primary being above 0.5; the Core-fit direction does not depend on Transfer labels, so the permutation is exact for Peak and Split.

**Core (A1, A4a).** Folds are the unit. Report mean and standard deviation over folds and the paired per-fold difference between arms with a sign test.

**Frozen Gemma numbers.** No per-episode scores were kept, so `scripts/jspace_offline_stats.py` attaches Hanley-McNeil standard errors (36 collusion, 36 control episodes) to each recorded AUROC and tests each J-only arm against 0.5 and against its complement with the normal approximation, treating arms as independent (conservative for the paired difference).

## 6. Readings fixed in advance

Let full, J, C, R (randsub), S (shuffled), P (pca) denote band-mean primary AUROCs at k = 25.

1. **J-space separates colluders in distribution** if, in A1, J_25 is at least 0.90 and the per-fold difference full minus J_25 is below 0.05 in mean. Otherwise, in distribution J-space is a weaker carrier than the full residual.
2. **The transferable signal is not in J-space** if, in A2, (i) the bootstrap interval of C minus J excludes zero and the point estimate is at least 0.15, (ii) the interval of J minus R contains zero or J is below R, and (iii) the interval of C minus full contains zero. Condition (ii) is what the frozen Gemma result lacks: it separates "J-space specifically fails" from "any 25 dimensions fail". If J is clearly above R and S but still far below C, the reading is "J-space carries a part of the signal, the complement carries most of it".
3. **"Complement equals full" is uninformative on its own** if C_randsub equals full within the bootstrap interval; this is expected and is stated so that the complement result is not read as evidence about J-space. *Amendment, 31 October 2026, after the Qwen3-32B run:* "equals" is read as "the interval of C_randsub minus full contains zero, or the difference is smaller than 0.02 in absolute value". On Qwen3 the difference was +0.006 with interval [+0.001, +0.011]: removing 25 random dimensions out of 5,120 changed the band mean by half a point and the interval missed zero by 0.001, which the rule as first written would have called "unexpected". The tolerance is recorded here, before the DeepSeek run finished, and the code applies it to every model.
4. **Role information is present in J-space on Transfer but the Core direction does not transfer** if, in A3, J_25 within Transfer reaches at least 0.80 while in A2 it is at most 0.65. **Role information is absent from J-space on Transfer finals** if A3's J_25 is also near its permutation null. These two readings are mutually exclusive and are the main new content of the analysis.
5. **Phase versus family.** If A4a's J_25 (Core discussion to Core final) drops as much as A2's, the loss is driven by the phase change: the final turn says different things, so its frame holds different directions. If A4a holds up and A4b drops, the loss is driven by the family change.
6. **Dimension is not the explanation** if contrast_8 or pca_k for small k reaches the full probe's transfer AUROC while J_k at the same k does not.

A reading that no official model and Gemma agree on is reported as model-dependent, not as a finding.

## 7. What this analysis does not do

It does not estimate Jacobians for the official models (no GPU here), does not use the Stego tier, does not touch Llama-3.1-70B, and does not make a causal claim. It does not change the probe pipeline; a different probe (for example, one fit directly on episode labels) is a different study. k = 25 is the primary frame size because it is the one the frozen Gemma numbers used; the sweep is reported in full.

## 8. Outputs

`results/jspace/<model>/summary.json` (every number with its interval), `results/jspace/<model>/RESULTS.md` (tables), `results/jspace/<model>/episode_scores.json` (per-episode Peak, Split and Asymmetry scores per arm, layer and k, so the bootstrap can be rerun without the activations), and `results/jspace/RESULTS.md` across models. Figures go to `results/jspace/figures/`.
