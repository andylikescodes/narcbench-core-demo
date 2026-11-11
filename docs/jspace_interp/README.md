# jspace-interp: J-space interpretability, k sweep and steering for the colluder study

Branch for the next J-space direction on NARCBench (Gemma-2-9B-it, instructed collusion). Started Nov 8, 2026.

## What is here
| Path | What it does | Status |
|---|---|---|
| `jspace_interp/contexts.py` | Context conditions: original, swap (honest system prompt + colluder-only blocks cut), nosys, prompt_only | tested |
| `jspace_interp/positions.py`, `extract.py` | Gemma chat layout with character spans; one forward pass reads last, eot, prevote, frac25/50/75, reply_mean, prompt_end | tested on tiny model |
| `jspace_interp/jacobian.py` | Exact Jacobian-lens estimation (jlens save format) and unembedding basis export | tested on tiny model |
| `jspace_interp/sweep.py`, `detectors.py`, `rows.py` | One pursuit to k_max with nested k prefixes (`method` = `nnomp`, `gp` or `gpfull`), the non-negative combination `j_recon`, random-subspace / random-atom / PCA / fixed-frame / shuffled arms, episode detectors and protocols. `gradient_pursuit_full` is gradient pursuit over the whole vocabulary; `TorchVocabSearch` moves its selection step to a torch device (the Mac's GPU, `mps`) and decides each pick in CPU arithmetic | tested on tiny model; whole-vocabulary pursuit checked against the one-activation reference and, on the device, against the CPU version on 160 real activations |
| `jspace_interp/concepts.py`, `stats.py` | Frozen concept word lists (Nov 8), permutation tests, BH | used |
| `jspace_interp/steer.py` | Experiment 3: matched-prefix steering of the final vote, pre-registered bars; since Nov 8 (late) the prefix is run once and only the tail is re-run on a copy of its KV cache, and only the sign the bars use is run per role | tested on tiny model (cached = full passes to 0.0) |
| `jspace_interp/matched_vocab.py` | Matched-vocabulary control wired into scoring: out-of-fold (leave-one-Core-domain-out) role scores for warned honest, unwarned honest, colluders and control agents; agent AUROCs, echo ratio, episode detectors on warned vs control and collusion vs warned episodes; concept-atom mass of warned agents | tested on tiny model |
| `generation/hf_batched.py` | Thread-safe batched `generate(system, user)` so warned episodes run several at a time through one model | tested on tiny model |
| `jspace_interp/nov5_jspace_decompose.py` | Copy of the Nov 5 decomposition script, recovered from the local session | reference |
| `scripts/jspace_interp_text_baseline.py` | Text-only baselines, behaviour split, concept rates | **ran on real transcripts** |
| `scripts/jspace_interp_extract.py` | Extraction CLI (4 conditions x 3 splits: core, transfer, warned; `--export-basis` writes the unembedding basis + tokenizer.json) | needs GPU for Gemma |
| `scripts/jspace_interp_analyze.py` | Experiments 1 and 2 on saved activations; scores every arm on the warned episodes when `<root>/warned/<condition>/` exists; writes role directions for Experiment 3; no bootstrap on the random draws. `--method gpfull [--device mps]` = whole-vocabulary gradient pursuit; `--decomp-cache` saves and reuses the pursuit (`--decompose-only`, `--require-cache`, `--k-max`); `--norm-kind none` leaves the final-norm gain out of the J-lens vectors; `--reuse-random-from` copies the random draws of an earlier sweep; each (layer, k) also writes `boot_*.npz`, the bootstrap draws of every arm | CPU once activations exist; the whole-vocabulary pursuit wants a GPU (about 2 min per layer of a cell on an M1) |
| `scripts/jspace_interp_role_dir.py` | Role direction (k = 25 filled J component, colluders minus honest) and median ‖h_J‖ for a layer, without the full sweep | CPU |
| `scripts/jspace_interp_steer.py` | Experiment 3 | needs GPU for Gemma |
| `scripts/jspace_interp_gen_matched_vocab.py` | Matched-vocabulary control episodes (warned honest seats 0-1); `--workers`/`--batch` run episodes concurrently through the batched generator | needs GPU; dry run and a real tiny-model run tested |
| `scripts/runpod_launch_jspace_interp.py`, `scripts/runpod_jspace_interp_boot.sh`, `scripts/runpod_range_server.py` | One-pod rebuild on RunPod: estimate, preflight (balance, running pods, GPU stock, pushed SHA), launch, status, watch, pull (verified), terminate; the boot script runs every stage and the analysis as parallel CPU jobs sized from the cgroup limits; `--mode steer` reruns only the steering chain | **ran Nov 9** (full run 7.7 h on an A40; steering rerun) |
| `scripts/jspace_interp_summarize.py` | SUMMARY.json, RESULTS.md and the paper's LaTeX table fragments from the sweep, steering and matched-vocabulary JSON; lists every planned cell that did not run | ran Nov 9 |
| `scripts/jspace_interp_verify_jlens.py` | Runs Anthropic's `jlens` code (`JacobianLens.load/transport`, `HFLensModel.unembed`) on saved activations and compares it with this repo's numpy lens arithmetic; writes `results/jspace_interp/jlens_check/jlens_check.json` | ran Nov 10: identical to float32 rounding |
| `scripts/jspace_interp_solver_check.py` | Scores the J-space arm on the headline cell under the readings of the paper's definition: non-negative OMP or gradient pursuit, span projection or non-negative reconstruction, J-lens vectors with or without the final-norm gain; `--extra` adds the coefficient refit and the whole-vocabulary search; `--merge` writes `results/jspace_interp/solver_check/solver_check.json` with paired differences to the whole-vocabulary reading | ran Nov 10–11 |
| `scripts/jspace_interp_merge_k.py` | Merges two runs of one cell that differ only in `--ks` (the pursuit is nested in k) into one `sweep_results.json`; checks that everything they share is identical | ran Nov 11 |
| `scripts/jspace_interp_paper_assets.py` | The paper's figures (`docs/jspace_interp/figs/fig_*.pdf`) and compact tables (`tables/tabp_*.tex`), read from SUMMARY.json, the sweep files and the steering files; `--numbers <file>` dumps the numbers the text quotes | ran Nov 10 |
| `scripts/jspace_interp_finish_cells.py` | Runs analysis cells a time cap cut, on CPU from the pulled activations; the summariser merges them | tested (dry run) |
| `scripts/jspace_interp_convert_paper_cache.py` | Turns the Nov 5 Gemma caches (activations_gen.npz + metadata_gen.json) into this layout so Exp 1 and 2 (last token, original) run on CPU | tested on synthetic data |
| `scripts/jspace_interp_tiny_e2e.py` | Whole pipeline on a random tiny Gemma-2 (numbers meaningless) | ran |
| `generation/core_protocol.py` | Optional `system_prompt_hook` and `run_tag` on `run_scenario` (defaults unchanged) | used by matched-vocab |

## Results
- `results/jspace_interp/gemma2_9b_gpfull/<condition>_<position>/sweep_results.json`: **the sweep the paper reports** (Nov 11, laptop GPU): gradient pursuit over the whole vocabulary, J-space arm = `J{k}_recon` (non-negative combination), span form = `J{k}_fill`; random draws copied from the Nov 9 files. `gemma2_9b_gpfull_nogain/original_last` is the headline cell with J-lens vectors that leave out the final-norm gain (random lens-atom null redrawn).
- `results/jspace_interp/gemma2_9b_gp/`: the same with the pursuit restricted to the 1,024 best-aligned tokens (Nov 10–11, laptop CPU; the four last-token cells and `original_prevote`); `gemma2_9b_gp_nogain/original_last` its no-gain cell. Kept as the comparison: the shortlist lowers the headline band mean by 0.05.
- `results/jspace_interp/gemma2_9b/<condition>_<position>/sweep_results.json`: **real, Nov 9 pod run** (pod n8tc4b9iwvm5lt, A40, commit 5c300ad): 16 files, 320 cells (original and swap at 7 positions, prompt_only at prompt_end, nosys at last; layers 19-23; k 8/25/50/100; 200 random subspaces and 200 random atom sets per cell). `SUMMARY.json`, `RESULTS.md` and `docs/jspace_interp/tables/tab_*.tex` come from `scripts/jspace_interp_summarize.py`.
- `results/jspace_interp/matched_vocab_runs/`: the 50 warned control episodes generated on the pod (`batch_meta.json` has the decoding settings).
- `results/jspace_interp/role_dirs/role_dirs.json`: role directions for steering (the `.npy` vectors are git-ignored; regenerate with `scripts/jspace_interp_role_dir.py` from the activations).
- `results/jspace_interp/steer/`: Experiment 3, **real, Nov 9** (steering-only pod lz7gdscc09o0on, `--mode steer`): `steer_core_swap_L22.json`, `steer_core_original_L22.json`, `steer_core_swap_L23.json` (400 Core final-vote prefixes each; original L23 skipped by the 4-hour cap). The four runs on the first pod had failed with CUDA OOM (fixed in bdd30cd). `steer_core_swap_L23_lensnull.json` (pod rt34fhh908ja7p, `--mode steer_null --random-kind lens`) repeats swap L23 against 20 random lens-atom directions instead of Gaussian ones.
- `results/jspace_interp/text_baseline/text_baseline.json`: real, Nov 8.
- `results/jspace_interp/matched_vocab/vocab_check.json`: real, Nov 8.
- `results/jspace_interp/tiny_pipeline_check/`: pipeline check only, random model (refreshed Nov 8 late with the warned-episode generation, extraction and matched-vocabulary analysis; `analysis_mv_sweep_results.json`).
- `results/jspace_interp/pod_runs/<pod>/`: launch record, pull record, status and job log of each pod (archives are git-ignored); `pod_logs/` has the per-stage logs. Activations live in `results/jspace_interp/acts/` (git-ignored, 2.3 GB) after `--pull`.

## GPU run, in order (what `scripts/runpod_jspace_interp_boot.sh` does on the pod)
```bash
python scripts/jspace_interp_gen_matched_vocab.py --model google/gemma-2-9b-it --all --workers 8 --batch 8
python scripts/jspace_interp_extract.py --model google/gemma-2-9b-it --splits warned --conditions original prompt_only \
  --export-basis data/j_basis/gemma2_9b --out results/jspace_interp/acts
python scripts/jspace_interp_extract.py --model google/gemma-2-9b-it --splits core transfer --conditions <one condition> \
  --out results/jspace_interp/acts            # original, swap, prompt_only, nosys
python scripts/jspace_interp_analyze.py --root results/jspace_interp/acts --lens <gemma-2-9b-it_jacobian_lens.pt> \
  --basis-dir data/j_basis/gemma2_9b --tokenizer data/j_basis/gemma2_9b/tokenizer.json --layers 19-23 \
  --ks 8,25,50,100 --positions <one position> --conditions <one condition> --n-random 200 \
  --out results/jspace_interp/gemma2_9b/<condition>_<position>   # one CPU job per condition x position
python scripts/jspace_interp_role_dir.py --root results/jspace_interp/acts --condition swap --position last --layers 22,23 \
  --lens <lens.pt> --basis-dir data/j_basis/gemma2_9b --out results/jspace_interp/role_dirs
python scripts/jspace_interp_steer.py --lens <lens.pt> --basis-dir data/j_basis/gemma2_9b --layer 22 --condition swap \
  --role-direction results/jspace_interp/role_dirs/role_dir_J25_swap_last_L22.npy --hj-norm <median ||h_J||>
```
Launch and pull with `scripts/runpod_launch_jspace_interp.py` (estimate by default; `--preflight`; `--launch --watch`; `--pull <pod>`).
Needs HF_TOKEN for the gated Gemma repo and RUNPOD_API_KEY, both from the environment only.

## Anthropic's `jlens`: what we use and what is ours
The reference implementation of the Jacobian lens is Anthropic's `jlens` (github.com/anthropics/jacobian-lens, v0.1.0,
one commit `581d398`, "reference implementation, not maintained"). It provides the lens fit (`jlens.fit`), the transport
(`JacobianLens.transport`, J_l h) and the readout (`HFLensModel.unembed`: final norm, LM head, softcap), plus a slice
visualisation. It contains **no J-space decomposition**, and the paper links no code for one.

| Part | Source |
|---|---|
| The lens J_l for Gemma-2-9B-it | Anthropic's `jlens`, as fitted and published by Neuronpedia (HF `neuronpedia/jacobian-lens`, 453 WikiText prompts, layers 0-40). Used as published; we fit no lens for Gemma. `jspace_interp/jacobian.py` is only for the tiny random model of the pipeline check. |
| Transport and readout | Re-implemented in numpy (`jspace_interp/nov5_jspace_decompose.py`) so the analysis runs on saved activations without the model. Checked against `jlens` with `scripts/jspace_interp_verify_jlens.py` on 480 activations: identical matrices and transport, logits within 2.8e-5 on a scale of 30, same top-1 and top-50 tokens for every activation. |
| J-lens vectors | Ours: v_i = J_l^T ((1 + gamma) * W_U[i]), the readout direction of token i under the official readout (checked in the same script). The paper writes "rows of W_U J_l". |
| J-space decomposition | Ours, from the paper's text, in the form of its Methods: gradient pursuit over the whole vocabulary, k = 25, J-space part = non-negative combination of the chosen vectors (the span form is reported next to it). `scripts/jspace_interp_solver_check.py` compares the readings on the headline cell: form and refit barely matter (0.80 to 0.82), a shortlist of 1,024 candidate tokens lowers the arm by 0.05 (0.771; the Nov 9 version of the paper had it), the solver does not matter given the search. |
| Probe, detectors, nulls, controls, steering | Ours. |

Not done: `JacobianLens.apply` on the model itself (needs the 9B model in memory); steering with role directions from the
present decomposition (the steering runs used the Nov 9 one). To install the package for the check: `pip install "git+https://github.com/anthropics/jacobian-lens@581d398"`.

## The paper
`docs/jspace_interp/jspace_colluder_study.tex` / `.pdf` (restructured Nov 10 for the Nov 11 group meeting; the draft
before that is commit `af7afae`). Since Nov 11 every J-space number comes from the whole-vocabulary sweep
(`gemma2_9b_gpfull`), except the steering; the Nov 10 version (commit `586495a`) had the shortlist decomposition. The
appendix "Changes since earlier drafts" lists what moved (headline J-space 0.773 -> 0.819; the verdict did not). A standard LaTeX article: abstract; introduction with a table of the seven findings;
background; the method in ten steps (what, why, details) and a section on what comes from Anthropic's `jlens`; seven
findings in order of importance, each opening with the question it answers; what did not run; limitations; conclusion;
references; appendices with the full tables, the earlier results and the list of changes. Set `\findingbreak` to
`\clearpage` in the preamble to start every finding on a new page.

Rebuild:
```bash
python scripts/jspace_interp_summarize.py --sweep-dir gemma2_9b_gpfull --primary recon   # SUMMARY.json, RESULTS.md, tables/tab_*.tex
python scripts/jspace_interp_verify_jlens.py --lens <lens.pt>   # optional: check against Anthropic's jlens
python scripts/jspace_interp_solver_check.py --lens <lens.pt> && python scripts/jspace_interp_solver_check.py --extra --lens <lens.pt> \
  && python scripts/jspace_interp_solver_check.py --merge   # optional
python scripts/jspace_interp_paper_assets.py --sweep-dir gemma2_9b_gpfull   # figs/fig_*.pdf, tables/tabp_*.tex (needs matplotlib)
cd docs/jspace_interp && tectonic jspace_colluder_study.tex
```
Every figure and every table in the findings is generated from the result files. Chart colours are the first three slots
of a palette checked for colour-vision deficiency: blue = full residual, orange = J-space, green = PCA; grey = random nulls.

Two things to know when reading the result files:
- The read position `prevote` is the last token of the reply text for the rows the probes use (Core discussion turns and
  Transfer final turns have no vote line, so the "token before the vote line" falls back to the last reply token). The three
  end positions `prevote`, `eot`, `last` are consecutive tokens.
- `exp1.*.top_tokens` in each `sweep_results.json` holds the 30 tokens with the smallest p per cell (`SUMMARY.json` keeps 15):
  token-level differences in atom selection by role. With the instruction in context these include scheme words
  ("subtly", "disinformation"); under swap and nosys no token differs on Transfer.
