# jspace-interp: J-space interpretability, k sweep and steering for the colluder study

Branch for the next J-space direction on NARCBench (Gemma-2-9B-it, instructed collusion). Started Nov 8, 2026.

## What is here
| Path | What it does | Status |
|---|---|---|
| `jspace_interp/contexts.py` | Context conditions: original, swap (honest system prompt + colluder-only blocks cut), nosys, prompt_only | tested |
| `jspace_interp/positions.py`, `extract.py` | Gemma chat layout with character spans; one forward pass reads last, eot, prevote, frac25/50/75, reply_mean, prompt_end | tested on tiny model |
| `jspace_interp/jacobian.py` | Exact Jacobian-lens estimation (jlens save format) and unembedding basis export | tested on tiny model |
| `jspace_interp/sweep.py`, `detectors.py`, `rows.py` | NNOMP once to k_max, nested k prefixes, random-subspace / random-atom / PCA / fixed-frame / shuffled arms, episode detectors and protocols | tested on tiny model |
| `jspace_interp/concepts.py`, `stats.py` | Frozen concept word lists (Nov 8), permutation tests, BH | used |
| `jspace_interp/steer.py` | Experiment 3: matched-prefix steering of the final vote, pre-registered bars; since Nov 8 (late) the prefix is run once and only the tail is re-run on a copy of its KV cache, and only the sign the bars use is run per role | tested on tiny model (cached = full passes to 0.0) |
| `jspace_interp/matched_vocab.py` | Matched-vocabulary control wired into scoring: out-of-fold (leave-one-Core-domain-out) role scores for warned honest, unwarned honest, colluders and control agents; agent AUROCs, echo ratio, episode detectors on warned vs control and collusion vs warned episodes; concept-atom mass of warned agents | tested on tiny model |
| `generation/hf_batched.py` | Thread-safe batched `generate(system, user)` so warned episodes run several at a time through one model | tested on tiny model |
| `jspace_interp/nov5_jspace_decompose.py` | Copy of the Nov 5 decomposition script, recovered from the local session | reference |
| `scripts/jspace_interp_text_baseline.py` | Text-only baselines, behaviour split, concept rates | **ran on real transcripts** |
| `scripts/jspace_interp_extract.py` | Extraction CLI (4 conditions x 3 splits: core, transfer, warned; `--export-basis` writes the unembedding basis + tokenizer.json) | needs GPU for Gemma |
| `scripts/jspace_interp_analyze.py` | Experiments 1 and 2 on saved activations; scores every arm on the warned episodes when `<root>/warned/<condition>/` exists; writes role directions for Experiment 3; no bootstrap on the random draws | CPU once activations exist |
| `scripts/jspace_interp_role_dir.py` | Role direction (k = 25 filled J component, colluders minus honest) and median ‖h_J‖ for a layer, without the full sweep | CPU |
| `scripts/jspace_interp_steer.py` | Experiment 3 | needs GPU for Gemma |
| `scripts/jspace_interp_gen_matched_vocab.py` | Matched-vocabulary control episodes (warned honest seats 0-1); `--workers`/`--batch` run episodes concurrently through the batched generator | needs GPU; dry run and a real tiny-model run tested |
| `scripts/runpod_launch_jspace_interp.py`, `scripts/runpod_jspace_interp_boot.sh`, `scripts/runpod_range_server.py` | One-pod rebuild on RunPod: estimate, preflight (balance, running pods, GPU stock, pushed SHA), launch, status, watch, pull (verified), terminate; the boot script runs every stage and the analysis as parallel CPU jobs | estimate/preflight run Nov 8 |
| `scripts/jspace_interp_convert_paper_cache.py` | Turns the Nov 5 Gemma caches (activations_gen.npz + metadata_gen.json) into this layout so Exp 1 and 2 (last token, original) run on CPU | tested on synthetic data |
| `scripts/jspace_interp_tiny_e2e.py` | Whole pipeline on a random tiny Gemma-2 (numbers meaningless) | ran |
| `generation/core_protocol.py` | Optional `system_prompt_hook` and `run_tag` on `run_scenario` (defaults unchanged) | used by matched-vocab |

## Results
- `results/jspace_interp/text_baseline/text_baseline.json`: real, Nov 8.
- `results/jspace_interp/matched_vocab/vocab_check.json`: real, Nov 8.
- `results/jspace_interp/tiny_pipeline_check/`: pipeline check only, random model (refreshed Nov 8 late with the warned-episode generation, extraction and matched-vocabulary analysis; `analysis_mv_sweep_results.json`).
- `results/jspace_interp/pod_runs/<pod>/`: launch record, pull record and pod logs of the RunPod rebuild (archives are git-ignored).

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
Needs HF_TOKEN for the gated Gemma repo and RUNPOD_API_KEY, both from the environment only. The paper draft is
`docs/jspace_interp/jspace_colluder_study.tex`.
