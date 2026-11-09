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
| `jspace_interp/steer.py` | Experiment 3: matched-prefix steering of the final vote, pre-registered bars | tested on tiny model |
| `jspace_interp/nov5_jspace_decompose.py` | Copy of the Nov 5 decomposition script, recovered from the local session | reference |
| `scripts/jspace_interp_text_baseline.py` | Text-only baselines, behaviour split, concept rates | **ran on real transcripts** |
| `scripts/jspace_interp_extract.py` | Extraction CLI (4 conditions x 2 splits) | needs GPU for Gemma |
| `scripts/jspace_interp_analyze.py` | Experiments 1 and 2 on saved activations | CPU once activations exist |
| `scripts/jspace_interp_steer.py` | Experiment 3 | needs GPU for Gemma |
| `scripts/jspace_interp_gen_matched_vocab.py` | Matched-vocabulary control episodes (warned honest seats 0-1) | needs GPU; dry run tested |
| `scripts/jspace_interp_convert_paper_cache.py` | Turns the Nov 5 Gemma caches (activations_gen.npz + metadata_gen.json) into this layout so Exp 1 and 2 (last token, original) run on CPU | tested on synthetic data |
| `scripts/jspace_interp_tiny_e2e.py` | Whole pipeline on a random tiny Gemma-2 (numbers meaningless) | ran |
| `generation/core_protocol.py` | Optional `system_prompt_hook` and `run_tag` on `run_scenario` (defaults unchanged) | used by matched-vocab |

## Results
- `results/jspace_interp/text_baseline/text_baseline.json`: real, Nov 8.
- `results/jspace_interp/matched_vocab/vocab_check.json`: real, Nov 8.
- `results/jspace_interp/tiny_pipeline_check/`: pipeline check only, random model.

## GPU run, in order
```bash
python scripts/jspace_interp_extract.py --model google/gemma-2-9b-it --out results/jspace_interp/acts
python scripts/jspace_interp_analyze.py --root results/jspace_interp/acts --lens <gemma-2-9b-it_jacobian_lens.pt> \
  --basis-dir <basis> --tokenizer <tokenizer.json> --layers 19-23 --out results/jspace_interp/gemma2_9b
python scripts/jspace_interp_gen_matched_vocab.py --all
python scripts/jspace_interp_steer.py --lens <lens.pt> --basis-dir <basis> --layer 22 --condition swap
```
Needs HF_TOKEN for the gated Gemma repo. The paper draft lives in the project's shared jspace-interp folder.
