# RunPod sketch — Gemma Core regen + static presenter

## Roles

| Workload | Needs GPU? | Notes |
|----------|------------|-------|
| **Regen** (`scripts/regen_gemma_core.py`) | Yes | HF transformers (or vLLM later). Writes `upstream/scenarios/gemma2_*/core/<run_id>/`. |
| **Presenter** (`app/server.py`) | No | Static replay of finished transcripts. CPU / tiny pod / local :8765. |

Do **not** keep a GPU pod warm just to host the demo.

## Balance / existing tooling

- Secrets: `/home/box/.secrets/interp-explorer.env` (`RUNPOD_API_KEY`, optional `HF_TOKEN`).
- GraphQL helper (browser UA): `/workspace/collusion-exp/scripts/runpod_graphql.py`.
- Shared volume (HF cache + explorer venv): `cr1kjtzjs2`.
- Prior job launchers: `collusion-exp/scripts/runpod_launch_*.py`, `runpod-study` job pods.

Estimate-first launcher (default does **not** spend):

```bash
cd /workspace/narcbench-core-demo
python3 scripts/runpod_launch_gemma_core.py --smoke --model google/gemma-2-2b-it
# after Andy approves:
python3 scripts/runpod_launch_gemma_core.py --smoke --model google/gemma-2-2b-it --launch
```

## Cost estimates (ballpark)

Protocol turns per scenario×mode ≈ 18 upper bound (2 private + 12 public + 4 final; control skips private).

| Job | Scenarios | Calls (upper) | GPU | ≈ hours | ≈ USD |
|-----|-----------|---------------|-----|---------|-------|
| **2B smoke** | 3 × 2 | ~108 | RTX 2000 Ada (~$0.24/h) | 0.3–0.5 | **~$0.10–0.15** |
| **2B full** | 50 × 2 | ~1800 | same | 3–5 | **~$0.80–1.40** |
| **9B smoke** | 3 × 2 | ~108 | ≥24–32 GB (A40/4090/…) | 1–1.5 | **~$0.50–1.10** |
| **9B full** | 50 × 2 | ~1800 | same | 10–16 | **~$5–12** |

Rates from prior runpod-study notes; confirm live stock/pricing before launch. Model weights should hit the shared volume cache after first pull (gemma-2-2b ~8–11 GB peak).

## Box status

This executor box has **no NVIDIA GPU** (`nvidia-smi` missing). Local smoke must use RunPod (or another GPU host). Use `--dry-run` only to validate directory shape / indexer — stubs are not science.

## Fetch results back

Pod boot packs `upstream/scenarios` into `results.tgz` on the status server. After job completes, copy into:

```text
/workspace/narcbench-core-demo/upstream/scenarios/gemma2_2b/core/<run_id>/
```

Then:

```bash
python3 scripts/build_index.py
# model picker will enable gemma2_2b when paired runs exist
python3 app/server.py   # CPU is enough
```

## Full / 9B — next commands (after smoke OK)

```bash
# 2B full (ask before spending ~$1)
python3 scripts/runpod_launch_gemma_core.py --full --model google/gemma-2-2b-it   # estimate
python3 scripts/runpod_launch_gemma_core.py --full --model google/gemma-2-2b-it --launch --max-minutes 300

# 9B smoke
python3 scripts/runpod_launch_gemma_core.py --smoke --model google/gemma-2-9b-it
python3 scripts/runpod_launch_gemma_core.py --smoke --model google/gemma-2-9b-it --launch --max-minutes 120
```

Presenter hosting: any CPU machine or RunPod **CPU**/network-volume pod serving `app/server.py` on :8765 — no GPU.
