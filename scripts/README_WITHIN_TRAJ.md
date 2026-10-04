# Within-trajectory mid-private smoke (RunPod)

Core smoke for teacher-forced mid-private intervention (attention layer 22 and residual layer 21). **Code on the pod comes from git**, not from a tarball in the GraphQL create body.

The pod clones `https://github.com/andylikescodes/narcbench-core-demo` and checks out the SHA you pass, then runs `scripts/within_traj_mid_private.py`. Scenario transcripts and direction `.npy` files stay on network volume `cr1kjtzjs2`. Results are written on that volume. `RUNPOD_API_KEY` is read from the environment (or an existing secrets file). It is not hardcoded.

`--git-sha` must already be pushed to GitHub. The launcher refuses to build a create request that would put `JOB_PAYLOAD_B64`, a payload tarball, or script source in the create env.

## Estimate (no pod)

```bash
python3 scripts/runpod_launch_within_traj.py --git-sha <40-hex>
```

Prints hours and cost and writes `results/transfer_stable/WITHIN_TRAJ_MID_PRIVATE_ESTIMATE.json`. This does not create a pod.

## Launch

```bash
python3 scripts/runpod_launch_within_traj.py --launch --git-sha <40-hex> \
  --volume-run-dir /workspace/jobs/narcbench-data/gemma2_9b/core/20261001T012639Z \
  --volume-directions /workspace/jobs/narcbench-data/transfer_stable/directions
```

`--launch` still checks the cost gate (default $1.50), account balance, and that no other billable GPU pod is up. Defaults if you omit the path flags:

| Flag | Default on the volume |
|------|------------------------|
| `--volume-run-dir` | `/workspace/jobs/narcbench-data/gemma2_9b/core/20261001T012639Z` |
| `--volume-directions` | `/workspace/jobs/narcbench-data/transfer_stable/directions` |
| `--volume-out` | `/workspace/jobs/narcbench-results/within_traj_mid_private` |

Direction files used by the runner: `lr_role_attn_L22.npy`, `lr_role_L21.npy`, and optionally `lr_role_perp_mode_L21.npy`. The model cache is `HF_HOME=/workspace/jobs/narcbench-hf` on the volume.

Status: `https://<pod-id>-8765.proxy.runpod.net/status.json`. Each run writes `smoke_<utc>/` plus `latest.tgz` under the output directory.

## Image that already contains the SHA

```bash
python3 scripts/runpod_launch_within_traj.py --launch --git-sha <40-hex> \
  --image <registry/image:tag> \
  --image-repo /opt/narcbench-core-demo
```

Boot skips the clone and exits unless `git rev-parse HEAD` in that directory equals `--git-sha`.

## Offline protocol check

No GPU and no pod:

```bash
python3 scripts/within_traj_mid_private.py --dry-run --max-scenarios 6
```
