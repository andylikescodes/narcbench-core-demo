# NARCBench Core demo

Zeabur presenter (no GPU): see [README_DEPLOY.md](README_DEPLOY.md).

## Within-trajectory mid-private smoke

RunPod code comes from git. The pod clones this repo and checks out a SHA you supply. Scenario transcripts and direction `.npy` files stay on the network volume. Details: [scripts/README_WITHIN_TRAJ.md](scripts/README_WITHIN_TRAJ.md).

Estimate only (does not create a pod):

```bash
python3 scripts/runpod_launch_within_traj.py --git-sha <40-hex>
```

Launch after that estimate looks right. The SHA must already be on GitHub:

```bash
python3 scripts/runpod_launch_within_traj.py --launch --git-sha <40-hex>
```
