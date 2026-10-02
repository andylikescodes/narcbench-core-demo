#!/usr/bin/env python3
"""Download ONLY NARCBench Core scenario JSON (no activations)."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "upstream"
REPO = "aaronrose227/narcbench"
PREFIX = "scenarios/qwen3_32b/core"
FILES = ("run_config.json", "agent_prompts.json", "results.json")


def main() -> None:
    api = HfApi()
    items = list(
        api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=PREFIX, recursive=False)
    )
    runs = sorted(getattr(i, "path", "").rsplit("/", 1)[-1] for i in items)
    print(f"{len(runs)} runs under {PREFIX}")
    n = 0
    for run in runs:
        for fn in FILES:
            repo_path = f"{PREFIX}/{run}/{fn}"
            dest = LOCAL / repo_path
            if dest.exists() and dest.stat().st_size > 0:
                continue
            hf_hub_download(REPO, repo_path, repo_type="dataset", local_dir=str(LOCAL))
            n += 1
    complete = sum(
        1
        for run in runs
        if all((LOCAL / PREFIX / run / f).exists() for f in FILES)
    )
    print(f"downloaded {n} new files; complete runs {complete}/{len(runs)}")
    print(f"local dir: {LOCAL / PREFIX}")


if __name__ == "__main__":
    main()
