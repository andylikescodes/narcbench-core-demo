"""Guards for the git-boot within-traj launcher. Does not create a pod."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import runpod_launch_within_traj as launch  # noqa: E402

SHA = "a" * 40


def _args(**overrides):
    values = dict(
        git_sha=SHA,
        repo_url=launch.REPO_URL,
        image=launch.IMAGE,
        image_repo="",
        volume_id=launch.VOLUME_ID,
        volume_run_dir=launch.DEFAULT_VOLUME_RUN,
        volume_directions=launch.DEFAULT_VOLUME_DIRECTIONS,
        volume_out=launch.DEFAULT_VOLUME_OUT,
        hf_home=launch.DEFAULT_HF_HOME,
        model="google/gemma-2-9b-it",
        max_scenarios=6,
        k_frac=0.5,
        max_new_tokens=64,
        seed=0,
        max_minutes=90,
        grace_minutes=6,
        gpu=launch.GPU,
        cloud="SECURE",
    )
    values.update(overrides)
    return launch.argparse.Namespace(**values)


class CreateEnvTests(unittest.TestCase):
    def test_create_env_has_sha_and_no_payload(self):
        create_input = launch.build_create_input(_args(), "k" * 40)
        env = {item["key"]: item["value"] for item in create_input["env"]}
        self.assertEqual(env["JOB_GIT_SHA"], SHA)
        self.assertEqual(env["JOB_REPO_URL"], launch.REPO_URL)
        self.assertEqual(env["JOB_RUN_DIR"], launch.DEFAULT_VOLUME_RUN)
        self.assertEqual(env["JOB_DIRECTIONS"], launch.DEFAULT_VOLUME_DIRECTIONS)
        self.assertNotIn("JOB_PAYLOAD_B64", env)
        self.assertNotIn("JOB_BOOT_B64", env)
        self.assertNotIn("JOB_TGZ_NAME", env)
        blob = json.dumps(create_input)
        self.assertNotIn("JOB_PAYLOAD_B64", blob)
        self.assertNotIn("base64 -d", create_input["dockerArgs"])
        self.assertIn("git clone", create_input["dockerArgs"])
        self.assertIn("checkout --detach", create_input["dockerArgs"])
        self.assertIn("JOB_GIT_SHA", create_input["dockerArgs"])
        self.assertIn("runpod_within_traj_boot.sh", create_input["dockerArgs"])
        nbytes = launch.validate_create_input(create_input)
        self.assertLess(nbytes, 20_000)
        self.assertLess(nbytes, launch.MAX_CREATE_BODY_BYTES)

    def test_refuses_payload_b64_and_tarball_in_env(self):
        create_input = launch.build_create_input(_args(), "k" * 40)
        poisoned = json.loads(json.dumps(create_input))
        poisoned["env"].append({"key": "JOB_PAYLOAD_B64", "value": "H4sIAAAAAAAA"})
        with self.assertRaises(launch.LaunchRefused) as raised:
            launch.validate_create_input(poisoned)
        self.assertIn("JOB_PAYLOAD_B64", str(raised.exception))

        tar = json.loads(json.dumps(create_input))
        tar["env"].append({"key": "JOB_TGZ_B64", "value": "H4sI" + ("A" * 80)})
        with self.assertRaises(launch.LaunchRefused):
            launch.validate_create_input(tar)

        stuffed = json.loads(json.dumps(create_input))
        for item in stuffed["env"]:
            if item["key"] == "JOB_RUN_DIR":
                item["value"] = "\x1f\x8b" + ("x" * 32)
        with self.assertRaises(launch.LaunchRefused) as raised:
            launch.validate_create_input(stuffed)
        self.assertIn("tarball", str(raised.exception))

    def test_refuses_script_source_in_env(self):
        create_input = launch.build_create_input(_args(), "k" * 40)
        poisoned = json.loads(json.dumps(create_input))
        for item in poisoned["env"]:
            if item["key"] == "JOB_MODEL":
                item["value"] = "def run_scenario(model, tok): pass"
        with self.assertRaises(launch.LaunchRefused):
            launch.validate_create_input(poisoned)

    def test_image_checkout_skips_requiring_volume_for_repo_path(self):
        create_input = launch.build_create_input(
            _args(image="registry.example/narcbench:pinned", image_repo="/opt/narcbench-core-demo"),
            "k" * 40,
        )
        env = {item["key"]: item["value"] for item in create_input["env"]}
        self.assertEqual(env["JOB_IMAGE_REPO"], "/opt/narcbench-core-demo")
        self.assertIn("JOB_IMAGE_REPO", create_input["dockerArgs"])
        launch.validate_create_input(create_input)
        with self.assertRaises(launch.LaunchRefused):
            launch.assert_checkout_path("/workspace/interp-demo", "--image-repo")

    def test_refuses_interp_demo_paths(self):
        with self.assertRaises(launch.LaunchRefused):
            launch.assert_volume_path("/workspace/interp-demo/models/hf", "--hf-home")
        with self.assertRaises(launch.LaunchRefused):
            launch.assert_volume_path("/tmp/run", "--volume-run-dir")

    def test_estimate_does_not_launch_or_leak_key(self):
        out = Path("/tmp/within_traj_estimate.json")
        card = Path("/tmp/within_traj_launch_card.json")
        env = os.environ.copy()
        env["RUNPOD_API_KEY"] = "super-secret-runpod-key"
        env.pop("HF_TOKEN", None)
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/runpod_launch_within_traj.py"),
                "--git-sha",
                SHA,
                "--estimate-out",
                str(out),
                "--launch-card",
                str(card),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("NOT launching", proc.stderr)
        self.assertFalse(card.exists())
        report = json.loads(out.read_text())
        self.assertEqual(report["git_sha"], SHA)
        self.assertFalse(report["payload_tarball_in_create_env"])
        self.assertFalse(report["job_payload_b64_in_create_env"])
        self.assertFalse(report["script_source_in_create_env"])
        self.assertEqual(report["code_source"], "git_clone_sha")
        self.assertLess(report["graphql_post_body_bytes"], launch.MAX_CREATE_BODY_BYTES)
        self.assertNotIn("super-secret-runpod-key", out.read_text())
        self.assertNotIn("super-secret-runpod-key", proc.stdout)
        self.assertNotIn("JOB_PAYLOAD_B64", proc.stdout)


class BootScriptTests(unittest.TestCase):
    def test_boot_script_uses_volume_paths_and_does_not_patch(self):
        text = (ROOT / "scripts/runpod_within_traj_boot.sh").read_text()
        self.assertIn("JOB_RUN_DIR", text)
        self.assertIn("JOB_DIRECTIONS", text)
        self.assertIn("within_traj_mid_private.py", text)
        self.assertIn("JOB_OUT_ROOT", text)
        self.assertNotIn("JOB_PAYLOAD_B64", text)
        self.assertNotIn("option_labels", text)
        self.assertNotIn("tarfile", text)
        self.assertNotIn("/upload", text)
        self.assertNotIn("interp-demo/.venv", text)
        self.assertIn("refusing to use interp-demo", text)
        self.assertLess(
            text.index("refusing to use interp-demo"),
            text.index('mkdir -p "$HF_HOME"'),
        )
        proc = subprocess.run(["bash", "-n", str(ROOT / "scripts/runpod_within_traj_boot.sh")], check=False)
        self.assertEqual(proc.returncode, 0)


class RunnerDryRunTests(unittest.TestCase):
    def test_dry_run_on_core_transcripts(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        import numpy as np

        directions = Path("/tmp/within_traj_dirs")
        directions.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for name in ("lr_role_attn_L22.npy", "lr_role_L21.npy", "lr_role_perp_mode_L21.npy"):
            vec = rng.normal(size=8).astype(np.float32)
            np.save(directions / name, vec)
        out = Path("/tmp/within_traj_dry_run")
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/within_traj_mid_private.py"),
                "--dry-run",
                "--max-scenarios",
                "2",
                "--run-dir",
                str(ROOT / "upstream/scenarios/gemma2_9b/core/20261001T012639Z"),
                "--directions",
                str(directions),
                "--out",
                str(out),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        meta = json.loads((out / "meta.json").read_text())
        self.assertEqual(meta["protocol"], "within_traj_mid_private_intervene")
        self.assertGreaterEqual(meta["n_scenarios"], 1)
        self.assertIn("ablate_role_attn_L22", meta["arms"])
        self.assertTrue((out / "DRY_RUN.md").is_file())


if __name__ == "__main__":
    unittest.main()
