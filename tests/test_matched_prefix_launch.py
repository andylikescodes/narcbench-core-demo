"""Guards for the git-boot matched-prefix launcher. Does not create a pod."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import runpod_launch_matched_prefix as launch  # noqa: E402

SHA = "b" * 40


def _args(**overrides):
    values = dict(
        git_sha=SHA,
        pod_name="job-narcbench-matched-prefix-final-resid",
        pairs_on_volume=launch.SUITE_META["core"]["pairs_on_volume"],
        volume_out="/workspace/jobs/narcbench-results/matched_prefix_final_resid",
        mode_name="final-resid-controls",
        max_pairs=12,
        model="google/gemma-2-9b-it",
        max_new_tokens=8,
        seed=0,
        max_minutes=90,
        grace_minutes=6,
        gpu=launch.GPU,
        cloud="SECURE",
        final_resid_controls=True,
        role_perp_confirm=False,
        extras=False,
        extras_only=False,
    )
    values.update(overrides)
    return launch.argparse.Namespace(**values)


class CreateEnvTests(unittest.TestCase):
    def test_create_env_has_sha_and_no_boot_or_payload(self):
        create_input = launch.build_create_input(_args(), "k" * 40)
        env = {item["key"]: item["value"] for item in create_input["env"]}
        self.assertEqual(env["JOB_GIT_SHA"], SHA)
        self.assertEqual(env["JOB_REPO_URL"], launch.REPO_URL)
        self.assertEqual(env["JOB_PAIRS"], "/workspace/jobs/narcbench-data/matched_prefix/pairs.jsonl")
        self.assertEqual(env["JOB_DIRECTIONS"], launch.DIRECTIONS_ON_VOLUME)
        self.assertEqual(env["JOB_MODE"], "final-resid-controls")
        self.assertNotIn("JOB_PAYLOAD_B64", env)
        self.assertNotIn("JOB_BOOT_B64", env)
        blob = json.dumps(create_input)
        self.assertNotIn("JOB_PAYLOAD_B64", blob)
        self.assertNotIn("JOB_BOOT_B64", blob)
        self.assertNotIn("base64 -d", create_input["dockerArgs"])
        self.assertIn("git clone", create_input["dockerArgs"])
        self.assertIn("checkout --detach", create_input["dockerArgs"])
        self.assertIn("runpod_matched_prefix_boot.sh", create_input["dockerArgs"])
        nbytes = launch.validate_create_input(create_input)
        self.assertLess(nbytes, 20_000)
        self.assertLess(nbytes, launch.MAX_CREATE_BODY_BYTES)

    def test_refuses_payload_and_boot_script_in_env(self):
        create_input = launch.build_create_input(_args(), "k" * 40)
        poisoned = json.loads(json.dumps(create_input))
        poisoned["env"].append({"key": "JOB_PAYLOAD_B64", "value": "H4sIAAAAAAAA"})
        with self.assertRaises(launch.LaunchRefused):
            launch.validate_create_input(poisoned)

        boot = json.loads(json.dumps(create_input))
        boot["env"].append({"key": "JOB_BOOT_B64", "value": "IyEvYmluL2Jhc2g="})
        with self.assertRaises(launch.LaunchRefused):
            launch.validate_create_input(boot)

        stuffed = json.loads(json.dumps(create_input))
        for item in stuffed["env"]:
            if item["key"] == "JOB_PAIRS":
                item["value"] = "\x1f\x8b" + ("x" * 32)
        with self.assertRaises(launch.LaunchRefused) as raised:
            launch.validate_create_input(stuffed)
        self.assertIn("tarball", str(raised.exception))

    def test_final_resid_boot_does_not_require_direction_files(self):
        text = (ROOT / "scripts/runpod_matched_prefix_boot.sh").read_text()
        self.assertIn("final-resid-controls", text)
        self.assertIn("PASS_DIRECTIONS=0", text)
        self.assertNotIn("JOB_PAYLOAD_B64", text)
        self.assertNotIn("JOB_BOOT_B64", text)
        self.assertIn("refusing to use interp-demo", text)
        self.assertLess(text.index("refusing to use interp-demo"), text.index('mkdir -p "$HF_HOME"'))
        proc = subprocess.run(["bash", "-n", str(ROOT / "scripts/runpod_matched_prefix_boot.sh")], check=False)
        self.assertEqual(proc.returncode, 0)

    def test_estimate_final_resid_stays_small(self):
        out = Path("/tmp/matched_prefix_final_estimate.json")
        env = os.environ.copy()
        env["RUNPOD_API_KEY"] = "super-secret-runpod-key"
        env.pop("HF_TOKEN", None)
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/runpod_launch_matched_prefix.py"),
                "--final-resid-controls",
                "--estimate-out",
                str(out),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("NOT launching", proc.stderr)
        report = json.loads(out.read_text())
        self.assertEqual(report["job_mode"], "final-resid-controls")
        self.assertFalse(report["job_boot_b64_in_create_env"])
        self.assertFalse(report["job_payload_b64_in_create_env"])
        self.assertLess(report["graphql_post_body_bytes"], 20_000)
        self.assertNotIn("super-secret-runpod-key", out.read_text())
        self.assertNotIn("super-secret-runpod-key", proc.stdout)
        self.assertNotIn("JOB_BOOT_B64", proc.stdout)
        self.assertNotIn("JOB_PAYLOAD_B64", proc.stdout)


if __name__ == "__main__":
    unittest.main()
