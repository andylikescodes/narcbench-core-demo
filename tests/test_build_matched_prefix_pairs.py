"""The v2 pair builder must be deterministic and must satisfy the matched-prefix invariants."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_matched_prefix_pairs as bp  # noqa: E402

COMMITTED = ROOT / "data/matched_prefix/pairs_core_v2.jsonl"
RECORDED_META = ROOT / "docs/handoff/2026-10-04-matched-prefix/final-resid-meta.json"
RECORDED_PER_PAIR = ROOT / "docs/handoff/2026-10-04-matched-prefix/final-resid-per_pair.json"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class PairBuilderTests(unittest.TestCase):
    def test_committed_file_matches_a_fresh_build(self):
        out = Path("/tmp/pairs_core_v2_rebuilt.jsonl")
        proc = subprocess.run([sys.executable, str(ROOT / "scripts/build_matched_prefix_pairs.py"), "--out", str(out)], cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(out.read_text(), COMMITTED.read_text(), "data/matched_prefix/pairs_core_v2.jsonl is stale; rerun the builder")

    def test_v2_lines_up_with_the_recorded_v1_set(self):
        rows = _rows(COMMITTED)
        meta = json.loads(RECORDED_META.read_text())
        self.assertEqual([r["pair_id"] for r in rows], [p.replace("_v1", "_v2") for p in meta["pair_ids"]])
        recorded = {p["pair_id"].replace("_v1", "_v2"): p for p in json.loads(RECORDED_PER_PAIR.read_text())}
        for r in rows:
            self.assertEqual(r["target_option"], recorded[r["pair_id"]]["target_option"])
            self.assertEqual(r["fair_option"], recorded[r["pair_id"]]["fair_option"])
            self.assertEqual(r["source_scenario_id"], recorded[r["pair_id"]]["source_scenario_id"])

    def test_invariants_and_harness_composition(self):
        import matched_prefix_interchange as mp  # noqa: E402

        for r in _rows(COMMITTED):
            self.assertEqual(bp.char_checks(r), [], r["pair_id"])
            pc, ph = mp.compose_prompt(r, "colluder"), mp.compose_prompt(r, "honest")
            stem = r["shared_stem"] + "\n"
            self.assertTrue(pc.startswith(stem) and ph.startswith(stem))
            self.assertTrue(pc.endswith("\nRECOMMENDATION:") and ph.endswith("\nRECOMMENDATION:"))
            self.assertNotEqual(pc, ph)
            self.assertIn(f"Option {r['target_option']}", r["divergence"]["arm_a"]["text"])
            self.assertNotIn("Option A", r["divergence"]["arm_b"]["text"])
            self.assertNotIn("Option B", r["divergence"]["arm_b"]["text"])
            self.assertEqual(r["pair_version"], "v2")
            self.assertEqual(r["suite"], "core")

    def test_widen_set_builds_every_letter_target(self):
        out = Path("/tmp/pairs_core_v2_all.jsonl")
        proc = subprocess.run([sys.executable, str(ROOT / "scripts/build_matched_prefix_pairs.py"), "--all-letter-targets", "--out", str(out)], cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = _rows(out)
        self.assertEqual(len(rows), 50)
        self.assertEqual(len({r["pair_id"] for r in rows}), 50)
        labels = {tuple(r["option_labels"]) for r in rows}
        self.assertEqual(labels, {("A", "B"), ("1", "2")})
        for r in rows:
            self.assertEqual(bp.char_checks(r), [], r["pair_id"])

    def test_unknown_scenario_is_refused(self):
        proc = subprocess.run([sys.executable, str(ROOT / "scripts/build_matched_prefix_pairs.py"), "--scenarios", "audit_01,nope_99", "--out", "/tmp/pairs_bad.jsonl"], cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("nope_99", proc.stderr)


if __name__ == "__main__":
    unittest.main()
