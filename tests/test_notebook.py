"""The Colab notebook must stay valid nbformat 4, free of stored outputs and secrets, and compile cell by cell.

Set NARCBENCH_NB_EXECUTE=1 (with nbclient and ipykernel installed) to also execute it end to end in DRY_RUN mode,
which uses a tiny random Gemma-2 on the CPU; the repository's default test run stays stdlib-only and fast.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks/narcbench_jspace_colab.ipynb"

SECTIONS = ["## 0.", "## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8.", "## 9.", "## 10."]
INPUTS = [
    "results/transfer_stable/WEEKEND_REVIEW/tables/hard_metrics.json",
    "results/transfer_stable/WEEKEND_REVIEW/tables/layer_sweep_residual.json",
    "results/transfer_stable/WEEKEND_REVIEW/tables/mlp_attn_site_aurocs.json",
    "results/transfer_stable/WEEKEND_REVIEW/tables/cross_phase.json",
    "results/transfer_stable/WEEKEND_REVIEW/tables/domain_holdout_L21.json",
    "results/transfer_stable/WEEKEND_REVIEW/tables/family_auroc_L21.json",
    "results/transfer_stable/WEEKEND_REVIEW/tables/controls_residualize.json",
    "results/transfer_stable/WEEKEND_REVIEW/tables/causal_nulls.json",
    "results/transfer/FULL_core_to_transfer_full/metrics.json",
    "results/transfer/FULL_core_to_transfer_wu_k25_j_only/metrics.json",
    "results/transfer/FULL_core_to_transfer_wu_k25_complement/metrics.json",
    "docs/NEXT-SCIENCE.md",
    "upstream/scenarios/gemma2_9b/core/20261001T012639Z",
    "upstream/scenarios/gemma2_2b/core/20260930T215454Z",
    "upstream/scenarios/gemma2_9b/transfer/RUNPOD",
]
SCRIPT_IMPORTS = [
    "extract_activations_from_transcripts", "train_collusion_probe", "eval_transfer_from_core", "project_j_complement",
    "project_j_variants", "transfer_stable_residualize", "transfer_stable_controls", "estimate_jacobian_jlens",
]


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads(NB.read_text())
        cls.code = [c for c in cls.nb["cells"] if c["cell_type"] == "code"]
        cls.markdown = [c for c in cls.nb["cells"] if c["cell_type"] == "markdown"]
        cls.all_text = "\n".join(_src(c) for c in cls.nb["cells"])

    def test_format_and_colab_metadata(self):
        self.assertEqual(self.nb["nbformat"], 4)
        self.assertEqual(self.nb["metadata"]["accelerator"], "GPU")
        self.assertIn("colab", self.nb["metadata"])
        self.assertGreater(len(self.code), 30)

    def test_code_cells_compile_and_carry_no_outputs(self):
        for i, c in enumerate(self.code):
            src = _src(c)
            self.assertFalse(src.lstrip().startswith(("%", "!")), f"cell {i} uses a shell/magic line; keep cells plain Python")
            compile(src, f"<cell {i}>", "exec")
            self.assertEqual(c.get("outputs", []), [], f"cell {i} has stored outputs; commit the notebook clean")
            self.assertIsNone(c.get("execution_count"), f"cell {i} has an execution count; commit the notebook clean")

    def test_sections_in_order(self):
        text = "\n".join(_src(c) for c in self.markdown)
        positions = [text.find(s) for s in SECTIONS]
        self.assertTrue(all(p >= 0 for p in positions), f"missing sections: {[s for s, p in zip(SECTIONS, positions) if p < 0]}")
        self.assertEqual(positions, sorted(positions))

    def test_no_secrets(self):
        self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{30,}", self.all_text))
        self.assertIsNone(re.search(r"RUNPOD_API_KEY\s*=\s*['\"]\w", self.all_text))
        self.assertNotIn("JOB_PAYLOAD_B64", self.all_text)

    def test_reads_only_committed_inputs(self):
        for rel in INPUTS:
            self.assertTrue((ROOT / rel).exists(), rel)
            self.assertIn(Path(rel).name, self.all_text, f"{rel} is in the inventory but the notebook never names it")

    def test_reuses_repository_scripts(self):
        for mod in SCRIPT_IMPORTS:
            self.assertTrue((ROOT / "scripts" / f"{mod}.py").exists(), mod)
            self.assertRegex(self.all_text, rf"from {mod} import", mod)

    def test_dry_run_switch_and_clone_target(self):
        self.assertIn('DRY_RUN = os.environ.get("NARCBENCH_NB_DRYRUN") == "1"', self.all_text)
        self.assertIn("https://github.com/andylikescodes/narcbench-core-demo.git", self.all_text)
        self.assertNotIn("--launch", self.all_text, "the notebook must never launch a pod")

    @unittest.skipUnless(os.environ.get("NARCBENCH_NB_EXECUTE") == "1", "set NARCBENCH_NB_EXECUTE=1 to execute the notebook in DRY_RUN mode")
    def test_executes_end_to_end_in_dry_run(self):
        code = (
            "import os, nbformat\n"
            "from nbclient import NotebookClient\n"
            "os.environ['NARCBENCH_NB_DRYRUN'] = '1'\n"
            f"nb = nbformat.read({str(NB)!r}, as_version=4)\n"
            f"NotebookClient(nb, timeout=900, kernel_name='python3', resources={{'metadata': {{'path': {str(NB.parent)!r}}}}}).execute()\n"
            "print('executed', sum(c.cell_type == 'code' for c in nb.cells), 'code cells')\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        self.assertIn("executed", proc.stdout)


if __name__ == "__main__":
    unittest.main()
