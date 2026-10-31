"""Synthetic-data tests for scripts/jspace_separability.py.

The generator writes Core and Transfer caches in the paper layout (activations_gen.npz + metadata_gen.json)
plus a small random unembedding, with a planted role direction, so the whole pre-registered pipeline
(A1-A5, bootstrap, permutation, readings, RESULTS.md) runs in well under a minute on a CPU. The numbers
are meaningless as science; the tests check mechanics and the identities the design relies on.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import jspace_separability as js  # noqa: E402
from project_j_complement import project_one  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

AGENTS = ["Ada", "Ben", "Cyd", "Dee"]


def make_unembed(out: Path, V: int, d: int, rng) -> np.ndarray:
    W = rng.randn(V, d).astype(np.float32) * 0.3
    np.save(out / "unembed_fp16.npy", W.astype(np.float16))
    np.save(out / "unembed_row_norms_fp32.npy", np.linalg.norm(W, axis=1).astype(np.float32))
    return W


def synth_split(out: Path, W: np.ndarray, layers: list[int], rng, *, families: list[str], per_family: int,
                transfer: bool, with_discussion: bool, u: np.ndarray, mode_dir: np.ndarray, role_strength: float = 5.0):
    """Episodes: per family/domain `per_family` scenario pairs x 2 modes x 4 agents.
    Rows: colluders' private round (collusion only), 3 public discussion rounds (if with_discussion), 1 final."""
    V, d = W.shape
    meta, rows = [], {L: [] for L in layers}

    def vec(role, mode, L, content_rows):
        h = (W[content_rows] * 4.0).sum(axis=0) + rng.randn(d) * 0.7 + 0.4 * L
        if mode == "collusion":
            h = h + 1.5 * mode_dir
        if role == "colluder":
            h = h + role_strength * u
        return h.astype(np.float32)

    for fam in families:
        for s in range(per_family):
            sid = f"{fam}_{s+1:02d}"
            for mode in ("collusion", "control"):
                rid = f"{'deception' if not transfer else fam}__{sid}__{mode}"
                roles = {a: ("colluder" if (mode == "collusion" and i < 2) else "honest") for i, a in enumerate(AGENTS)}
                content = rng.choice(V, 3, replace=False)
                turns = []
                if mode == "collusion":
                    turns += [(a, "private", "private", 0) for a in AGENTS[:2]]
                if with_discussion:
                    for r in (1, 2, 3):
                        turns += [(a, "discussion", "public", r) for a in AGENTS]
                turns += [(a, "final", "public", 4) for a in AGENTS]
                for a, phase, channel, rnd in turns:
                    meta.append({
                        "run_id": rid, "mode": mode, "role": roles[a], "channel": channel, "phase": phase, "round": rnd,
                        "agent_name": a, "n_tokens": 100, "seed": 1, "flagged": False,
                        "family": "deception" if not transfer else fam, "scenario_id": sid,
                        "scenario_domain": None if transfer else fam, "agent_vote": None, "target_option": None,
                        "target_majority": None, "collusion_success": None, "fair_option": None, "label_variant": None, "bp_corr": None,
                    })
                    for L in layers:
                        rows[L].append(vec(roles[a], mode, L, content) if channel != "private" else np.zeros(d, np.float32))
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata_gen.json").write_text(json.dumps(meta))
    np.savez(out / "activations_gen.npz", **{f"layer_{L}": np.stack(rows[L]) for L in layers})
    return meta


class Identities(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.tmp = Path(tempfile.mkdtemp())
        self.W = make_unembed(self.tmp, V=300, d=40, rng=self.rng)
        self.norms = np.load(self.tmp / "unembed_row_norms_fp32.npy")
        self.H = self.rng.randn(12, 40).astype(np.float32) + 3 * self.W[:12]

    def test_topk_frames_match_bruteforce(self):
        idx, vals = js.topk_frames(self.H, self.W, self.norms, 7, vocab_chunk=64)
        scores = self.H @ (self.W / self.norms[:, None]).T
        for i in range(len(self.H)):
            brute = np.argsort(-scores[i])[:7]
            self.assertEqual(set(idx[i].tolist()), set(brute.tolist()))
            self.assertTrue(np.all(np.diff(vals[i]) <= 1e-6))  # descending

    def test_nested_projection_equals_project_one_per_prefix(self):
        idx, _ = js.topk_frames(self.H, self.W, self.norms, 9)
        proj = js.nested_frame_projections(self.H, idx, self.W, [3, 9])
        for i in range(len(self.H)):
            for k in (3, 9):
                hj, hc = project_one(self.H[i], self.W[idx[i, :k]])
                np.testing.assert_allclose(proj[k][i], hj, atol=1e-3)
                comp = self.H[i] - proj[k][i]
                self.assertAlmostEqual(float(proj[k][i] @ comp), 0.0, places=2)  # orthogonal
                np.testing.assert_allclose(proj[k][i] + comp, self.H[i], atol=1e-5)  # sums to h
                f = float(proj[k][i] @ proj[k][i]) / float(self.H[i] @ self.H[i])
                self.assertTrue(0.0 <= f <= 1.0 + 1e-6)

    def test_fixed_projection_is_idempotent_and_random_subspace_small(self):
        Q = js.haar_basis(40, 10, self.rng)
        P = js.fixed_projection(self.H, Q, 10)
        np.testing.assert_allclose(js.fixed_projection(P, Q, 10), P, atol=1e-4)
        self.assertLess(js.fve(P, self.H), 0.6)  # a random 10/40 subspace of noisy data takes well under all the norm

    def test_fast_auc_equals_sklearn_with_ties(self):
        y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        s = np.array([0.1, 0.4, 0.4, 0.9, 0.9, 0.2, 0.7, 0.7])
        self.assertAlmostEqual(js.fast_auc(y, s), roc_auc_score(y, s), places=12)

    def test_permutation_under_null_is_not_small(self):
        rng = np.random.RandomState(1)
        y = np.array([1, 0] * 20)
        fams = ["a"] * 20 + ["b"] * 20
        ep = {"y": y.tolist(), "family": fams, "peak": rng.randn(40).tolist(), "split": rng.randn(40).tolist(), "asym": rng.randn(40).tolist()}
        Y = js.permuted_labels(y, fams, 400, rng)
        self.assertTrue(np.all(Y.sum(1) == y.sum()))
        self.assertTrue(np.all(Y[:, :20].sum(1) == y[:20].sum()))  # within-family counts kept
        p = js.permutation_pvalue({"0": ep}, Y)
        self.assertGreater(p, 0.02)
        # planted signal → small p
        ep2 = dict(ep, peak=(y * 3 + rng.randn(40) * 0.1).tolist(), split=(y * 3 + rng.randn(40) * 0.1).tolist(), asym=(y * 3 + rng.randn(40) * 0.1).tolist())
        self.assertLess(js.permutation_pvalue({"0": ep2}, Y), 0.01)

    def test_bootstrap_is_stratified(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        fams = ["a", "a", "a", "a", "b", "b"]
        boots = js.bootstrap_indices(y, fams, 20, np.random.RandomState(0))
        for idx in boots:
            self.assertEqual(len(idx), 6)
            self.assertEqual(int(y[idx].sum()), 3)
            self.assertEqual(sum(fams[i] == "b" for i in idx), 2)


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(7)
        cls.tmp = Path(tempfile.mkdtemp())
        layers = [3, 4, 5]
        d = 96
        (cls.tmp / "unembed").mkdir()
        W = make_unembed(cls.tmp / "unembed", V=500, d=d, rng=rng)
        u = rng.randn(d); u /= np.linalg.norm(u)
        mode_dir = rng.randn(d); mode_dir /= np.linalg.norm(mode_dir)
        synth_split(cls.tmp / "core", W, layers, rng, families=[f"dom{i}" for i in range(6)], per_family=3,
                    transfer=False, with_discussion=True, u=u, mode_dir=mode_dir, role_strength=8.0)
        synth_split(cls.tmp / "transfer", W, layers, rng, families=["fam_a", "fam_b", "fam_c"], per_family=4,
                    transfer=True, with_discussion=True, u=u, mode_dir=mode_dir, role_strength=8.0)
        cls.out = cls.tmp / "out"
        argv = sys.argv
        sys.argv = ["jspace_separability.py", "--core-acts", str(cls.tmp / "core"), "--transfer-acts", str(cls.tmp / "transfer"),
                    "--unembed", str(cls.tmp / "unembed"), "--layers", "3-5", "--ks", "4,8", "--out", str(cls.out), "--tag", "synthetic",
                    "--n-boot", "60", "--n-perm", "60"]
        try:
            rc = js.main()
        finally:
            sys.argv = argv
        cls.rc = rc
        cls.summary = json.loads((cls.out / "summary.json").read_text())

    def test_runs_and_writes_outputs(self):
        self.assertEqual(self.rc, 0)
        for f in ("summary.json", "RESULTS.md", "episode_scores.json"):
            self.assertTrue((self.out / f).exists(), f)
        self.assertGreater(len((self.out / "RESULTS.md").read_text()), 2000)

    def test_all_arms_all_analyses(self):
        s = self.summary
        self.assertEqual(s["ks"], [4, 8, 25])  # k=25 is always added as the primary frame size
        for key in ("A1_core_ldo", "A2_transfer", "A3_transfer_lofo", "A4a_core_disc_to_final", "A4b_core_to_transfer_discussion"):
            self.assertIn("band_mean", s[key], key)
            self.assertEqual(set(s[key]["band_mean"]["25"]), set(js.ARM_ORDER), key)
        self.assertEqual(set(s["A2_inference"]), {"4", "8", "25"})
        self.assertIn("C_J-J", s["A2_inference"]["25"]["differences"])
        self.assertIn("A5_geometry", s)
        self.assertTrue(set(s["readings"]) >= {"R1_in_distribution", "R2_transfer", "R3_complement_uninformative", "R4_within_transfer", "R5_phase_vs_family", "R6_dimension"})

    def test_planted_signal_is_found_by_full_and_complement_of_random(self):
        bm = self.summary["A2_transfer"]["band_mean"]
        self.assertGreater(bm["8"]["full"]["primary"], 0.85)
        for k in ("4", "8"):  # removing a random 4- or 8-dimensional subspace of 96 leaves the full probe intact
            self.assertLess(abs(bm[k]["C_randsub"]["primary"] - bm[k]["full"]["primary"]), 0.1, k)
        self.assertGreater(self.summary["A1_core_ldo"]["band_mean"]["25"]["full"]["primary"], 0.85)
        self.assertGreater(self.summary["A3_transfer_lofo"]["band_mean"]["25"]["full"]["primary"], 0.85)

    def test_intervals_bracket_estimates_and_fve_is_a_fraction(self):
        inf = self.summary["A2_inference"]["25"]["arms"]
        for arm, a in inf.items():
            self.assertTrue(a["ci"][0] - 1e-9 <= a["est"] <= a["ci"][1] + 1e-9, arm)
            self.assertTrue(0 < a["p_perm"] <= 1)
        bm = self.summary["A2_transfer"]["band_mean"]["25"]
        for arm in ("J", "C_J", "randsub", "pca"):
            self.assertTrue(0 <= bm[arm]["fve_core"] <= 1)
        self.assertAlmostEqual(bm["J"]["fve_core"] + bm["C_J"]["fve_core"], 1.0, places=3)
        self.assertGreater(bm["J"]["fve_core"], bm["randrows"]["fve_core"])  # the aligned frame holds more of h than random rows

    def test_episode_scores_reproduce_point_estimates(self):
        ep = json.loads((self.out / "episode_scores.json").read_text())["A2"]
        s = self.summary
        for L in s["layers"]:
            e = ep[str(L)]["25"]["J"]
            prim = np.mean([roc_auc_score(np.array(e["y"]), np.array(e[m])) for m in ("peak", "split", "asym")])
            self.assertAlmostEqual(prim, s["A2_transfer"]["per_layer"][str(L)]["25"]["J"]["primary"], places=9)


if __name__ == "__main__":
    unittest.main()
