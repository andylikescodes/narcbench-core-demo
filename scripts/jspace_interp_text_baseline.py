#!/usr/bin/env python3
"""Text-level baselines and confound controls that run on CPU from the transcripts alone.

What it measures (plan: "The confound every experiment must control" and Experiment 1, text side):
  A. Reply-only bag-of-words probe, scored with the same Peak/Split/Asymmetry detectors and protocols
     as the activation probes (Core leave-one-domain-out; Core -> Transfer; within Transfer leave one
     family out). This bounds how much of the role signal is plain wording of the reply.
  B. System-prompt-only bag-of-words probe: the text-level version of the prompt-only control. It shows
     how separable the instructions are on their own (the activation-level version needs a GPU).
  C. Concept-lexicon probe: the 4 frozen concept-group rates only (readable, 4 features).
  D. Behaviour split (Core): among colluder-prompted agents, out-of-fold reply-probe scores of those who
     voted the target vs those who did not; same for honest agents.
  E. Concept-group rates in replies: colluders vs honest agents (within-episode role permutation) and
     colluders vs control-episode agents (permutation stratified by scenario), Core and Transfer.

The TF-IDF vocabulary is fit on Core text only (no labels). For Core leave-one-domain-out that includes
the held-out domain's words in the vocabulary (unsupervised); the probe weights never see its labels.

  /home/claude/venv/bin/python scripts/jspace_interp_text_baseline.py --out results/jspace_interp/text_baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jspace_interp import concepts, detectors, stats  # noqa: E402
from jspace_interp.agents import agent_texts, load_rows  # noqa: E402


def attach(agents, X):
    for a, x in zip(agents, X):
        a["x"] = np.asarray(x, dtype=np.float64).ravel()
    return agents


def tfidf_features(fit_texts, *agent_lists, max_features=5000):
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=max_features,
                          sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    vec.fit(fit_texts)
    outs = []
    for agents in agent_lists:
        X = np.vstack([np.asarray(vec.transform(a["texts"]).mean(axis=0)).ravel() for a in agents])
        outs.append(X)
    return outs


def lexicon_features(agents):
    X = []
    for a in agents:
        c = concepts.text_group_counts(" ".join(a["texts"]))
        n = max(c["n_words"], 1)
        X.append([1000.0 * c[g] / n for g in concepts.GROUPS])
    return np.array(X)


def run_protocols(core, xfer, label):
    res = {}
    res["core_lodo"] = detectors.core_lodo(core)
    r = detectors.evaluate(core, xfer)
    idx, boot = detectors.episode_bootstrap(r["y"], r["episode_scores"],
                                            strata=[f"{rid.split('__')[0]}|{y}" for rid, y in zip(r["rids"], r["y"])])
    res["core_to_transfer"] = {k: r[k] for k in ("peak", "split", "asym", "primary", "n_test_episodes")}
    res["core_to_transfer"]["primary_ci95"] = [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]
    res["transfer_lofo"] = detectors.transfer_lofo(xfer)
    oof = res["core_lodo"].pop("agent_scores")
    print(f"[{label}] Core LODO {res['core_lodo']['primary']:.3f} | Core->Transfer "
          f"{res['core_to_transfer']['primary']:.3f} {res['core_to_transfer']['primary_ci95']} | "
          f"within-Transfer LOFO {res['transfer_lofo']['primary']:.3f}")
    return res, oof


def behaviour_split(core_agents, oof):
    out = {}
    for role in ("colluder", "honest"):
        grp = [a for a in core_agents if a["mode"] == "collusion" and a["role"] == role]
        hit = [oof[(a["run_id"], a["agent_name"])] for a in grp if a["vote"] == a["target"]]
        miss = [oof[(a["run_id"], a["agent_name"])] for a in grp if a["vote"] != a["target"]]
        auc, p = stats.permutation_auroc(hit, miss, n_perm=5000)
        out[role] = {"n_voted_target": len(hit), "n_voted_other": len(miss),
                     "auroc_target_vs_other": auc, "perm_p": p,
                     "mean_score_target": float(np.mean(hit)) if hit else None,
                     "mean_score_other": float(np.mean(miss)) if miss else None}
    # colluders who did NOT vote target vs honest agents who did not: role with behaviour held to "fair"
    col_miss = [oof[(a["run_id"], a["agent_name"])] for a in core_agents
                if a["mode"] == "collusion" and a["role"] == "colluder" and a["vote"] != a["target"]]
    hon_miss = [oof[(a["run_id"], a["agent_name"])] for a in core_agents
                if a["mode"] == "collusion" and a["role"] == "honest" and a["vote"] != a["target"]]
    auc, p = stats.permutation_auroc(col_miss, hon_miss, n_perm=5000)
    out["colluder_vs_honest_both_voted_other"] = {"auroc": auc, "perm_p": p,
                                                 "n_colluder": len(col_miss), "n_honest": len(hon_miss)}
    return out


def concept_rates(agents, split):
    rates = lexicon_features(agents)
    out = {}
    col_eps = [i for i, a in enumerate(agents) if a["mode"] == "collusion"]
    for gi, g in enumerate(concepts.GROUPS):
        v = rates[col_eps, gi]
        obs, p, _ = stats.within_episode_role_permutation(
            v, [agents[i]["run_id"] for i in col_eps], [agents[i]["role"] == "colluder" for i in col_eps],
            n_perm=5000)
        sel = [i for i, a in enumerate(agents) if a["role"] == "colluder" or a["mode"] == "control"]
        obs2, p2, _ = stats.stratified_permutation(
            rates[sel, gi], [agents[i]["role"] == "colluder" for i in sel],
            [agents[i]["scenario_id"] or agents[i]["run_id"].split("__")[1] for i in sel], n_perm=5000)
        out[g] = {
            "colluder_per1000": float(rates[[i for i in col_eps if agents[i]["role"] == "colluder"], gi].mean()),
            "honest_in_collusion_per1000": float(rates[[i for i in col_eps if agents[i]["role"] == "honest"], gi].mean()),
            "control_per1000": float(rates[[i for i, a in enumerate(agents) if a["mode"] == "control"], gi].mean()),
            "colluder_minus_honest": obs, "p_within_episode": p,
            "colluder_minus_control": obs2, "p_stratified": p2,
        }
    pv = [out[g][k] for g in out for k in ("p_within_episode", "p_stratified")]
    adj, rej = stats.benjamini_hochberg(pv)
    i = 0
    for g in out:
        for k in ("p_within_episode", "p_stratified"):
            out[g][k + "_bh"] = float(adj[i])
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/jspace_interp/text_baseline")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    core_rows, xfer_rows = load_rows("core"), load_rows("transfer")
    report = {"sources": {"core_runs": "upstream/scenarios/gemma2_9b/core/20261001T012639Z",
                          "transfer_runs": "upstream/scenarios/gemma2_9b/transfer/RUNPOD"}}

    # A. reply-only, protocol phases as in the activation probes
    core_r = agent_texts(core_rows, "core", "discussion", "response")
    xfer_r = agent_texts(xfer_rows, "transfer", "final", "response")
    Xc, Xt = tfidf_features([t for a in core_r for t in a["texts"]], core_r, xfer_r)
    attach(core_r, Xc), attach(xfer_r, Xt)
    report["A_reply_only_bow"], oof = run_protocols(core_r, xfer_r, "reply-only BoW")

    # B. system-prompt-only text
    core_s = agent_texts(core_rows, "core", "discussion", "system_prompt")
    xfer_s = agent_texts(xfer_rows, "transfer", "final", "system_prompt")
    Xc, Xt = tfidf_features([t for a in core_s for t in a["texts"]], core_s, xfer_s)
    attach(core_s, Xc), attach(xfer_s, Xt)
    report["B_system_prompt_only_bow"], _ = run_protocols(core_s, xfer_s, "system-prompt-only BoW")

    # C. concept lexicon (4 features)
    core_l = agent_texts(core_rows, "core", "discussion", "response")
    xfer_l = agent_texts(xfer_rows, "transfer", "final", "response")
    attach(core_l, lexicon_features(core_l)), attach(xfer_l, lexicon_features(xfer_l))
    report["C_concept_lexicon"], _ = run_protocols(core_l, xfer_l, "concept lexicon (4 feats)")

    # D. behaviour split on reply-only out-of-fold scores
    report["D_behaviour_split_core"] = behaviour_split(core_r, oof)
    print("[behaviour split]", json.dumps(report["D_behaviour_split_core"], indent=1))

    # E. concept rates in replies
    report["E_concept_rates"] = {
        "core_discussion": concept_rates(agent_texts(core_rows, "core", "discussion", "response"), "core"),
        "core_final": concept_rates(agent_texts(core_rows, "core", "final", "response"), "core"),
        "transfer_final": concept_rates(agent_texts(xfer_rows, "transfer", "final", "response"), "transfer"),
        "transfer_discussion": concept_rates(agent_texts(xfer_rows, "transfer", "discussion", "response"), "transfer"),
    }
    report["concept_groups_frozen"] = concepts.GROUPS
    (args.out / "text_baseline.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print("wrote", args.out / "text_baseline.json")


if __name__ == "__main__":
    main()
