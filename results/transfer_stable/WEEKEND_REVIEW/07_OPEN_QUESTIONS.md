# Open questions + next experiments (ranked)

**Written:** 2026-10-01 ~5:10 PM PT  
**Causal GPU:** **paused** pending matched-prefix rethink. Soft-stop active.

**Standing claim to pressure-test:** monitorable transferable coordination signature (attn_L22 / non-J); **editability unproven**. Open Qs below are ranked to turn “unproven” into either a causal win under matched prefixes or a sharper negative.

---

## Ranked open questions

| Rank | Question | Why it matters | Next experiment | GPU? |
|---:|---|---|---|---|
| 1 | Is attn_L22 / residual role direction **causally usable under matched prefixes**? | Resolves “readable but free-regen null” | Matched-prefix interchange smoke (design ready) | Later, tiny |
| 2 | At **which token / turn** does collusion commitment become linearly / behaviorally locked? | Targets where to intervene | Checkpoint assays + within-trajectory teacher-force | Later |
| 3 | Does **role_perp** outperform ambient role under a fair causal test? | Specificity vs mode entanglement | Same as (1) with role_perp vs mode vs random | Later |
| 4 | Is the Transfer-stable geometry the **same direction** as Core in-distribution causal substrate? | Core nulls were also free-regen; matched-prefix on Core first may be cleaner | Matched-prefix on Core letter-target scenarios first | Later |
| 5 | Multi-position / subspace (contrast PCA k=4–8) vs single LR vector? | Low-rank correlational win unused causally | Only after (1) shows any causal movement | Later |
| 6 | Family heterogeneity (news / pump_dump weaker under dim)? | Writing / robustness | Offline family tables already exist; causal later | Offline ok |
| 7 | Cross-phase Transfer (train discussion → eval discussion)? | Gate card leftover | Offline from existing Transfer discussion acts | **$0 now** |
| 8 | Residual-α / larger free-regen n? | — | **Do not do** under current design | Paused |

---

## Next experiments (action list)

### Do now (offline)

1. Review this pack (`01` → `06`).
2. Edit / approve matched-prefix micro-world wording in `design/matched_prefix_spec.md` + example pairs.
3. Run `scripts/matched_prefix_scaffold.py` (dry-run) to list Core seeds usable as stems.
4. Optional offline: Transfer discussion-phase eval; Core domain-holdout → Transfer families (`GATE_CARDS.md` leftovers).

### Do after Andy/Paige OK (GPU smoke)

1. Matched-prefix path-patch smoke on Core (letter `target_option`), attn_L22 + residual L21/L22, n≈6–12, random + role_perp controls.
2. If movement: within-trajectory mid-turn on real Core transcripts.
3. Checkpoint assays to map commitment timing.

### Explicitly paused

- Further free-regen last-token residual/attn ablate (any α, any n) on Transfer-stable lane.
- Track I / Paper 2 / interp-demo / family-house.

---

## Note on standing greenlight

A standing greenlight for Transfer-stable work exists, but **this pack’s mandate is consolidate + design**. Causal GPU stays paused until the matched-prefix (or equivalent) redesign is the approved next spend.
