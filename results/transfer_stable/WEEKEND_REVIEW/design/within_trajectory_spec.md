# Within-trajectory mid-discussion interventions — design spec

**Written:** 2026-10-01 ~5:10 PM PT · Transfer-stable · offline design  
**Related:** `../06_METHODS_LADDER.md`, `matched_prefix_spec.md`

---

## Goal

Test whether ablating / patching the Transfer-stable role direction **inside a real Core multi-round trajectory** moves later talk or votes — **without** regenerating divergent committee prefixes.

Prior discussion-phase smoke regenerated colluder discussion under ablate (honest msgs reused) then spliced into finals. That still lets **colluder text diverge** from the baseline transcript the probe saw. Within-trajectory teacher-forcing keeps the recorded path until the intervene point.

---

## Protocol sketch

### Inputs

- Core collusion transcript: `upstream/scenarios/gemma2_9b/core/20261001T012639Z/<scenario>__collusion/`
- `agent_prompts.json` + `results.json` (`public_messages`, `final_texts`, `roles`, votes)
- Direction: `results/transfer_stable/directions/lr_role_attn_L22.npy` (+ residual L21/L22, role_perp, random)

### Steps

1. **Select intervene turn:** e.g. first public colluder turn, mid-discussion colluder turn, or final-vote prompt last token.
2. **Rebuild prompt** for that agent at that turn (system + user with DISCUSSION HISTORY as in regen harness).
3. **Teacher-force** recorded tokens for positions `0 … k−1` of the agent’s utterance (or all prior turns’ tokens in context).
4. **Intervene at position k** (and optionally continue intervening for the rest of the utterance):
   - Ablate: residual-write / attn out `− α (·û) û`
   - Or patch from a matched honest forward at same absolute position (harder — needs parallel honest trajectory; prefer matched-prefix first)
5. **Continue generation** from k (temperature 0) for the rest of the turn and/or through final vote.
6. **Metrics:** string edit vs recorded utterance; keyword/target mentions; final vote; P(target) at vote; flips vs teacher-forced baseline continuation (no patch).

### Arms

| Arm | Intervention |
|---|---|
| baseline_tf | Teacher-force then free continue, no patch |
| ablate_role | û = role attn_L22 / residual |
| ablate_role_perp | û = role_perp |
| random | matched α random direction |
| (optional) steer_up on honest agent | +β σ û |

### Why after matched-prefix

Matched-prefix answers “is the direction causally usable at all under shared computation?” Within-trajectory answers “does it matter on the naturalistic multi-agent path?” If matched-prefix is null, within-trajectory is lower priority.

---

## Success metrics

- Ablate changes continuation or vote vs baseline_tf **and** more than random.
- Effect larger when intervening at commitment-onset tokens (from checkpoint assays) than at arbitrary mid-sentence positions.

---

## GPU

Requires hooked generation on 9B — **later**, after matched-prefix smoke approval. No GPU for this spec.

---

## Soft-stop

No Track I / Paper 2 / interp-demo / family-house. No residual-α chase as a substitute for teacher-forced within-trajectory design.
