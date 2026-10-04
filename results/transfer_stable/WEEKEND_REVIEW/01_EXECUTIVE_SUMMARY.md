# Executive summary — a Transfer-stable collusion study

**For:** Andy · **Written:** 2026-10-01 ~5:15 PM PT  
**Lane:** NARCBench Transfer-stable · Gemma-2-9B-it · Core discussion → Transfer final  
**Soft-stop / GPU:** honored; causal GPU paused; balance last known ~$26.24.

This is not a scoreboard of AUROCs. It is a short study of **where multi-agent collusion leaves a transferable signature in the model**, what that signature is *of*, and what we can claim for **monitoring vs control**.

---

## Lead answer (safety-relevant)

We find a **monitorable, Transfer-stable coordination signature**: colluder vs honest role is near-ceiling linearly readable after Core→Transfer shift, concentrated in **non-J residual geometry** and especially **attention write-in at L22 (~0.999)**. That signature is entangled with **run-level collusion setup** (role↔mode), so it looks more like a **coordination / coalition-setup state** than a pure “lying atom.”  

Under free regeneration with last-token residual-write ablation, **we could not edit votes** — so **readability ≠ control**. The responsible claim is: **detectable transferable coordination geometry; editability unproven**. Next causal work must use **matched-prefix interchange** and **checkpoint assays**, not residual-α chasing.

---

## The arc (five beats)

### 1. Probeable (prior) → where + transfer (ours)

Prior NARCBench-style work showed collusion is **probeable** from hidden states (full residual / activation probes). That establishes *existence* of a linear readout. It does not locate the signal relative to verbalizable / J-space structure, does not say which **block write-in** carries Transfer OOD signal, and does not separate **role deception** from **collusion-condition setup**.

**Our positive:** Core-trained role probes **transfer** to Transfer finals (full band primary **0.902**; residual L21 **0.941**). After MLP/attn extracts, the best site is **attn_L22 primary ≈ 0.999**. Low-rank directions (LR weights, contrast PCA k≈4–8) match or beat ambient — the transferable geometry is a **shared moderately low-rank direction**, not a diffuse full-space quirk.

### 2. J fails Transfer; non-J / attn_L22 is Transfer-stable

HARD Core→Transfer (n=72): full **0.902** vs J-only **~0.55–0.60**; complement **≈ full (~0.89–0.90)**. So the Transfer-stable story is **not** “collusion lives in J.” Whatever J captures in-distribution does **not** carry the OOD role detector. The complement — residual mid-band, then attention L22 — does.

**Where (summary):** J ✗ for Transfer · residual L21–L22 ✓ · **attn_L22 ✓✓** · mlp useful but weaker than attn at matched layers.

### 3. Role↔mode entanglement → coordination/setup vs pure lying atom

Role (colluder vs honest) and mode (collusion-run vs control-run) **cross-predict** on Transfer (mode→role ~**0.90**; cos(role, mode_runavg) ~**0.70**). After residualizing mode out of role, **role still transfers** (L21 0.859→**0.796**; L22 0.873→**0.837**) while mode leak drops.

**Mechanism reading (correlational):** the Transfer-stable direction is better described as a **coordination / coalition-setup signature** — jointly sensitive to “this is a collusion episode” and “this agent is a colluder” — than as an isolated token-level lying detector. That is AI-safety-relevant: monitors may catch **setup state**, not only explicit deceptive utterances.

### 4. Causal null under divergent histories → readability ≠ control

Five smoke batteries (Transfer residual α=1 and α-sweep; Core residual; attn_L22 final; attn_L22 discussion±final) all returned **null** on n=6: votes / collusion_success tied to baseline and random. Discussion null closes the cheap escape hatch that “we only missed because the decision locked at final.”

**Interpretation we endorse:** free-regen last-token ablate on **divergent histories** is a **weak causal test**. It does **not** prove the signature is epiphenomenal. It **does** block the claim “we can steer collusion by scrubbing this direction at last token under free regen.”

### 5. Safety claim + next methods

| We can claim | We cannot claim (yet) |
|---|---|
| **Monitorable** Transfer-stable coordination signature (attn_L22 / non-J residual) | That scrubbing it **controls** collusive votes |
| Correlational localization beyond the probe paper (J vs residual vs attn) | That the axis is a pure deception atom |
| Role survives mode peel (not “just task”) | Editability under matched computation |

**Next (design-first, no GPU now):** matched-prefix micro-worlds → path-patch / interchange; within-trajectory teacher-forced mid-discussion edits; checkpoint assays for *when* commitment appears. **Do not:** residual-α chase; Track I / Paper 2 / interp-demo / family-house.

---

## Fridge line

**Transfer-stable attn_L22 reads coordination setup almost perfectly after shift; ablating it on free regen never flipped votes — monitor yes, edit unproven; redesign causal tests around matched prefixes.**
