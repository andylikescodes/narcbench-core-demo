# Controls + residualize — key numbers & interpretation

**Written:** 2026-10-01 ~5:10 PM PT  
**Sources:** `../CONTROLS_OFFLINE.md`, `../RESIDUALIZE_MODE.md` (+ JSON)  
**Offline only** (no GPU). Soft-stop honored.

**Role in the study (beat 3):** evidence that the Transfer-stable geometry is a **coordination / coalition-setup signature** (role↔mode entanglement) rather than a pure lying atom — while role_perp shows deception-role is not *only* mode.

---

## Question

Is the Transfer-stable L21–L22 / PCA-contrast geometry **deception-specific** (colluder vs honest), **task/mode-generic** (collusion-run vs control-run), or **mixed**?

Labels used: **mode** (collusion / control) and **role** (colluder / honest). Never `collusion_success` (all false on Transfer finals for success labeling in this control pass).

---

## Protocol (brief)

| Piece | Choice |
|---|---|
| Fit | Core discussion public |
| Eval | Transfer final (zero-shot) |
| Role direction | LR colluder vs honest, Core **collusion-mode only** |
| Mode direction | LR on **agent-average per run**; label = run mode |
| Residualize | ambient unit: `role_perp = role − proj_mode(role)` |
| Layers | L21, L22 (primary); PCA-contrast @ L23 in controls |

Scripts: `scripts/transfer_stable_controls.py`, `scripts/transfer_stable_residualize.py`

---

## Key numbers — controls

### Cross-test specificity @ L21 (Transfer final)

| Direction → target | AUROC | Interpretation |
|---|---:|---|
| Role → role | **0.859** | deception transfers |
| Role → mode (run-avg) | 0.766 | moderate mode leak |
| Role → mode (scenario primary) | **0.941** | strong via asymmetry scores |
| Mode → mode (run-avg) | **0.886** | task/mode transfers |
| Mode → role | **0.900** | **strong role leak** |

L22: role→role **0.873**; mode→role **0.887**; mode→mode **0.831**.

### Cosines (ambient)

| Pair | L21 | L22 |
|---|---:|---:|
| Role LR vs mode run-avg | **+0.694** | **+0.707** |
| Role LR vs mode sample-level | ≈0 | ≈0 |

### PCA-contrast @ L23 (same mixed pattern)

| k | Role→role | Mode→role | cos(role,mode) |
|---:|---:|---:|---:|
| 4 | 0.935 | 0.898 | 0.869 |
| 8 | 0.882 | 0.912 | 0.762 |

---

## Key numbers — residualize

| Layer | role→role | **role_perp→role** | role→mode | **role_perp→mode** |
|---:|---:|---:|---:|---:|
| 21 | 0.859 | **0.796** | 0.766 | **0.575** |
| 22 | 0.873 | **0.837** | 0.767 | **0.604** |

| Pair | L21 | L22 |
|---|---:|---:|
| role vs mode (before) | 0.694 | 0.707 |
| role_perp vs mode (after) | **0.000** | **0.000** |

Symmetric peel: mode_perp→mode stays high (0.819 / 0.836); mode_perp→role drops to ~0.55.

Directions: `../directions/lr_role_perp_mode_L21.npy`, `lr_role_perp_mode_L22.npy`, `meta.json`

---

## Interpretation (plain English)

1. **Mixed, not pure.** If the geometry were deception-only, mode→role would be near chance. If mode-only, role→role would be near chance. We see **both directions transfer to both targets**.
2. **Role is not reducible to mode.** After orthogonalizing out mode, role_perp still separates colluder vs honest on Transfer (~0.80–0.84) while mode leak falls toward chance-ish (~0.58–0.60).
3. **Practical:** steer/ablate along ambient role will likely move **both** role and mode geometry unless using role_perp (or collusion-only contrasts). Entanglement is a **specificity** issue for writing claims — it does **not** by itself explain the causal nulls (random and perp arms were also null).
4. **For matched-prefix redesign:** prefer interventions that flip **role** (or target commitment) while holding scenario/mode scaffolding fixed; report role_perp as the specificity check when claiming deception-axis causality.

---

## One-line claim (from source)

> YES — after removing mode, role still transfers (mode leak reduced): L21 role→role 0.859 → role_perp→role 0.796; role→mode 0.766 → role_perp→mode 0.575; cos(role,mode) 0.694 → 0.000.

## Safety bridge

Ambient mixed geometry → monitors may fire on **collusive configuration** (setup + role). Residualize → a **role-without-mode** axis still transfers, so the signature is not dismissed as task-ID only. Neither result implies we can **edit** collusion by scrubbing the ambient direction (`04`, `02`).
