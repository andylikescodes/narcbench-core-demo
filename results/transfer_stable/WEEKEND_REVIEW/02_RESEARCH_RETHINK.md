# Research rethink — beyond J, toward a coordination-signature claim

**Written:** 2026-10-01 ~5:15 PM PT · Transfer-stable lane · paper-shaped one-pager  
**Audience:** Andy weekend review · deeper than a single-site probe result

---

## Claim in one paragraph

Multi-agent collusion in NARCBench leaves a **Transfer-stable linear signature** that is **not** carried by J-space under Core→Transfer shift, **is** carried by mid-band residual geometry and especially **attention write-in at L22**, and is **entangled with run-level collusion setup** (role↔mode) in a way that suggests a **coordination / coalition-setup state** rather than a pure lying atom. That signature is **monitorable** after distribution shift; **editability is unproven** — free-regen last-token ablations were null, which we read as **weak causal evidence under divergent histories**, not as proof of epiphenomenality. The next science is **matched-prefix interchange** and **checkpoint timing**, not α-chasing.

---

## Beat 1 — From “probeable” to “where + what transfers”

**Prior bar (probe paper / NARCBench family):** colluding vs non-colluding structure is linearly readable from hidden activations. That is necessary but shallow for mechanism and for safety ops: a monitor needs to know **which subspace / site survives shift**, and an editor needs a **causal** handle.

**Our bar:** (i) locate relative to **J vs complement**, (ii) localize **residual vs mlp vs attn** write-in, (iii) test **Transfer** (Core fit → Transfer eval), (iv) stress **role vs mode**, (v) attempt **causal** edit under honest protocols — and when that fails, redesign the causal protocol rather than inflate correlational claims.

Positive correlational stack: HARD transfer full **0.902**; residual L21 **0.941**; **attn_L22 ≈ 0.999**; low-rank contrast PCA k=8 @ L23 **0.964**. Pointers: `03_CORRELATIONAL_CLAIMS.md`.

---

## Beat 2 — J fails Transfer; non-J residual / attn_L22 is Transfer-stable

| Locus | Transfer role readout | Reading |
|---|---|---|
| J-only (W_U / jac k25) | ~0.55–0.60 | Does **not** transfer |
| J complement | ~0.89–0.90 ≈ full | OOD signal outside J |
| Residual L21–L22 | 0.941 / 0.930 | Transfer-stable mid-band |
| MLP L19–23 | ≤0.915 (best L22) | Present, dominated by attn |
| **Attn L22** | **≈0.999** | **Primary Transfer-stable site** |

**Implication for “where”:** verbalizable / J-aligned features are the wrong Transfer monitor for this protocol. The transferable collusion-role information is a **non-J residual direction**, preferentially visible as an **attention residual-write at L22**. That is a localization claim the original full-activation probe paper does not make.

---

## Beat 3 — Role↔mode entanglement as mechanism evidence

Cross-tests (Transfer final @ L21): role→role **0.859**, mode→role **0.900**, cos(role, mode_runavg) **≈0.70**. Residualize mode out of role: role_perp→role still **0.796** (L22 **0.837**); mode leak collapses toward ~0.58–0.60.

**What this is not:** “the probe is just detecting the collusion task label.” Role survives an explicit mode peel.

**What this is:** ambient geometry mixes **episode setup** (we are in a collusion run) with **agent role** (I am a colluder). Safety-relevant gloss: the Transfer-stable readout tracks a **coordination / coalition-setup signature** — useful for **monitoring whether the model is in a collusive configuration** — more than a clean scalar of “this token is a lie.”

Writing discipline: say **coordination signature** (correlational); do not say “we found the collusion circuit” or “pure deception direction.”

---

## Beat 4 — Ablate nulls = weak causal evidence under divergent histories

All tested arms (residual α=1 and {2,4,8}; Core residual; attn_L22 final; attn_L22 discussion±final) left committee votes unchanged vs baseline/random (n=6). Votes often **literally identical** across arms.

Why this fails as a strong negative for “causal substrate exists”:

1. **Divergent prefixes** — free regen (esp. discussion) walks off the transcript the probe was fit on.
2. **Last-token residual-write only** — may miss earlier partner-read / commitment positions.
3. **Coarse discrete votes on tiny n** — zero flips is informative against large effects under *this* protocol, not against all protocols.
4. Discussion null rules out only the narrow story “final-only missed early lock-in” **for this intervention class**.

**Correct slogan:** **readability ≠ control** (under tested free-regen last-token ablations).  
**Incorrect slogan:** “attn_L22 is epiphenomenal.”

---

## Beat 5 — AI-safety claim (state this carefully)

### Claim (endorsed)

> **Monitorable transferable coordination signature.** On Gemma-2-9B-it NARCBench Core→Transfer, colluder-role information is linearly readable after shift, outside J, peaking at attention L22, entangled with collusion-run setup but not reducible to it after residualize. This supports **monitoring** for collusive *configuration* under distribution shift.

### Non-claim (explicit)

> **Editability unproven.** We do **not** claim that ablating or steering this direction controls collusive behavior. Free-regen last-token interventions were null; matched-prefix / interchange / within-trajectory tests are required before any control or “unlearning” narrative.

### Operational consequence

Monitors can **score** Transfer-stable attn_L22 / residual role (and role_perp) directions on held-out protocols. **Intervention / mitigation** research must change design (see `06_METHODS_LADDER.md`): matched-prefix micro-worlds, within-trajectory teacher-forced edits, checkpoint assays for commitment timing. **Paused:** residual-α chase; same free-regen ablate at larger n.

---

## Soft-stop

Track I / Paper 2 / interp-demo / family-house untouched. No GPU this pack.
