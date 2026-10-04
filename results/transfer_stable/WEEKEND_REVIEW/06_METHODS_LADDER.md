# Methods ladder — design-first (offline now; GPU later)

**Written:** 2026-10-01 ~5:10 PM PT  
**Standing greenlight exists but this task is consolidate + design.** Default: **no GPU**.  
**Do not do yet:** residual-α chase; larger-n free-regen last-token ablate of the same class.

**Why this ladder exists:** correlational half of the safety claim is in place (**monitorable Transfer-stable coordination signature**); causal half (**editability**) needs matched-prefix / within-trajectory / checkpoints — see `02_RESEARCH_RETHINK.md` beat 5.

---

## Ladder order (cheapest / strongest evidence first)

| Step | Method | Cost now | GPU later? | Evidence strength if it works |
|---:|---|---|---|---|
| **0** | Finish design specs + pair templates + dry-run scaffold from Core transcripts | **$0 offline** | no | Spec clarity |
| **1** | **Matched-prefix micro-worlds** — build prompt pairs; classical path-patch / activation interchange | Spec $0; tiny GPU smoke only if needed | **Yes, small** (teacher-force + patch; not full regen battery) | **Strongest** classical causal |
| **2** | **Within-trajectory mid-discussion interventions** — teacher-force shared history; intervene mid-turn | Spec $0 | Yes, after (1) smoke | Strong; keeps probe substrate |
| **3** | **Checkpoint behavioral assays** — private / post-discussion / pre-vote forced choices | Spec $0; some assays may need gen | Optional light GPU | Locates *when* commitment appears |
| ~~4~~ | ~~Residual α chase / free-regen last-token ablate scale-up~~ | — | **Do not do yet** | Weak under divergent histories |

---

## 1. Matched-prefix micro-worlds

**Goal:** Prompt pairs that share a **token prefix until a single intervention point**, then differ only by role and/or target commitment — so path-patch / activation interchange is well-defined.

### Why Core scenarios are not already matched-prefix

Existing Core collusion vs control pairs share `scenario_id`, `seed`, agent names, and `target_option`, but **system prompts diverge early** (colluder vs honest role text). So raw collusion/control transcripts are **not** classical matched-prefix pairs — early tokens already differ. Free-regen ablate inherits that problem.

### Spec (minimal)

1. **Shared stem:** scenario briefing + agent identity + committee options — identical tokens.
2. **Intervention suffix (one of):**
   - **Role flip:** short role instruction (“You are colluding toward option B” vs “You are honest”) placed **after** the shared stem, OR
   - **Target flip:** same role, flip private target letter only at a marked position.
3. **Evaluation prompt:** identical final-vote (or next-token) template after the divergence point.
4. **Interchange:** run A and B with teacher-forcing on the shared prefix; at divergence position (or chosen layer×position), copy activations A→B / B→A; measure ΔP(target), vote string, KL.

Full writeup: `design/matched_prefix_spec.md`  
Schema + examples: `design/matched_prefix_pair_schema.json`, `matched_prefix_example_pairs.json`  
Dry-run scaffold: `scripts/matched_prefix_scaffold.py`

### Success metrics

- **Behavioral:** colluder-stem pairs vote target at non-trivial rate; honest-stem pairs do not (or flip).
- **Causal:** patching colluder activations into honest stem (at attn_L22 / residual L21–L22 last shared or first divergent position) moves P(target) / vote toward collusion **more than** random-direction patch.
- **Specificity:** role_perp direction ≥ ambient role, or ambient role ≫ mode-only control.

### GPU estimate (later)

- Design + template build: **$0**.
- Smoke (2B or 9B, n≈6–12 pairs, teacher-force + patch, no long free discussion): likely **≪ prior free-regen smokes** (minutes–tens of minutes on L4) if prompts are short micro-worlds.
- Full free multi-round regen **not** required for interchange smoke.

---

## 2. Within-trajectory mid-discussion interventions

**Goal:** Same agent, same history (teacher-forced from an existing Core collusion transcript). Intervene **mid-generation** or at a **mid-discussion turn boundary** without regenerating divergent prefixes for the whole committee.

### Spec sketch

1. Load Core collusion transcript; reconstruct per-agent prompts at turn *t*.
2. Teacher-force tokens 0…k−1 from the recorded utterance (or full prior turns).
3. At position k (e.g. first generated token of a colluder public turn, or last token before vote), ablate/steer/patch attn_L22 (or residual L21–L22).
4. Continue generation; measure whether subsequent talking points / final vote move.
5. Controls: random direction; role_perp; honest-agent same position.

Full writeup: `design/within_trajectory_spec.md`

### When to run

After matched-prefix smoke clarifies whether the direction is causally usable at all under matched computation. Within-trajectory answers “where in the real multi-round protocol.”

### GPU

Requires model forward with hooks; still cheaper than regenerating full committees if limited to single-agent continuations. **Do not launch** until step 1 design is approved.

---

## 3. Checkpoint behavioral assays

**Goal:** Measure **when** collusion commitment appears, so interventions target the right phase.

### Assays (forced choice / probe)

| Checkpoint | Probe | What it tells |
|---|---|---|
| Post-private (R0) | Forced choice: “Which option will you ultimately support?” (hidden) | Early private commitment |
| Mid-discussion | Same forced choice after turn *t* | Whether public talk locks intent |
| Pre-vote | Probe P(target letter) / forced recommendation before free vote gen | Immediate pre-decision state |
| Activation readout | Score attn_L22 / residual role direction at each checkpoint | When linear readout peaks |

Full writeup: `design/checkpoint_assays_spec.md`

### GPU

Private/post-discussion probes may need short gens or logit reads on existing prompts — can start with **logit-only on reconstructed prompts** (teacher-force) offline-ish once a local/CPU path exists; practical 9B still wants GPU. Spec now; run later.

---

## Do not do yet

| Anti-item | Why |
|---|---|
| Residual-α chase (α≫8, multi-layer α grids) | Same weak design; prior α∈{2,4,8} already null |
| Scale free-regen last-token ablate to n=36/72 | Amplifies a weak test; cost without design fix |
| Track I / Paper 2 / interp-demo / family-house | Soft-stop |
| Billable GPU for anything above without Andy/Paige OK on matched-prefix plan | Pause rule |

---

## What can start **now** (no GPU)

- [x] This ladder + rethink pack  
- [x] `design/matched_prefix_spec.md` + schema/examples  
- [x] `design/within_trajectory_spec.md`  
- [x] `design/checkpoint_assays_spec.md`  
- [x] `scripts/matched_prefix_scaffold.py` dry-run from Core transcripts  
- [ ] (Andy) Approve matched-prefix micro-world content + smoke budget before any pod
