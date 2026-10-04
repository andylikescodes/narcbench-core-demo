> **GitHub snapshot, 2026-10-02.** Paths that begin with `/workspace/narcbench-core-demo/` are this repository. Paper PDF: [papers/transfer-stable-collusion-2026-10/main.pdf](../papers/transfer-stable-collusion-2026-10/main.pdf). Study pack: [results/transfer_stable/WEEKEND_REVIEW/](../results/transfer_stable/WEEKEND_REVIEW/). Attachment inventory: [SNAPSHOT.md](../results/transfer_stable/WEEKEND_REVIEW/SNAPSHOT.md).

# NARCBench Weekend — FINAL HANDOFF

**Date:** 2026-10-01 ~5:21 PM PT  
**For:** Andy  
**Status:** **CLOSED for weekend handoff** — science freeze folded into PDF; no GPU launched; Track I soft-stopped; causal paused.

---

## One-line safety claim

**Monitorable, Transfer-stable coordination signature (`attn_L22` / non-J residual) — editability unproven under tested free-regen interventions; matched-prefix + checkpoint assays are the next causal designs (not residual-α chasing).**

---

## Canonical paths

| Item | Absolute path |
|---|---|
| **PDF (14 pp, solid; science freeze folded)** | `/workspace/andy-biz/papers/transfer-stable-collusion-2026-10/main.pdf` |
| Paper build dir | `/workspace/andy-biz/papers/transfer-stable-collusion-2026-10/` |
| **WEEKEND_REVIEW pack** | `/workspace/narcbench-core-demo/results/transfer_stable/WEEKEND_REVIEW/` |
| **CODE_INDEX (code review map)** | `/workspace/narcbench-core-demo/results/transfer_stable/WEEKEND_REVIEW/CODE_INDEX.md` (symlink: `/workspace/andy-biz/papers/transfer-stable-collusion-2026-10/CODE_INDEX.md`) |
| Pack README (cross-links PDF) | `/workspace/narcbench-core-demo/results/transfer_stable/WEEKEND_REVIEW/00_README.md` |
| **Andy-biz skim** | `/workspace/andy-biz/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md` |
| Cache inventory | `/workspace/andy-biz/NARCBench-Transfer-Stable-Cache-Inventory-2026-10-01.md` |
| **Claims number sheet (Mira lock)** | `/workspace/narcbench-core-demo/results/transfer_stable/WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET.md` (+ `.json`) |
| **Claims checklist** | `/workspace/narcbench-core-demo/results/transfer_stable/WEEKEND_REVIEW/CLAIMS_CHECKLIST.md` |
| Track I soft-stop note | `/workspace/track-i-intention/notes/SOFT_STOP_2026-10-01.md` |
| This FINAL handoff | `/workspace/andy-biz/NARCBench-Weekend-FINAL-HANDOFF-2026-10-01.md` |
| Results root | `/workspace/narcbench-core-demo/results/transfer_stable/` |

---

## Verification checklist (this close-out)

| Check | Result |
|---|---|
| Paper exists | **Yes** — `main.pdf` 440368 bytes, **14 pages**, A4; rebuilt 2026-10-01 **5:21 PM PT**; title *Transfer-Stable Collusion Representations…*; Author Andy Liang |
| Paper solid | **Yes** — clean `pdflatex` finish; no undefined refs/citations; WEEKEND_REVIEW figures + freeze tables folded |
| Mira CLAIMS_NUMBER_SHEET / CLAIMS_CHECKLIST | **Found + locked** — `CLAIMS_NUMBER_SHEET.md` / `.json` + `CLAIMS_CHECKLIST.md` (~5:20 PM PT) |
| Science freeze → PDF | **Folded in** — cross-phase discussion residual **0.959**, attn_L22 discussion **0.997**, domain-holdout L21 mean **0.939**, family AUROC @ L21; figs `fig_cross_phase_residual`, `fig_causal_null_flat`, WEEKEND_REVIEW j/attn exports |
| Claims sheet | **Yes** — frozen sheet is canonical number source for typeset |
| WEEKEND_REVIEW complete | **Yes** — spine `00`–`07`, `PACK_MANIFEST.json`, `design/`, `scripts/`, `figures/`, `tables/`, claims sheet, **`CODE_INDEX.md`** |
| PDF cross-linked in `00_README.md` | **Yes** |
| Balance | **~$26.24** (last known after discussion-phase attn_L22 smoke ~4:37 PM PT; live API not queried) |
| Causal GPU | **Paused** pending matched-prefix redesign |
| Track I | **Soft-stopped** — not touched; marker at `track-i-intention/notes/SOFT_STOP_2026-10-01.md`; no GPU / Track I process launched |
| Paper 2 / interp-demo / family-house | Soft-stop honored (untouched) |

---

## What is closed (process)

1. Mechanism skim rewritten and packed.
2. Weekend review PDF built and verified (**14 pp** after science-freeze fold-in).
3. WEEKEND_REVIEW consolidated study pack complete + PDF path in README.
4. Mira science freeze locked (`CLAIMS_NUMBER_SHEET` + checklist + tables/figures) and **folded into** `main.tex` / `main.pdf`.
5. Causal free-regen battery documented as null; further same-design GPU paused.
6. Soft-stops held (Track I soft-stop note written); no new spend / no GPU this close-out.

---

## Truly open? (only if blocking)

**No blocking open todos for this weekend handoff.**

Deferred research (not weekend blockers; already designed, GPU later only with approval):

- Matched-prefix interchange smoke (design ready under `WEEKEND_REVIEW/design/`).
- Within-trajectory + checkpoint assays (specs ready).
- Offline leftovers (discussion-phase eval; domain-holdout; family AUROC) — **done and in PDF** (Section science-freeze robustness).

Explicitly **do not** do: residual-α / larger free-regen n on the current harness; Track I / Paper 2 / interp-demo work this weekend.

---

## Suggested weekend read order

1. This FINAL handoff (paths + status).
2. PDF: `/workspace/andy-biz/papers/transfer-stable-collusion-2026-10/main.pdf`
3. Or skim: `/workspace/andy-biz/NARCBench-Transfer-Stable-Weekend-Review-2026-10-01.md`
4. Number check: `WEEKEND_REVIEW/CLAIMS_NUMBER_SHEET.md`
5. Deep dive: `WEEKEND_REVIEW/01` → `07` as needed.
6. Code review map: `WEEKEND_REVIEW/CODE_INDEX.md` (or paper-dir symlink `papers/transfer-stable-collusion-2026-10/CODE_INDEX.md`).

---

## Soft-stop reminder

- **Do not launch GPU** for Transfer-stable causal until matched-prefix (or equivalent) is approved.
- **Do not touch Track I** / Paper 2 / interp-demo / family-house under this soft-stop.
