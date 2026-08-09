# Verbalized-Arm Gap-Closure Campaign — Report of Record (2026-08-08 → 2026-08-09)

One-page index + headline results for the P0/P1/P2 campaign that closed Paper B's
verbalized-arm evidence gaps ahead of v15. Every number here is a pointer — the
per-phase docs (§6) and run artifacts (§7) are authoritative.

## 1. What the campaign was

Paper B v14 carried three disclosed gaps: (i) a scaffold confound measured only on the
superseded token-slice/k=4 arm ("documented rather than bounded"); (ii) an ε-floor
sensitivity sweep promised but never reported (and run on only 2/8 leg pairs); (iii) a
defined-answer control named and not run. Governing directives: P1 (logit experiments
superseded by verbalized equivalents), P2 (k=4 deprecated; coverage rule k ≥ #categories),
P3 (no logit-channel discussion in the paper except others' work, e.g. DACA),
serving-parity (API only at matched precision; prefer self-hosted vLLM both legs).
Plan of record: `judex/spec/plan_2026_08_08_verbalized_arm_gap_closure.md`; pre-specified
reads: `judex/spec/memo_2026_08_08_phase0_prespecified_reads.md` (both umbrella spec/,
written before the corresponding data existed).

## 2. Headline results

| # | Result | Where |
|---|---|---|
| R1 | ε-floor sweep completed 8/8: every shipped τ_v reproduces at ε=0.005; T_J (1.1531) identical at the two lowest floors; ≤2 clustering rule floor-invariant. Boundary notes: maverick 1.4862 at ε=0.001 (above the published band edge it defines); qwen leaves its 1.2811 grid point at ε≥0.0125 | phase0 doc §Task-1 |
| R2 | On-pair τ-range sweep (first ever on the run of record): F4 acceptance region contiguous [1.0477, 2.000]; T_J=1.153 inside (bootstrap pass 0.931); band's lowest 7.3% of log-width fails; instrument reproduces the frozen 9-point band_profile EXACTLY and the arm-2 supervised fit (2.5446 ≈ 2.545) | phase0 doc §Task-2 |
| R3 | **Scaffold sensitivity (k=5 coverage-preserving perturbations, V1 alt-set / V2 rev-order, full 4-family panel): the INCREMENT collapses — V1 panel all four τ_v = 1.0252 (point collapse; published band gone); V2 {0.9518, 1.0252×3} (lower edge exits the band); the transferred-constant analog → 1.0252 under both = the grid point that fails F4 on the on-pair sweep. The ABSOLUTE survives — ≤2 ratio holds with margin (1.241/1.116 vs 1.296), LOFO holds (0.1820/0.0856 ≤ 0.191), medians 2.2402/2.2032 vs 2.353** | phase1b doc §4.3–4.5 |
| R4 | Pooled scaffold bound: **B = 0.2971 (BOUNDED-LARGE)**, three-way tie at exactly 4 grid steps (grid quantization); gate-restricted computation equals the full one; max Δargmax 16.67 pp. Family typology: qwen τ_v-sensitive/argmax-stable; gemma31 τ_v-inert/argmax-sensitive; glm τ_v-inert/T_abs-moves; maverick both-move/argmax-stable. alt_set inflates T_abs(pre) on every family | phase1a §read + phase1b §3–4 |
| R5 | Serving-transport diagnostic (free): OpenRouter vs self-hosted bf16 vllm-chat, same weights/scaffold — identical τ_v grid point, Δln T_abs +0.0006, argmax −3.63 pp; aggregate-indistinguishable, supports the shipped baseline pairing | phase1a doc §transport |
| R6 | **MMLU control E1 (base ECE level): UNRESOLVED — all four base legs fail the parse gate.** Bases deterministically skip the reasoning span on the MMLU prompt (stage-1 ~1 char, stage-2 empty; same box/model/flow yields 842-char reasoning on AIReg). A harness–task interaction finding, not a null; survivorship subsets fenced, not promoted (glm .079 / qwen .100 / maverick .138 / gemma31 .290) | phase2c doc §E1 + diagnostics |
| R7 | MMLU control E2 (direction): post-training improves ECE 4/4 (CI-significant 1/4: gemma31 −0.256) — AIReg's direction, opposite DACA's | phase2c doc §E2 |
| R8 | **DACA-gap: gate-passing post legs (ECE .034–.075) sit INSIDE DACA's scaffold-free base band at matched K=4** — the verbalized harness does not manufacture AIReg-scale miscalibration on defined answers. Capability contrast: bases 0.68–0.90 argmax over the 0.25 floor (competent test-takers, weak judges) | phase2c doc §DACA |
| R9 | Panel-annotated MMLU exemplar store: 237 rows, panel accuracy .857–1.000/subject, correct-answer filter documented; organic store much sharper than AIReg's (median peak 0.95, 38% one-hot) — a real inter-scaffold difference to name in v15 | corpus manifest + phase2c doc |

Bound discipline held throughout: no control outcome revises B or any Phase-1 verdict.

## 3. v15 consequences (agreed direction; RE-SCOPE §6 of the design before authoring — it presumed a delivered E1)

- P3 removal pass (all token-slice content out; DACA-style citations stay) — removal
  inventory in plan §1h; scoping-table traps still apply (readout τ ≠ logit; log-prob
  proxies ≠ logit arm).
- Scaffold appendix: B = 0.2971 headline; increment claims (τ_v set, band, T_J transfer)
  restated scaffold-conditional; absolute-estimand robustness a positive result.
- App. A gains the ε outcome (R1); App. C/G re-anchored to the on-pair sweep (R2).
- Control appendix: E2 + DACA-gap + the base-collapse as a named scope-limiting result;
  Limitations "We did not run that control" replaced accordingly; §6.2 closing sentence
  stays "present before alignment" strengthened by R8.
- Single ledger remap (baseline v14 per the claims.yaml header note); twin lockstep;
  twin-abstract staleness fixes (pre-v11 wording; A–H vs A–I listing).
- OPEN (user, undecided): labeled single-stage (--no-reason) base rerun ≈$55 to rescue a
  fenced base-ECE read — harness-variant, reported as such if run. Current lean: skip.

## 4. Spend ledger (invoice-exact; all instances destroyed, fleet verified [] after each phase)

| Phase | Spend | Note |
|---|---:|---|
| P0 | $0 | analyses of stored data |
| P1a | $7.104 | qwen + gemma31 variants; incl. $0.174 orphan API cells (annex) |
| P1b | $132.843 | glm complete; incl. $22.699 dead rental (top-up-rate incident) |
| P1b closure | $32.571 | maverick post pair (machine-cache hit) |
| P2b exemplars | $1.46 | 7-seat panel annotation, 251/252 |
| P2c | $109.742 | 8 control legs; no dead rentals |
| **Total** | **$283.72** | vs the plan's ~$220–350 batched envelope |

Ops lessons recorded: autobill $5-increments cannot feed an 8×H200 download (need ≥$50
single-increment float); teardown-time invoices under-report ~17% (poll until two reads
agree); no SSH keys on the vast account (no warm-box model swaps); same-machine
re-rental can hit provider disk cache (time win only); build ALL harnesses before
renting shared checkpoints ([[joint-infrastructure-campaign-planning]]).

## 5. Gates and known blemishes (for any later audit)

- P1a: qwen rev_order post contract 0.80 FAIL (recorded, not repaired); P1b: glm V1 post
  0.8833 FAIL; glm V2 post exactly 0.9000 PASS; maverick closure V1 post = the campaign's
  only contract-1.0000 leg. Bound is independent of gate-failing legs (phase1b §3).
- P2c gates 3/8 PASS (the three post legs excl. qwen post 81.67% — JSON syntax);
  all base legs FAIL (R6).
- Stale-T_rel collateral: 6 study_b reports refreshed a stale auxiliary T_rel during the
  ε pass (cause = committed Murphy bin fix ca13157; field unused by any paper number;
  backups `study_b_report.pre_eps_backup_20260808.json` retained in each dir).
- glm-4.5 permanently fails one exemplar annotation item (unquoted JSON; recorded in
  annotation_failures.json; item keeps 6 seats).
- alt_set infeasible on the v1 control store (≥2 source items per letter per subject
  needed); guard fires correctly; rev_order available.

## 6. Per-phase documents (all committed on judex-calibration branch `claude-verbalized-gap-closure` unless noted)

| Doc | Content |
|---|---|
| `docs/phase0_verbalized_sensitivity_20260808.md` | ε-sweep 8/8 tables, on-pair τ-range, integrity attestations, draft v15 sentences |
| `docs/phase1a_scaffold_sensitivity_results_20260808.md` | V1/V2 qwen+gemma31 legs, transport diagnostic, cost ledger conventions |
| `docs/phase1b_scaffold_sensitivity_results_20260808.md` | full campaign incl. closure: per-leg tables, V1/V2 PANEL TABLES (§4.3–4.5), pooled bound, top-up incident (§6), ledgers |
| `docs/phase2_defined_answer_control_design.md` | D1–D8 (user-confirmed), pre-specified E1/E2 interpretation tables (§3), build plan |
| `docs/phase2c_control_results_20260809.md` | control results: per-leg tables, E1/E2 reads, DACA table, base-collapse diagnostics, ledger |
| `judex/spec/plan_2026_08_08_verbalized_arm_gap_closure.md` (umbrella, UNCOMMITTED) | the governing plan: P1/P2/P3 directives, removal inventory, phasing |
| `judex/spec/memo_2026_08_08_phase0_prespecified_reads.md` (umbrella, UNCOMMITTED) | pre-specified interpretation tables, written before data |
| `judex-corpus/leaf_exemplars/mmlu_control_exemplars_v1/{README.md,manifest.json,cost_ledger.json}` (branch `claude-mmlu-control-exemplars`) | exemplar construction provenance, roster, filter stats |

## 7. Run artifacts on disk (gitignored; MAIN checkout only)

judex-calibration `runs/`:
- Scaffold variants: `study_b_{qwen,gemma31,glm,maverick}_k5v{1,2}/` (legs + meta +
  reports + READMEs), `study_b_gemma31_vllmchat_base/` (matched-transport baseline post),
  `study_b_gemma31_api_k5v1/` (74-cell orphan API annex, transport-labeled).
- Panel reads: `estimand_k5variants/estimand_k5variants.json` (all baseline/V1/V2 panels).
- Phase 0: `q4_range_onpair_20260808/` (+eps0005 variant); per-leg
  `study_b_*/study_b_report.json` now with `epsilon_sensitivity` (backups alongside).
- Control: `control_mmlu_{qwen,gemma31,glm,maverick}/` ({pre,post}_control.json + meta +
  control_report.json), `control_scaffold_probe.json`, `scaffold_variant_probe*.json`.

judex-corpus: `leaf_exemplars/mmlu_control_exemplars_v1/` (prompt snapshots, 251 raw
annotations, store sha `c4174d45…`). judex-calibration `data/`:
`mmlu_control_slice_v1.json` (sha `31822879…`), `mmlu_control_exemplar_candidates_v1.json`
(sha `1a477bb8…`).

## 7a. E1 remediation arc (2026-08-09, post-dating §2's R6; supersedes R6's open status)

Full record: `docs/e1_recollection_results_20260809.md`, `docs/e1_gate_qwen_v2_20260809.md`,
`docs/e1_k8_iteration_20260809.md`, `docs/e1_final_panel_results_20260809.md` (the arc doc).

- Diagnosis: Phase-2c base collapse = Mode B empty-reasoning 86% (short exemplar spans;
  left-edge skip on short symbolic items) + Mode A LaTeX 8% (escapes + brace-hijack of
  the first-{ JSON scan).
- Fixes (all scaffold-content or parsing; two-stage harness untouched): store v2
  (span-matched median 810, math-safe, $2.59; organic credences preserved), band_400_800
  selection, last-object parsing. k=8 density iteration CONCEDED (82.50%; redistributes
  failure mass, formal_logic −5).
- **Final panel: E1 = 1 of 4 gate-passing.** qwen base 95.00% PASS (ECE10 0.0802,
  DACA-like row — not promotable as a panel); gemma31 80.00% / glm 31.67% / maverick
  77.50% FAIL via the same left-edge mechanism = a CHECKPOINT PROPERTY across 3/4
  families, robust to span length, density, and parsing.
- **R10 (new negative result): measured base ECE on defined answers is
  scaffold-entangled** — the v2 scaffold moves both-scaffold-item accuracy
  glm −13.3 / maverick −13.8 pts at unchanged confidence (~2× ECE) while gaining
  qwen +3.0 / gemma31 +7.0. Sign-level family dependence: the defined-answer control
  independently reproduces the Phase-1 lesson that base-model measurements are
  scaffold-conditional.
- qwen E2 (only fully-v2 pair): +0.0017 [−0.0461, +0.0752] — null. Other families' E2
  is unrescuable within this design (their PRE legs fail gates regardless of post
  re-collection).
- E1-arc spend: $2.59 + $0.888 + $0.853 + $0.821 + $30.013 = **$35.17**; campaign
  total **$318.89**. Ops: invoice under-report reproduced 8–17% (poll to stability);
  the $50-float rule is arithmetically unsatisfiable below ~$65 credit — a documented
  runway-check substitution (§7.1 of the arc doc) is the working alternative.

## 8. Commit chain (branch `claude-verbalized-gap-closure`, PUSHED)

`19f6f11` P0 instrument+results → `82cd122` scaffold variants → `3e1c4ae` P1a executed →
`07ed19b` P1b $0 prep → `8a4c184` P1b 6/8 → `9d7d770` P1b closed 8/8 →
`7240eed` P2a design → `4073523` P2a revised → `42bbac4` P2b harness →
`22f7c76` P2c results → this report. Corpus: `68e3cea` on
`claude-mmlu-control-exemplars` (PUSHED). Neither branch merged to develop yet.
