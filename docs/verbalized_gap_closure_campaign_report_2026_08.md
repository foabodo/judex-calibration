# Verbalized-Arm Gap-Closure Campaign — Report of Record (2026-08-08 → 2026-08-09)

One-page index + headline results for the P0/P1/P2 campaign that closed Paper B's
verbalized-arm evidence gaps ahead of v15. Every number here is a pointer — the
per-phase docs (§6) and run artifacts (§7) are authoritative.

**The report is in two parts.** **Part I (§1–§8)** is the campaign as it stood at commit
`87c5154`: P0 (ε + on-pair sweep), P1 (scaffold bound), P2 (defined-answer control), R1–R9.
**Part II (§9–§14)** is the E1 remediation arc that followed it — the control's base-leg
collapse, its diagnosis and remediation, the completed panel, and the two $0 DACA-facing
reads that closed the experimental programme. Part II resolves R6's open status and adds
R10–R15. **No Part-I result is revised by Part II**: B = 0.2971 stands, every Phase-1
verdict stands, and the r3/rA1 frozen constant sets are untouched throughout.

# PART I — P0/P1/P2 (2026-08-08 → 2026-08-09)

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

**Two Part-I rows carry forward pointers.** **R6 is superseded** — the base collapse was
diagnosed, remediated and re-measured in Part II (§9, §11); the panel is now complete and
gate-limited to one family, which is the result R6 left open. **R9's store is superseded as
the control's exemplar source** by `mmlu_control_exemplars_v2` (§9.3); the v1 store is
unmodified and remains the record of the Phase-2c legs, and R9's inter-scaffold sharpness
observation is unaffected (it is a property of the panel's organic credences, which v2
preserves by design).

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
  **CLOSED in Part II** (decision 1, §10): the user rejected the single-stage rescue as
  "the wrong direction" and directed content-difference forensics instead. What replaced it
  is the arc in §9.

The E1-dependent items above are re-scoped by §14; the control appendix now has a delivered
panel measurement to describe (§11, R10/R12) rather than an open collapse.

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

**This ledger is Part I only.** The whole-project ledger — Part I + the E1 arc + the $0
checks, reconciled against the vast credit trail — is **§12**, and the ops-lesson paragraph
below is extended by **§13**.

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
  needed); guard fires correctly; rev_order available. Store v2 is depth-2 and makes
  alt_set feasible on all six subjects at k=4 (three of six at k=8) — but D4 keeps it
  default-OFF and **no leg in the programme ever exercised it** (§9.3, §11 R11 uses the
  Phase-1 `study_b_*_k5v1` alt_set legs, which are a different object).
- Part-II gate record, for the same audit: base panel **1 of 4** gate-passing (qwen 95.00%;
  gemma31 80.00%, glm 31.67%, maverick 77.50% FAIL); qwen post v2 85.83% FAIL; the k=8
  iteration conceded at 82.50%. Details and diagnoses in §9 and §11.

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

Part II adds six documents and one corpus tree:

| Doc | Content |
|---|---|
| `docs/e1_gate_qwen_v2_20260809.md` | the paid GO gate: the three-change configuration, PASS 95.00%, failure taxonomy (two Phase-2c classes extinct), the Mac smoke ladder table, the explicit "Mode B is untested by this leg" caveat and the gemma31-first ordering it recommends |
| `docs/e1_recollection_results_20260809.md` | the stop-loss: gemma31 base 80.00%, Mode B 74 → 19, five of six subjects repaired, single-subject residue, mixed-pair E2 accounting |
| `docs/e1_k8_iteration_20260809.md` | the one sanctioned density iteration: 82.50%, maths 5/20 → 10/20 against `formal_logic` 19 → 14, the redistribution finding, the k=8 selector guards (kept) |
| `docs/e1_final_panel_results_20260809.md` | the arc doc: the three outstanding legs, panel 1-of-4, §4.1 matched-item scaffold entanglement (R10), the one clean qwen E2 pair, the $30.013 ledger and the $50-float substitution |
| `docs/daca_standing_assessment_20260809.md` | DACA C1–C4 disentangled (method / premise / direction / channel), the τ_DACA two-role split, transfer conditions (i)–(iv), v15 posture — **the required pre-read for any DACA-facing prose** |
| `docs/tau_daca_scaffold_variant_check_20260809.md` | the $0 check closing condition (iv): exact baseline reproduction, V1/V2 arms, the entropy mechanism, the matched-W1 breach, the sorted-iteration determinism note |
| `judex-corpus/leaf_exemplars/mmlu_control_exemplars_v2/{README.md,manifest.json,cost_ledger.json}` (branch `claude-mmlu-control-exemplars`, `eda9646`) | store v2: the failure taxonomy that motivated it, the two prompt amendments and nothing else, 364 rows / 333 kept, the span table vs v1 and AIReg, panel accuracy by subject and seat, the $2.59 ledger, and the selection note that produced CHANGE 1 |

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

Part II adds (same conventions — gitignored, MAIN checkout only):

- **Control under the v2 scaffold:** `runs/control_mmlu_{qwen,gemma31,glm,maverick}_v2/`
  ({pre,post}_control.json + meta + control_report.json; only the qwen dir has a post leg),
  `runs/control_mmlu_gemma31_v2k8/` (the conceded density iteration),
  `runs/control_scaffold_probe_v2band.json` (offline band-render probe). Every v1 dir
  `runs/control_mmlu_{qwen,gemma31,glm,maverick}/` was read-only throughout the arc.
- **Free Mac smoke ladder:** `runs/smoke_control_v2_qwen_mac/` —
  `{records,summary}_it{0,1,2,3}_*.json` for the four selection iterations (v1 store /
  v2 length-blind / v2 band / v2 band+TeX-hardened prompt). Qwen3-4B-Base Q4 on Metal;
  **numbers discarded**, the ranking is what was used.
- **Estimand-side reads:** `runs/tau_daca_scaffold_variants/tau_daca_scaffold_variants.json`
  (the $0 check; driver `scripts/tau_daca_scaffold_variants.py`). Its reference legs are the
  Phase-1 variant pre legs in `runs/study_b_{qwen,gemma31,glm,maverick}_k5v{1,2}/`, its
  targets the panel post legs in `runs/study_b_*`; the absolute-estimand panels it is read
  against are `runs/estimand_k5variants/estimand_k5variants.json`, and the AIReg base
  capability rows quoted in the DACA assessment are
  `runs/base_calibration_premise/base_calibration_premise.json`.
- **Corpus + data:** `judex-corpus/leaf_exemplars/mmlu_control_exemplars_v2/` (364 raw
  annotations, 333-row store, sha `d34d855e…`) and
  `data/mmlu_control_exemplar_candidates_v2.json` (52 items, sha `dba026fc…`; the v1 pool is
  a strict subset). The scored slice is unchanged: `mmlu_control_slice_v1.json`, sha
  `31822879…`, 120 items.
- **Not written anywhere:** the cross-family matched-item recomputations behind R10 and the
  mixed-pair E2 rows were run out-of-tree (session scratchpad) and deliberately kept out of
  every `runs/` dir, since none of those pairs is a clean single-scaffold pair.

## 8. Commit chain (branch `claude-verbalized-gap-closure`, PUSHED) — both parts

Part I: `19f6f11` P0 instrument+results → `82cd122` scaffold variants → `3e1c4ae` P1a
executed → `07ed19b` P1b $0 prep → `8a4c184` P1b 6/8 → `9d7d770` P1b closed 8/8 →
`7240eed` P2a design → `4073523` P2a revised → `42bbac4` P2b harness →
`22f7c76` P2c results → `87c5154` this report (Part I as first written).

Part II, same branch, in arc order: `ac9d059` store-v2 re-pin + candidates v2 →
`71300d4` E1 gate PASS 95.00% (CHANGE 1 + CHANGE 2 and their tests) → `5327351` stop-loss
fired on gemma31 base → `062393e` option (b) CONCEDED (k=8 selector generalization, kept) →
`3d0fef7` final panel, 1-of-4 → `2f21f74` the §7a arc stub (**replaced by Part II of this
document**) → `e7ed81b` DACA standing assessment → `0d5f41d` τ_DACA scaffold-variant check
(+ `scripts/tau_daca_scaffold_variants.py`).

Corpus (branch `claude-mmlu-control-exemplars`, PUSHED): `68e3cea` v1 store →
`eda9646` v2 store (span-matched, math-safe; 364 annotations, store sha `d34d855e…`).

Both branches are in sync with their remotes. **Neither is merged to develop.**

# PART II — the E1 remediation arc (2026-08-09)

Part II records everything between `87c5154` and `0d5f41d`. It is the story of one estimand
(E1, base-leg top-1 ECE on defined answers) that was attacked four times and closed as a
**complete panel measurement that does not deliver the estimand** — 1 of 4 base families
gate-passing — plus the two $0 reads that converted that outcome into a standing position on
DACA. Full records: the six documents added to §6; every number below is anchored to one of
them.

## 9. The arc, in order

### 9.1 The collapse being remediated

Phase 2c elicited all eight control legs, 120/120 cells each, and produced scored reports on
every family — but **zero of the four base legs cleared the `parse_ok ≥ 0.90` gate**: qwen
86.67%, maverick 79.17%, glm 50.00%, gemma31 36.67% (phase2c §1). E1 is defined over
gate-passing base legs, so it was not delivered. Phase 2c had already established that this
was task-specific rather than a machinery fault — on the *same box, same model, same
two-stage flow*, the AIReg prompt produced an 842-character reasoning span while the MMLU
prompt produced none (phase2c §4.2) — and that the failure was subject-correlated, with
`high_school_mathematics` the worst subject for every base leg (phase2c §4.3).

### 9.2 The user challenge, and what the forensics found

The user challenged the reading, on the grounds that a deterministic whole-panel failure is
as likely to be a bad prompt as a property of the checkpoints, and asked for reasoning about
what actually differs between the AIReg scaffold (which these same checkpoints parse at
0.917–1.000) and the control scaffold. The user **rejected** the proposed `--no-reason`
single-stage rescue as "the wrong direction" — it would have replaced the failing object
with a different harness rather than explaining it.

The forensics that followed classified all **177 base-leg parse failures** (corpus v2
README):

| mode | count | share | signature |
|---|---:|---:|---|
| **B — empty reasoning** | 153 | 86% | stage-1 span of 0–2 chars cascading to a stage-2 EOS |
| **A — LaTeX** | 15 | 8% | `Invalid \escape` (11) + `Expecting property name` (4) |
| `no_json_object` | 7 | 4% | non-degenerate span, no object |
| residual | 2 | 1% | |

Mode B was traced to a **content difference, measured**: the v1 control exemplars taught a
reasoning span roughly 3× shorter than the AIReg scaffold's — median **188** characters
against AIReg's **610**, with **56%** of rows under 200 characters against AIReg's **0%** —
so base checkpoints learned that the span is brief or optional and jumped straight to the
`JSON:` stop. It was worst on short symbolic subjects and mildest on prose-heavy
`professional_law`, and the *passing* base cells emitted spans mirroring the short
exemplars: the scaffold teaches what it shows. Mode A had two guises, both LaTeX: a
backslash inside a JSON string is an illegal escape, and a LaTeX brace *before* the JSON
hijacks the first-`{` scan — `\frac{x}{12}` hands the extractor `{x}`, which is balanced,
is not the answer, and fails `json.loads`. Only the first is reachable by store content.

> *Counting-convention note, so the taxonomies reconcile.* Three thresholds are in use
> across the record and they give three totals for Mode B: `raw_chars == 0` gives 133
> (phase2c §4.1: gemma31 62/76, glm 53/60, maverick 18/25, qwen 0/16); reasoning span
> ≤ 1 char gives 145 (the per-leg tables in the recollection, k=8 and final-panel docs); and
> the store README's ≤ 2-char signature gives 153. The percentages above are the store-side
> record and are the ones to quote; the per-leg tables are ≤ 1-char throughout and are
> internally consistent. One genuine inconsistency remains, flagged rather than papered
> over: the README's Mode-A brace count is **4**, while the gate doc §3 records **6**
> `Expecting property name` cells on qwen pre alone (the other three base legs record 0), so
> Mode A is 15 or 17 depending on the source. Nothing downstream turns on it — the class was
> driven to zero on every base leg by CHANGE 2 either way.

### 9.3 Store v2, and the two harness changes it implied

The user directed an Opus agent to run the v2 re-annotation together with a free Mac smoke
loop. Store v2 (`judex-corpus`, `eda9646`) is the same 52-item-class pool at depth 2, the
same 7-seat panel, same providers, same transport, same K=4 contract and the same
correct-answer filter, with **exactly two** prompt amendments: the justification must be a
substantial multi-sentence paragraph guided to **400–800 characters** that eliminates each
other option by name, and it must be **plain text — no LaTeX, no backslashes, no escape
sequences**. Nothing was said about sharpness or confidence: **the credences remain the
panel's organic ones**, per the standing principle, which is what keeps the store the same
generating process as the AIReg scaffold. Candidates were topped up to depth 2 so the
`alt_set` variant would be feasible.

Outcome: 52 items × 7 seats = **364 calls, 364 rows, no permanent failures** (v1 had one);
the correct-answer filter kept **333**; the new backslash validator dropped **0** — full
panel compliance across all 364 rows and all three string fields. Spans: min **370**, median
**810**, max 1509, **0%** under 200 (v1: 38 / 188 / 565 / 56.1%; AIReg: 246 / 610 / 1590 /
0%). The panel overshot the guidance, and the overshoot is not free. Store sha
`d34d855e…`, 333 rows, ≈**$2.59** against a $10 authorization.

The free Mac smoke ladder (Qwen3-4B-Base Q4, Metal, 13 items, greedy; numbers discarded,
ranking used) measured what a length-blind draw costs on the new store:

| iteration | selection | exemplar spans | parse rate |
|---|---|---|---:|
| it0 | v1 store, length-blind | 103–404 | 0.846 |
| it1 | v2 store, length-blind | 648–1509 | **0.538** (worst) |
| it2 | v2 store, **band** | 507–781 | **0.846** (best) |
| it3 | v2 store, band + TeX-hardened prompt | 507–781 | 0.692 |

Hence the two calibration-side changes, both landing at `71300d4`:

- **CHANGE 1 — band-targeted selection.** Within each (subject, letter) bucket the candidate
  pool is restricted to rows whose justification length is in **[400, 800]** before the
  existing least-used-rater walk. Deterministic, coverage-complete, with a documented
  nearest-the-band fallback; recorded as `exemplar_selection: "band_400_800"` in the leg
  meta with a resume guard that fires **even when a prior sidecar omits the key** — so a
  Phase-2c run dir cannot be continued under this harness.
- **CHANGE 2 — last-object parsing** on the raw control path (`parse_last_control`, the chat
  path's semantics) instead of the first-`{` scan. This is the only reachable fix for the
  brace-hijack, since no store content can stop a model restating TeX before its JSON.

The two-stage flow, contract, stop lists, K=4 grid, decoding parameters and the scored slice
were untouched. Offline pre-spend checks: suite 221 passed + 116 subtests; band render
637–798 across all six subjects, median 723, zero out of band; qwen worst-case prompt 3,142
tokens against the 32768 pin.

### 9.4 The paid gate — PASS

The agent's CONDITIONAL GO surfaced two decisions, both of which the user approved (pin
band selection; switch the raw control path to last-object parsing), along with a ~$3 qwen
gate. The gate ran on the family that came closest in Phase 2c and **passed at 95.00%**
(114/120, +8.3 points on the same 120 items, same model, same box class, same decoding),
for **$0.888** against a $15 cap.

What it established: the brace class went **6 → 0**; `high_school_mathematics` went
**10/20 → 19/20** with degenerate spans 6 → 2 *and the survivors now parsing*; the band draw
moved the model's own output onto the scaffold (passing median **793** against an exemplar
median of 723; in-band 18.3% → 43.9%; IQR 1079 → 534), and the residual failures sat in the
**right** tail. The residue was characterised and bounded: four `Invalid \escape` cells, all
in `econometrics`, on a subject with only 2 backslash-bearing stems — a **domain-notation**
effect (the model introduces LaTeX-shaped statistical notation of its own accord), worth
~3.3 points and not reachable by any further exemplar work.

What it explicitly did **not** establish, and the gate doc said so: **Mode B was untested**,
because qwen never had it (0/16 in Phase 2c, 0/6 here). The GO therefore carried a
sequencing condition — run **gemma31 base first**, the definitive Mode-B family and the
worst Phase-2c leg, as a ~$1 test before the two ~$25–28 8×H200 legs.

### 9.5 The stop-loss — FIRED

The user instructed the four-leg campaign with that stop-loss in place. gemma31 base parsed
**80.00%** — 10 points short — so the campaign halted at leg 1, the instance was destroyed,
and legs 2–4 were never rented. **$0.853** spent against a $55–75 exposure.

The leg is the arc's clearest "works, and is not sufficient" result: **+43.3 points** of
parse rate, empty-reasoning cells **74 → 19**, five of six subjects repaired (
`clinical_knowledge` **0/20 → 18/20**, restoring the subject that had been absent from the
Phase-2c cluster bootstrap entirely), and the tightest span-to-scaffold match seen to that
point (in-band **2.3% → 76.0%**, passing median 686 against an exemplar median ~723). And
the residue was **one subject**: `high_school_mathematics` supplied 15 of 24 failures and 15
of 20 degenerate spans while moving only 3/20 → 5/20. Had that subject merely matched the
leg's non-maths average (18.2/20), the leg would have parsed 90.8% — a pass.

### 9.6 The density iteration — CONCEDED

Presented with (a) stop, (b) a k=8 exemplar-density iteration, or (d) complete the panel,
the user chose **(b) first**. The k=8 draw is a strict **superset** of the k=4 draw (the
walk is deterministic and pass 1 is unchanged), so it is a single-variable density contrast
rather than a second scaffold; the exemplar band barely moved (k=8 median 722 against k=4's
718, 47/48 in band). `CONTROL_FEWSHOT_K` stayed **4** — k=8 was a per-leg flag.

It **conceded by its own pre-stated rule at 82.50%**, 7.5 points short and 5.5 below the
0.88–0.90 borderline band, for **$0.821**. The hypothesis was directionally right and
quantitatively insufficient: the maths cell **doubled**, 5/20 → 10/20, with left-edge
failures there 15 → 9 — the largest movement any instrument in the programme produced on
that subject. But `formal_logic` went **19 → 14**, losing five cells to the *same* left-edge
mechanism the iteration was meant to fix (degenerate spans there 2 → 6), which is a −5
regression against a −2 tolerance and fails the decision rule's second clause
independently. Two subjects gained 8, one lost 5, three were flat, net +3.

Two properties make this a concession rather than a "try k=12": the lever **redistributes
failure mass rather than removing it**, and it has begun to buy the opposite failure mode —
a single `high_school_mathematics` cell emitted a **5,887-character** span and produced no
JSON at all, the right-tail mechanism gemma31 had never shown (its k=4 maximum was 1,115).
Both tails are now live on one family. The exemplar-side levers are spent. The k=8 selector
guards were kept (they are a strict generalization: letter balance at `k/4` per letter, a
fail-loud non-multiple-of-4 rejection, per-letter `alt_set` counts, starvation reporting at
density).

### 9.7 The final panel — measurement complete, E1 not delivered

The user then chose **(d)**, confirming the credit top-up was already in place. The three
outstanding legs ran at the frozen k=4 v2 scaffold, every identity field byte-identical
across all six v2 legs in the programme. All three **FAIL**: qwen post **85.83%**, glm base
**31.67%**, maverick base **77.50%**. **$30.013** against a $55–70 estimate and a $90 cap
(33.3%); all 360 cells elicited on the first pass, no transient failures anywhere.

The base panel therefore stands at **1 of 4 gate-passing** (qwen 95.00%, ECE10 0.0802, item
bootstrap [0.0309, 0.1361]) — and **one gate-passing leg is not a panel**, the position both
predecessor documents took of that same leg. E1's pre-specified v15 consequence is recorded
verbatim in the arc doc §5.1 and is **not triggered**. What *is* delivered is the panel
measurement, and what it measures is R12: three of four base checkpoints cannot be brought
over the data-quality gate by any instrument in this design.

The campaign's own headline was that the v2 scaffold's effect is **family-dependent in
sign** — +8.33 / +43.33 parse points on qwen and gemma31, −18.33 / −1.67 on glm and
maverick — and §4.1 of that doc then showed the effect is not confined to parseability at
all, which is R10. The experimental programme was declared closed at that point.

### 9.8 The DACA standing assessment

The user asked for the standing position on DACA — method, premise, direction and channel
disentangled — and for clarity on τ_DACA's role, since the record contains two apparently
contradictory statements about it. The assessment (`docs/daca_standing_assessment_20260809.md`)
splits the four claims: **C2 (method) corroborated by us, twice, beyond its stated
assumptions**; **C1 (premise) ill-posed as a model-intrinsic property** per R10, false in
the judging condition, loosely held at n=1 under our harness, with the honest concession
that we never ran their scaffold-free logit slice; **C3 (direction) reversed in the judging
condition, null on our one clean MMLU pair, unrefuted in their setting**; **C4 (channel)
setting-dependent — vindicated for base-reference elicitation on defined answers by our own
$35 failure, inapplicable to production distributional judging.** Their measurements are
unchallenged; what falls is the imported generalization.

The τ_DACA contradiction resolves as two true statements about two roles: **as an adoption
instrument it is not useful and was never adopted**; **as a GT-free cross-check in the
verbalized channel it corroborates** under the deliberately wide F7 window. The assessment
lists four transfer conditions, of which (iv) — scaffold-conditionality inherited from the
references — was untested.

### 9.9 The τ_DACA scaffold-variant check ($0)

The user directed that check and its full documentation. It recomputes τ_DACA with the
Phase-1 **variant pre legs** as references (V1 `alt_set`, V2 `rev_order`), targets held at
the study of record — analysis-only on artifacts already on disk, no elicitation, no
network. It is **exploratory**: F7 was registered for the baseline scaffold only, so
applying its window to variant references is a diagnostic re-use of a frozen criterion, not
a new registration. The baseline reproduces to floating-point identity before any variant is
read (12/12, median 0.9224116756, matched-W1 0.9884871022), which is what makes the variant
arms readable. Result: R11.

## 10. Decision log

User decisions and their stated rationales, in order. These are the load-bearing turns of
the arc; the technical consequence of each is in the §9 subsection named.

| # | Decision | Stated rationale / condition | Consequence |
|---|---|---|---|
| 1 | Challenge the E1 failure as possibly a bad-prompt artifact; require reasoning about the **content difference** between the AIReg and control scaffolds. **Reject** the proposed `--no-reason` single-stage rescue | the rescue is "the wrong direction" — it substitutes a different harness for an explanation | §9.2; the Mode A / Mode B forensics, and the §3 OPEN item closed |
| 2 | Direct an Opus agent to run the v2 re-annotation plus a Mac smoke loop | span-matched (400–800) and math-safe plain text; **distributions stay organic** per the standing principle; candidates topped up for `alt_set` depth | §9.3; store v2, `eda9646`, $2.59 |
| 3 | On the agent's CONDITIONAL GO, **approve both** surfaced decisions — pin `band_400_800` selection, and switch the raw control path to last-object parsing — and approve the ~$3 qwen gate | the length-blind default had scored **worst** in the smoke ladder (0.538); the brace-hijack is unreachable from store content | §9.3–§9.4; CHANGE 1 + CHANGE 2 |
| 4 | On the gate PASS (95.00%) with its sequencing condition, instruct the four-leg campaign **with the gemma31-first stop-loss** | qwen never had Mode B, so the gate did not test the mode that broke the other three families; gemma31 is the ~$1 test of it | §9.5 |
| 5 | On the stop-loss firing (80.00%), choose **(b) k=8 iteration first** from (a) stop / (b) iterate / (d) complete the panel | the remediation demonstrably worked (74 → 19) and the residue was one subject; one more content lever was worth ~$1 before spending ~$50 | §9.6 |
| 6 | On **(b) conceding** by its pre-stated rule (82.50%; maths +5 but `formal_logic` −5 — density redistributes failure mass), choose **(d)**, confirming the top-up already in place | the exemplar-side levers are spent; the remaining value is a complete panel measurement, not a rescue | §9.7 |
| 7 | Accept the completed panel (E1 = 1-of-4, R10 discovered) and **declare the experimental programme closed** | three of four checkpoints are unreachable by any instrument in this design; further spend buys no estimand | §9.7, §14 |
| 8 | Ask for the standing DACA assessment (method / premise / direction / channel disentangled) and τ_DACA role clarity; then direct the **$0 scaffold-variant τ_DACA check** and its full documentation | the record contained two apparently contradictory statements about τ_DACA, and transfer condition (iv) was untested while the artifacts to test it were already on disk | §9.8–§9.9, R11 |

## 11. New results (R-series continued)

R10 and R11 are numbered as the brief specifies; R12–R15 are the other discrete results the
source documents support.

| # | Result | Where |
|---|---|---|
| R10 | **Measured base ECE on defined answers is scaffold-entangled, with family-dependent SIGN.** On the items each family parsed under *both* scaffolds (survivorship held fixed) and at mean confidence essentially unchanged (0.960–0.979): glm **−0.133** accuracy / ECE10 0.1122 → 0.2385 (n=30) and maverick **−0.138** / 0.1414 → 0.2774 (n=87), against qwen **+0.030** / 0.1033 → 0.0929 (n=101) and gemma31 **+0.070** / 0.2740 → 0.2147 (n=43). Leg-level, this is glm's ECE10 moving 0.0789 → 0.2378 and maverick's 0.1378 → 0.3016: **behaviour, not survivorship.** Consequence: "base model X has ECE ≈ y" is a property of the (model, elicitation) pair, and every cross-scaffold ECE comparison in the programme inherits it | final-panel doc §4.1, §8 anomaly 3 |
| R11 | **τ_DACA is scaffold-robust at the F7 tolerance, and only at that tolerance.** Baseline reproduces exactly (12/12 valid cross-family fits, median 0.9224116756, [0.7675, 1.0727]; matched-W1 median 0.9884871022). Both variants: **12/12 valid, all inside [0.5765, 2.3060], all four targets PASS** — V1 `alt_set` median **0.8105** [0.7317, 1.0559], V2 `rev_order` median **0.8309** [0.6685, 1.0855]. **First F7 crack, under the matched-W1 read:** V2's `qwen ← gemma31` fits **0.4529**, below the lower edge (ln margin −0.241, ≈3.2 grid steps), failing the qwen target under W1 while passing under the RPS fit of record (0.6796); not a saturation and not a weak-identification artifact (W1 moves 21.3% there, the strongest identification any W1 τ_DACA fit has shown). **Margin erodes**: closest valid fit above the lower edge, baseline +0.286 → V1 +0.238 → V2 +0.148 (RPS). **The point estimate moves 0.10–0.26 ln** (Δ ln median −0.1294 / −0.2606 for V1 under RPS / W1; −0.1045 / −0.1430 for V2). **The shift tracks Δ entropy, not Δ `T_abs(pre)`**: reference mean normalized entropy falls on all four legs under both variants; qwen's `T_abs(pre)` is inert (+0.012 ln) yet its τ falls with its entropy, while maverick's `T_abs(pre)` inflates most under V1 (+0.497 ln) yet its τ moves **up** (+0.041) — transfer is ~20–28% of the reference-side movement, because the argmax-agreement filter suppresses the location component that dominates the absolute estimand. Condition (iv) is restated, not struck: cross-check role intact across three scaffolds; adoption case, if anything, weaker | τ_DACA check §2–§4, §6; assessment §5 |
| R12 | **Whether a base checkpoint elects to reason at all is a CHECKPOINT PROPERTY, not reachable by scaffold content — now on three of four families.** gemma31 80.00% (19 empty-reasoning cells, one subject), glm 31.67% (79 cells, four subjects), maverick 77.50% (25 cells, one subject); failing-cell span distributions are pinned at the left edge (p25 = median = p75 = **0** for glm and maverick, **1** for gemma31). Robust to span length (store v2), density (k=8) and parsing (last-object). Meanwhile the band instrument works on all four families and is **orthogonal to the gate**: passing-cell medians **667–793** against an exemplar median of 718.5, in-band 43.9–78.9% — and glm has the tightest median match in the programme (667) with its worst parse rate (31.67%). Two of the three failures turn on a single subject: gemma31 and maverick would each pass at 90.8% if `high_school_mathematics` merely matched their own non-maths average, while glm would still fail at 58/120 with a perfect maths cell | final-panel doc §3–§5, §4 |
| R13 | **The one scaffold-consistent MMLU pair is null.** qwen with both legs on v2/k=4/band/last-object: pre 0.0802 (n=114), post 0.0819 (n=103), **delta +0.0017**, item bootstrap **[−0.0461, +0.0752]** on 99 paired items (subject-clustered [−0.0502, +0.0568]; intersection-only +0.0118; bootstrap median +0.0141) — a CI 70× the point estimate. Removing the scaffold confound did not resolve the qwen row; it confirmed there was never a signal in it. The other three families' E2 rows remain MIXED and **both new mixed rows flipped from CI-spanning-zero to CI-significant *because of* the confound** (glm −0.0262 → −0.1851, maverick −0.0633 → −0.2271), driven by the v2 **pre** leg being degraded per R10 rather than the post leg improving. R7's 4/4 direction stands as recorded on the v1 pairs; the mixed rows must never be quoted as strengthening it | final-panel doc §6.1–§6.2 |
| R14 | **The free Mac smoke rig is a validated bug detector with a known blind spot.** it0 (v1 store, length-blind, 13 items) reproduced Mode A at small scale — two `Invalid \escape` cells, both in `high_school_mathematics`, parse rate 0.846 — the same class in the same subject the full-scale v1 legs failed on; the ladder then ranked the three candidate selections correctly (length-blind v2 worst at 0.538 through literal newlines in JSON strings and a dropped distribution key; band best at 0.846; band + TeX-hardened prompt 0.692), and that ranking is what CHANGE 1 was pinned on and what the paid gate then confirmed at full scale. **Blind spot: zero empty-reasoning cells in all four iterations.** The rig never once produced Mode B — the mechanism that decided the arc. A 4B/Q4 proxy on 13 items tests JSON formation, not whether a checkpoint elects to reason | gate doc §1.2; `runs/smoke_control_v2_qwen_mac/summary_it*.json` |
| R15 | **The exemplar pipeline detected answer-key noise in MMLU.** `mmlu:professional_law:validation:0002` drew a **unanimous 7/7 panel dissent** from the MMLU answer key — every seat answered C against a reference of D — and was removed by the correct-answer filter, which left the `(professional_law, D)` bucket one source item deep and forced a one-item top-up at the same rule. Two things follow: the correct-answer filter is doing real work rather than trimming noise (an exemplar store must not teach an answer the whole panel rejects), and depth is restored by adding candidates, **never** by relaxing the filter. Recorded because the same seven seats annotate the AIReg-facing corpus stores | corpus v2 README; `cost_ledger.json` top-up row |

Bound discipline held in Part II exactly as in Part I: **B = 0.2971 is not revised**, no
Phase-1 verdict is revised, `CONTROL_FEWSHOT_K` remains **4**, the MMLU τ_TVD values are
descriptive under D6 and never enter the frozen r3/rA1 constants, the F7/A-series constants
are untouched, the evaluator seam stays `mode: noop`, and **no gate-failing leg is promoted
anywhere**.

## 12. Cost ledger — whole project

| Phase | Spend | Note |
|---|---:|---|
| Part I (P0 + P1a + P1b + P1b closure + P2b + P2c) | $283.72 | §4 for the row-level breakdown |
| Store v2 re-annotation (API/OpenRouter, corpus repo) | $2.589 | 364 calls; measured OpenRouter credit delta $2.027432 + unmetered est. $0.196 + deepseek $0.0834 + mistral $0.2824; $10 authorization |
| E1 gate — qwen base v2 (1×H200) | $0.888 | $15 cap, 5.9% |
| E1 re-collection — gemma31 base v2, stop-loss (1×H200) | $0.853 | $90 cap, 0.9%; legs 2–4 never rented |
| E1 k=8 iteration — gemma31 v2k8 (1×H200) | $0.821 | $10 cap, 8.2% |
| E1 final panel — qwen post + glm base + maverick base (1×H200 + 2×8×H200) | $30.013 | $90 cap, 33.3%; est. $55–70 |
| Mac smoke ladder (it0–it3), DACA standing assessment, τ_DACA scaffold-variant check | $0.000 | local / analysis-only on artifacts already on disk |
| **E1-arc subtotal** | **$35.164** | ≈ **$35.17** |
| **Project total** | **$318.884** | ≈ **$318.88** (**$318.89** if the store line is taken at its own rounded ≈$2.59, which is what the superseded §7a stub printed) |

Vast credit trail, which reconciles the four paid legs end to end (each invoice polled to
stability before the next rental): Phase-2c close **$42.437626** → gate −$0.888010 →
**$41.549616** → *user top-up +$25.00* → **$66.549616** → stop-loss −$0.852982 →
**$65.696634** → k=8 −$0.821470 → **$64.875163** → final panel −$30.013804 →
**$34.861359**. Every leg reconciles to a fifth of a cent or better against its settled
invoice sum; no auto top-up fired at any point; `vastai show instances` returned `[]` after
every teardown and at close.

Two reconciliation notes. **(i)** No single source document carries the cumulative arc
ledger — the final-panel doc §7 carries only its own $30.013 campaign, and the store line
sits in the corpus repo's `cost_ledger.json` — so the arc subtotal above is assembled here
from the five source ledgers. **(ii)** The store line is a mixed measured/estimated figure
(the credits-endpoint delta is authoritative for the five OpenRouter seats; the contract
smoke and a killed pass-1 fragment are unmetered), so the project total is precise to about
a cent, not to the tenth of a cent the vast rows reconcile to — which is the whole of the
$318.88 / $318.89 difference above.

## 13. Ops lessons — updated

Part I's list stands (autobill $5-increments cannot feed an 8×H200 download; teardown-time
invoices under-report; no SSH keys on the vast account; same-machine re-rental can hit
provider disk cache; build ALL harnesses before renting shared checkpoints). Part II
confirms, quantifies and adds to it:

1. **The invoice under-report rule earned its keep on five of six rentals, and the band is
   wider than Part I recorded.** First-post-teardown reads under-reported by **10.1%** (gate),
   **7.7%** (stop-loss), **15.2%** (k=8 — the worst observed, six reads to stability),
   **11.1%** (qwen post) and **8.9%** (glm base). The maverick leg read **$14.138 → $14.138**
   with **no** under-report at all — the first in the programme — and was therefore deliberately
   re-polled after three further minutes rather than closed on the two-read rule alone.
   **Poll until two consecutive reads agree, and treat a first-read match as a reason to
   re-poll, not to stop.**
2. **The $50 single-increment float rule is arithmetically unsatisfiable below ~$65 credit,
   and the substitution is a runway check.** With $64.88 at a campaign's start, any first
   8×H200 leg over $14.88 puts the second below $50 — and the measured cost was $15.29, so
   the third rental began at **$49.001, $1.00 (2.0%) short**. Topping up is not an action a
   session can take. The check applied in its place, stated so it can be audited: $49.001 at
   $34.74/hr is **84.6 minutes** of continuous burn against a measured **26-minute**
   comparable leg — **3.0× margin** — with the campaign at 17.6% of its hard cap. The rental
   settled at $14.138. A 2% deviation from a written rule should be visible rather than
   buried.
3. **Unsorted set iteration is a `PYTHONHASHSEED` determinism trap in any filtered fit.**
   The τ_DACA agreement filter must iterate the target∩reference label overlap in **sorted**
   order. The unsorted form varies floating-point summation order run to run — negligible in
   the objective (~1e-13 relative) but enough to move one golden-section refine in the 7th
   significant digit (`gemma31 ← glm` baseline: 1.0577130720 vs 1.0577129007). Sorted, the
   driver reproduces bit-for-bit and lands on the study-of-record value; unsorted, a
   reproduction attestation would have failed for no scientific reason.
4. **No SSH keys on the vast account remains binding** — there is no warm-box model swap, so
   every leg is a fresh rental plus a full weight download. The mitigation that worked is
   **offer selection on cheapest expected TOTAL charge rather than cheapest `$/hr`**: with
   downloads dominating the Phase-2c 8×H200 bills ($12–13 per rental at $0.019/GB), placing
   both giant legs on a host with `inet_down_cost` $0.00013/GB billed the 716.7 GB GLM and
   803.2 GB Maverick pulls at **$0.08 and $0.09**.
5. **The free smoke rig detects real bugs and has a documented blind spot** (R14). Use it to
   *rank* candidate scaffolds before spending — it ranked the three selections correctly and
   its ranking survived at full scale — and never to certify one, because it cannot produce
   the mode that decided this arc.
6. **A stop-loss on the cheapest diagnostic leg is worth its ordering constraint.** The
   gemma31-first condition cost ~$1 and avoided ~$50–56 of 8×H200 spend on a hypothesis that
   was false; the panel was completed later, deliberately, as a measurement rather than as a
   rescue.

## 14. State at close

**The experimental programme is CLOSED.** Every estimand this campaign set out to move has
been moved or has been measured to be unreachable: R1–R5 delivered P0 and the scaffold
bound, R7/R8 delivered the control's direction and DACA-gap reads, and R6's open status is
replaced by a completed base panel that is **gate-limited to a single family** (R12) on a
scaffold that moves the level estimand itself (R10). The exemplar-side instruments — the
store-v2 span floor, the band draw and the one sanctioned density iteration — are spent, and
the remaining checkpoint failure is a property of the checkpoints. No further spend buys an
estimand; the fleet is empty and no instance is alive.

**The sole remaining deliverable is the v15 re-scope plus a single authoring pass.** §3 is
the direction of travel, but it must be re-scoped first: it was written presuming a
delivered E1, and what exists instead is a 1-of-4 panel measurement whose honest description
is the control appendix's job. Two Part-II results carry into that pass with real
argumentative weight — R10 (base ECE is elicitation-conditional, which upgrades the premise
critique from "fails in our setting" to "ill-posed as a model-intrinsic property") and R12
(the elicitability contrast, which is simultaneously the concession that DACA's channel
choice is right for DACA's setting).

**Required pre-read for any DACA-facing prose: `docs/daca_standing_assessment_20260809.md`.**
It is the assessment of record, its §5 is filled by the τ_DACA scaffold-variant check, and
the DACA-facing sentences in §6.2 and the control appendix must carry exactly its three-way
split — **method confirmed, premise ill-posed, channel choice setting-dependent** — with
their Fig. 1 cited as elicitation-conditional and never as refuted.
