# Absolute-mechanism decision protocol — rA1

**Date frozen:** 2026-07-24 · **Status:** FROZEN at commit time. Test against A1–A8; never edit.
**Estimand:** the **absolute** correction `T_abs` — the RPS-optimal temperature that carries a
judge's verbalized credence distribution onto the AIReg-Bench human ground truth.
**Relation to r3:** r3 (`e6_onpair_decision_protocol_r3.md`) registers the **ratio** estimand
`τ_v` (post→base, no ground truth) and its transferred constant `T_J = 1.153`. rA1 is a separate,
parallel registration for a different estimand on the same deployed pair. r3 stays frozen and
untouched; rA1 does not modify, reinterpret, or supersede any r3 constant.

**Reference object (stated explicitly, because it was previously mislabelled).** Every fit in this
protocol scores against `judex_calibration.aireg.load_cells()` → `Cell.gt_probs`: the AIReg-Bench
ground truth, a **continuous credence distribution reconciled by the MG-MFRM cumulative-consistency
model over three human legal-expert annotators** (`rater_id` 0/1/2), read out at τ = 0.675. It is
**not** the seven-seat **LLM** exemplar panel in `judex-corpus`, which supplies few-shot material
only and is firewall-disjoint from the ground truth. Reference properties, recomputed 2026-07-24:
mean max-probability 0.4607, median 0.3798, 0/120 cells above 0.9, mean normalized entropy 0.7686,
Murphy uncertainty 0.06814, argmax majority-class rate 0.3333.

**Validation status of the reference is `unavailable`** (max R-hat 1.0123 > 1.01; prior-predictive
not computed). Safe to use; must never be described as a validated benchmark. The reference's
dispersion is in part a modelling choice (the τ = 0.675 readout) — see §5.

---

## 0. Frozen constants

| id | decision | value |
|---|---|---|
| **A1** | Panel membership | **{qwen, gemma31, glm, maverick}** — the r3 F1 panel, carried over **verbatim** from the 2026-07-21 user decision. Not re-selected here. Registered sensitivity (report-only, non-gating): the all-six set adding {llama31, gemma26}. |
| **A2** | Capability floor | each panel member's **verbalized post leg** must show Murphy **resolution > 0 at every bin count in {3, 5, 10, 15, 20}**, with the minimum **≥ 0.03**. Margin provenance: the Study A capability gate's observed pass/fail separation (passing legs 0.0327–0.0509, failing legs 0.0122–0.0326). Diagnostic, non-gating: argmax above the **0.333** GT majority-class floor. |
| **A3** | Clustering gate | the panel `T_abs(post)` set must satisfy **max/min ≤ 2** — the same factor-2 rule adopted in Study A Q3 (where 3.05 > 2 produced the NEGATIVE verdict) and mirrored by r3 F5. Carried over unchanged; **not** derived from the spread observed here. |
| **A4** | Transferred constant | **T_A = median of the panel `T_abs(post)` set.** |
| **A5** | Sensitivity band | the panel range **[min, max]** of `T_abs(post)`; the closed read is published at both edges as well as at T_A. |
| **A6** | Saturation | no panel member's fit may sit at a `T_BOUNDS = (0.25, 20.0)` endpoint, and T_A must be strictly interior. A boundary value is a peg, not a fit. |
| **A7** | Acceptance on the deployed pair | `study_a.closed_side_check(preds, cells, T_A)` at `MURPHY_BINS = 10`: **Murphy Reliability strictly better ∧ mean RPS no worse (≤ +1e-9) ∧ Resolution within 10 % of uncalibrated.** This is r3 F4 carried over unchanged. |
| **A8** | Objective discipline | **RPS throughout** — panel fits, T_A, and the closed acceptance read. No W1-fitted quantity may be divided by, or compared as commensurable with, an RPS-fitted one; no cross-objective decomposition may be reported. |

### Constants instantiated at freeze (panel side only — no closed-pair quantity was computed first)

| quantity | value |
|---|---:|
| `T_abs(post)` qwen · Qwen3.5-35B-A3B | 2.6365 |
| `T_abs(post)` gemma31 · Gemma-4-31B | 2.2417 |
| `T_abs(post)` glm · GLM-4.5 | 2.4639 |
| `T_abs(post)` maverick · Llama-4-Maverick | 2.0351 |
| **A2** min resolution over members × bins | **0.0536** (≥ 0.03 → PASS, 1.8× margin) |
| **A2** diagnostic: all members above 0.333 argmax floor | yes |
| **A3** ratio max/min | **1.2956** (≤ 2 → PASS) |
| **A4** **T_A** | **2.3528** |
| **A5** band | **[2.0351, 2.6365]** |
| **A6** any member saturated | no; T_A strictly interior |
| A1 sensitivity: all-six median / ratio | 2.3779 / 1.2956 |

The all-six sensitivity moves T_A by 1.1 %, and the clustering ratio is unchanged (the extremal
members are the same two families). Panel membership is therefore not load-bearing.

---

## 1. The confirmation read

**Target:** the executed E6 on-pair sweep, `judex-evaluator/runs/stage9-onpair-e6-20260721`
(120 cells, Sonnet 4.6 + GPT 5.4 — the deployed closed evaluator pair).
**Instrument:** `e6_r3_arms.py`'s `load_closed` and `study_a.closed_side_check`, reused **unforked**.
`load_closed` reads only `prediction` from the run's `metrics_report.json`; ground truth always
comes from `aireg.load_cells()`.

**Verdict rule.** rA1 returns **CONFIRMED** iff A2, A3 and A6 all pass on the panel **and** A7
passes on the deployed pair at T_A. Any single failure is terminal for this revision: the constant
is not transferable, and the absolute mechanism stays a reported finding rather than a confirmed
one. Band-edge and sensitivity reads are published alongside but never rescue a failed A7.

---

## 2. What rA1 does and does not establish

| claim | status if rA1 passes |
|---|---|
| The absolute correction transfers from the open panel to the deployed closed pair | **confirmed against this benchmark**, in the registered form of §3 |
| The magnitude of the correction is a property of the task and its human reference | **supported, not proven** — invariance across scale/lineage/architecture is evidence, not identification |
| The correction generalizes to other documents, other rubrics, or other closed pairs | **not established** — see §3 |
| T_A is a precise constant | **no** — a ±20 % single-measurement noise band applies (bf16 CoT jitter, Study A finding); read T_A as ≈ 2.35 |

---

## 3. The honest limit (read before citing rA1)

**This is a registered-form confirmation, not a blind one.** The analyst who wrote this protocol had
prior exposure to a closed-pair read at T ≈ 2.353 (carried in the 2026-07-24 session handoff) before
freezing these constants. Every constant here is nevertheless defined by a rule fixed independently
of that read — A1 is a prior user decision carried over verbatim, A3 is the Study A Q3 factor-2 rule,
A7 is r3 F4 unchanged, A4 is a plain median — and no constant was tuned to a closed-side outcome. But
the protocol cannot claim the blindness that T_J's registration had. **rA1 must never be described
as pre-registered in the sense r3 was.** The accurate phrasing is: *the absolute mechanism satisfies
a registered acceptance criterion on the deployed pair, under constants carried over from prior
frozen decisions.*

**It is also in-sample for the generalization the deployment argument needs.** The panel constant and
the closed read are both fit and scored against the same 120 AIReg-Bench cells and the same
three-expert reference. What is out-of-sample is the *model population* (open panel → closed pair);
what is in-sample is the *task, the documents, and the reference*. The deployment claim leans on
both, and only the first is tested here.

**What would make it genuinely out-of-sample** (none executed; costs are estimates, all require
explicit user authorization to spend):

| option | what it buys | cost |
|---|---|---|
| Held-out document split of the 24 AIReg documents (fit T_A on 12, read on the other 12, doc-clustered) | removes item-level in-sampleness; cheap and immediate | **$0** — analysis-only on existing artifacts; weakest of the three (same reference, same rubric) |
| A second deployed closed pair (a different vendor pairing) read at the same T_A | tests transfer across the closed population, the actual deployment claim | one E6-scale sweep; order **$20–60** in API spend, plus wall-clock |
| A different benchmark with a dispersed human reference, run through this harness | tests whether the magnitude is a property of *this* task or of dispersed judging generally — the strongest test, and the one the invariance interpretation actually predicts | largest: requires a second GT construction, or an existing multi-annotator distributional benchmark; the MMLU control named in `analysis_2026_07_21_base_calibration_premise.md` is **not** a substitute (one-hot reference, opposite ECE optimum) |

---

## 4. Explicit non-interference

rA1 changes no config, adopts nothing, and does not touch the evaluator seam (stays `mode: noop`).
It does not modify r3, and it does not license any edit to the papers: promoting the absolute
mechanism to "confirmed primary correction" in `judex_paper_v6_absolute_calibration.tex` is a
separate user decision, taken with §3's limits in hand.

## 5. Standing caveat on the reference

The reference's dispersion (mean max-probability 0.461) is jointly a fact about three experts
disagreeing and a consequence of the MG-MFRM readout temperature τ = 0.675, which was tuned blind
by entropy-matching and re-pinned by user decision. Any absolute temperature measured against it —
T_A included — is measured against *that* object and is not comparable to temperatures fit against
the pre-2026-07-09 (τ = 1, snapped, pooled) labels. A reviewer may reasonably ask how much of T_A
tracks the readout choice; rA1 does not answer that, and the question is not rhetorical.
