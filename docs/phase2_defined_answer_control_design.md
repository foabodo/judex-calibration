# Phase 2a — Defined-Answer Control: Design (2026-08-09)

Status: DESIGN FOR USER REVIEW — no build, no spend. Decisions D1–D8 below need user
confirmation; recommendations are marked. Companion recon: the file-precise engineering
spec is summarized in §5 (full details in the 2026-08-09 recon report, reproduced in the
session record).

## 0. Objective and scope

Run the adoption-panel base/post pairs through the **identical verbalized elicitation
harness** on a benchmark with defined answers, and report top-1 ECE + accuracy there.
This is the control named (and declared un-run) in Paper B v14 §6.2 and Limitations, and
demanded verbatim by the adversarial-reviewer construction in
`spec/analysis_2026_07_21_base_calibration_premise.md:273` ("Run your bases on MMLU under
this same harness and show the ECE is still 0.3, or scope every claim").

What it licenses: the choice between two readings of the AIReg base-miscalibration
finding — task/reference/scaffold-specific vs generic-under-this-harness — and the
standing of §6.2's direction claim (post-training improves ECE 6/6 on AIReg; DACA finds
the opposite on MMLU).

What it does NOT do:
- It is **not** the "second dispersed-reference benchmark" open item
  (`docs/absolute_mechanism_protocol_rA1.md:134` — one-hot reference, opposite ECE
  optimum; that control remains separate and out of scope).
- It is **not** a DACA replication. DACA's Fig. 1 MMLU numbers (base ECE 0.034–0.069,
  post 0.190–0.242) come from a **direct next-token slice over A–D with no scaffold**
  (`analysis_2026_07_21_base_calibration_premise.md:245`). Our control elicits through
  the verbalized contract + few-shot scaffold. DACA is a qualitative comparand — the
  *within-harness* AIReg-vs-MMLU contrast is the deliverable. (P3 note: DACA is other
  researchers' work; discussing its logits method is the sanctioned exception.)

## 1. The two estimands

- **E1 (LEVEL):** base-leg top-1 ECE on MMLU under the identical harness, against the
  AIReg values as printed in v14 (six bases 0.175–0.443; the five floor-clearing legs
  0.175–0.398, median 0.260 — NOT the pre-v13 0.175–0.445/median-0.282 figures from the
  2026-07-21 analysis doc).
- **E2 (DIRECTION):** paired post−pre ECE delta on MMLU per family
  (`boot_paired_ece_delta`, B=2000, seed 20260721), against the AIReg direction (post
  improves in 6/6, CI-significant in 4) and DACA's MMLU direction (post degrades 3–7×,
  logit-slice elicitation).

Conventions (frozen, = the existing battery): confidence = max_j p_j after the ε-floor;
correct = 1{argmax pred = key}; ECE10 equal-width / AECE5 equal-mass / MCE /
over-confidence gap / AUROC; T_ece and T_aece by grid argmin over `study_a.GRID`;
cross-fitting only if reported (in-sample binned argmins are optimistically biased).

## 2. Design decisions

**D1 — Benchmark: MMLU (recommended).** It is the benchmark the paper already compares
against (DACA Fig. 1), the one the reviewer construction names, and the literature's
default calibration setting. MedMCQA/MathQA add nothing here; MMLU-Pro's 10 options
would break both the coverage-rule scaffold economy and the DACA comparand. Note K=4
matches DACA's K exactly — the current v14 text compares K=5 AIReg ECE to K=4 DACA ECE
across chance floors (1/5 vs 1/4); the control at K=4 is *better* matched to the
comparand than the paper's existing contrast.

**D2 — Slice: 120 items, 6 subjects × 20 (recommended).** 120 mirrors the AIReg cell
count. Six subjects spanning domains and difficulty (proposed:
`formal_logic`, `high_school_mathematics`, `professional_law`, `clinical_knowledge`,
`philosophy`, `econometrics` — user may swap; criterion: domain spread + a difficulty
gradient so accuracy isn't pinned at ceiling or chance). `criterion_id = subject` so the
per-criterion few-shot cache builds one exemplar block per subject (6 blocks × k=4 = 24
exemplars to author — vs 228 if we stratified across all 57 subjects, which is the real
reason not to). Deterministic selection: first 20 test-split items per subject by
canonical index after excluding any item textually colliding with an exemplar. The slice
is **vendored as committed JSON** with a pinned `slice_sha256` (stdlib-only doctrine; no
`datasets` dependency in the elicitation path), sidecar-pinned with an explicit
resume-guard clause.

**D3 — Contract: 4-option nominal, confidence channel KEPT, findings DROPPED
(recommended).** The control contract emits `answer_distribution` over {A,B,C,D} on the
0.05 grid, `answer_letter`, `answer_justification`, plus the unchanged 3-level
`confidence_distribution` + `confidence_justification`. `findings` (requirement/status/
evidence against a legal criterion) has no defined-answer analogue and is dropped rather
than stubbed. Consequences, stated honestly in the paper: the identical-harness claim is
scoped to *scaffold structure + two-stage flow + decoding params + parse_ok-tier gate*,
not to the six-field contract_complete tier (which cannot be the same object without
findings). Keeping the confidence channel costs nothing and makes B-Q4-style correctness
calibration *cleaner* on MMLU than on AIReg (the correctness event is natively defined);
its W1-risk companion is swapped for 0/1-error risk (two call sites).

**D4 — Exemplar policy: the load-bearing decision.**
- Structure: k=4, one exemplar per option letter (the coverage rule k ≥ #categories,
  one-per-category — the same rule that made k=5 mandatory on the 5-level task; this is
  not a return to the deprecated k=4, which was defective only because 4 < 5).
- Source: MMLU dev/validation split for the six subjects, never the scored slice
  (firewall by construction). Dev has 5 items/subject; where the four answer letters
  aren't all covered, fill from validation.
- **Exemplar distributions + authoring — REVISED 2026-08-09 (user question resolved
  this):** the earlier draft proposed hand-authoring 24 rationales with distributions
  tuned to match the AIReg store's sharpness profile. SUPERSEDED. **Recommended route:
  adapt the corpus-v2 annotation pipeline** (`judex-corpus/leaf_exemplars/`: cli.py
  annotate → synthesize; validation + provenance layers reused). MMLU dev/validation
  items enter as the excerpts (no authoring stage — the questions exist); the SAME
  7-seat rater panel (deepseek-v4-pro, mistral-large-2512, qwen3.5-35b-a3b, maverick,
  glm-4.5, kimi-k2-thinking, gemma-4-26b) annotates them under an MCQ-adapted contract;
  synthesis emits store rows with full panel provenance. This DISSOLVES the
  one-hot-vs-invented-dispersion tension: exemplar distributions are the panel's organic
  credences — the same generating process that produced the AIReg scaffold's
  distributions, nothing invented or tuned. Rater overlap with elicited families
  (qwen/maverick/glm/gemma seats) matches the AIReg scaffold's existing, sanctioned
  provenance property (corpus-side authorship, firewall-irrelevant) — so it makes the
  harness MORE identical, not less. Format-coupled build items: a new MCQ task template
  (the current one has no options slot), an MCQ system frame, K=4 key tuples in a
  control synthesis path (`_DISTRIBUTION_KEYS` is a module constant), and the store-row
  level-key becomes the answer letter. `_validate_findings` is already lenient (absent
  findings → []), matching D3. Est. annotation cost ~$2–4 (7 raters × ~24–40 candidate
  rows). Selection then reuses the stratified fixed_set walk with `rater_model`
  balancing intact. Fallback route (if the pipeline adaptation stalls): the superseded
  sharpness-matched hand-authoring. The alt_set/rev_order variant machinery remains
  available on the control scaffold for a $0-code robustness check (OPTIONAL, default
  off).

**D5 — Legs: pairs on the four panel families (recommended) = 8 legs.** E2 (the
direction claim) is a paired claim; bases-only (~half the cost) would test E1 alone and
leave §6.2's direction contrast uncontrolled. Transport per the serving-parity directive,
replicating each family's AIReg transport exactly: qwen raw-completions bf16 both legs;
glm and maverick raw bf16 both legs; gemma31 raw bf16 pre + **self-hosted vllm-chat bf16
post** (the matched-transport pairing established in Phase 1a — the OpenRouter route is
not used). llama31 stays excluded; gemma26 stays capability-gated (its AIReg exclusion
reasons don't transfer to MMLU, but panel identity does — the control controls the
panel, not the excluded families).

**D6 — τ_v companion on MMLU: descriptive-only, TVD objective, appendix-grade
(recommended: compute, don't headline).** `fit_tau_oc` is GT-free and would run, but its
W1 objective is order-dependent on nominal options — meaningless as shipped. A TVD
objective drops into the same grid-argmin pattern and is permutation-invariant. Even so,
an MMLU τ_v is a different estimand (different task, different reference regime) and
**must not** enter the frozen r3/rA1 cluster/band/LOFO constants. Cost: $0 beyond the
legs. Deliverable status: one descriptive table, clearly fenced.

**D7 — Scoring + uncertainty: item-level bootstrap primary, subject-clustered
sensitivity (recommended).** MMLU items are independent draws in a way AIReg cells
(24 documents × 5 articles) are not; a 6-subject cluster bootstrap has too few clusters
for stable CIs. Both run through the existing machinery unchanged (`document_id` = item
id for i.i.d., = subject for the sensitivity read). Metrics per §1; nothing ordinal is
emitted (no RPS, no W1-vs-key, no Murphy — `score_variant` is never called on control
records; a dedicated scoring module enforces this).

**D8 — Cross-K comparability: state, don't correct.** ECE at K=4 sits over a 0.25 chance
floor vs 0.20 at K=5. The paper's contrast sentences quote both numbers with the floors
named and decline a numeric equality — same posture v14 already takes toward the
DACA comparison, now with one fewer axis of difference.

## 3. Pre-specified reads (written before any datum exists)

**E1 interpretation table** (base legs, MMLU ECE10, gate-passing families):

| Outcome | Read | v15 consequence |
|---|---|---|
| MMLU base ECE ≲ 0.10 (DACA-like) | AIReg base miscalibration is task/reference/scaffold-specific; the harness does not manufacture it on defined answers | §6.2 keeps the reversal, now same-harness; Limitations scope *tightens to the judging condition* — the strongest honest version of the current text |
| 0.10–0.20 (intermediate) | partial harness contribution; scoping language stays conditional | §6.2 reports the split; "present before alignment" stands |
| ≳ 0.20 (AIReg-like) | miscalibration is generic under this harness — incl. on defined answers | strengthens "present before alignment"→"harness-general"; BUT invites "the verbalized harness inflates ECE" — adjudicated against DACA's scaffold-free 0.034–0.069: a large gap over DACA at matched K implicates the elicitation mode, and the paper says so |

**E2:** if post improves ECE on MMLU too, the AIReg direction finding generalizes across
tasks under this harness (and the DACA-direction contrast becomes an elicitation-mode
contrast, not a task contrast). If post degrades on MMLU (matching DACA), the AIReg
improvement is task-specific and §6.2 is rescoped accordingly. Either outcome is
reportable; neither threatens the sharpness-gap, oracle-control, or repairability
results (v14 already notes those "hold without it").

**Bound discipline:** no outcome of this control revises B = 0.2971 or the Phase-1 panel
verdicts; scaffold sensitivity and the defined-answer control are separate appendices.

## 4. Elicitation plan and cost (real-cost basis from Phases 1a/1b)

8 legs × 120 items, --workers 8, fresh run dirs `runs/control_mmlu_<family>/`
({pre,post}_control.json), one rental per checkpoint with ALL that checkpoint's control
legs on the warm box (joint-planning rule: this is the last planned elicitation on these
checkpoints — nothing further to batch; if any future experiment needing these boxes is
proposed before 2c runs, it must join this campaign).

| box | legs | est. |
|---|---|---:|
| qwen base + qwen post (1×H200 ×2) | 2 | ~$3–5 |
| gemma31 base raw + post vllm-chat (1×H200 ×2) | 2 | ~$5–8 |
| glm base + post (8×H200 ×2) | 2 | ~$70–80 |
| maverick base + post (8×H200 ×2) | 2 | ~$60–70 |
| **total** | 8 | **~$140–165** |

Hard cap proposal: **$220**. Credit-float rule from Phase 1b: ensure ≥$50 available in a
single increment before each 8×H200 rental (the $5 autobill trickle cannot feed a
download); same-machine re-rental is worth attempting for disk-cache wall-clock savings,
never relied on. Teardown + fleet-empty verification per standing practice; invoice-exact
per-leg ledger.

## 5. Build plan (Phase 2b — one focused session, $0)

From the 2026-08-09 engineering recon (sizes S/M; file-precise anchors in the recon):
1. **`src/judex_calibration/mmlu.py`** (S–M): vendored-slice loader → duck-typed items
   (`item_label`, `evidence_text`=stem, `criterion_text`=rendered options,
   `criterion_id`=subject; `gt_labels`=(A..D), `gt_argmax`, one-hot `gt_probs`,
   `document_id` per D7). Slice JSON committed with sha256.
2. **Control exemplar store via the corpus pipeline** (M, per revised D4): adapt
   `judex-corpus/leaf_exemplars/` — new MCQ task template + system frame + K=4 control
   synthesis constants; run `annotate` (7-seat panel, ~$2–4) + `synthesize` on MMLU
   dev/validation items for the six subjects; rows carry the rendering-mandatory fields
   (text, answer letter, option distribution, 1..K order key, confidence distribution,
   both justifications) + selection fields (id, source_item_label, rater_model) + panel
   provenance. Fallback: hand-authored rows per the superseded D4 draft.
3. **`src/judex_calibration/elicit_control.py`** (M): fork-with-shared-primitives —
   OPTION_LABELS, control prompt preamble (replacing the one EU-AI-Act sentence),
   render/parse for the D3 contract; imports the transport, two-stage flow, JSON
   extraction, dist validation, ε-floor, and run_variant machinery unchanged; emits the
   same record keys so gate summary/views work untouched. Explicit resume-guard clauses
   for `contract` + `slice_sha256` (the guard iterates prior-sidecar keys — new identity
   fields are silently unguarded without their own clause).
4. **`src/judex_calibration/ece.py`** (M): lift ECE10/AECE5/MCE/AUROC/boot_* out of
   `scripts/base_calibration_premise_probe.py` with a byte-identity regression against
   the shipped probe artifact; `score_control.py` assembles E1/E2 (+ D6 companion) and
   never calls `score_variant`/Murphy/W1.
5. **`scripts/run_control_leg.py`** (+ API-twin shim reusing the --base-url vllm-chat
   path) (S–M): fork of the Study-B driver; `{leg}_control.json` filenames so control
   dirs can never be analyzed by the AIReg driver; gate = parse_ok ≥ 0.90 (re-derived,
   D3).
6. **Tests** (mirroring the existing suites): K=4 parser tier battery, render→parse
   round-trip, floor on 4-vectors, run_variant seam incl. the new guard clauses,
   store coverage + firewall disjointness, ECE toy cases + lift regression, a guard test
   that the control path never emits ordinal metrics. Offline scaffold probe
   (token budget vs the 32768 pin) before any spend.

## 6. v15 sentence map (what this control changes, by location)

- §6.2 `:883-885` — the closing "present before alignment / caused by pre-training"
  sentence is rewritten per the E1/E2 outcome row.
- §6.2 `:891-905` — the DACA contrast gains the same-harness MMLU column; the
  "task, reference, metric, and scaffold all differ" scoping sentence weakens or falls.
- Limitations `:1233-1235` — "We did not run that control." deleted; replaced by the
  result + residual scope.
- New appendix (control detail: slice, contract, exemplar policy, tables) — appendices
  are uncounted; body cost ≈ 2–3 reworked sentences, inside the p. 9 camera-ready budget.
- Ledger: new claims for every printed control number; remap in the same v15 pass as the
  P3 removal + scaffold appendix (single remap, v14 baseline per the claims.yaml header).

## 7. Decision list for the user

| # | Decision | Recommendation |
|---|---|---|
| D1 | benchmark | MMLU |
| D2 | slice | 120 = 6 subjects × 20; subject list open to swaps |
| D3 | contract | 4-option + confidence kept + findings dropped; gate scoped to parse_ok |
| D4 | exemplar store | corpus-pipeline panel annotation of MMLU dev items (organic panel credences; REVISED per user Q2, 2026-08-09); fallback = hand-authoring |
| D5 | legs | pairs, 4 families, 8 legs, transports per serving parity |
| D6 | MMLU τ_v | compute descriptive-only under TVD; never touches frozen constants |
| D7 | uncertainty | item-level bootstrap primary, subject-clustered sensitivity |
| D8 | cross-K | state chance floors, decline numeric equality |
| — | budget | ~$140–165 est., $220 hard cap, ≥$50 single-increment float per big rental |
| — | sequencing | 2b build + tests + offline probe BEFORE any rental; then 2c in one campaign; then v15 (single authoring pass + single ledger remap) |
