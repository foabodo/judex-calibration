# Verbalized-first calibration plan — the four-arm architecture

Status: **DRAFT for user adoption** (2026-07-21). This document rebuilds the Stage-2
three-arm calibration architecture verbalized-first/verbalized-only, per the user
verdict of 2026-07-21: the logit-based approaches of Study A — supervised and
unsupervised — are failures as the project's calibration contribution, and the
reframing is built on Study B's verbalized-channel results.

Nothing in this document is self-executing: **protocol r3 adoption, the E6 sweep
spend, and any `pipeline.yaml` calibration block are user decisions.** The evaluator
calibration seam stays `mode: noop` until then. Study A artifacts are rescoped, never
deleted or rewritten (provenance rules in §6).

Companion documents:
- R0 evidence: `docs/verbalized_r0_analyses_2026_07_21.md` (the four free analyses).
- Study B record: `docs/study_b_results_2026_07_20.md` + `docs/study_b_design.md`
  (§4 pre-registered gates, §6 B-Q4).
- Paper rewrite map: umbrella
  `spec/plan_2026_07_21_verbalized_first_paper_rewrite_map.md`.
- Logit-era protocol of record (r2, remains the adopted protocol until r3 is
  adopted): `docs/e6_onpair_decision_protocol.md`.

---

## 1. Premise — the channel-selection argument

The production JUDEX evaluators emit contract-0.2.0 **verbalized distributions** (a
5-level compliance distribution + a 3-level confidence distribution on the 0.05
grid). Study A measured post-training overconfidence in the **logit channel**
(token-slice over A–E) and its clustering gate failed decisively: gate-passing τ_oc
{1.60, 1.86, 4.88}, max/min 3.05 > 2 — no transferable constant, and a
logit↔verbalized bridge is impossible in the strong sense (2026-07-12 feasibility
analysis). Study B repeated the same design in the **production channel** and the
same methodology came back alive:

- τ_v (W1-aligned post→pre inverse-softmax temperature on ε-floored verbalized
  vectors) ∈ [1.025, 1.380] over the gate-passing panel, none saturated;
- clustering **passes**: max/min = 1.346 ≤ 2 at n=4 in-mode ({qwen 1.281, llama31
  1.025, glm 1.025, maverick 1.380}), n=5 adding gemma31_api's cross-mode 1.025;
- gate outcomes track the channel, not the weights (all three Study A gate failures
  re-tested: two reverse cleanly, one — gemma26 — is the lone partial corroboration
  and is capability-excluded);
- the closed pair natively emits this channel, so a verbalized-derived constant is
  **channel-matched** to the evaluators it would correct — the transport
  impossibility that blocked Study A's Q4 simply does not arise.

Narrative position (user-directed): Study A appears exactly once in the
contribution's story, as **the channel-selection negative result** — logits are the
wrong-channel measurement for a system that emits distributions as text. The
verbalized program is the original clustered-constant plan finally executed in the
right channel. (Provenance stays factual everywhere; see §6.)

Honesty clauses that ride with the premise (from R0 analysis A):

- The 60-pt log grid does not contain 1.0 (neighbors 0.952 / 1.025); three of five
  gate-passing τ_v are the first grid point above 1. "All τ_v ≥ 1" therefore reads:
  no family under-confident, two clearly sharpened.
- Post-training sharpening is a **subpopulation phenomenon**: τ_v ≈ 1 wherever
  post/pre modes agree, 2.7–4.2 where they disagree; the scalar entropy flip in
  llama31/maverick is stratum composition, not a contradiction. The W1-aligned and
  entropy-matched temperatures are different estimands; the production system is
  W1/RPS-scored, so the W1-aligned one is metric-matched.
- The in-channel correction is **modest** (oracle tempering gain ≈ 1% of the
  post→pre W1 gap). The paper story is the channel contrast (ratio 1.35 vs 3.05 on
  the same lineages) and the resulting *usable* mechanism, not a large effect size.

## 2. The four arms

### Arm 1 — transferred median(τ_v) (PRIMARY candidate)

The reborn arm 1: a clustered cross-family constant from open pre/post pairs,
measured in the production channel. Adoption gates, all in-channel and all already
pre-registered in `study_b_design.md` §4:

1. **Contract-complete ≥ 0.90** per leg (the B-Q1 gated rate, FULL six-field tier);
2. **Resolution-primary capability floor** (Murphy resolution > 0 with margin, both
   legs) — gemma26's base fails this (0.0134) and its τ_v is excluded as
   weak-reference, permanently;
3. **Clustering max/min ≤ 2** over the gate-passing families — PASSES (1.346);
4. **No peg** — any `T_BOUNDS`-boundary temperature is a boundary artifact, never
   adoptable.

Candidate constant and band:

- gate-passing τ_v: in-mode n=4 {1.025, 1.025, 1.281, 1.380}; n=5 with gemma31_api
  (cross-mode, channel-labeled) adds 1.025;
- **verbalized sensitivity band [1.03, 1.38]** (the gate-passing range) — note it
  does not even overlap the logit-era band [1.60, 4.88]; that contrast is the paper
  exhibit;
- the numeric median is **convention-dependent and must be frozen in protocol r3
  before on-pair data exists**: interpolated 1.153 / study_a upper-median 1.281 on
  n=4; 1.025 on n=5. R0's LOFO check (analysis B) shows every convention lands
  within the band of every held-out family (max |log ratio| 0.297), transfer is
  safe (worst case ≈ 1.9% W1 cost) and modestly beneficial GT-side, so the
  convention choice is not outcome-critical — but it is exactly the kind of degree
  of freedom that must be closed pre-hoc. Recommendation to take into r3: the
  **interpolated median on the in-mode n=4 set (1.153)**, with the n=5 value (1.025)
  reported as channel-sensitivity.

Arm-1 adoption is decided on the R1 sweep via `closed_side_check` (reliability
strictly better, RPS no worse, resolution within tolerance), now well-posed because
the constant and the target share a channel.

### Arm 2 — supervised held-out T* (DEMOTED to triangulator/validation)

The logit-era E6 mechanism (accuracy-gated held-out supervised T* + band) is
**demoted from mechanism to validation**: once the E6 sweep produces on-pair
verbalized outputs, the supervised fit on held-out docs validates (or impeaches) the
arm-1 constant. Divergence is published, not reconciled. Estimand caveat carried
from R0-B: T*-vs-GT absorbs base-vs-panel miscalibration on top of the post-training
increment, so T* > median(τ_v) is expected, not a failure; the comparison metric is
decision-relevant improvement under each, not numeric equality.

### Arm 3 — GT-free triangulators, retried in-channel

Both logit-era GT-free machines re-run on verbalized data. Their logit-era failures
do not pre-judge the retry; neither does Study B pre-judge success. Both publish
either way. R0 prototypes (smoke-grade, 15 closed items) set expectations:

- **τ_DACA-verbalized** (`fit_tau_daca`/`daca_triangulation`, verbalized base
  references): the R0 prototype does **not** corroborate arm 1 — valid-fit median
  0.38 (sharpen) vs τ_transfer 1.15 (flatten), two of five references pegged at the
  lower bound. The suspected mechanism is the familiar reference
  non-exchangeability, now with sharp verbalized bases vs a hedging production
  pair. Expectation for R1: the logit-era failure mode is the base case; a real
  E6-data verdict either way is a publishable data point about GT-free calibration.
- **Verbalized dispersion pool** (rebuilt from the gate-passing verbalized base
  legs): arrives healthy — mixture entropy 1.18 nats (≥ the 1.00 vetting floor),
  no LOBO fragility (min 1.109), decorrelated by construction and measurement (min
  pairwise TVD 0.33–0.39; the correlated-internal-pool negative does not recur).
  Prototype closed-run T_raw **lands inside the verbalized band** bases-only (1.17 /
  1.26) — the first time a raw pool fit has concurred with the primary arm without
  a shrinkage guard (logit-era raw fits overshot to 7.3–7.8). Two things must be
  frozen in r3 before the sweep: **pool membership** (bases-only vs cfp1-augmented —
  the augmented variant fits 1.70–1.78, above-band, so this choice is load-bearing
  in-channel) and the **anchor/λ question** (the logit-era shrinkage constants 2.794
  / 0.35 are dead with their band; either re-derive against [1.03, 1.38] by the same
  frozen rule, or adopt the raw fit with a concurrence criterion — r3 decides).

### Arm 4 — the confidence-distribution instrument (NEW; no logit analog)

B-Q4's one-parameter T_c on the 3-level confidence distribution, with the
**association-diagnostic adoption bar**: LODO-"calibratable" alone never suffices
(LODO fires even for base-rate flattening); adoption of any per-family T_c requires
the confidence signal to carry per-cell information (Kendall τ_b between p̂(C) and
realized W1, negative with margin). The measured panel resolves into a gradient —
interior T_c 4.5–4.9 where confidence carries signal (gemma31_api 4.53, qwen 4.88),
7–13 where weak/level-only (maverick, glm, gemma26_api), a wrong-sign 20.0 peg
(llama31). Stated confidence is universally overconfident in level (p̂(C) 0.72–0.95
vs realized 0.48–0.68). R1 application: fit T_c for the closed pair on the E6 sweep,
apply the same association bar, and report the confidence-channel calibration as the
novel instrument of the contribution. This also owns the S09-scalar risk surfaced by
the preamble twin (confidence collapse on the GPT seat is a channel property the
instrument must detect, not assume away).

## 3. R0 evidence base (executed 2026-07-21, $0)

See `docs/verbalized_r0_analyses_2026_07_21.md` for the full record:

| analysis | outcome |
|---|---|
| A. τ_v-vs-entropy reconciliation | RESOLVED — subpopulation/mode-relocation mechanism; standalone paper material |
| B. LOFO transfer on median(τ_v) | PASS — safe (≤1.9% worst-case W1 cost), modest benefit, convention flagged for r3 |
| C. τ_DACA-verbalized prototype | NON-CORROBORATING at n=15 — logit-era failure shape; real test = R1 |
| D. Verbalized dispersion pool | HEALTHY + band-concordant (bases-only); membership + anchor/λ = r3 freezes |

## 4. Phasing

- **R0 (done, free)**: the four analyses above; this plan; the paper rewrite map.
- **R1 (user-gated, ~$445 — revised quote per `sweep-readiness` reconciliation)**:
  the on-pair E6 sweep (24 docs, Sonnet 4.6 medium + GPT 5.4 medium,
  `configs_v2exemplars`, contract outputs). Unblocks: arm-2 validation T*, the real
  τ_DACA-verbalized, the pool fit on real on-pair data, `closed_side_check` with
  median(τ_v), range-robustness re-run against [1.03, 1.38], and the closed-pair
  confidence fit (arm 4). Prerequisite: **protocol r3 adopted first** (else the
  sweep's degrees of freedom are open when data arrives).
- **R2 (free)**: protocol r3 finalization for user adoption + paper rewrite
  execution per the rewrite map (likely its own handoff).

## 5. Protocol r3 — decisions to freeze (draft checklist for the user)

r3 replaces r2's instrument ladder with a verbalized-first one. Open decisions r3
must close, each currently a free parameter:

1. Median convention for the arm-1 constant (recommended: interpolated, in-mode n=4
   ⇒ 1.153; n=5 reported as sensitivity).
2. Gate-passing set identity (in-mode n=4 primary vs n=5 with the channel-labeled
   API leg).
3. Pool membership for arm 3b (bases-only vs cfp1-augmented) and the anchor/λ
   question (re-derive vs raw-fit-with-concurrence).
4. Concurrence criteria: arm-2 T* vs arm-1 constant; pool T_raw vs band; τ_DACA
   validity gates (reference-entropy floor to pre-empt the non-exchangeability
   mechanism?).
5. Arm-4 adoption bar constants (Kendall threshold/margin for "carries signal").
6. `closed_side_check` acceptance criteria (inherit E6's: reliability strictly
   better, RPS no worse, resolution within 10%?).
7. Failure ladder + terminal position (r2's "band-only, never worse than doing
   nothing" translated to [1.03, 1.38]).

r2 stays the protocol of record until the user adopts r3; r3 will be drafted as a
separate document in R2, leaving r2's text intact (same discipline as the band
amendment: amendments are appended records, not rewrites).

## 6. Integrity provisions (non-negotiable, from the mandate)

- **Narrative never falsifies provenance.** Methods/prereg/provenance keep true
  dates and collection order: gemma26 was a 2026-07-21 retest and may be *framed*
  as motivating the Gemma-31B collection, but its provenance record stays factual;
  the band-amendment history (`17ffd41`/`61da68a`) stays factual; Study B's
  pre-registered n=2 verdict stays distinguished from extension reads (GLM,
  gemma-API legs, Study-A-failure retests are labeled extensions).
- **Study A artifacts are rescoped, not deleted**: the guide, the r2 protocol, the
  frozen umbrella specs and their amendment chain remain the audit trail; their
  status lines change (see the rewrite map), their content does not.
- **Smoke discipline**: every closed-side number derived from
  `stage9-claude-gpt-medium` (15 items) is prototype-labeled and non-adoptable.
- **Channel labels are load-bearing**: gemma31_api/gemma26_api results carry
  cross-mode labels in every table; gemma26(-api) never enters a gate-passing set
  (capability-excluded base).
- **No adoption without the user**: no `pipeline.yaml` calibration block, no
  protocol adoption, no spend. The seam stays `mode: noop`.
- Inherited mechanics: ε = 0.005 before every inverse-softmax op; `T_BOUNDS =
  (0.25, 20.0)` explicit; peg ≠ fit; AIReg = validation only; corpus-v2 k=5
  firewall-disjoint few-shot; run-id identity; doc-clustered bootstrap (seeded);
  panel-robust corrections only.
