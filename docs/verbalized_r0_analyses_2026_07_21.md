# R0 analyses — verbalized-first reframe (2026-07-21)

Status: **executed, $0** — the four free analyses of the verbalized-first reframe
(handoff `spec/handoff_2026_07_21_verbalized_first_reframe.md` §5·R0, items 1–4).
Instrument: `scripts/verbalized_reframe_r0.py`; artifact of record:
`runs/verbalized_r0/r0_analyses.json` (gitignored — the numbers below are the
committed record). Data: the eight on-disk Study B legs (`runs/study_b_*`) + the one
existing on-pair closed run `stage9-claude-gpt-medium` (3 graded docs / **15 items**
— every closed-side number here is **PROTOTYPE / smoke-grade** by the <120-cell rule
and is never adoptable).

**Panel definition (USER DECISION 2026-07-21, applies to every adoption-relevant
analysis below):** the arm-1 panel is **{qwen 1.281, gemma31 1.025, glm 1.025,
maverick 1.380}**. gemma31's measurement is its chat-template collection (leg
`study_b_gemma31_api`) — the greedy-continuation collapse under vLLM is a serving
pathology of the Gemma-4 family, not a property of the weights, so the chat-template
leg is *the* gemma31 τ_v and the serving transport is methods-level provenance (leg
meta sidecars), not a headline distinction. **llama31 is excluded** from every
adoption-relevant set (accumulated instrument anomalies — notably the sole
wrong-sign B-Q4 association with a pegged T_c — carry caveat burden
disproportionate to its contribution); it remains in §A as measured, non-panel
evidence. The exclusion is **numerically inert**: llama31's τ_v (1.025) duplicates
gemma31's grid point, so the panel multiset {1.025, 1.025, 1.281, 1.380} — and with
it the median, the band, and the cluster ratio — is unchanged by the swap. Decided
before any on-pair data exists (pre-hoc for adoption).

Discipline: ε = 0.005 floors before every inverse-softmax op; `T_BOUNDS = (0.25, 20.0)`
explicit; boundary = peg, never a fit; gemma26's base is capability-excluded
(resolution 0.0134) and appears only as labeled context, never in an adoption set.

**One caveat that applies to every temperature below (grid resolution):** the 60-point
log grid over (0.25, 20) does not contain T = 1.0 exactly; the bracketing grid points
are **0.952 and 1.025** (log-step ≈ 7.7%). A fitted value of 1.025 (or 0.952) therefore
means "indistinguishable from 1 at grid resolution". Two of the four panel τ_v values
are exactly the 1.025 grid point, so the honest reading of "τ_v ≥ 1 everywhere" is
"no family shows *under*-confidence, and two (qwen, maverick) show clearly-resolved
sharpening" — not "four separately-resolved positive effects".

---

## A. τ_v vs the entropy scalar — reconciled

Question (handoff §1 nuance): the W1-aligned temperature τ_v detects post-training
sharpening in every family, while the mean normalized-entropy summary flips sign in
llama31 and maverick. Which view is right, and why do they disagree?

**Answer: both are right about different summaries of a heterogeneous per-cell
population, and the disagreement is fully explained by two strata.** Per family, on
the paired parse-ok overlap:

| family | n | τ_v | τ_H (entropy-match) | mean H pre→post | share post-sharper | τ_v on post-sharper | τ_v on post-flatter | τ_v mode-agree | τ_v mode-disagree | Kendall(ΔH, W1-gain) |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen | 109 | 1.281 | 1.025 | 0.488→0.490 | 0.54 | 2.321 | 0.952 | 0.952 | 3.903 | +0.52 |
| gemma31 | 118 | 1.025 | 1.025 | 0.537→0.518 | 0.48 | 2.321 | 0.952 | 1.025 | 4.204 | +0.45 |
| glm | 112 | 1.025 | 1.104 | 0.555→0.526 | 0.48 | 2.500 | 1.025 | 1.025 | 3.364 | +0.48 |
| maverick | 120 | 1.380 | **0.952** | 0.538→0.574 | **0.39** | 3.903 | 1.025 | 1.025 | 3.364 | +0.25 |
| llama31 (non-panel) | 115 | 1.025 | **0.820** | 0.516→0.583 | **0.33** | 2.321 | 1.025 | 0.952 | 2.692 | +0.16 |
| gemma31-vLLM† | 57 | 1.857 | 1.189 | 0.523→0.440 | 0.53 | 4.204 | 1.025 | 0.952 | 4.204 | +0.49 |
| gemma26‡ | 114 | 1.189 | 0.952 | 0.442→0.464 | 0.43 | 2.900 | 0.952 | 0.952 | 20.0 (peg) | +0.36 |

Panel families first; the last three rows are measured, non-panel evidence.
† the collapsed vLLM serving artifact of the same weights (n=57 parsed) — context
only; the panel gemma31 row above is the valid chat-template measurement.
‡ gemma26's base is capability-excluded; row is labeled context (its raw-completion
leg has n_overlap = 2 — omitted).

Reading, in order of force:

1. **The sharpening is a subpopulation phenomenon.** In every family the cells split
   into a stratum where the post leg genuinely over-concentrated (τ_v refit on that
   stratum alone: **2.3–4.2** — logit-era-sized temperatures) and a complementary
   stratum that is temperature-calibrated to the base (τ_v ≈ 0.95–1.03, i.e. ≈ 1 at
   grid resolution). The family-level τ_v is a W1-weighted compromise between the two.
2. **The mode-agreement split localizes it further**: wherever post and pre argmax
   agree, τ_v ≈ 1 (0.95–1.03 in all eight rows); the entire flattening demand comes
   from the **mode-disagreement cells** (τ_v 2.7–4.2), where the post leg has both
   moved and concentrated its mass. W1 is location-sensitive, so it prices that
   relocation; Shannon entropy is location-free, so it cannot see it.
3. **The scalar flip is a composition effect, not a contradiction.** The flip families
   are exactly the ones where the post-sharper stratum is a *minority* (maverick 0.39,
   and non-panel llama31 0.33, vs ≈ 0.5 elsewhere): the signed mean of ΔH flips when flattened cells
   outnumber sharpened ones, while the W1 objective — dominated by the large-movement
   sharpened/relocated cells — still fits τ_v ≥ 1. The per-cell association has one
   sign everywhere: Kendall(ΔH, per-cell W1 gain from flattening) is positive in all
   families (+0.16 to +0.52) — the cells where post is sharper than pre are precisely
   the cells flattening helps. Only the unweighted scalar summary flips.
4. **τ_H makes the estimand contrast exact.** τ_H (the temperature equating mean post
   entropy to mean pre entropy — the pure-spread analog of τ_v, and the verbalized
   descendant of the retired 2(c) entropy-accounting machinery) sits at 0.82/0.95 for
   llama31/maverick (sign-opposed to their τ_v) and ≈ 1.0–1.2 elsewhere (well below
   τ_v for qwen). Temperature-as-spread and temperature-as-W1-alignment are
   **different estimands on real data**; the production system is scored in W1/RPS,
   so the W1-aligned one is the channel- and metric-matched choice. This is the
   publishable reconciliation: report both, adopt neither blindly, and state that a
   scalar entropy summary is not a calibration diagnostic for location-shifted
   distributions.
5. Subset composition (parse-failure selection) moves the *pre* means slightly
   (gemma31-vLLM own-set 0.536 vs overlap 0.523) but does not drive any flip — the
   flips survive on the paired overlap.

---

## B. Leave-one-family-out transfer of median(τ_v) — the free adoption analog

Panel: **{qwen 1.281, gemma31 1.025, glm 1.025, maverick 1.380}** (user decision
2026-07-21). Cluster ratio 1.346 ≤ 2 — the clustering gate passes. Sensitivity set:
+llama31 (n=5), whose τ_v duplicates gemma31's grid point.

**The median convention is a live protocol-r3 decision** (it did not matter at n=3
in Study A; it matters now):

| set | interpolated median | upper median (`sorted[n//2]`, study_a convention) |
|---|---|---|
| panel n=4 | **1.153** | 1.281 |
| +llama31 n=5 (sensitivity) | 1.025 | 1.025 |

LOFO (hold out each panel family; transfer the median of the rest):

| held-out | own τ_v | LOFO transfer (interp.) | in-channel W1(post→pre): raw / at-own / at-transfer | oracle-gain recovered | GT-side mean RPS: T1 → transfer | GT-side reliability: T1 → transfer |
|---|---|---|---|---|---|---|
| qwen | 1.281 | 1.025 | .7653 / .7577 / .7641 | 0.161 | .0587 → .0575 | .0473 → .0463 |
| gemma31 | 1.025 | 1.281 | .6006 / .6009 / .6084 | n/a (no oracle gain) | .0333 → .0255 | .0214 → .0135 |
| glm | 1.025 | 1.281 | .5704 / .5711 / .5814 | n/a (no oracle gain) | .0457 → .0360 | .0330 → .0230 |
| maverick | 1.380 | 1.025 | .8577 / .8555 / .8573 | 0.152 | .0358 → .0350 | .0217 → .0218 |

max |log(τ_transfer/τ_own)| = 0.297 (both sets) — every transferred constant lands
within a factor e^0.297 ≈ 1.35 of the held-out family's own fit, i.e. **inside the
[1.03, 1.38] band itself**. That is the LOFO pass, and it is also its content: the
band is narrow enough that transfer cannot miss badly.

Honest effect-size read (do not oversell):

- **In-channel, the τ_v correction is small.** The W1(post→pre) gaps (0.57–0.86) are
  dominated by *content* differences between legs, not scale; the oracle tempering
  gain is ≤ 0.0076 W1 (≈ 1%), and the LOFO transfer recovers ~15–16% of that where it
  exists and costs at most 0.011 W1 (≈ 1.9%) where the held-out family is already
  ≈ calibrated. Transfer is **safe but modest** — exactly what a tight near-1 cluster
  implies. The contrast with the logit channel is the story (ratio 1.35 vs 3.05), not
  the magnitude.
- **GT-side (labeled supervised read, reported not gated):** mild flattening at the
  transferred constant improves mean RPS for all four held-out post legs and Murphy
  reliability for three (maverick flat), with resolution unchanged (±0.001). Note the
  GT-side optimum is *larger* than τ_v for qwen/maverick (own-τ column beats transfer
  on RPS): τ_v isolates the post-training increment while T*-vs-GT also absorbs the
  base's own miscalibration vs the human panel — the two estimands should not be
  conflated in the paper, and closed-side adoption still runs through the R1
  `closed_side_check`, not through this read.

---

## C. τ_DACA-verbalized — PROTOTYPE (15 items, smoke-grade)

Setup: `study_a.daca_triangulation` unchanged; references = the four panel families'
verbalized **base** legs (ε-floored); closed side = `stage9-claude-gpt-medium`
reconciled finals (the right pair, but 3 docs / 15 items ⇒ prototype-labeled,
never adoptable; the real version rides the E6 sweep).

| reference | τ_DACA | saturated | agreement rate (n) | ref argmax acc (gate only) |
|---|---|---|---|---|
| qwen | 0.284 | no | 0.54 (7) | 0.46 |
| gemma31 | 0.250 | **peg (lower bound)** | 0.47 (7) | 0.53 |
| glm | 0.870 | no | 0.50 (7) | 0.57 |
| maverick | 0.250 | **peg (lower bound)** | 0.33 (5) | 0.40 |

Two valid fits, **{0.284, 0.870}** (at n_valid = 2 a "median" is barely meaningful —
the summary reports 0.870 by the upper convention; both values are the result), vs
τ_transfer 1.153 and vs the 15-item supervised prototype T* = 1.503.

**The prototype does not corroborate arm 1 — and the failure shape is the familiar
one.** Every τ_DACA lands on the *other side of 1*: aligning the closed pair to
verbalized base references says "sharpen", while both the transferred constant and
the (proto) supervised fit say "flatten". Mechanism, most likely: **reference
non-exchangeability recurs in-channel** — the verbalized bases emit fairly sharp
distributions (mean normalized entropy ≈ 0.44–0.56) while the production closed
pair's reconciled outputs hedge (the entropy-flag phenomenon), so the GT-free
alignment target is sharper than the thing being calibrated, and two of four fits
peg at the lower bound. Caveats
cutting both ways: agreement strata are 4–7 cells (n=15 closed items), so nothing
here is resolved; but nothing here licenses optimism either. Publish the retry either
way (handoff §2); expectation for R1 set accordingly: **the logit-era τ_DACA failure
mode is the base case, in-channel data notwithstanding.**

---

## D. Verbalized dispersion pool — re-derivation

Pool = the panel families' verbalized **base** legs. Diagnostics on the full common
parse-ok cell set (bases only, no closed run needed); prototype T_raw on the 15-item
closed run (smoke-grade). **No shrinkage applied**: the logit-era anchor (2.794) and
λ (0.35) were frozen against the logit band [1.60, 4.88]; carrying either into this
channel would be an unfrozen retune. Anchor/λ re-derivation against the verbalized
band is an R1/protocol-r3 decision.

| pool | n cells | mixture entropy (nats) | ≥ 1.00 floor | LOBO min | min pairwise TVD | proto T_raw bases-only | proto T_raw cfp1+bases |
|---|---|---|---|---|---|---|---|
| P4 panel bases | 103 | 1.178 | yes | 1.090 | 0.328 | **1.278** (n=12) | 1.753 (n=12) |
| P5 +llama31 (sensitivity) | 103 | 1.188 | yes | 1.134 | 0.329 | 1.260 (n=11) | 1.696 (n=11) |

- **The decorrelation lesson holds in-channel by construction and by measurement**:
  cross-family bases, min pairwise TVD ≈ 0.33 (nowhere near the correlated-internal
  regime of the negative result), mixture entropy comfortably above the logit-era
  1.00-nat vetting floor, and no wide-member fragility (LOBO min 1.090 — no member's
  removal drops the pool below floor).
- **Basin-level concurrence, for the first time**: the bases-only prototype T_raw
  (1.28, sensitivity 1.26) lands **inside the verbalized band [1.03, 1.38]** — against the
  logit-era pattern where raw closed-run fits overshot to 7.3–7.8 and needed the
  shrinkage guard to be usable. The cfp1-augmented variant (1.70–1.75) sits above the
  band; membership (bases-only vs augmented) is therefore load-bearing in-channel and
  must be fixed in protocol r3 *before* the E6 sweep, not after seeing it.
- All closed-side fits here are n=12 (n=11 for the sensitivity pool): the 15 items
  ∩ each pool's all-bases parse-ok set — smoke.

---

## Cross-analysis synthesis (what R0 establishes)

1. **Arm 1 evidence strengthened with sharper honesty**: clustering passes (1.346),
   LOFO transfer is safe-but-modest, and the τ_v ≥ 1 headline must be stated with the
   grid-resolution and subpopulation qualifications of §A. The candidate constant is
   convention-dependent (1.025 / 1.153 / 1.281) — a protocol-r3 decision to freeze
   *before* on-pair data exists.
2. **The reconciliation (§A) is standalone paper material**: sharpening is a
   mode-relocation subpopulation phenomenon; W1-aligned and entropy-matched
   temperatures are different estimands; the scalar entropy flip is composition, not
   contradiction.
3. **Arm 3 splits**: the dispersion pool arrives in-channel healthy and concordant
   (D); τ_DACA arrives with its logit-era pathology apparently intact (C). Both get
   their real test on E6 data; both get published either way.
4. **Nothing is adopted.** No config, no protocol edit, no constant paste — every
   adoption path runs through the user (protocol r3 draft = R2; sweep = R1 gate).
