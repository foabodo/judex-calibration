# τ_DACA under scaffold-variant references — check of 2026-08-09

**Status:** EXPLORATORY / DIAGNOSTIC. Closes transfer-condition (iv) of
`docs/daca_standing_assessment_20260809.md` §3. F7 was registered for the **baseline
scaffold only**; nothing here is a new registered result, nothing is adopted, r3 F1–F8
and rA1 A1–A8 remain frozen and untouched. The evaluator seam stays `mode: noop`.

**Cost:** $0 — analysis-only on artifacts already on disk. No elicitation, no network.
**Branch:** `claude-verbalized-gap-closure` @ `2f21f74`.
**Driver:** `scripts/tau_daca_scaffold_variants.py` (new, labeled variant read; the frozen
drivers `scripts/tau_daca_verbalized_fullscale.py`, `scripts/absolute_vs_ratio_estimand.py`
and `scripts/objective_contamination_audit.py` were read, never edited).
**Artifact:** `runs/tau_daca_scaffold_variants/tau_daca_scaffold_variants.json` (gitignored).

---

## 1. Pre-specified read (written before computing)

### 1.1 Question

τ_DACA is the program's GT-free cross-check: an agreement-filtered alignment of a POST
("closed") leg to a BASE ("reference") leg, adapted to our objective. Its full-scale
verbalized study of record
(`spec/analysis_2026_07_21_tau_daca_verbalized_fullscale.md`) used the **baseline-scaffold**
verbalized base legs as references. Phase 1a/1b then showed that those reference legs move
materially under coverage-preserving scaffold perturbation — `alt_set` inflates
`T_abs(pre)` by **+0.46 to +0.72 ln** on three of four families (gemma31 +0.721, maverick
+0.497, glm +0.455; qwen inert at +0.012). The corroboration was therefore
**scaffold-conditional in an untested way**. This check tests it: recompute τ_DACA with
the **variant pre legs as references**, holding the targets fixed.

### 1.2 Methodology — inherited verbatim from the study of record

| element | value |
|---|---|
| targets ("closed" analogs) | the four panel **POST** legs of record: `study_b_qwen`, `study_b_gemma31_api`, `study_b_glm`, `study_b_maverick` |
| references | the four panel **PRE** legs, swapped per arm (baseline / V1 `alt_set` / V2 `rev_order`) |
| agreement filter | strict argmax agreement between target and reference on the shared parse-ok cells |
| ε floor | 0.005 at analysis time, then renormalize (Study B convention) |
| fit of record | `judex.calibration.fit_temperature`, RPS objective, 49-pt log grid + golden refine, `T_BOUNDS = (0.25, 20.0)` |
| matched-W1 variant | the same filter on the 60-pt `study_a.GRID` under W1 — the objective T_J itself was fitted under (`scripts/objective_contamination_audit.py` A2) |
| validity | fit not saturated (not on a T_BOUNDS edge) **and** reference not below chance (reference argmax accuracy ≥ 0.2; gate-only, GT never enters the fit) |
| F7 read | per target, over the **3 cross-family** references: corroborates iff `n_valid ≥ 3` and every valid fit lies in `[T_J/2, 2·T_J] = [0.5765, 2.3060]`, T_J = 1.153 |

Own-family fits are computed and printed as diagnostics only; they never count toward F7,
exactly as in the study of record.

Baseline values to reproduce: **12/12 valid cross-family fits, median 0.9224116756116703,
range [0.7674842, 1.0727147]** (RPS of record), and **median 0.9884871, range [0.7616868,
1.0251785]** (matched-W1). If the baseline does not reproduce, the check STOPS.

### 1.3 Pre-specified interpretation

- **SURVIVES** for a variant iff the F7 bar holds for every one of the four targets under
  that variant's references — i.e. ≥3 valid cross-family fits, all inside
  [0.5765, 2.3060]. Phrase the outcome as "the corroboration **is / is not**
  scaffold-robust", never as a new registered result.
- Report per variant: n valid, median, range, per-target window verdict, and the shift of
  the median against the baseline 0.9224 (RPS) / 0.9885 (W1).

### 1.4 Direction of shift — mechanics, predicted before looking

The fit applies temperature T to the **target** (post) distribution and aligns it to the
**reference** (pre). T > 1 flattens the target; T < 1 sharpens it. So:

> τ_DACA measures how much the post leg must be flattened to look like the base leg on
> the cells where they already agree on the mode. A **sharper reference** demands a
> **smaller** τ_DACA.

`T_abs(pre)` is the absolute correction the pre leg needs against the human GT: a larger
`T_abs(pre)` means the leg must be flattened more, i.e. it is **sharper / more
overconfident**. `alt_set` inflated `T_abs(pre)` on gemma31, glm and maverick, so — to
first order — **alt_set references are sharper, and τ_DACA fits taken against them should
move DOWN.** Since the baseline median (0.922) already sits *below* T_J = 1.153 and the
F7 window's lower edge is 0.5765, a downward shift is the direction that **threatens**
the corroboration: the headroom below is ln(0.922/0.5765) = 0.470 in log units, and the
alt_set reference-side inflation is +0.46–0.72 ln. **If the shift transferred one-for-one
from T_abs(pre) to τ_DACA, the check would fail.** That is the pre-specified stake.

Two reasons the transfer may be less than one-for-one, both stated in advance so neither
is a post-hoc rescue:

1. `T_abs(pre)` is fitted against a *dispersed human GT* over all cells and is
   location-sensitive (W1/RPS price mode relocation, not only spread); part of its
   inflation may be *location* error rather than sharpening. τ_DACA is fitted on the
   *argmax-agreement* subpopulation, where location error is by construction suppressed.
2. The agreement filter re-composes under a new reference: which cells survive changes
   with the reference's argmax, so the estimand's subpopulation is not held fixed.

Accordingly the check also reports, per reference leg, the **mean normalized entropy** on
the shared overlap — a direct, objective-free read of whether the variant references
actually sharpened — and the agreement rate, so a median shift can be attributed to
sharpening vs. re-composition rather than assumed.

`rev_order` moved `T_abs(pre)` far less (+0.005 to +0.151 ln), so the pre-specified
expectation for V2 is a **small** downward shift, well inside the window.

### 1.5 Caveats carried in advance

- **Gate blemishes.** The two Phase-1 B-Q1 contract-gate failures are `study_b_qwen_k5v2`
  **post** (0.80) and `study_b_glm_k5v1` **post** (0.8833). The primary check swaps
  **references only**, and references are **pre** legs — so neither blemished leg enters
  the primary computation. Every variant **pre** leg passes B-Q1 (qwen k5v1/k5v2, gemma31
  k5v1/k5v2, glm k5v1 0.9917 / k5v2 0.9583, maverick k5v1/k5v2 0.9667). A labeled
  secondary arm swaps **both** sides (variant references *and* variant targets), where the
  blemished legs do appear; that arm is reported with and without them.
- **Cross-mode pairing for gemma31.** gemma-4 post cannot be collected on raw
  continuation (greedy-repetition collapse), so every gemma31 τ_v-like quantity is
  cross-mode: raw-completions **pre** against chat-template **post**. Crucially for *this*
  check, all gemma31 **pre** legs — baseline (`study_b_gemma31_api/pre_verbalized.json`, a
  provenance copy of the vast bf16 raw base leg) and both variants — are the **same
  transport**: bf16 vast raw `/v1/completions`. The reference swap is therefore
  transport-matched for gemma31, and no transport difference is absorbed into the
  scaffold delta. The target side is held at the published OpenRouter post leg (the study
  of record's target); a labeled sensitivity substitutes the matched-transport
  `study_b_gemma31_vllmchat_base` post.
- **Exploratory status.** F7's window and min-valid threshold were registered against the
  baseline scaffold. Applying them to variant references is a diagnostic re-use of a
  frozen criterion, not a new registration.

---

## 2. Baseline reproduction attestation

**REPRODUCES — exactly, both objectives.** The new driver, run with `references =
baseline`, returns the study of record's numbers to floating-point identity (tolerance
5e-9):

| objective | n valid (cross-family) | median | range | of record | match |
|---|---:|---:|---|---|:--:|
| RPS (fit of record) | 12 / 12 | 0.9224116756 | [0.7674842, 1.0727147] | 12, 0.9224116756, [0.7674842, 1.0727147] | ✔ |
| matched-W1 (60-pt grid) | 12 / 12 | 0.9884871022 | [0.7616868, 1.0251785] | 12, 0.9884871022, [0.7616868, 1.0251785] | ✔ |

Per-target baseline table (RPS, `(o)` = own-family diagnostic, not counted):

| target POST | ref qwen | ref gemma31 | ref glm | ref maverick | F7 |
|:---|---:|---:|---:|---:|:--|
| qwen | 0.876 (o) | 0.881 | 0.960 | 1.073 | PASS |
| gemma31 | 0.908 | 0.965 (o) | 1.058 | 0.937 | PASS |
| glm | 0.849 | 0.880 | 0.993 (o) | 1.005 | PASS |
| maverick | 0.767 | 0.828 | 0.952 | 1.008 (o) | PASS |

All four targets pass F7. The reproduction is exact, so the variant arms are readable.

**Determinism note.** The agreement filter iterates the target∩reference label overlap in
**sorted** order, as `scripts/tau_daca_verbalized_fullscale.py` does. Python set iteration
over string keys is `PYTHONHASHSEED`-dependent, and the unsorted form
(`scripts/objective_contamination_audit.py`'s `daca_fit`) varies the floating-point
summation order run to run — negligible in the objective (~1e-13 relative) but enough to
move one golden-section refine in the 7th significant digit (`gemma31 ← glm` baseline:
1.0577130720 vs 1.0577129007). Sorting removes it: this driver reproduces bit-for-bit
across runs, and lands on the study-of-record value.

---

## 3. Per-variant results

### 3.1 V1 = `alt_set` references

| target POST | ref qwen | ref gemma31 | ref glm | ref maverick | n valid | F7 (RPS) |
|:---|---:|---:|---:|---:|---:|:--|
| qwen | 0.774 (o) | 0.732 | 0.856 | 1.056 | 3 | **PASS** |
| gemma31 | 0.778 | 0.809 (o) | 0.974 | 1.039 | 3 | **PASS** |
| glm | 0.803 | 0.797 | 0.844 (o) | 1.042 | 3 | **PASS** |
| maverick | 0.776 | 0.749 | 0.818 | 0.928 (o) | 3 | **PASS** |

Cross-family pool: **n = 12 valid**, median **0.8105**, range **[0.7317, 1.0559]**, all
twelve inside [0.5765, 2.3060]. Matched-W1: n = 12, median **0.7617**, range [0.6565,
1.0252], all inside — **all four targets pass F7 under both objectives**.

**V1 verdict: the corroboration is scaffold-robust under `alt_set` references.**

### 3.2 V2 = `rev_order` references

| target POST | ref qwen | ref gemma31 | ref glm | ref maverick | n valid | F7 (RPS) |
|:---|---:|---:|---:|---:|---:|:--|
| qwen | 0.770 (o) | 0.680 | 0.881 | 1.086 | 3 | **PASS** |
| gemma31 | 0.813 | 0.779 (o) | 1.048 | 1.074 | 3 | **PASS** |
| glm | 0.741 | 0.726 | 0.901 (o) | 1.015 | 3 | **PASS** |
| maverick | 0.669 | 0.746 | 0.848 | 0.977 (o) | 3 | **PASS** |

Cross-family pool: **n = 12 valid**, median **0.8309**, range **[0.6685, 1.0855]**, all
twelve inside the window.

**V2 verdict under the fit of record: the corroboration is scaffold-robust under
`rev_order` references.**

**One qualification, and it is the honest headline of this check.** Under the *matched-W1*
variant, V2 produces one fit outside the window: `qwen POST ← gemma31 rev_order BASE` fits
**0.4529**, below the lower edge 0.5765 (ln margin −0.241, ≈ 3.2 grid steps). That single
fit makes the **qwen** target FAIL the F7 window test under W1 (2 of 3 in range); the other
three targets pass, so the W1 pool is n = 12 with median 0.8567 but *not* all-inside. The
same pair under the RPS fit of record is 0.6796 — inside, with 0.148 ln of headroom.

Two things must be said about that, both of them limiting rather than excusing:

- It is **not** a saturation artifact (T_BOUNDS lower edge is 0.25; the fit is a genuine
  interior optimum) and **not** a weak-identification artifact — W1 moves 21.3% between
  T = 1 and the optimum on that filtered subpopulation, the strongest identification any
  W1 τ_DACA fit in this family of computations has shown. At baseline the W1 objective was
  nearly non-identifying (movement −9.98% to +3.29%, median −3.12%; negative means exact
  T = 1, which is off-grid, beats the grid optimum), which is precisely *why* every
  baseline W1 fit sat on the two grid points adjacent to 1.0. Under variant references W1
  starts to identify (movement up to +25.7%), and when it identifies it points lower than
  RPS does.
- The window it fails is anchored on T_J = 1.153, a **W1-fitted** constant, so
  `fit_W1 × window_W1` is the *matched* pair in the objective-contamination audit's sense
  — the one that pair passed at baseline and fails here for one target. The mixed pair of
  record (`fit_RPS × window_W1`) passes everywhere.

### 3.3 Summary of the pre-specified read

| arm | n valid (RPS) | median | range | window verdict | survives? |
|:---|---:|---:|---|:--|:--|
| baseline (reproduction) | 12 | 0.9224 | [0.7675, 1.0727] | all 4 targets PASS | — (record) |
| **V1 `alt_set`** | 12 | **0.8105** | [0.7317, 1.0559] | all 4 targets PASS | **SURVIVES** |
| **V2 `rev_order`** | 12 | **0.8309** | [0.6685, 1.0855] | all 4 targets PASS | **SURVIVES** under the fit of record; qwen target FAILS under matched-W1 |

Headroom is real but shrinking. The smallest valid fit in each arm, expressed as log
distance above the window's lower edge: baseline **+0.286**, V1 **+0.238**, V2 **+0.148**
(RPS); under W1, baseline +0.279, V1 +0.130, V2 **−0.241** (the breach above). A third
perturbation of the same size in the same direction would put the RPS read at the edge
too.

---

## 4. Direction of shift vs. prediction

**Predicted (§1.4): DOWN. Observed: DOWN, in both arms, under both objectives.**

| arm | Δ ln median (RPS) | Δ ln median (W1) |
|:---|---:|---:|
| V1 `alt_set` | **−0.1294** (0.9224 → 0.8105) | **−0.2606** (0.9885 → 0.7617) |
| V2 `rev_order` | **−0.1045** (0.9224 → 0.8309) | **−0.1430** (0.9885 → 0.8567) |

**The mechanism checks out on the objective-free read.** Mean normalized predictive entropy
of every reference leg fell under both variants — the variant references really are
sharper, not merely differently located:

| reference | baseline h̄ | V1 h̄ (Δ) | V2 h̄ (Δ) |
|:---|---:|---:|---:|
| qwen | 0.4899 | 0.4239 (−0.0660) | 0.3766 (−0.1133) |
| gemma31 | 0.5364 | 0.3707 (−0.1657) | 0.4888 (−0.0476) |
| glm | 0.5544 | 0.4642 (−0.0901) | 0.5336 (−0.0207) |
| maverick | 0.5377 | 0.4662 (−0.0715) | 0.4872 (−0.0505) |

**But the transfer is strongly attenuated, exactly as pre-specified.** The reference-side
`T_abs(pre)` inflation under `alt_set` was +0.46 to +0.72 ln on gemma31/glm/maverick; the
τ_DACA median moved only −0.129 ln — roughly **20–28% of the reference-side movement**.
Per-reference (mean Δ ln τ over the three cross-family targets that use it):

| reference | V1: Δ ln τ_DACA | V1: Δ h̄ | V1: Δ ln T_abs(pre) | V2: Δ ln τ_DACA | V2: Δ h̄ | V2: Δ ln T_abs(pre) |
|:---|---:|---:|---:|---:|---:|---:|
| qwen | −0.067 | −0.066 | +0.012 | −0.128 | −0.113 | +0.005 |
| gemma31 | −0.129 | −0.166 | +0.721 | −0.186 | −0.048 | +0.151 |
| glm | −0.116 | −0.090 | +0.455 | −0.070 | −0.021 | +0.134 |
| maverick | **+0.041** | −0.072 | +0.497 | **+0.053** | −0.051 | +0.068 |

The τ_DACA shift tracks **Δ entropy**, not Δ `T_abs(pre)`: qwen's `T_abs(pre)` is inert
(+0.012 ln) yet its entropy falls and its τ_DACA falls with it; maverick's `T_abs(pre)`
inflates most under V1 (+0.497) yet its τ_DACA moves *up*. That is the pre-specified
reason 1 confirmed on data — a large part of the `T_abs(pre)` inflation is **location**
error against the dispersed human GT, and the argmax-agreement filter suppresses location
by construction, so it never reaches τ_DACA. Reason 2 is also live: agreement rates move
by −0.16 to +0.09 across the twelve pairs, so the estimand's subpopulation re-composes
under a new reference and some of the per-pair movement is composition, not sharpening
(maverick-as-reference is the clearest case — its agreement rate with the qwen and
maverick targets *rises* under both variants while its entropy falls, and its τ goes up).

**Reading:** τ_DACA inherits the reference legs' scaffold-conditionality, in the predicted
direction, at roughly a quarter of the magnitude the absolute reference-side estimand
shows. The factor-2 F7 window absorbs it — with less margin than before.

---

## 5. Gate-blemish and pairing sensitivity

### 5.1 The two B-Q1 gate failures do not touch the primary check

`study_b_qwen_k5v2` post (contract 0.80) and `study_b_glm_k5v1` post (0.8833) are **post**
legs. The primary check swaps **references**, and references are **pre** legs, so neither
enters any number in §2–§4. Every variant **pre** leg used here passes B-Q1: qwen k5v1/k5v2,
gemma31 k5v1/k5v2, glm k5v1 0.9917 / k5v2 0.9583, maverick k5v1/k5v2 0.9667. Reference
argmax accuracy (the gate-only below-chance guard, threshold 0.2) stays far above chance in
every arm — lowest is maverick under V2 at 0.303 — so **no reference was excluded in any
arm**; all 12 cross-family fits are valid in all three arms.

### 5.2 Labeled secondary arm: both sides swapped

Because the blemished legs are the natural targets of a *full* scaffold swap, the driver
also computes a labeled secondary arm in which variant POST legs are the targets and the
same-variant PRE legs are the references (V1×V1, V2×V2). This is where the blemishes bite,
so it is reported with and without them:

| arm | targets | n valid | median | range | window | F7 all targets |
|:---|:---|---:|---:|---|:--|:--|
| V1 both-sides | all four | 12 | 0.9237 | [0.7923, 1.0714] | all inside | PASS |
| V1 both-sides | excl. glm (gate 0.8833) | 9 | 0.9591 | [0.7923, 1.0714] | all inside | PASS |
| V2 both-sides | all four | 12 | 0.8736 | [0.6629, 1.0979] | all inside | PASS |
| V2 both-sides | excl. qwen (gate 0.80) | 9 | 0.8291 | [0.6629, 1.0979] | all inside | PASS |

The verdict is insensitive to the blemished legs in both directions: dropping them moves
the median by +0.037 (V1) and −0.053 (V2) in raw units and changes no window verdict.
Note also that swapping *both* sides partly cancels the reference-side shift (V1 median
0.924 vs 0.811 references-only) — the variant post legs sharpen alongside their bases,
which is the expected behaviour of a *ratio* estimand and further evidence that the
references-only arm is the conservative (harder) test.

### 5.3 gemma31 cross-mode pairing

All gemma31 **pre** legs — baseline (`study_b_gemma31_api/pre_verbalized.json`, a
provenance copy of the vast bf16 raw base leg) and both variants — are the same transport
(bf16 vast raw `/v1/completions`), so the reference swap carries no transport confound.
The gemma31 **target** is cross-mode by necessity (raw pre / chat post; gemma-4 post
collapses under greedy raw continuation). Substituting the matched-transport
`study_b_gemma31_vllmchat_base` post for the published OpenRouter post as the gemma31
target:

| arm | n valid | median | range | F7 all targets |
|:---|---:|---:|---|:--|
| baseline | 12 | 0.9112 | [0.7675, 1.0727] | PASS |
| V1 `alt_set` | 12 | 0.8105 | [0.7317, 1.0559] | PASS |
| V2 `rev_order` | 12 | 0.8381 | [0.6685, 1.0869] | PASS |

No verdict turns on the pairing choice; the median moves by at most 0.011.

---

## 6. What this does to the τ_DACA transfer-conditions list

Transfer condition **(iv) is now tested and downgraded from "untested" to "measured, and
survivable within the diagnostic's own tolerance."** The corroboration is scaffold-robust
in the sense F7 asks about: with the references re-elicited under two different
coverage-preserving scaffold perturbations, all twelve cross-family fits stay valid and
inside [0.5765, 2.3060] for every target under the fit of record. But the condition does
not disappear — it changes character. τ_DACA **inherits** the reference legs'
scaffold-conditionality with a consistent sign (sharper references ⇒ lower τ_DACA) and
about a quarter of the magnitude the absolute reference-side estimand shows, and the
inherited movement eats real margin: the closest valid fit sits +0.286 ln above the lower
edge at baseline, +0.238 under V1 and +0.148 under V2, and under the *objective-matched*
W1 read V2 breaches it outright on one target. So condition (iv) should be restated, not
struck: *the estimator is robust to scaffold perturbation of its references at the
factor-2 tolerance F7 was given, and only at that tolerance — it is not robust in the
sense of returning a stable point estimate, and the tolerance is what is doing the work.*
This sharpens the standing role split rather than changing it: as a **GT-free cross-check**
τ_DACA continues to corroborate, now demonstrably across three scaffolds; as an **adoption
instrument** it is if anything worse off, because the same perturbations that leave the
window verdict intact move the point estimate by 0.10–0.26 ln and its objective-matched
variant by enough to cross a registered boundary. Conditions (i) elicitable references,
(ii) shared entropy regime and (iii) full-scale n are untouched by this check and stand as
written — though (ii) is worth watching, since the variant references opened the
target-reference entropy gap by 0.02–0.17 and that is the mechanism carrying the shift.

