# Phase 0 — verbalized-channel sensitivity results (2026-08-08)

Cost: $0. Pure re-analysis of stored artifacts; no elicitation, no API spend, no GPU.
Scope: the two verbalized sensitivity items of
`spec/plan_2026_08_08_verbalized_arm_gap_closure.md` §1b — the ε-floor sweep, completed from
2 of 8 leg pairs to 8 of 8, and the τ range-robustness sweep, moved off 15-item/off-pair
proxies onto the on-pair confirmation run of record.

Reading rules for both were fixed in advance in
`spec/memo_2026_08_08_phase0_prespecified_reads.md`.

---

## 1. ε-floor sweep — all eight verbalized leg pairs

### 1.1 What was run

`scripts/run_study_b_leg.py --analyze --eps-sensitivity` on the six leg dirs that lacked an
`epsilon_sensitivity` block (`study_b_gemma26`, `study_b_gemma26_api`, `study_b_gemma31`,
`study_b_gemma31_api`, `study_b_glm`, `study_b_llama31`). `study_b_qwen` and
`study_b_maverick` already carried the block and were not re-run; their values below are read
from the existing reports.

The floor replaces exact contract zeros before any temperature machinery, because the
inverse-softmax seam maps a zero to the seam's own 1e-12 clip (≈ −27.6 in logit units). The
registered value is ε = 0.005, one tenth of the 0.05 grid the elicited credences lie on.

τ_v is an exhaustive argmin over a 60-point logarithmic grid on [0.25, 20.0]. One grid step
is a factor 1.0771, i.e. |Δln| = 0.0743. Movements smaller than that are not resolvable.

### 1.2 τ_v across the four floors

| Family | ε=0.001 | ε=0.005 | ε=0.0125 | ε=0.025 | max/min | grid steps | panel member |
|---|---:|---:|---:|---:|---:|---:|:--:|
| qwen | 1.2811 | **1.2811** | 1.1894 | 1.1042 | 1.1601 | 2 | yes |
| maverick | 1.4862 | **1.3798** | 1.1894 | 1.0252 | 1.4497 | 5 | yes |
| glm | 1.0252 | **1.0252** | 1.0252 | 1.0252 | 1.0000 | 0 | yes |
| gemma31 (API post) | 1.0252 | **1.0252** | 1.0252 | 1.0252 | 1.0000 | 0 | yes |
| gemma31 (raw post; gate-failing) | 2.3207 | 1.8571 | 1.6008 | 1.3798 | 1.6819 | 7 | no |
| gemma26 (API post) | 1.3798 | 1.1894 | 1.0252 | 1.0252 | 1.3459 | 4 | no |
| gemma26 (raw post; gate-failing) | 0.2500 † | 0.3365 | 1.0252 | 1.0252 | 4.1007 | 19 | no |
| llama31 (excluded pre-hoc) | 1.1894 | 1.0252 | 1.0252 | 1.0252 | 1.1601 | 2 | no |

† pegged on the lower search bound — a saturation flag, not a fit. It is the only saturated
τ_v anywhere in the sweep, and it belongs to the leg whose post twin parses 1 of 120 cells.

At ε = 0.005 every family reproduces its shipped τ_v exactly (8/8, to floating-point
identity). The registered floor is therefore not a re-derivation: it is the value every
published τ_v was already computed at.

### 1.3 T_c(post) across the four floors

| Family | ε=0.001 | ε=0.005 | ε=0.0125 | ε=0.025 | max/min |
|---|---:|---:|---:|---:|---:|
| qwen | 5.6582 | 4.8772 | 4.2040 | 3.9030 | 1.4497 |
| maverick | 8.2028 | 7.0705 | 6.5644 | 6.5644 | 1.2496 |
| glm | 11.8916 | 8.8352 | 7.0705 | 6.0945 | 1.9512 |
| gemma31 (API) | 6.0945 | 4.5281 | 3.9030 | 3.3643 | 1.8115 |
| gemma31 (raw) | 17.2393 | 13.7959 | 11.8916 | 10.2501 | 1.6819 |
| gemma26 (API) | 17.2393 | 12.8084 | 10.2501 | 8.2028 | 2.1016 |
| gemma26 (raw) | 20.0000 ‡ | 20.0000 ‡ | 20.0000 ‡ | 20.0000 ‡ | 1.0000 |
| llama31 | 17.2393 | 20.0000 ‡ | 20.0000 ‡ | 20.0000 ‡ | 1.1601 |

‡ pegged at the upper search bound. T_c moves monotonically downward with the floor on every
family that fits at all: a higher floor compresses the stated-confidence link's dynamic
range, so less softening is required. T_c is a diagnostic, not a published constant.

### 1.4 The read that matters: panel-derived quantities per floor

The adoption panel is {qwen, gemma31 (API), glm, maverick}. Recomputing the panel's derived
quantities at each floor:

| ε | panel τ_v (qwen, gemma31, glm, maverick) | T_J (interpolated median) | band | max/min | ≤ 2 rule |
|---|---|---:|---|---:|:--:|
| 0.001 | 1.2811, 1.0252, 1.0252, 1.4862 | 1.1531 | [1.0252, 1.4862] | 1.4497 | PASS |
| **0.005** | 1.2811, 1.0252, 1.0252, 1.3798 | **1.1531** | **[1.0252, 1.3798]** | **1.3459** | PASS |
| 0.0125 | 1.1894, 1.0252, 1.0252, 1.1894 | 1.1073 | [1.0252, 1.1894] | 1.1601 | PASS |
| 0.025 | 1.1042, 1.0252, 1.0252, 1.0252 | 1.0252 | [1.0252, 1.1042] | 1.0771 | PASS |

Three facts follow.

1. **The clustering result is floor-invariant.** The max/min ratio stays at or below 1.45 at
   every floor, comfortably inside the ≤ 2 rule, at every point of the sweep. The study's
   central qualitative claim — the verbalized channel clusters where the other channel did
   not — does not depend on the floor at all.
2. **T_J is unchanged at the registered floor and its neighbour below**, 1.1531 at both
   ε = 0.001 and ε = 0.005, and moves only at floors 2.5× and 5× larger.
3. **The band contracts monotonically as the floor rises**, and it is maverick that drives
   both ends of that movement: maverick alone spans 1.4862 → 1.0252 across the sweep, five
   grid steps, and it is the family that defines the band's upper edge.

### 1.5 Published-value boundary crossings (the flag the task asks for)

- **maverick, ε = 0.001**: τ_v = 1.4862 sits **above** the published band's upper edge
  1.3798. Since the band is the panel range and maverick defines its upper edge, the band
  would read [1.0252, 1.4862] at that floor.
- **qwen, ε ≥ 0.0125**: τ_v leaves its published grid point 1.2811 (→ 1.1894, → 1.1042).
- **glm and gemma31 (API)**: invariant at 1.0252 across all four floors; the band's lower
  edge is stable in every case.
- **T_J**: unchanged at ε ∈ {0.001, 0.005}; 1.1073 at 0.0125; 1.0252 at 0.025.
- Outside the panel, the largest movement is the gate-failing raw gemma26 leg (19 grid steps,
  including a lower-bound peg) — an artifact of a leg that parses 1 of 120 cells, not a
  measurement.

Per the pre-specified reading, none of this licenses changing ε. The floor is derived from
the contract's own grid, and re-tuning an analysis constant to protect a published value is
precisely the move the panel-robustness principle forbids. What the sweep licenses is a
reported magnitude.

---

## 2. τ range-robustness on the on-pair confirmation run

### 2.1 Instrument

New file `scripts/range_robustness_verbalized.py` — the verbalized-native successor of the
logit-era `scripts/q4_range_robustness.py`, which is left unedited as that channel's record.
The scratchpad script the 2026-07-21 analysis used no longer exists; this one is rebuilt from
that document's stated specification and is committed.

- **Data**: `judex-evaluator/runs/stage9-onpair-e6-20260721/metrics_report.json` — the
  deployed pair, 120 items, 24 documents, contract 0.2.0, `configs_v2exemplars`.
- **Join**: on `item_label` identity, 120/120 matched, zero unmatched on either side. Level
  order is read from each prediction's own label names, never from emission order.
- **Ground truth**: `aireg.load_cells()` only.
- **Criterion**: F4 — Reliability strictly better AND mean RPS no worse (1e-9) AND
  |Resolution change| within 10% of the uncalibrated Resolution, on the 10-bin Murphy
  decomposition. Every number is produced by `study_a.closed_side_check`, the production
  helper the protocol names.
- **Grid**: 95 log-spaced points over [0.800, 2.000], 30 more across [1.0252, 1.3798], plus
  markers at T_J, both exact band edges, the four panel τ_v values, identity, and the nine
  across-band profile points — 134 unique points.
- **Bootstrap**: 2000 replicates, cluster = document (24 clusters, 5 Articles each), seed
  20260720.
- **Identity handling**: the uncalibrated baseline is the stored vectors. `apply_temperature`
  at T = 1 is a *near*-identity, not an identity — the inverse-softmax seam clips the 144
  exact zeros in the stored vectors at 1e-12 before renormalising — so the round trip is
  measured and reported (max |Δp| = 1.5e-12, Δ mean RPS = −2.4e-14) rather than asserted.
- **Vectorized replica**: the bootstrap's weighted Murphy path is asserted to reproduce
  `closed_side_check` exactly at unit weights at every one of the 134 grid points.

Output: `runs/q4_range_onpair_20260808/range_robustness_verbalized.json` (gitignored,
host-local), with an ε = 0.005 sensitivity variant under `eps0005/`.

### 2.2 Cross-check against the committed record

The nine across-band profile points in
`runs/e6_r3_stage9-onpair-e6-20260721/e6_r3_arms.json`
(`arm1_transferred_T_J.band_profile`) are reproduced **exactly** — all four reported fields
(reliability improvement, RPS change, resolution change, F4 verdict) identical at all nine T
values, including the FAIL at the band's lower edge. Independently, the supervised full-sample
fit comes back at **T\* = 2.5446**, matching the committed arm-2 value 2.545.

### 2.3 Result

Uncalibrated baseline: Reliability 0.02092, Resolution 0.04916, mean RPS 0.03350, mean W1
0.51036, argmax accuracy 0.39167.

**F4 acceptance region within [0.800, 2.000]: [1.0477, 2.000], contiguous**, 102 of 134 grid
points passing.

- The **lower edge is 1.0477**, and the criterion that binds below it is *Reliability not
  strictly better* — the correction becomes too small to register an improvement, the same
  mechanism the proxy datasets showed in 2026-07-21.
- There is **no upper break inside the sweep**: F4 still passes at 2.000 and beyond, so the
  upper edge reported here is the sweep ceiling, not a crossing. The 2026-07-21 proxy
  expectation of an RPS crossing near ~1.40 does not materialise on the run of record; at 120
  items and 24 documents the RPS improvement keeps growing across the whole range.
- **T_J = 1.153 is inside the region**, 0.0958 in log units above the lower edge — about 1.3
  τ_v grid steps of margin.
- **The band [1.0252, 1.3798] is not fully covered.** Its lower 7.3% (in log width),
  [1.0252, 1.0477], fails; the rest passes. This restates as a region what the committed
  single-point read already said: the band's lower edge fails F4 while T_J and the upper edge
  pass.
- **Identity (T = 1.0) fails**, with reliability improvement −0.00034 — the run is not already
  calibrated.

Point reads:

| T | reliability improvement | RPS change | resolution change | F4 |
|---|---:|---:|---:|:--:|
| 1.0000 (identity) | −0.00034 | −0.00000 | +0.00076 | FAIL |
| 1.0252 (band low) | −0.00033 | −0.00042 | +0.00120 | FAIL |
| 1.1530 (T_J) | +0.00151 | −0.00218 | +0.00086 | **PASS** |
| 1.2811 (τ_v qwen) | +0.00379 | −0.00345 | −0.00260 | **PASS** |
| 1.3798 (band high) | +0.00485 | −0.00419 | −0.00406 | **PASS** |

### 2.4 Bootstrap (2000 doc-clustered replicates, seed 20260720)

| Quantity | Median | 95% interval |
|---|---:|---|
| Lower edge of the acceptance region | 1.0208 | [1.0000, 1.2285] |
| Upper edge | 2.0000 | [1.2526, 2.0000] |

F4 pass rate across replicates: **T_J 0.931**, qwen's τ_v 0.846, band high 0.679, band low
0.450, identity 0.430. P(band fully covered) = 0.263.

Two honest caveats. First, 949 of 2000 replicates produce a non-contiguous acceptance set —
the region has ragged edges under resampling, because the Murphy decomposition's calibration
groups are re-populated by every reweighting. Second, the pass rate at T_J, 0.931, sits just
below the ≥ 0.964 the 2026-07-21 analysis projected from its 120-item off-pair proxies; the
direction was right, the separation is slightly weaker on the real pair. Two replicates
produce an empty region.

### 2.5 ε = 0.005 sensitivity on the closed-side vectors

Applying the channel's floor to the *closed pair's* stored vectors before the sweep (a
convention the committed instrument does not use — `e6_r3_arms.load_closed` reads them as
emitted) moves the lower edge to 1.0352, leaves T_J inside, still leaves the band's lower edge
outside, and raises the bootstrap pass rate at T_J to 0.988. It also moves the supervised fit
from 2.5446 to 1.7028, which is precisely why the record's convention is the unfloored one:
2.545 is the value the frozen arm-2 read carries. The variant is reported as a sensitivity and
changes no verdict about T_J or the band.

---

## 3. Integrity attestation

Before any re-analysis, each of the eight `runs/study_b_*` leg dirs had its
`study_b_report.json` copied to `study_b_report.pre_eps_backup_20260808.json` in the same
directory. Those backups are on disk.

A field-by-field semantic diff of the six regenerated reports against their backups gives:

| Dir | epsilon_sensitivity fields added | Other fields added | Fields removed | Other fields changed |
|---|---:|---|---|---|
| study_b_gemma26 | 12 | none | none | none |
| study_b_gemma26_api | 12 | none | none | none |
| study_b_gemma31 | 12 | none | none | `legs/{pre,post}/score/T_rel` |
| study_b_gemma31_api | 12 | none | none | `legs/pre/score/T_rel` |
| study_b_glm | 12 | none | none | `legs/{pre,post}/score/T_rel` |
| study_b_llama31 | 12 | none | none | `legs/{pre,post}/score/T_rel` |

**The `T_rel` movement is not caused by this pass.** Three controls, all run on copies in a
scratch directory so that no `runs/` artifact was touched:

1. `--analyze` **without** `--eps-sensitivity` on a copy of `study_b_glm` produces the *same*
   two `T_rel` changes and nothing else. The ε flag contributes zero collateral change.
2. Re-analysing copies of `study_b_qwen` and `study_b_maverick` — the two dirs left untouched
   — reproduces their committed reports field-for-field **including the entire
   `epsilon_sensitivity` block**, with the single exception of the same `T_rel` fields. The ε
   instrument is deterministic and reproduces the record exactly.
3. The cause is dated and committed: `ca13157` ("Objective-contamination audit + Murphy
   bin-count hygiene fix", 2026-07-24) unified the Murphy reliability bin count behind
   `MURPHY_BINS = 10`; the `T_rel` fitter previously used `bins=3`. The Study B reports were
   written 2026-07-20, four days earlier. Every one of the eight dirs therefore carried a
   stale `T_rel`, and the two that were not re-run still do.

`T_rel` is an auxiliary reliability temperature. It is not a published constant, appears in no
paper version, feeds no gate, and is not an input to τ_v, T_J, the band, the cluster ratio, or
T_abs. Every field that does feed those — including the entire `murphy` block, `T_rps`, the
contract-compliance gates, `tau_v`, and the B-Q4 confidence block — is bit-identical across
the six dirs.

The regenerated reports are left in place, because they are what the current code produces and
the pre-existing value was stale under a fix that predates this work. The backups remain on
disk if the alternative is preferred.

**Concurrent-edit window.** A separate Phase-1a preparation change to `scripts/run_study_b_leg.py`
(adding `leg_meta` / `scaffold_variant` provenance echoes to each leg block, plus a
`--scaffold-variant` flag) landed in the working tree at 21:39:44 on the same day. All six
reports were written 21:38:37–21:38:47 and all three scratch controls 21:39:38–21:39:43, i.e.
before that edit; none of the regenerated reports contains a `leg_meta` or `scaffold_variant`
field, verified by grep. The concurrent edits to `src/judex_calibration/elicit_verbalized.py`
and `fewshot.py` touch only the elicitation path (`build_fewshot_by_criterion`, `run_variant`,
`elicit_cell`), which `--analyze` does not call. Re-running `--analyze` on these dirs after the
Phase-1a change lands will legitimately add those two provenance fields per leg.

---

## 4. Paper-facing sentences (drafts for v15)

Prose convention: sequence by logic, not by discovery chronology; no registration or freezing
rhetoric.

**Appendix A — the ε outcome sentence** (replacing the bare "swept as a sensitivity" clause,
which has stood unreported in every version):

> Sweeping the floor over {0.001, 0.005, 0.0125, 0.025} leaves the clustering result intact
> at every value: the panel's ratio of largest to smallest increment stays between 1.08 and
> 1.45, well inside the factor of two the transfer requires. The transferred constant is
> unchanged at 0.001 and 0.005 and falls to 1.107 and 1.025 at the two coarser floors, and
> the panel's range narrows with the floor, driven by the single family that defines its
> upper end. Two of the four panel families are invariant across the whole sweep. The floor
> is fixed by the contract's grid rather than chosen for its effect, so we report this
> dependence rather than tuning against it.

**Appendix C/G — re-anchoring the τ-range sensitivity to the run of record:**

> On the deployed pair's own run, the temperatures that satisfy the acceptance criterion form
> a contiguous interval whose lower end is 1.048; above it, no criterion binds anywhere below
> 2.0. The transferred constant sits inside that interval with roughly 0.10 in log units of
> margin, while the lower end of the panel's range does not: the correction there is too small
> to improve reliability at all. Resampling documents, the constant satisfies the criterion in
> 93% of replicates and the interval's lower end has a 95% interval of [1.00, 1.23].

> The binding constraint at the low end is reliability, not resolution. Resolution moves by
> less than a tenth of its uncalibrated value across the entire swept range, so the concern
> that a correction of this kind buys calibration by discarding discrimination does not
> arise at these temperatures.

**Limitations / §6.1, one sentence of scope:**

> The acceptance interval is a property of one run, one reference, and one pair; what
> transfers is the observation that the constant sits inside it with margin while the panel
> range's lower end does not.

---

## 5. Files

Created:
- `scripts/range_robustness_verbalized.py` — the committed verbalized-native instrument.
- `docs/phase0_verbalized_sensitivity_20260808.md` — this document.
- `runs/q4_range_onpair_20260808/range_robustness_verbalized.json` (+ `eps0005/`) — gitignored.
- `runs/study_b_*/study_b_report.pre_eps_backup_20260808.json` × 8 — gitignored.

Modified:
- `runs/study_b_{gemma26,gemma26_api,gemma31,gemma31_api,glm,llama31}/study_b_report.json` —
  gitignored; `epsilon_sensitivity` added, `T_rel` refreshed as documented in §3.

Unchanged by this work: `scripts/e6_r3_arms.py`, `scripts/absolute_vs_ratio_estimand.py`,
`scripts/absolute_mechanism_rA1.py`, `scripts/q4_range_robustness.py`, every `src/` module,
and the `study_b_qwen` / `study_b_maverick` run dirs. Other tracked files that appear modified
in the working tree (`scripts/run_study_b_leg.py`, `src/judex_calibration/{fewshot,
elicit_verbalized,elicit_api_verbalized}.py`) and the untracked
`scripts/run_study_b_api_leg.py`, `scripts/scaffold_variant_probe.py`,
`tests/test_scaffold_variants.py` belong to the concurrent Phase-1a preparation, not to this
pass (see §3).

Also written outside this repo: `spec/memo_2026_08_08_phase0_prespecified_reads.md` in the
umbrella.
