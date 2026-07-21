# Full-scale tau_DACA in the verbalized channel (Study B data)

**Date:** 2026-07-21  
**Protocol:** r3 (`docs/e6_onpair_decision_protocol_r3.md`, ADOPTED 2026-07-21)  
**Driver:** `scripts/tau_daca_verbalized_fullscale.py`  
**Report of record:** `runs/tau_daca_fullscale/tau_daca_fullscale.json`  
**Cost:** $0 (analysis-only on existing Study B artifacts)

## 1. Purpose

Remove the 15-item scale caveat from the R0 tau_DACA prototype (section C of
`docs/verbalized_r0_analyses_2026_07_21.md`). The prototype found
NON-CORROBORATION: 2 valid fits {0.284, 0.870} with 2/4 references pegged low,
suspected cause = reference non-exchangeability in-channel. This study tests
whether the failure persists at the full 120-cell Study B scale — and if so,
whether it is structural or repairable.

## 2. Method

Each panel POST leg (qwen, gemma31_api, glm, maverick) serves as the "closed"
evaluator analog. Each panel BASE leg serves as a reference. This gives 4x4 =
16 tau_DACA fits (4 own-family + 12 cross-family). The F7 verdict is evaluated
per target on the 3 cross-family fits (matching the real E6 arm3a structure
where the closed pair is distinct from all panel families).

All distributions are epsilon-floored (eps=0.005) at analysis time. Temperature
fits use `T_BOUNDS = (0.25, 20.0)` on the standard 60-point log grid. The
agreement filter requires argmax agreement between closed (POST) and reference
(BASE).

Frozen r3 constants for F7:
- T_J = 1.153 (F2)
- DACA_RANGE = [T_J/2, 2*T_J] = [0.577, 2.306]
- DACA_MIN_VALID = 3

## 3. F7 verdict: CORROBORATING (overturns R0)

**All four panel targets pass F7.** Every cross-family tau_DACA fit is valid
(unsaturated, reference above chance) and falls within [0.577, 2.306].

| Target       | Ref: qwen | Ref: gemma31 | Ref: glm | Ref: maverick | F7     |
|:-------------|----------:|-------------:|---------:|--------------:|:-------|
| qwen POST    |  0.876(o) |        0.881 |    0.960 |         1.073 | PASS   |
| gemma31 POST |     0.908 |     0.965(o) |    1.058 |         0.937 | PASS   |
| glm POST     |     0.849 |        0.880 | 0.993(o) |         1.005 | PASS   |
| maverick POST|     0.767 |        0.828 |    0.952 |      1.008(o) | PASS   |

(o) = own-family fit (diagnostic only; not counted toward the cross-family F7).

Cross-family summary: n=12 valid fits, range [0.767, 1.073], median 0.922.

**The R0 prototype's NON-CORROBORATION was a scale artifact.** At 15 items the
agreement filter retained too few cells per reference, and 2/4 references
pegged at the lower bound (0.25). At 120 cells, none peg — every reference
yields a stable fit inside the F7 range.

### 3.1 Context: tau_v (post->pre within-family)

| Family     | tau_v |
|:-----------|------:|
| qwen       | 1.281 |
| gemma31    | 1.025 |
| glm        | 1.025 |
| maverick   | 1.380 |

tau_DACA median (0.922) < T_J (1.153) < tau_v median (1.153). The systematic
offset is explained by the agreement-filter's subpopulation selection (section 4).

## 4. Diagnosis: agreement-filter subpopulation bias

### 4.1 Selection rates

Agreement rates range 36-53% across the 12 cross-family pairs (mean ~44%).
The filter excludes over half the cells — those where POST and BASE argmax
disagree.

### 4.2 Selected-population characterization (pooled across 12 cross-family pairs)

| Quantity                   | Agree (n=615) | Disagree (n=760) |
|:---------------------------|:-------------:|:----------------:|
| Mean h_post (norm entropy) |     0.497     |      0.550       |
| Mean h_base (norm entropy) |     0.459     |      0.588       |
| Mean W1(post, GT)          |     0.486     |      0.617       |
| GT argmax match rate       |     0.69      |      0.52        |

The agreement filter selects the **sharp, well-calibrated** subpopulation:
lower entropy, lower W1 to GT, higher GT-argmax accuracy. These are cells where
both POST and BASE are confident and usually correct — precisely the cells where
calibration correction is least needed. Excluded cells are hedged and less
accurate, driving the tau_v > 1 signal.

### 4.3 Own-family tau_DACA vs tau_v

| Family    | tau_DACA (own) | tau_v | ratio | log_ratio |
|:----------|---------------:|------:|------:|----------:|
| qwen      |          0.876 | 1.281 | 0.684 |    -0.380 |
| gemma31   |          0.965 | 1.025 | 0.942 |    -0.060 |
| glm       |          0.993 | 1.025 | 0.968 |    -0.032 |
| maverick  |          1.008 | 1.380 | 0.730 |    -0.314 |

Own-family tau_DACA is systematically below tau_v. The agreement filter
excludes the cells where post-training shifted the mode — exactly the cells
driving the overconfidence signal. On the remaining "stable" cells, both legs
are already similar and the fit temperature is near 1.0.

The gap is largest for qwen (log_ratio -0.38) and maverick (-0.31) — the two
families with the strongest post-training sharpening (tau_v 1.28, 1.38). For
gemma31 and glm (tau_v = 1.025, near-unit), the gap is negligible.

### 4.4 Mode-agreement stratum decomposition

Within the agreement-selected cells, further splitting by whether the POST
argmax matches GT:

| Target → Ref        | mode_agree_gt (n) | tau   | mode_disagree_gt (n) | tau   |
|:---------------------|------------------:|------:|---------------------:|------:|
| qwen → gemma31      |            37     | 0.816 |               13     | 1.089 |
| qwen → glm          |            39     | 0.909 |               11     | 1.209 |
| qwen → maverick     |            23     | 1.022 |               19     | 1.148 |
| gemma31 → qwen      |            41     | 0.930 |               10     | 0.834 |
| gemma31 → glm       |            48     | 1.046 |               12     | 1.123 |
| gemma31 → maverick  |            30     | 0.907 |               14     | 1.017 |
| glm → qwen          |            38     | 0.821 |               14     | 0.927 |
| glm → gemma31       |            34     | 0.785 |               16     | 1.072 |
| glm → maverick      |            26     | 1.001 |               22     | 1.010 |
| maverick → qwen     |            32     | 0.747 |               14     | 0.811 |
| maverick → gemma31  |            34     | 0.723 |               27     | 0.949 |
| maverick → glm      |            40     | 0.869 |               21     | 1.170 |

The mode-agree-with-GT stratum tends lower (tau 0.72-1.05) and the
mode-disagree stratum higher (tau 0.81-1.21). Both strata remain within the F7
range [0.577, 2.306]. The R0 analysis A's mode-relocation prediction holds:
tau ~ 1 on mode-agree cells, higher on mode-disagree — but the magnitudes are
much milder than the tau_v strata (R0: 2.7-4.2 on mode-disagree).

### 4.5 Agreement-filter threshold sweep

Example: qwen POST -> maverick BASE:

| Filter          | n_pass | tau   | saturated |
|:----------------|-------:|------:|:---------:|
| argmax agree    |     42 | 1.073 | False     |
| top-2 overlap   |    101 | 2.261 | False     |
| top-3 overlap   |    118 | 3.016 | False     |
| no filter       |    118 | 3.016 | False     |

Relaxing the filter from strict argmax to top-2 overlap doubles the included
cells and triples the temperature. The top-3 and no-filter results converge
because most distributions are concentrated enough that top-3 covers the
support. This confirms the agreement filter excludes the cells driving high
temperatures — the disagreement cells need 2-3x flattening, while the
agreement cells need none.

### 4.6 Per-reference entropy profile

| Target POST     | h_post | qwen BASE h | gemma31 BASE h | glm BASE h | maverick BASE h |
|:----------------|-------:|------------:|---------------:|-----------:|----------------:|
| qwen            |  0.495 |       0.488 |          0.542 |      0.535 |           0.559 |
| gemma31         |  0.540 |       0.494 |          0.548 |      0.536 |           0.553 |
| glm             |  0.534 |       0.493 |          0.547 |      0.537 |           0.554 |
| maverick        |  0.584 |       0.490 |          0.544 |      0.537 |           0.554 |

All entropy values are in the [0.49, 0.58] range — **POST and BASE occupy the
same entropy regime**. The gap |h_ref - h_target| ranges 0.003-0.094. Contrast
with the logit channel where base distributions were dramatically sharper than
the production pair. This narrow entropy gap is the structural reason tau_DACA
works in the verbalized channel: references are exchangeable-enough that the
agreement-filter bias stays mild.

The gradient: qwen BASE is the sharpest (h = 0.49), maverick POST is the most
hedged (h = 0.58). Maverick POST → qwen BASE produces the lowest tau_DACA
(0.767), consistent with the largest entropy gap driving the strongest
sharpening signal. This is the only fit below 0.8 and the closest to the F7
lower bound (0.577).

## 5. Structural vs repairable

**Structural, but benign in-channel.** The agreement filter estimates a
different conditional quantity than T_J:

- T_J (and tau_v): temperature on ALL cells, including the mode-shifted hard
  cells that carry the overconfidence signal.
- tau_DACA: temperature on the agreement-selected subpopulation, where both
  target and reference are sharp, correct, and already well-calibrated.

This is by design — DACA Prop. 3.3 mandates excluding disagreement cells to
avoid conflating miscalibration with dispute. The resulting stratum bias
(documented as `agreement_rate` in the fit output) is inherent.

**Why this is benign in the verbalized channel but fatal in the logit channel:**
In the logit channel, T_J fell in the 1.60-4.88 range, and the agreement-filter
bias toward ~1.0 placed tau_DACA far outside the factor-2 window. In the
verbalized channel, T_J = 1.153 is close enough to 1.0 that the bias (median
tau_DACA 0.922) stays within [0.577, 2.306]. The verbalized channel's narrow
POST/BASE entropy gap (section 4.6) keeps the estimator in-regime.

### 5.1 Repair prototype: entropy-matched reference selection

As a labeled exploratory variant, an entropy-matched reference selector picks
the BASE family closest in entropy to the target POST cell, then applies the
standard argmax-agree filter:

| Target    | gap <= 0.10: n, tau | gap <= 0.25: n, tau |
|:----------|--------------------:|--------------------:|
| qwen      |    53,  1.070       |    61,  1.063       |
| gemma31   |    49,  0.961       |    70,  0.997       |
| glm       |    67,  1.005       |    72,  1.021       |
| maverick  |    52,  0.996       |    60,  1.008       |

The repair pulls tau toward 1.0 (well-calibrated by construction — matching
entropy is matching spread). It does NOT converge to T_J (1.153) because it
still operates on the agreement-selected subpopulation. The repair is
unnecessary: standard tau_DACA already passes F7 without modification.

## 6. llama31 diagnostic

llama31 is excluded from adoption-relevant sets but tested as a diagnostic
target (its tau_v = 1.025 duplicates gemma31's grid point).

| Ref family | tau_DACA | agree rate | saturated |
|:-----------|---------:|-----------:|:---------:|
| qwen       |    0.795 |       0.41 | False     |
| gemma31    |    0.896 |       0.42 | False     |
| glm        |    0.861 |       0.43 | False     |
| maverick   |    1.019 |       0.38 | False     |

F7 on cross-family references: n_valid=4, taus=[0.795, 0.861, 0.896, 1.019] —
**CORROBORATING**. Its inclusion/exclusion is numerically inert for the F7
verdict, consistent with its documented inertness for tau_v.

## 7. Recommended fate of arm 3a (tau_DACA) in E6 protocol

**KEEP arm 3a as the registered report-only cross-check.**

1. tau_DACA CORROBORATES at full scale (120 cells) in the verbalized channel.
   The R0 prototype's NON-CORROBORATION was a 15-item scale artifact — pegged
   fits at n=15 become stable fits at n=120.

2. The systematic downward bias (median tau_DACA 0.922 vs T_J 1.153) is the
   documented agreement-filter subpopulation effect. It is structural (the
   estimator targets a different conditional quantity than T_J) but benign —
   all 12 cross-family fits are well within F7's [0.577, 2.306] range.

3. No protocol change is needed. The r3 frozen constants (F7 range and
   min-valid threshold) remain appropriate. The entropy-matched repair variant
   is documented as exploratory but NOT recommended for adoption — standard
   tau_DACA passes F7 without modification.

4. The POSITIVE result strengthens the E6 protocol's expected outcome: when the
   real on-pair sweep runs, arm 3a is likely to corroborate (the actual
   Claude+GPT closed pair will have entropy in the same regime as the panel
   POST legs, and the agreement filter will produce stable fits at 120 cells).
   However, the real closed pair is instruction-tuned for a different task
   (cross-family evaluation, not few-shot compliance scoring), so its
   agreement rate with the open BASE references may differ from the 36-53%
   observed here.

5. This result supersedes the logit-channel tau_DACA failure (`spec/
   analysis_2026_07_12_deterministic_tau_bridge_feasibility.md`) — that failure
   was channel-specific (driven by the logit vs verbalized entropy gap), not a
   property of the estimator itself.

## 8. Summary for paper integration

The finding informs `judex-paper/judex_paper_v4_verbalized_calibration.tex`:

- **Section on channel selection:** the logit-channel tau_DACA failure (published
  negative) vs verbalized-channel tau_DACA success (this analysis) is evidence
  that the verbalized channel produces more coherent pre/post distributions for
  calibration purposes.

- **Section on E6 cross-checks:** at full scale, the GT-free agreement-filtered
  estimator corroborates the transferred constant T_J within a factor of 2.
  The systematic downward bias (median 0.922 vs 1.153) is the agreement
  filter's subpopulation effect — documented, expected, and within tolerance.

- **Methodological note:** the agreement filter selects the well-calibrated
  subpopulation (GT-argmax match 0.69 vs 0.52 on excluded cells). This is a
  known property (DACA Prop. 3.3) that limits the estimator's sensitivity to
  the calibration correction it is meant to corroborate. The factor-2 F7 range
  absorbs this limitation.
