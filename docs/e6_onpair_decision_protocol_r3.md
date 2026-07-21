# E6 on-pair sweep — decision protocol r3 (verbalized-first)

**ADOPTED 2026-07-21 (user confirmation of F1–F8, same day as the draft).** r3 is
now the protocol of record and SUPERSEDES r2
(`docs/e6_onpair_decision_protocol.md`) in full; r2 is retained unedited as the
logit-era record, and its frozen inputs (band [1.60, 4.88], anchor 2.794, λ 0.35,
the A1–A4 pool arm on token-slice bases) are retired with their channel — none of
them is referenced by any r3 instrument. Provenance: drafted 2026-07-21 under the
user's agreement to proceed to r3 (the reframe mandate + plan
`docs/verbalized_calibration_plan.md`); the frozen constants below were confirmed
explicitly and close before any on-pair datum exists.

Scope: the decision logic for the held E6 on-pair sweep (Sonnet 4.6 medium + GPT
5.4 medium, `configs_v2exemplars`, 24 docs × 5 Articles = 120 cells, ~$445,
user-gated). All calibration instruments operate on the closed pair's native
**verbalized** contract-0.2.0 outputs — the channel the constant was derived in.
Every temperature below: ε = 0.005 floor before inverse-softmax,
`T_BOUNDS = (0.25, 20.0)` explicit, boundary = peg = never adoptable,
doc-clustered bootstrap seed 20260720 where a CI is called for.

## 0. Frozen constants (the six decisions of plan §5, resolved)

| # | decision | FROZEN value |
|---|---|---|
| F1 | Arm-1 panel | {qwen, gemma31, glm, maverick} (user decision 2026-07-21; llama31 excluded, gemma31 = chat-template collection) |
| F2 | Arm-1 constant | **τ\* = 1.153** — the interpolated median of the panel τ_v {1.025, 1.025, 1.281, 1.380} |
| F3 | Sensitivity band | **[1.025, 1.380]** (the panel range; written [1.03, 1.38] in prose) |
| F4 | Arm-1 acceptance (`closed_side_check` at τ\*) | Murphy Reliability strictly better ∧ mean RPS no worse (≤ +1e-9) ∧ Resolution within 10% of uncalibrated — E6's registered criteria, carried over unchanged |
| F5 | Arm-2 concurrence | held-out supervised T\* concurs with τ\* iff \|ln(T\*/τ\*)\| ≤ ln 2 (the factor-2 rule, mirroring the adopted clustering gate) |
| F6 | Pool arm (arm 3b) | membership = the four panel families' verbalized **BASE legs only** (no cfp1 augmentation); estimator = **raw** `fit_dispersion_temperature` — **no shrinkage** (the logit anchor/λ are dead with their band; no re-derivation, to keep the arms decoupled); vetting: pool mixture entropy ≥ 1.00 nats on the sweep's parsed cells ∧ LOBO min ≥ 1.00 ∧ fit unsaturated; concurrence = T_raw ∈ [1.025, 1.380] |
| F7 | τ_DACA arm (arm 3a) | report-only; per-reference validity = unsaturated ∧ reference not below chance (unchanged); **corroboration verdict iff ≥ 3 valid fits AND their full range ⊂ [τ\*/2, 2τ\*]**; the reference-vs-closed mean-normalized-entropy gap is published per reference as the non-exchangeability diagnostic |
| F8 | Arm-4 adoption bar | per-seat + pooled T_c on the closed pair's confidence distributions; "carries signal" iff LODO calibratable (mean OOS ΔBrier < 0, 95% doc-clustered CI excluding 0) ∧ full-sample T_c unsaturated ∧ **Kendall τ_b(p̂(C), realized W1) ≤ −0.10**; arm 4 is report-only within E6 — any production use is a separate user decision |

(F5's factor-2 is deliberately generous: arm 2 estimates a *different estimand* —
T\*-vs-GT absorbs base-vs-panel miscalibration on top of the post-training
increment, so T\* ≥ τ\* is expected. Divergence beyond factor-2 is published, not
reconciled.)

## 1. The instrument ladder

1. **Arm 1 — transferred τ\* = 1.153** (primary candidate mechanism).
   `study_a.closed_side_check(closed_preds, cells, T=1.153)` on the sweep's
   reconciled verbalized outputs, plus the same check at the band edges
   {1.025, 1.380} as the frozen sensitivity sweep. Passing F4 at τ\* is the
   adoption condition (adoption itself = the user pasting the emitted block;
   this protocol only recommends).
2. **Arm 2 — supervised held-out T\*** (validation, demoted from r2's primary).
   Held-out doc-clustered fit on the sweep vs AIReg GT; reports T\*, saturation,
   F5 concurrence, and its own F4-style acceptance read.
3. **Arm 3b — verbalized dispersion pool** (GT-free cross-check, F6). Reports
   T_raw, vetting gates, band concurrence. Never gates arm 1.
4. **Arm 3a — τ_DACA-verbalized** (GT-free cross-check, F7). Reports per-reference
   fits + the corroboration verdict. Never gates anything.
5. **Arm 4 — confidence instrument** (novel exhibit, F8). Per-seat + pooled T_c,
   the association diagnostic, risk-coverage before/after. Report-only.
6. **Band-only sensitivity reporting** (always emitted): `closed_side_check`
   deltas across [1.025, 1.380], the r3 analog of r2's band caveat.

## 2. What each instrument gates vs. reports

| instrument | gates adoption? | reports |
|---|---|---|
| Arm 1 (τ\* = 1.153) | Yes — F4 at τ\* is the recommendation condition | reliability/RPS/resolution deltas at τ\* and both band edges |
| Arm 2 (supervised T\*) | No — validation only | T\*, saturation, F5 concurrence, estimand-gap note |
| Arm 3b (pool) | No | T_raw, vetting, band membership |
| Arm 3a (τ_DACA) | No | per-reference fits, validity, entropy-gap diagnostic, F7 verdict |
| Arm 4 (T_c) | No (report-only in E6) | T_c per seat, LODO, association, AURC |
| Band-only | No — always-on caveat | delta profile across the band |

## 3. The failure ladder (terminal state)

```
closed_side_check at tau* = 1.153 on the sweep (F4)
   |
   +-- PASSES -> RECOMMEND adoption of tau* (user pastes the emitted calibration
   |             block; provenance JSON kept alongside the run). Report arm 2/3a/3b
   |             concurrence as corroboration (non-gating) + the band profile.
   |
   +-- FAILS
          |
          +-- arm 2's T* passes its own accuracy gate, is unsaturated, AND lands
          |   in the band [1.025, 1.380]
          |     -> REPORT T* as the candidate (user decision; not auto-adopted).
          |        This is the r2 mechanism surviving as a fallback, now
          |        band-constrained in-channel.
          |
          +-- otherwise
                -> TERMINAL: band-only sensitivity reporting on [1.025, 1.380];
                   no point correction adopted; the evaluator seam stays
                   mode: noop. The protocol never produces a worse outcome than
                   not having run it.
```

## 4. Concurrence table (publish alongside the sweep)

| instrument | T | vs τ\* = 1.153 | gate/verdict |
|---|---|---|---|
| Arm 1 τ\* | 1.153 (frozen) | — | F4 pass/fail |
| Arm 2 supervised T\* | (measured) | \|ln ratio\| vs ln 2 (F5) | own accuracy gate; band membership |
| Arm 3b pool T_raw | (measured) | band membership (F6) | vetting pass/fail |
| Arm 3a τ_DACA | (measured, per ref) | F7 range criterion | validity per ref |
| Arm 4 T_c | (measured, per seat) | n/a (different object) | F8 bar |
| r2 logit-era instruments | — | — | retired with their channel (r2 retained as record) |

## 5. Explicit non-interference

- The Study B pre-registration (`docs/study_b_design.md` §4, §6) is FROZEN and
  untouched; r3 consumes its gate definitions verbatim.
- The r2 protocol and every logit-era freeze it composes
  (`dispersion_pool_promotion.md` A1–A4, `q4_range_robustness.md` band framing,
  the 2(d) composite) are retained UNEDITED as the Study A record; r3 references
  none of their constants.
- The R0 prototype numbers (15-item closed run) informed F6's membership choice
  and F7's expectation-setting but are smoke-grade and appear in no acceptance
  criterion.
- All six frozen decisions (F1–F8) close before any on-pair datum exists —
  nothing in this protocol is tuned against the sweep it governs.
- **Terminal-state guarantee (carried from r2):** worst case degrades to
  band-only reporting with the seam at `mode: noop` — never worse than doing
  nothing.
