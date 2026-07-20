# E6 on-pair sweep — composite decision protocol

**ADOPTED 2026-07-19 (user).** This is the single source of truth for the decision logic of the
held E6 on-pair sweep (Sonnet 4.6 medium + GPT 5.4 medium, 24 docs × 5 Articles = 120 cells,
~$290–330, user-gated). It is the §5 protocol of the band-mechanism composite analysis
(umbrella `spec/analysis_2026_07_19_band_mechanism_composite.md`, "the 2(d) doc"), adopted as
drafted. It composes three FROZEN, independently pre-registered instruments — it edits none of
them:

- the **A1–A4 pool-arm pre-registration** — umbrella
  `spec/analysis_2026_07_19_dispersion_pool_promotion.md` §6 (instrument:
  `scripts/dispersion_pool_unsupervised_arm.py`), and
- the **[1.60, 8.84] sensitivity-band framing** — umbrella
  `spec/analysis_2026_07_19_q4_range_robustness.md` (instrument:
  `scripts/q4_range_robustness.py`).

Direction-safety qualifiers below ("§4(a)", "§4(b)") refer to the 2(d) doc's §4, whose
multi-target correction is the reason the fixed-3.761 fallback carries a near-calibrated
exclusion: only the band's lower edge (T = 1.60) is safe on all 7 validated targets; the band
interior harms a near-calibrated target (qwen_post, T\* = 1.353, safe zone ≈ [1.30, 1.75]).

## 1. The instrument ladder

1. **Supervised held-out T\*** (primary mechanism). `fit_temperature` on the closed pair's own
   predictions vs AIReg GT, `T_BOUNDS=(0.25,20.0)`, `study_a.saturated()` checked. This is what
   actually gets used if it passes its own accuracy gate — nothing in this protocol changes that.
2. **Frozen pool arm, A1–A4** (unsupervised *validation* arm — reports concurrence, per
   `spec/analysis_2026_07_19_dispersion_pool_promotion.md` §6, UNCHANGED). Gates on A1∧A2∧A2b∧A3;
   A4 secondary/non-gating.
3. **Fixed-3.761 zero-fit fallback** (characterized in the 2(d) doc). Activates **only** as a
   fallback — see §3.
4. **Band-only sensitivity reporting** (always reported, per `q4_range_robustness`'s framing):
   "any τ in [1.60, 8.84] improves reliability/RPS on the open panel, subject to the qualification
   in the 2(d) doc's §4(a)."

## 2. What each instrument gates vs. reports

| instrument | gates production adoption? | reports |
|---|---|---|
| Supervised T\* | Yes — the accuracy-gated production mechanism, unconditionally the first choice if it passes its own gate | T\*, saturation flag, band membership |
| Pool arm (A1–A4) | Only as an unsupervised **cross-check** on T\* — never gates T\* itself | T̂, gap-closure vs T\*, A1–A4 pass/fail, A4 concurrence ratio |
| Fixed-3.761 | **Activates only in the failure branch** (§3) — never overrides a passing supervised fit | gap-closure vs T\* (post-hoc, once T\* is known), §4(a)/(b) qualifiers |
| Band-only | Never gates — always reported as the sensitivity-band caveat on whatever point is adopted | benefit-band coverage of the adopted T |

## 3. The failure ladder (terminal state)

```
supervised T* fit, accuracy-gated
   |
   +-- passes gate, not saturated -> ADOPT T* (primary), report band membership + pool
   |                                   concurrence as corroboration (non-gating)
   |
   +-- fails gate OR saturated
          |
          +-- pool arm A1 ∧ A2 ∧ A2b ∧ A3 all pass
          |     -> pool T-hat is the fallback estimate; still wrap in the band caveat;
          |        REPORT (not silently adopt) -- this is a user decision per the pool
          |        doc's own decision rule, unchanged here
          |
          +-- pool arm's null clause fires (A1/A2/A2b fail) AND supervised T* (even if
          |   gate-failed) is itself within the band [1.60, 8.84]
          |     -> FIXED-3.761 ZERO-FIT FALLBACK activates, wrapped in the explicit
          |        caveat from S4(a): this fallback is validated only for targets that are
          |        not near-calibrated; if there is independent reason to believe the closed
          |        pair is close to calibrated (T* estimate, even a failed/noisy one, near or
          |        below ~1.6-1.8), DO NOT apply 3.761 -- fall through to band-only.
          |
          +-- everything above fails, OR the closed pair's own (even failing) T* estimate
                sits near/below the band's lower edge
                -> TERMINAL: band-only sensitivity reporting, no point correction adopted.
                   This is the paper's CURRENT position -- the protocol never produces a
                   worse outcome than not having run it.
```

## 4. Concurrence table (publish alongside the sweep)

Columns: instrument, T, gap-closure vs whatever T\* the sweep measures, pass/fail against its
own gate:

| instrument | T | gap-closure vs on-pair T\* | gate status |
|---|---|---|---|
| Supervised T\* | (measured) | 1.00 by definition | accuracy-gated pass/fail |
| Pool arm (frozen, A1–A4) | (measured) | (measured) | A1/A2/A2b/A3 pass/fail, A4 reported |
| Fixed-3.761 | 3.761 | (measured, 2(d) doc's method) | direction-safety qualifier from §4(a) |
| 2(c) entropy-accounting | — | — | considered and rejected (fails LOFO, dominated by T̂₀.₃; `spec/analysis_2026_07_19_entropy_accounting.md`) |

## 5. Explicit non-interference

- The pool arm's A1–A4 pre-registration (`dispersion_pool_promotion.md` §6) is FROZEN: λ=0.30,
  anchor 3.761, pool membership, construction, and acceptance thresholds are not touched or
  retuned by this protocol.
- The E6 band framing (`q4_range_robustness.md`) is FROZEN: the band [1.60, 8.84] and its
  tolerant/strict criteria are not touched or retuned by this protocol.
- This protocol only sequences these frozen instruments and adds one new fallback (fixed-3.761,
  itself frozen by construction — it is a constant, nothing to retune) plus the qualifier that the
  fallback must not fire when the closed pair looks near-calibrated.
- **Terminal-state guarantee:** in the worst case (every unsupervised instrument fails, or the
  closed pair turns out to be near-calibrated), the protocol degrades exactly to
  supervised-T\*-or-band-only — the paper's status quo. It never recommends an action worse than
  doing nothing.
