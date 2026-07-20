# Study B design — pre/post overconfidence and confidence calibration on the VERBALIZED channel

Status: **Phase B0 deliverable (2026-07-20)** — design + free feasibility work. No paid leg
runs until the user approves B1. Author: Claude (handoff
`spec/handoff_2026_07_20_study_b_verbalized_channel.md`).

## 1. Scope and inheritance

Study A (CLOSED — conclusions frozen) measured pre→post overconfidence on the **token-slice
logit channel** and closed every GT-free calibration path it tested (Q3-NEGATIVE, τ_DACA dead,
entropy accounting dead; survivors: the frozen dispersion-pool E6 arm and the supervised
closed-pair T* wrapped in the **amended** sensitivity band [1.60, 4.88] under the adopted
composite protocol, calibration `17ffd41`). But the production evaluators (Sonnet 4.6 medium +
GPT 5.4 medium) do not emit logits: they emit the contract-0.2.0 verbalized output — a 5-level
`compliance_distribution` **and** a 3-level `confidence_distribution` on the 0.05 grid. Study A's
temperatures do not transport to that channel by construction (grid quantization ⇒ the
logit↔verbalized bridge is impossible in the strong sense, 2026-07-12 analysis; ~2× entropy,
2(c) audit §4). Study B measures the phenomenon **in the production channel** and adds the
object Study A never touched: **verbalized confidence**.

Correction to the handoff: contract 0.2.0 has **six** top-level fields (findings,
compliance_level, compliance_distribution, compliance_justification, confidence_distribution,
confidence_justification — verified against `judex-evaluator/configs/output_contract.yaml`);
"8-field" is a stale label from the pre-0.2.0 corpus vintage.

Unchanged inheritances: the 120 AIReg cells via `aireg.load_cells()` only; corpus-v2 dimension
store few-shot, k=5, firewall-disjoint; resolution-primary capability gate; `T_BOUNDS =
(0.25, 20.0)` passed explicitly, boundary = peg never a fit; single-measurement noise band
±10–15% per-leg T (τ ratios ≈ ±20%); doc-clustered bootstrap (24 docs, seeded); vast vLLM bf16
serving playbook (`--gpu-memory-utilization 0.85`, `--max-model-len 32768`, `--workers 8`);
run-id identity; panel {qwen, gemma31, llama31} gate-passing + GLM post-only; `develop`
integration branch; nothing from `runs/` committed.

## 2. Questions

- **B-Q1 (feasibility, load-bearing):** can pretrained BASES emit contract-shaped verbalized
  distributions at all, via k=5 few-shot JSON continuation on `/v1/completions`? Gate: §4.
  If a base fails, we STOP and report — a publishable boundary result. **Never** silently fall
  back to the logit channel.
- **B-Q2 (the τ_oc twin):** pre→post change in verbalized compliance distributions — **τ_v**,
  fit with the SAME machinery (`study_a.fit_tau_oc`: W1-aligned post→pre on the 60-point log
  grid over T_BOUNDS), on ε-floored vectors (§5). τ_v is measured fresh; it is never converted
  to or from τ_oc.
- **B-Q3 (clustering):** does τ_v cluster where τ_oc did not? Pre-registered stopping rule,
  same form as Study A's: adoptable only if **max/min ≤ 2 over the gate-passing τ_v set**
  (gate = §4 contract-compliance AND resolution-primary, both legs). Channel comparison
  (INV_SOFTMAX(verbalized) vs the already-paid token-slice legs, per family per leg) is free
  science once B1/B2 data exist.
- **B-Q4 (new object):** is verbalized **confidence** calibratable — does stated confidence
  track realized per-cell error, and does a learned one-parameter recalibration improve that
  tracking out of sample? Objective choice: §6.

## 3. Instrument (implemented in B0: `src/judex_calibration/elicit_verbalized.py`)

Both legs of a family run on the **same** vast vLLM `/v1/completions` endpoint class with the
**same** prompt scaffold — only the weights differ (Study A's isolation principle, ported).

- **Few-shot:** k=5 per Article from the corpus-v2 dimension store — the SAME rows Study A's
  scaffold selects (`fewshot.select_rows`, stratified fixed_set), re-rendered as
  (Evidence, Criterion, Reasoning, **reduced-contract JSON**). The v2 store rows carry the full
  contract shape (verified in B0), so the rendering is store-native, no synthesis.
- **Reduced contract (design decision):** the answer JSON is
  `{compliance_level, compliance_distribution, confidence_distribution}` — the two measured
  objects plus the argmax name. The prose fields (findings, both justifications) are dropped
  from the ANSWER: the reasoning span plays their role, and full-contract answers would add
  ~1–2k generated tokens/cell of parse-fragile prose. Production-fidelity trade-off is
  acknowledged: B-Q1 as gated here = "can bases emit the contract's *distributional core*".
  An optional full-contract probe on a 15-cell subset can be added in B1 as a diagnostic if
  the user wants the stricter reading (cheap, ~+10% of B1 cost).
- **Two stages per cell** (Approach C, verbalized): (1) generate reasoning, stop before
  `JSON:`; (2) re-feed `prompt + reasoning + "\nJSON:"`, greedy-generate ≤400 tokens.
- **Parse policy (pre-registered):** balanced-brace extraction → JSON decode → both
  distributions validated (all keys, non-negative, positive sum) and renormalized. One pass,
  **no resampling** (temperature 0 is deterministic), no repair beyond extraction. A failed
  cell is recorded (`parse_ok: false` + error class) and counts against the B-Q1 gate.
  Diagnostics recorded per cell, reported not gated: raw sums, 0.05-grid conformance,
  compliance_level↔argmax consistency.
- **Prompt budget (measured in B0):** contract rendering adds 1,052 chars (~250 tokens) per
  Article block over Study A's letter scaffold ⇒ worst case ≈ 23.6k required; the 32768 pin
  holds. Re-measure per family tokenizer before provisioning (inherited rule).
- **Artifacts:** `runs/study_b_<family>/{pre,post}_verbalized.json` = {item_label: full cell
  record}; meta sidecar pins (model, reason, budget, fewshot_k, **channel=verbalized,
  epsilon**) — mismatched resume hard-errors. `study_b_report.json` from
  `scripts/run_study_b_leg.py --analyze` (free, idempotent).

## 4. Pre-registered gates and constants (frozen before any paid leg)

| Constant | Value | Rationale |
|---|---|---|
| Contract-compliance gate (B-Q1) | parse rate ≥ **0.90** per leg | below this, the leg's parsed subset is a selected sample; τ_v on it is not the panel measurement |
| Capability gate | resolution-primary (Murphy resolution > 0 with margin) on the parsed compliance view, both legs | inherited verbatim |
| τ_v stopping rule (B-Q3) | max/min ≤ 2 over gate-passing families | same form as Study A's adopted rule |
| ε-floor | **0.005** (half the 0.05 grid step), applied at ANALYSIS time; stored vectors keep exact zeros | §5 |
| τ_v fit | `fit_tau_oc` machinery: W1 alignment post→pre, 60-pt log grid, T_BOUNDS (0.25, 20.0), saturation flagged | inherited verbatim |
| Elicitation | temperature 0, budget 2048, k=5, workers 8, no resampling | inherited / §3 |
| B-Q4 link + objective | §6 | — |
| Scoring | RPS/W1/Murphy against `aireg.load_cells()` only | inherited |

Smoke discipline inherited: `--limit`/`--no-reason` or <120 elicited cells auto-marks the
report `smoke` — numbers discarded, never integrated.

## 5. ε-floor (the zeros trap)

The contract REQUIRES zero-mass levels written as `0.0`; `ln 0 = −∞`. Every inverse-softmax
op (τ_v fit, `apply_temperature`, channel comparison) therefore needs a floor, and the floor is
results-affecting. Pre-registered: **ε = 0.005**, `floor_and_renormalize` applied to every
stored vector before any temperature machinery; the production op `judex.calibration
.apply_temperature` is then used **unchanged** (its internal 1e-12 clip is inert post-floor —
without the floor it would manufacture ≈ −27.6 logits out of grid zeros; verified against
`calibration.py::invert_softmax` in B0). ε is pinned in every leg's meta sidecar.
**ε-sensitivity check, once, on B1 data:** refit τ_v and the B-Q4 temperature at
ε ∈ {0.001, 0.005, 0.0125, 0.025}; report the spread in the B1 report; never revisited
per-analysis.

## 6. B-Q4: the confidence-calibration objective (choice + justification)

Confidence has no direct GT. Of the three candidate framings, the **primary pre-registered
objective is (i) confidence-conditional error calibration**, because it is the only one that
directly answers "does stated confidence track realized error, and does a learned map improve
the tracking":

- **Realized outcome per cell:** primary binary event **C = argmax agreement** of the parsed
  compliance distribution with GT (GT-anchored, no data-dependent thresholds); continuous
  companion W1 for the association diagnostic.
- **Fixed link (pre-registered):** p̂(C) = 0·p(low) + 0.5·p(medium) + 1·p(high) over the
  ε-floored confidence 3-vector. The weights are the natural ordinal midpoints; fitting them
  would spend the tiny 3-level support on link estimation instead of calibration.
- **Learned map:** one temperature T_c on the confidence 3-vector (inverse-softmax, ε-floored,
  T_BOUNDS, saturation flagged), fit by minimizing the **Brier score of C** under the link.
- **Out-of-sample (pre-registered):** doc-clustered **LODO** over the 24 documents; verdict
  "calibratable" iff mean OOS ΔBrier < 0 with a 95% doc-clustered bootstrap CI excluding 0.
- **Association diagnostic (reported):** Kendall τ_b between s = p̂(C) and realized W1
  (expected negative), doc-clustered bootstrap CI.
- **Secondary (router framing, per the E2 S09 finding):** risk–coverage curve ranking cells by
  s; area under the risk–coverage curve before/after the T_c map. Reported, not gated.
- **Descriptive only:** sharpness-consistency — corr(normalized entropy of the compliance
  distribution, s) — the contract instructs confidence ≠ compliance-sharpness; we measure
  whether bases/posts honor that.

A single temperature cannot move the confidence *location*, only its dispersion — if the
reliability diagram shows a location bias (e.g. uniform over-statement of `high`), that is a
FINDING about the channel, reported as such; we do not escalate to multi-parameter maps
without a user decision (panel-robustness principle).

Application boundary: Study B fits and validates confidence calibration on the **open panel
legs**. Applying it to the closed pair requires on-pair contract outputs — that is the E6
on-pair sweep's data (~$290–330, separately gated); the only existing on-pair contract run is
`stage9-claude-gpt-medium` (15 items), used in B0/B1 as free machinery-prototyping data only.

## 7. Phases and cost

| Phase | What | Cost | Gate to proceed |
|---|---|---|---|
| **B0** (this) | inventory, design doc, machinery + tests, Mac smoke | $0 | user approves design |
| **B1** | Qwen both legs, verbalized, 1×H200 (~$2.5–4/hr, proven box class) | **~$10–20** | B-Q1 gate on the base leg; STOP + report if failed |
| **B2** | Gemma-31B both legs (~$10–15); Llama-3.1-405B both legs on 8×H200 (~$70–100); GLM **post-only** via OpenRouter verbalized (~$3–8, no logprobs needed — channel-caveat labeled, contributes no τ_v) | **~$85–125** | user approval; B-Q3 verdict + B-Q4 fits after |
| **B3** | synthesis: τ_v vs τ_oc channel comparison, closed-pair implications, protocol-instrument recommendation (report only), paper write-up | $0 | — |

Note: the handoff estimated B2 at $30–90; the dominant term is Llama-3.1-405B (8×H200,
$33/hr, ~810 GB download + two legs). A descoped B2 without llama31 is ~$15–25 but drops the
gate-passing set to 2 families, weakening the B-Q3 verdict. User's call at the B1→B2 gate.

GLM base is NOT run (gate-marginal in Study A; its τ-type values are excluded — the 2026-07-20
band amendment exists because this rule was once violated).

## 8. What is free vs. not (verified in B0)

- **The $0 panel-wide channel comparison on existing post legs is NOT possible:** every Study A
  leg on disk (`runs/*/{pre,post}.json`) stores only token-slice 5-vectors; no verbalized JSON
  was ever collected (verified against the artifacts and `elicit_base.py` — the guide §4.4
  validity check needs data that does not exist yet). The verbalized side must be collected
  (B1/B2); the token-slice side of every comparison is already paid for.
- **Free now:** machinery + tests (landed, 65 passing); contract few-shot round-trip against
  the real v2 store (25/25 exemplar JSONs parse); prompt-budget delta (+250 tokens, pin holds);
  B-Q4 machinery prototyping against `stage9-claude-gpt-medium`'s 15 on-pair contract outputs
  (numbers labeled prototype, never results).
- **Mac smoke: EXECUTED (2026-07-20, numbers discarded).** The brew `llama-server` binary is
  gone from this Mac, but the judex-arm conda env carries `llama-cpp-python` 0.3.32 with its
  server module — `python -m llama_cpp.server --model
  ~/models/qwen3-4b-base-gguf-v2/Qwen3-4B-Base.Q4_K_M.gguf --n_gpu_layers -1 --n_ctx 32768`
  serves the same OpenAI-compatible `/v1/completions` the transport targets. Smoke results
  (`runs/smoke_b_qwen_mac{,_reason}`, auto-marked smoke): 3/3 no-reason cells + 1 full
  two-stage cell (reasoning span 1,001 chars) all parse; parse_rate 1.0, on-grid 1.0,
  level↔argmax 1.0, under the real k=5 contract few-shot. Machinery validated end-to-end;
  int4 4B numbers are meaningless and discarded per the smoke discipline.

## 9. Non-goals / invariants

No logit-channel fallback for a failed base (B-Q1 failure is the result). No deterministic
τ_oc↔τ_v conversion, ever. No `pipeline.yaml` calibration block from any Study B number
without an explicit user decision. No change to the E6 composite decision protocol (B3 may
*recommend* a verbalized-confidence instrument; adoption is the user's). AIReg never enters
few-shot. Corrections must be panel-robust — no per-family retuning of the contract scaffold.
