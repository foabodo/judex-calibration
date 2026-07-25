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
run-id identity; panel {qwen, gemma31, llama31} gate-passing, plus GLM-4.5 as a
user-directed extension (excluded at the B0 review, REVERSED later 2026-07-20 after the
llama31 T_c peg — both legs, self-hosted, labeled extension; see §7); `develop`
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
- **FULL contract (USER DECISION 2026-07-20, superseding the B0 reduced-contract draft):**
  the answer JSON is the complete six-field contract 0.2.0 shape in contract order —
  findings, compliance_level, compliance_distribution, compliance_justification,
  confidence_distribution, confidence_justification. Few-shot exemplars render all six
  fields from the store rows (findings restricted to the contract's
  requirement/status/evidence keys). B-Q1 is therefore the strict reading: "can bases emit
  the full production contract."
- **Two stages per cell** (Approach C, verbalized): (1) generate reasoning, stop before
  `JSON:`; (2) re-feed `prompt + reasoning + "\nJSON:"`, greedy-generate ≤1600 tokens.
- **Parse policy (pre-registered, two-tier):** balanced-brace extraction → JSON decode →
  `parse_ok` = both distributions valid (all keys, non-negative, positive sum; renormalized)
  — the minimum for τ_v scoring; `contract_complete` = additionally findings a non-empty
  array of {requirement, status∈{met, partially_met, unmet, indeterminate}, evidence}
  objects, a valid compliance_level, and non-empty justification strings — **the B-Q1 gated
  rate**. One pass, **no resampling** (temperature 0 is deterministic), no repair beyond
  extraction. Failures are recorded with error classes; per-field `contract_missing` counts
  are reported. Diagnostics recorded per cell, reported not gated: raw sums, 0.05-grid
  conformance, compliance_level↔argmax consistency, n_findings.
- **Prompt budget (re-measured under the FULL contract, Qwen tokenizer, real store):**
  worst-case stage-1 prompt **24,996 tokens** (Art 14 / Scenario B | Use 1); required
  context = prompt + 2048 reasoning + scaffold + 1600 JSON generation ≈ **28,652** ⇒ the
  32768 pin holds with ~4k headroom. Re-measure per family tokenizer before provisioning
  (inherited rule).
- **Artifacts:** `runs/study_b_<family>/{pre,post}_verbalized.json` = {item_label: full cell
  record}; meta sidecar pins (model, reason, budget, fewshot_k, **channel=verbalized,
  epsilon**) — mismatched resume hard-errors. `study_b_report.json` from
  `scripts/run_study_b_leg.py --analyze` (free, idempotent).

## 4. Pre-registered gates and constants (frozen before any paid leg)

| Constant | Value | Rationale |
|---|---|---|
| Contract-compliance gate (B-Q1) | **contract_complete rate ≥ 0.90** per leg (FULL six-field tier; parse_rate reported alongside) | below this, the leg's emissions are not the production contract; τ_v still scores the parse_ok subset for data efficiency |
| Capability gate | resolution-primary (Murphy resolution > 0 with margin) on the parsed compliance view, both legs | inherited verbatim |
| τ_v stopping rule (B-Q3) | max/min ≤ 2 over gate-passing families | same form as Study A's adopted rule |
| ε-floor | **0.005** (~~half~~ **one tenth** of the 0.05 grid step — see correction note below), applied at ANALYSIS time; stored vectors keep exact zeros | §5 |
| τ_v fit | `fit_tau_oc` machinery: W1 alignment post→pre, 60-pt log grid, T_BOUNDS (0.25, 20.0), saturation flagged | inherited verbatim |
| Elicitation | temperature 0, budget 2048, k=5, workers 8, no resampling | inherited / §3 |
| B-Q4 link + objective | §6 | — |
| Scoring | RPS/W1/Murphy against `aireg.load_cells()` only | inherited |

> **Correction [2026-07-25]:** the ε-floor row's original parenthetical read "(half the 0.05
> grid step)". That gloss is arithmetically wrong — half of 0.05 is 0.025. **0.005 is one
> tenth of the grid step** (equivalently, half of 0.01). The registered **value 0.005 is
> unchanged** and is what every leg ran with (`elicit_verbalized.EPSILON`, pinned in each leg's
> meta sidecar); only the description was wrong, so no result is affected. Note that 0.025 —
> the true half-step — is one of the four ε-sensitivity points in §5, which is a separate
> check and was never the registered floor. The row above has been corrected in place; this
> note records what it originally said.

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

> **APPROVED 2026-07-20 (user):** design approved with the FULL-contract amendment (§3);
> B1 **and** B2 spend approved, self-hosted vast only, GLM excluded. Execution plan:
> Qwen and Gemma boxes provision in parallel (cheap, ~$3–4/hr each — parallelism is
> cost-neutral in GPU-hours and halves wall-clock); the 8×H200 Llama box launches only
> after the FIRST base leg clears the B-Q1 contract-complete gate (the load-bearing
> feasibility unknown — gating the expensive box on it is the cost-effective ordering).
> A B-Q1 base failure still means STOP for that family and report, never a logit fallback.

| Phase | What | Cost | Gate to proceed |
|---|---|---|---|
| **B0** (this) | inventory, design doc, machinery + tests, Mac smoke | $0 | user approves design |
| **B1** | Qwen both legs, verbalized, 1×H200 (~$2.5–4/hr, proven box class) | **~$10–20** | B-Q1 gate on the base leg; STOP + report if failed |
| **B2** | Gemma-31B both legs (~$10–15); Llama-3.1-405B both legs on 8×H200 (~$70–100) — all self-hosted vast vLLM (user decision 2026-07-20) | **~$80–115** | user approval; B-Q3 verdict + B-Q4 fits after |
| **B3** | synthesis: τ_v vs τ_oc channel comparison, closed-pair implications, protocol-instrument recommendation (report only), paper write-up | $0 | — |

Note: the handoff estimated B2 at $30–90; the dominant term is Llama-3.1-405B (8×H200,
$33/hr, ~810 GB download + two legs). A descoped B2 without llama31 is ~$15–25 but drops the
gate-passing set to 2 families, weakening the B-Q3 verdict. User's call at the B1→B2 gate.
**Llama-3.1-405B stays in the plan** — it is not the family Study A ruled out (see §7b).

**GLM: excluded at the B0 review, REVERSED later 2026-07-20 (user decision) as a labeled
extension.** The original exclusion rationale stands on record: its base fails Study A's
token-slice resolution-primary gate, so it was excluded from the pre-registered campaign.
After the campaign delivered the llama31 T_c peg (B-Q4), the user directed GLM's addition to
see how it behaves in the verbalized setting. Terms of the extension: BOTH legs, self-hosted
vast 8×H200 TP8+EP (no API orphan), same instrument and gates; its verbalized-channel gate
status is measured fresh (the Study A gate failure was a different channel); results are
reported as an extension — the pre-registered B-Q3 ratio over the original gate-passing set
is not retroactively redefined, but the extended set's ratio is reported alongside.

### 7a. API-based cost alternative for POST legs — CONSIDERED AND REJECTED

> **RESOLVED (user decision 2026-07-20): every Study B leg runs self-hosted on vast vLLM.**
> The analysis below is retained as the record of what was considered and why it was not
> adopted (savings ~$4–10 total, contingent on unverified API token-billing behavior, at the
> price of reintroducing the serving-mode confound the isolation principle exists to prevent).
> GLM rows below are additionally moot — GLM is excluded from Study B entirely (§7).

Unlike Study A's token-slice channel — which needs `logprobs: true` and so is blocked on any
provider that doesn't expose them (GLM's `post_api` entry in `models.yaml` is `logprobs:
false`, ruling it out there) — Study B's verbalized channel needs only text generation. Any
OpenAI-compatible chat completions endpoint (OpenRouter, or a family's native API) can serve
the **POST (instruct)** leg. This does **not** extend to the **PRE (base)** leg: commercial
inference APIs serve chat/instruct-tuned checkpoints, not raw pretrained bases, so PRE always
needs self-hosted vLLM on vast regardless of this option — this section prices the POST leg
only, as a substitute for that leg's SHARE of the self-hosted box time (chiefly the ~810 GB
POST-weights download, which the PRE leg's box time does not amortize).

**Pricing (OpenRouter list rates, checked 2026-07-20 against OpenRouter's own
`/api/v1/models/<id>/endpoints` JSON, not just the rendered page — verify again before use,
list prices drift):**

| Family (post_repo) | Input $/1M | Output $/1M | Source |
|---|---|---|---|
| Qwen3.5-35B-A3B | $0.14 | $1.00 | openrouter.ai/qwen/qwen3.5-35b-a3b |
| Gemma-4-31B-it | $0.10 | $0.35 | openrouter.ai/google/gemma-4-31b-it |
| GLM-4.5 | $0.60 | $2.20 | openrouter.ai/z-ai/glm-4.5 |
| Llama-3.1-405B-Instruct | **NO VIABLE API ROUTE** | — | see below |

**Llama-3.1-405B-Instruct has no verified API route — checked directly, not estimated.**
OpenRouter's `/api/v1/models/meta-llama/llama-3.1-405b-instruct/endpoints` (and the `:free`
variant) both return an **empty `endpoints` array**: no provider is currently routed through
OpenRouter for this model at all, so there is no OpenRouter rate to quote — the previous
version of this table's "~$0.80–$3.50" range was built from third-party pages found via search
and should be disregarded. Checking one of those third parties directly compounds the
problem: DeepInfra's own listing for this exact checkpoint is reported (2026-07 search
results) to redirect low-demand requests to `NousResearch/Hermes-3-Llama-3.1-405B` — a
different fine-tune, not the weights this study self-hosts, so even that path would break the
same-weights isolation principle (§3) if used. **Conclusion: for Llama, self-hosting on vast
is not merely the preferred option — it is the only verified-available path for the POST leg**
(`models.yaml`'s own `post_api` fallback entry for this family already carried an
"availability NOT re-verified for 3.1" caveat; this checks that caveat and it does not clear).
Re-verify directly (`curl openrouter.ai/api/v1/models/<id>/endpoints`, or query the specific
third-party provider's own API) if this matters again before B2, rather than trusting a
search-snippet price.

**Two very different cost regimes, both real:**

- **Reasoning DISABLED, single call** (`elicit_post.elicit_verbalized`'s existing pattern —
  extend it with the `confidence_distribution` key): ≈21k input tokens/cell (the measured
  k=5 few-shot+evidence+criterion size) + ≈400 output tokens/cell (JSON only), × 120 cells:

  | Family | Est. cost, POST leg only |
  |---|---|
  | Qwen | ≈ $0.40 |
  | Gemma-31B-it | ≈ $0.27 |
  | GLM-4.5 | ≈ $1.62 |
  | Llama-3.1-405B-Instruct | **N/A — no viable route, see above** |

  Cheap, but **not what Study A/B's design otherwise specifies**: the live protocol is
  reasoning-ON (`elicit_post.py`'s own docstring flags disabling reasoning as a *different*,
  ~100×-cheaper, non-reasoning judgement — a legitimate but DIFFERENT measurement, not a
  cost-saving substitute for the live leg without saying so).

- **Reasoning ENABLED, two-stage** (matching this repo's live design — generate reasoning,
  then generate JSON, mirroring `elicit_verbalized.py`'s base-leg approach): input roughly
  doubles (the JSON-generation call re-sends the prompt + reasoning span) and output gains a
  reasoning span up to the 2048-token budget **per cell**. For a thinking-capable post model
  (Qwen3.5-35B-A3B, GLM-4.5), a provider that bills hidden reasoning tokens as output can push
  this materially higher and less predictable than the disabled-reasoning estimate above —
  this is the exact trap `elicit_post.py` was written to dodge for Study A's post leg. No
  clean estimate is offered here for this regime; if pursued, cap `max_tokens` explicitly and
  monitor live spend rather than trusting a pre-computed number.

**Verdict for B2 (now decided):** self-hosting all legs on vast is the plan — it is the only
path that measures reasoning-ON pre AND post in the identical channel/scaffold (the isolation
principle in §3), and the PRE leg needs the box regardless. The API option's realistic net
savings (~$2–5/family on Qwen/Gemma, given the download-dominated cost structure in
`docs/vast_quickstart.md`) did not justify the confound; for Llama no API route exists at all
(no OpenRouter endpoints; DeepInfra reportedly redirects to a Hermes fine-tune), so its full
cost (~$70–100, both legs, 8×H200) was always unconditional. **User adopted the self-hosted
plan 2026-07-20.**

### 7b. Why "Llama" was ruled out — and why that Llama is not this Llama

Nothing in Study B ruled out Llama. The rule-out is **inherited from Study A** and concerns a
**different model**: the *original* panel entry was **Llama-4-Maverick** (MoE), which failed
Study A's cheap-trial resolution-primary gate on 2026-07-19 — argmax accuracy 0.250 (≈ chance),
Murphy resolution ≈ 0.0197 (noise-level), both T* pegged at the 20.0 bound, and its τ_oc was
flagged reference-degenerate. The user then swapped it for **Llama-3.1-405B** (dense) as a
within-lineage rescue — the same swap pattern used for Gemma (26B-A4B → 31B dense). That dense
405B model subsequently **passed** the gate (argmax 0.392, resolution 0.0393, finite T*) and
is the family in every Study A/B cost table since (`configs/models.yaml`, memory
`study-a-cheap-trial-executed.md`). So "Llama" was never absent from Study B's plan — §7's B2
row has carried Llama-3.1-405B since B0 landed; the model that was actually excluded is one
Study B never touches.

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
