# Study B results — verbalized-channel pre/post calibration campaign (2026-07-20)

Executed under the approved design (`docs/study_b_design.md`, full-contract amendment).
All legs self-hosted vast vLLM bf16, k=5 full-contract few-shot, workers 8, temperature 0,
120 AIReg cells each; GLM excluded (user decision). Per-cell artifacts in gitignored
`runs/study_b_{qwen,gemma31,llama31}/`; reports of record are each dir's
`study_b_report.json` + `runs/study_b_cross/study_b_cross_family.json`. All six boxes
destroyed after Mac-side verification. Estimated spend ≈ $110–150 (llama's two 8×H200
boxes dominate; exact figure on the vast dashboard).

## B-Q1 — can the models emit the full production contract? LARGELY YES, with one sharp exception

Contract-complete rate (gate ≥ 0.90), full six-field 0.2.0 tier:

| Family | PRE (base) | POST (instruct) |
|---|---|---|
| Qwen3.5-35B-A3B | **91.7% PASS** | **95.8% PASS** |
| Gemma-4-31B | **100% PASS** | **30.8% FAIL** |
| Llama-3.1-405B | **95.0% PASS** | **98.3% PASS** |

The load-bearing feasibility unknown resolves POSITIVE: pretrained bases emit the full
contract via few-shot continuation, at 92–100%, with every parsed emission on the 0.05 grid
and (for parsed cells) essentially always complete — failures are all-or-nothing (no JSON
at all), never partial contracts.

**The Gemma-4-31B-it post failure is a decoding pathology, not a comprehension failure:**
the captured diagnostic (`runs/study_b_gemma31/diagnostics_post_failures.json`) shows it
open the JSON correctly (`{"findings": [{"`) and then collapse into a greedy-decoding
repetition loop ("la la la…") that never closes a brace. Its parsed-subset capability is
intact (argmax 0.439, resolution 0.0572). This is a temperature-0 long-context continuation
degeneration specific to this instruct model — a boundary result about the measurement
channel, recorded per the no-resampling rule.

## B-Q2 — τ_v (verbalized pre→post overconfidence temperature)

| Family | τ_v | status |
|---|---|---|
| Qwen | 1.2811 | clean fit |
| Llama-405B | 1.0252 | clean fit |
| Gemma-31B | 1.8571 | EXCLUDED (post gate fail; 57-cell selected sample) |

ε-sensitivity (pre-registered, once, on B1): τ_v(qwen) = 1.281 at ε ∈ {0.001, 0.005},
1.189 at 0.0125, 1.104 at 0.025 — stable at and below the pre-registered 0.005 floor.

## B-Q3 — does τ_v cluster? YES over the gate-passing set (with a survivorship caveat)

Gate-passers {qwen, llama31}: max/min = **1.25 ≤ 2 ⇒ the pre-registered rule PASSES** —
the sharpest contrast with Study A, whose token-slice τ_oc ratio was 3.05 (Q3-NEGATIVE).
Caveats stated up front: n = 2, and the family that stretched Study A's ratio (gemma31,
τ_oc 4.88) is excluded here by the post-leg contract gate — the verbalized clustering may
partly reflect survivorship of well-behaved emitters, not just channel narrowing. No
constant is adopted into any config from this result (report-only, per the design).

**Channel comparison (the B3 free science):** τ_v < τ_oc for every family — qwen 1.28 vs
1.60, llama31 1.03 vs 1.86, gemma31 1.86 (selected) vs 4.88. The verbalized channel
carries systematically LESS pre→post overconfidence than the token-slice channel,
consistent with the 2(c) audit (grid quantization ⇒ flatter, higher-entropy emissions).
For llama31 the verbalized channel shows essentially NO overconfidence (τ_v ≈ 1.03)
despite τ_oc 1.86 in logits. The channels remain incommensurable; these are parallel
measurements, never conversions.

## B-Q4 — is verbalized confidence calibratable? LEVEL yes, SIGNAL only for Qwen

Stated confidence is **universally overconfident in level**: mean p̂(C) vs realized
argmax-accuracy = 0.833 vs 0.602 (qwen), 0.715 vs 0.483 (llama31), 0.846 vs 0.439
(gemma31, selected subset).

Pre-registered verdict (doc-clustered LODO, ΔBrier CI excluding 0):

| Post leg | T_c (full) | saturated? | LODO ΔBrier [CI] | verdict | Kendall τ_b(s, W1) |
|---|---|---|---|---|---|
| Qwen | 4.877 | no (folds 3.9–6.1) | −0.052 [−0.090, −0.013] | **calibratable** | −0.174 (right sign) |
| Llama-405B | 20.0 | **PEGGED** (folds hit bound) | −0.071 [−0.108, −0.034] | fires, but degenerate | +0.069 (wrong sign) |
| Gemma-31B | 13.8 | near-peg (a fold pegs) | −0.169 [−0.277, −0.059] | fires; selected sample | −0.133 |

Honest reading under the inherited peg-is-not-a-fit discipline: only **Qwen** has a
meaningful confidence-calibration story — an interior temperature, stable across folds,
with a right-sign error association. **Llama's** Brier gain comes from flattening
essentially to the base rate (T_c on the bound; association ≈ 0): its stated confidence is
uninformative at the event level, and a pegged T_c must never be adopted. Gemma's fit
rides on the gate-failing selected subset. The T_c values do not cluster (4.9 / 20-peg /
13.8) — no shared confidence constant exists, mirroring Study A's Q3 lesson in the new
object. AURC is unchanged by T_c everywhere (monotone-ish map), so the router value of
confidence (E2's S09 finding) rests on the raw ranking signal: modest for qwen/gemma,
absent for llama31.

## Implications for the closed pair (report-only)

The production channel's own overconfidence (τ_v) is small for the two clean families —
smaller than the token-slice τ_oc that motivated the [1.60, 4.88] band. This does NOT
amend the band (that is closed-pair supervised territory under the adopted composite
protocol); it is off-pair evidence that verbalized emissions are flatter than logits.
The verbalized-confidence instrument (B-Q4) shows a real level-bias correction is
learnable per-family but not transferable — any closed-pair application requires on-pair
contract outputs (the E6 sweep) and its own fit. Protocol changes remain user decisions.

## Addendum (2026-07-21): Gemma post re-collected via 16-bit API — the panel completes

USER DECISION: rather than accept a 2-family panel, the Gemma-4-31B-it post leg was
re-collected through OpenRouter's chat API pinned to 16-bit providers (`quantizations
[bf16, fp16]`, `allow_fallbacks false`; every one of the 120 cells was served by
**OpenInference @ bf16** — the study's own vast precision). Serving-mode confound accepted
and labeled: channel `verbalized_api_chat` (chat template; no raw continuation), artifacts
in `runs/study_b_gemma31_api/` (the vast failure leg stands untouched). Cost ≈ $0.60.

- **The collapse is channel-bound, not weights-bound:** the same weights that degenerated
  under greedy raw-completions continuation emit **93.3% contract-complete** (112/120,
  gate PASS) under the chat template — and post argmax 0.678 is the best of the panel
  (resolution 0.056, T_rps 2.24 finite).
- **Cross-mode τ_v(gemma) = 1.0252** (vast pre × API post; confound-labeled). Extended
  gate-passing set {qwen 1.281, llama31 1.025, gemma31_api 1.025}: **ratio 1.25 ≤ 2 —
  the B-Q3 clustering rule still PASSES at n = 3**, and the survivorship caveat weakens
  materially: the family that broke Study A's token-slice clustering (τ_oc 4.88) clusters
  tightly in the verbalized channel once its emissions parse. (The pre-registered original
  2-family verdict stands; this is the extension read, per the amended §7 discipline.)
- **B-Q4 gains a second clean fit — and the two agree:** Gemma-API T_c = **4.528**
  (interior; folds 4.2–5.25, none saturated), LODO ΔBrier −0.069, CI [−0.111, −0.028]
  excluding 0 ⇒ calibratable; Kendall −0.169 (right sign). Against Qwen's T_c 4.877
  (folds 3.9–6.1): **ratio 1.08** — the two informative confidence temperatures sit
  practically on top of each other near ≈ 4.5–4.9, with llama31's peg the lone outlier.
  The B-Q4 story upgrades from "only Qwen" to "two clean, mutually consistent fits;
  llama31's stated confidence uninformative". Level overconfidence is again extreme
  (mean p̂(C) 0.95 vs realized 0.678). A shared confidence constant remains unadopted —
  n = 2 clean fits, one cross-channel — but the agreement is now a reportable regularity
  rather than a single point.

## Deviations from plan

None affecting the pre-registered quantities. One diagnostic addition: a labeled 2-cell
re-elicitation of Gemma post failures to capture raw text (stored separately; the recorded
failures stand). Llama's boxes provisioned faster than budgeted; spend landed at or below
the ~$80–115 B2 estimate plus ~$15–25 for B1.
