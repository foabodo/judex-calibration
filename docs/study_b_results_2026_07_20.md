# Study B results — verbalized-channel pre/post calibration campaign (2026-07-20/21, FINAL)

Executed under the approved design (`docs/study_b_design.md`, full-contract amendment) plus
three user-directed extensions: the Gemma-31B post re-collection via 16-bit API, the GLM-4.5
extension (§7 amendment; parallel session, `1cb98cd`), and the Study-A-failures retest
(Gemma-4-26B-A4B + Llama-4-Maverick, incl. a 26B API re-collection after its own raw-channel
collapse). All vast legs self-hosted vLLM bf16, k=5 full-contract few-shot, workers 8,
temperature 0, 120 AIReg cells each; API legs are OpenRouter pinned to 16-bit providers with
fallbacks disabled (gemma31: 120/120 OpenInference @ bf16; gemma26: 120/120 DekaLLM @ bf16).
Per-cell artifacts in gitignored `runs/study_b_*/`; the report of record is
`runs/study_b_cross/study_b_cross_family.json` (eight-way merge). All boxes destroyed after
Mac-side verification. Total campaign spend ≈ **$280–370** (four 8×H200 model-pairs dominate:
llama31, glm, maverick ×2 boxes each; API legs ≈ $1.20 combined).

## B-Q1 — full-contract emission: YES across the board, and gate outcomes are CHANNEL-BOUND

Contract-complete rate (gate ≥ 0.90), full six-field 0.2.0 tier:

| Family | PRE (base) | POST (instruct) | Channel note |
|---|---|---|---|
| Qwen3.5-35B-A3B | **91.7% PASS** | **95.8% PASS** | vast raw-completions |
| Gemma-4-31B | **100% PASS** | raw **30.8% FAIL** → chat-API bf16 **93.3% PASS** | reversal ① |
| Gemma-4-26B-A4B | **95.0% PASS** (capability-weak, see below) | raw **0.83% FAIL** → chat-API bf16 **100% PASS** | reversal ①′ |
| Llama-3.1-405B | **95.0% PASS** | **98.3% PASS** | vast raw-completions |
| Llama-4-Maverick | **98.3% PASS** | **98.3% PASS** | reversal ③ — clean BOTH legs |
| GLM-4.5 | **96.7% PASS** | **92.5% PASS** | reversal ② |

Structural findings:

- **Raw-completions post-leg collapse is a Gemma-4 family trait, not a size accident:**
  31B-it degenerates at 30.8% and 26B-A4B-it near-totally at 0.83% (same signature —
  `{"findings": [{"` then repetition; `no_json_object` dominant), and BOTH reverse
  completely under the chat template (93.3%, 100%). No other family's instruct twin shows
  it (Qwen 95.8, Llama-3.1 98.3, Maverick 98.3, GLM 92.5 — all on raw completions).
- **Study A's token-slice gate failures do not transfer to the verbalized channel — with one
  partial exception.** All three retested Study A failures pass the verbalized CONTRACT
  gate: Maverick cleanly on both legs (base argmax 0.325, resolution 0.038, finite T);
  GLM's base is the best base of the panel (0.543 / 0.047); but **Gemma-26B-A4B's base,
  while 95% contract-compliant, stays capability-weak** — resolution 0.0134 (below the
  0.0197 that failed Maverick in Study A) and argmax 0.298 (< the 0.333 majority floor).
  Under the inherited resolution-primary gate its base FAILS on capability, the one case
  where Study A's verdict partially survives the channel switch: the model can emit the
  form but not the discrimination. Its τ_v is therefore excluded from the gate-passing set.

## B-Q2 — τ_v (verbalized pre→post overconfidence temperature)

| Family | τ_v | status |
|---|---|---|
| Qwen | 1.2811 | clean, in-mode |
| Llama-3.1-405B | 1.0252 | clean, in-mode |
| GLM-4.5 | 1.0252 | clean, in-mode (extension) |
| Llama-4-Maverick | 1.3798 | clean, in-mode (extension) |
| Gemma-31B | 1.0252 | cross-mode (vast pre × API post), confound-labeled |
| Gemma-26B | 1.1894 | EXCLUDED — base fails resolution-primary (weak reference) |

ε-sensitivity (pre-registered, once, on B1): τ_v(qwen) stable at 1.281 for ε ∈ {0.001, 0.005}.

## B-Q3 — τ_v clusters at every panel width

- **Pre-registered verdict (original set, n=2):** ratio 1.25 ≤ 2 — **PASSES**.
- **Extension reads:** in-mode n=4 {qwen, llama31, glm, maverick} → ratio **1.346**;
  with cross-mode gemma31, n=5 → **1.346**. (Including even the excluded gemma26 value
  would not change the ratio — 1.189 is interior.) Every family measurable in this channel
  lands in [1.03, 1.38]: a genuinely tight verbalized-overconfidence band, against Study A's
  token-slice ratio of 3.05 on the same panel lineages.
- **Channel comparison:** τ_v < τ_oc for every family with both measurements (1.28 vs 1.60;
  1.03 vs 1.86; 1.03 vs 4.88-excised; extensions never had valid τ_oc). The verbalized
  channel is systematically flatter (consistent with the 2(c) grid-quantization audit).

No constant is adopted into any config (report-only).

## B-Q4 — verbalized confidence: universal LEVEL overconfidence; a signal-quality spectrum

Stated confidence overstates correctness everywhere — mean p̂(C) vs realized accuracy:
0.833/0.602 (qwen), 0.950/0.678 (gemma31-API), 0.945/0.575 (gemma26-API), 0.715/0.483
(llama31), 0.779/0.595 (glm), 0.755/0.500 (maverick).

| Post leg | T_c (full) | folds | LODO ΔBrier [CI] | Kendall τ_b(s,W1) | reading |
|---|---|---|---|---|---|
| Gemma-31B-API | 4.528 | 4.2–5.25 | −0.069 [−0.111, −0.028] | −0.169 | **calibratable w/ signal** |
| Qwen | 4.877 | 3.9–6.1 | −0.052 [−0.090, −0.013] | −0.174 | **calibratable w/ signal** |
| Maverick | 7.070 | 5.7–11.0 | −0.058 [−0.099, −0.013] | −0.095 | calibratable, weak signal |
| GLM | 8.835 | 7.1–12.8 | −0.058 [−0.104, −0.006] | −0.014 | level-only |
| Gemma-26B-API | 12.808 | 9.5–20.0 (a fold pegs) | −0.132 [−0.211, −0.050] | −0.022 | level-only, unstable folds |
| Llama-3.1-405B | **20.0 PEG** | at bound | −0.071 [−0.108, −0.034] | +0.069 | degenerate flattening |

The six-leg panel resolves into a clean gradient: **fitted T_c magnitude rises as per-cell
association quality falls** (4.5–4.9 with real signal → 7–13 with weak-to-none → the peg
with wrong-sign). The two signal-bearing temperatures agree tightly (ratio 1.08). The
practical reading for any future application: the LODO "calibratable" verdict alone is
insufficient — it fires even for pure base-rate flattening; the association diagnostic is
the adoption bar. No shared confidence constant exists (interior-T_c spread 4.5→12.8);
router value tracks the association column.

## Implications for the closed pair (report-only)

The production channel's own pre→post overconfidence is small and tightly banded
([1.03, 1.38]) — well below the token-slice values behind the [1.60, 4.88] band, which is
unaffected (closed-pair supervised territory under the adopted composite protocol). The
Gemma-4 raw-continuation pathology is irrelevant to the closed pair (chat-API-served) but
matters for any future open-model verbalized measurement: **serve instruct models through
their chat template.** Confidence: level-bias correction is learnable per-family; adoption
bar = association diagnostic, not the LODO verdict. Protocol changes remain user decisions.

## Deviations from plan

- GLM exclusion REVERSED by the user (§7 amendment); collected in a parallel session.
- The all-self-hosted decision was amended by the user for the two Gemma post legs after
  their raw-channel collapses (16-bit API re-collection, confound-labeled; the vast failure
  legs stand as the raw-channel record).
- Study-A-failures retest (gemma26 + maverick) added by user direction 2026-07-21; the
  archived Maverick serving spec (8×H100-80, $60) was found infeasible (803 GB bf16 weights
  > 640 GB VRAM) and re-provisioned on 8×H200.
- One mid-leg crash (OpenRouter HTTP-200 error body without `choices`) recovered via the
  checkpoint/resume discipline after a transport-retry patch (`4c2eab8`); 26 cached cells
  reused, none re-elicited.
- A labeled 2-cell diagnostic re-elicitation of the Gemma-31B vast-post failures (raw text
  capture; recorded failures stand).
- No pre-registered quantity was altered; extension reads are reported alongside, never in
  place of, the pre-registered verdicts.
