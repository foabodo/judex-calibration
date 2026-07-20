# Study B results — verbalized-channel pre/post calibration campaign (2026-07-20/21)

Executed under the approved design (`docs/study_b_design.md`, full-contract amendment) plus
two user-directed extensions: the Gemma post re-collection via 16-bit API (2026-07-21) and
the GLM-4.5 extension (§7 amendment; collected in a parallel session, `1cb98cd`). All vast
legs self-hosted vLLM bf16, k=5 full-contract few-shot, workers 8, temperature 0, 120 AIReg
cells each; the one API leg is OpenRouter pinned to 16-bit providers (all 120 cells served
OpenInference @ bf16 — the study's own precision). Per-cell artifacts in gitignored
`runs/study_b_{qwen,gemma31,gemma31_api,llama31,glm}/`; reports of record are each dir's
`study_b_report.json` + `runs/study_b_cross/study_b_cross_family.json` (five-way merge).
All boxes destroyed after Mac-side verification. Total spend ≈ $200–250 (llama31's and
GLM's 8×H200 boxes dominate; the API leg cost ≈ $0.60).

## B-Q1 — can the models emit the full production contract? YES, and gate outcomes are CHANNEL-BOUND

Contract-complete rate (gate ≥ 0.90), full six-field 0.2.0 tier:

| Family | PRE (base) | POST (instruct) | Channel note |
|---|---|---|---|
| Qwen3.5-35B-A3B | **91.7% PASS** | **95.8% PASS** | vast raw-completions |
| Gemma-4-31B | **100% PASS** | raw-completions **30.8% FAIL** → chat-API bf16 **93.3% PASS** | reversal ① |
| Llama-3.1-405B | **95.0% PASS** | **98.3% PASS** | vast raw-completions |
| GLM-4.5 (extension) | **96.7% PASS** | **92.5% PASS** | reversal ② |

The load-bearing feasibility unknown resolves POSITIVE — pretrained bases emit the full
contract at 92–100%, all parsed emissions on the 0.05 grid, failures all-or-nothing.

**The campaign's sharpest structural finding is a pair of opposite-direction reversals
showing that gate outcomes track the CHANNEL, not the weights:**

- **Reversal ① (Gemma-4-31B-it):** under greedy raw-completions continuation the instruct
  twin collapses into a repetition loop (`{"findings": [{"` then "la la la…"; diagnostic in
  `runs/study_b_gemma31/diagnostics_post_failures.json`) — 30.8% FAIL. The same weights
  under the chat template emit 93.3% complete contracts with the **best post argmax of the
  panel (0.678)**. The recorded vast failure stands as the raw-channel result; the API pair
  lives in `runs/study_b_gemma31_api/` (channel `verbalized_api_chat`, confound-labeled).
- **Reversal ② (GLM-4.5 base):** the family whose token-slice gate failure drove the
  2026-07-20 band amendment **passes both verbalized gates** — 96.7% contract-complete,
  argmax 0.543 (best BASE of the panel), resolution 0.047, T_rps 2.55 finite. Its Study A
  exclusion was a fact about the logit channel, not about the model's compliance judgement.

## B-Q2 — τ_v (verbalized pre→post overconfidence temperature)

| Family | τ_v | status |
|---|---|---|
| Qwen | 1.2811 | clean fit, in-mode |
| Llama-405B | 1.0252 | clean fit, in-mode |
| GLM-4.5 | 1.0252 | clean fit, in-mode (extension) |
| Gemma-31B | 1.0252 | cross-mode (vast pre × API post) — confound-labeled |

(The vast Gemma pair's 1.8571 is excluded: gate-failing selected sample.) ε-sensitivity
(pre-registered, once, on B1): τ_v(qwen) stable at 1.281 for ε ∈ {0.001, 0.005}.

## B-Q3 — τ_v clusters, robustly

- **Pre-registered verdict (original gate-passing set, n=2):** {qwen 1.281, llama31 1.025}
  → ratio **1.25 ≤ 2, rule PASSES.**
- **Extension read:** in-mode n=3 {qwen, llama31, glm} → ratio **1.25**; with the
  cross-mode Gemma point, n=4 → ratio **1.25**. Three families sit on the same grid point
  (1.0252) with Qwen mildly above. The survivorship concern raised at n=2 is now largely
  dissolved: BOTH families that were missing (gemma via channel failure, glm via the old
  token-slice gate) cluster once measured in a channel they can emit in.
- **Channel comparison:** τ_v < τ_oc for every family — qwen 1.28 vs 1.60, llama31 1.03 vs
  1.86, gemma31 1.03 (cross-mode) vs 4.88, glm 1.03 vs excised. The verbalized channel
  carries systematically less pre→post overconfidence than the token-slice channel
  (consistent with the 2(c) grid-quantization/entropy audit); three of four families are
  essentially uninflated verbally. Parallel measurements, never conversions.

No constant is adopted into any config from these results (report-only, per the design).

## B-Q4 — verbalized confidence: universally overconfident in LEVEL; calibratable with heterogeneous signal quality

Stated confidence overstates correctness everywhere: mean p̂(C) vs realized argmax-accuracy
= 0.833/0.602 (qwen), 0.950/0.678 (gemma-API), 0.715/0.483 (llama31), 0.779/0.595 (glm).

Pre-registered verdict (doc-clustered LODO, ΔBrier CI excluding 0) + peg discipline:

| Post leg | T_c (full) | folds | LODO ΔBrier [CI] | verdict | Kendall τ_b(s, W1) |
|---|---|---|---|---|---|
| Qwen | 4.877 | 3.9–6.1, none peg | −0.052 [−0.090, −0.013] | **calibratable** | −0.174 (right sign) |
| Gemma-API | 4.528 | 4.2–5.25, none peg | −0.069 [−0.111, −0.028] | **calibratable** | −0.169 (right sign) |
| GLM | 8.835 | 7.1–12.8, none peg | −0.058 [−0.104, −0.006] | calibratable, **level-only** | −0.014 (≈ 0) |
| Llama-405B | **20.0 PEG** | folds hit bound | −0.071 [−0.108, −0.034] | fires, but **degenerate** | +0.069 (wrong sign) |

Honest stratification: **Qwen and Gemma are the real confidence-calibration story** — two
interior temperatures that agree (4.877 vs 4.528, **ratio 1.08**), stable folds, right-sign
per-cell error association. **GLM** adds a third interior fit and the three-way interior
ratio is 1.95 (≤ 2), but its association is ≈ zero: its stated confidence carries a
correctable LEVEL bias and no per-cell ranking signal — a level-only calibration.
**Llama31's** peg is degenerate base-rate flattening (never adoptable). So the panel spans
the full spectrum: informative-and-calibratable (qwen, gemma), level-only (glm),
uninformative (llama). No shared confidence constant is adopted — but the qwen↔gemma
agreement is a reportable regularity, and the interior-T_c ratio landing inside the ≤2 form
is worth carrying to the paper as an exploratory observation (NOT a pre-registered pass;
the B-Q3 rule was registered for τ_v, not T_c). Router value (E2's S09 framing) mirrors the
association column: modest for qwen/gemma, absent for glm/llama; AURC is unmoved by T_c
everywhere (monotone-ish map).

## Implications for the closed pair (report-only)

The production channel's own pre→post overconfidence is small and clustered (three families
at ≈1.03, one at 1.28) — well below the token-slice τ_oc values that motivated the
[1.60, 4.88] band. This does not amend the band (closed-pair supervised territory under the
adopted composite protocol); it is off-pair evidence that verbalized emissions are flatter
than logits. On confidence: a per-family level-bias correction is learnable and two of four
families agree on its magnitude, but signal quality is family-heterogeneous — any
closed-pair application needs on-pair contract outputs (the E6 sweep) and its own fit, with
the association diagnostic (not just the LODO verdict) as the adoption bar. Protocol
changes remain user decisions; recommendation unchanged: per-family fit only, never
transferred.

## Deviations from plan

- GLM exclusion (B0 review) was REVERSED by the user (design §7 amendment) after the
  llama31 T_c peg; collected as a labeled extension in a parallel session (`1cb98cd`),
  both legs self-hosted per the amended terms.
- The all-self-hosted decision was amended by the user for ONE leg (Gemma post) after the
  raw-channel collapse: 16-bit API re-collection, confound-labeled; the vast failure leg
  stands as the raw-channel record.
- One diagnostic addition: a labeled 2-cell re-elicitation of the Gemma vast-post failures
  to capture raw text (stored separately; the recorded failures stand).
- No pre-registered quantity was altered; extension reads are reported alongside, never in
  place of, the pre-registered verdicts.
