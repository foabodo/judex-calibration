# Study A — Open Pre/Post-Pair Calibration: Implementation Guide

**Repo:** `judex/judex-calibration` (new) · **Driver:** Claude Code on local macOS · **Compute:** rented remote GPU (vLLM)
**Status:** plan / runbook. No spend until Phase 0 gate passes.

> **2026-07-01 integration remediation applied** (see `docs/integration_remediation_2026_07_01.md`);
> **last re-verified 2026-07-18** against corpus `9541c5b`, ground-truth `b2e4fe3`, evaluator
> `2b6322b` (see that doc's 2026-07-18 addendum — it supersedes the 2026-07-03 one, whose pins and
> store counts are historical). **Two carried numbers changed** and are corrected there: the GT
> bundle was re-materialized 2026-07-09 (freethresh / readout τ=0.675 / continuous, W1 mean 0.194,
> argmax untouched), and the sanity fit `T_rps 2.4434` re-derives to **3.5585**. GT is loaded
> canonically + reproducibly (`aireg.py` synthesizes the
> manifest-verified cumulative-consistency bundle — no gitignored run dependency); few-shot is drawn for
> real from the corpus store (`fewshot.py`, k from `models.yaml`); `study_a` emits a drop-in evaluator
> calibration block. An **optional** evaluator-side family-scoped seam is specified (see §4.6 and the
> remediation doc) for the cases where the *global* seam won't do; for the simple two-closed-family case
> the block drops into the global `calibration` key as-is.

---

> **Two workflows — never cross the recipes.** The same driver runs both:
> - **SMOKE** — free Mac plumbing check (**llama.cpp/Metal, int4/Q4, `--limit`/`--no-reason`**, one
>   small stand-in model) → τ_oc **MEANINGLESS, discard it** (`docs/local_smoke_quickstart.md`; Phase 0a).
> - **LIVE** — the real, paid experiment (**vast vLLM, bf16, all 120 cells, reasoning ON**, the seven
>   panel models) → the **real τ_oc** (Phases 1–4).
>
> Flags/host/model/dtype decide which; a `--limit`/`--no-reason`/<120-cell run auto-flags `smoke` in
> the report + calibration block. This guide is the **LIVE** plan; the smoke is the free precheck.

> **2026-07-02 — evaluator pair switched to Anthropic + GPT; panel is now SEVEN.** The collaborative
> (cross-family) evaluators are once again **Anthropic + GPT**, not Gemini/GPT. Consequences: (1) Google
> is no longer a reserved evaluator, so **`gemma-4-26B-A4B` joins as the seventh annotator** (see §2);
> (2) the transfer target in Q3/Q4 is now the **Claude + GPT** closed pair — the Q4 closed-side check
> needs a **Claude + GPT** AIReg run (the legacy `stage9-gemini-gpt-medium` run is Gemini/GPT). The
> firewall now bars only **Anthropic and OpenAI** models from the annotator panel. Historical passages
> below that describe the *earlier* Gemini/GPT failures are left as-is (that history is accurate).

## 0. Why this exists (carry-over from the calibration thread)

We need **overconfidence-correcting recalibration for the closed JUDEX evaluators on future out-of-sample documents with no ground truth**. (The pair was Gemini + GPT when the three routes below were measured; it switched to **Anthropic + GPT** on 2026-07-02 — the failures are pair-independent, but the *target* is now Claude + GPT.) Three GT-free routes were exhausted:

- **DACA** — wrong objective (top-1 ECE), argmax-agreement filter discards ambiguous cells, and no open base sibling for Gemini/GPT. Abandoned.
- **Internal permutation-gleaning dispersion** — merged to `judex-evaluator` develop as infra (`mode: noop`), but on the live runs it fits **T ≈ 1.0 (identity)** because the two evaluators are *correlated and confidently-wrong-in-agreement*, so their own re-draws don't reveal the overconfidence.
- **Supervised temperature scaling** — works (Murphy Reliability 0.028 → 0.0045) but T pegs at ~19 and is wildly per-doc unstable, because the accuracy deficit (44–52% argmax) makes RPS-min *and* Reliability-min both flatten to the marginal. **Objective swap does not help in isolation** — verified.

**What the open pre/post pairs uniquely add:** the pre-trained variant is (assumed) well-calibrated; the post-trained variant is overconfident. Running *both* on AIReg-Bench (which has independent human GT) lets us measure the **clean post-training overconfidence**, holding base capability fixed — the one signal neither the internal dispersion nor the accuracy-contaminated supervised fit can give. The base models also serve as a **decorrelated reference** to fix the dispersion-≈-identity failure.

### Scientific questions (answered in order)

Study A is **not** the generic JUDEX calibration arm. That arm fits temperatures against GT for models we control. Study A tests whether a **portable constant** derived from open hybrid pairs is scientifically defensible for **closed** models (**Anthropic + GPT**) whose base variants are inaccessible.

- **Q1 — Premise validation:** are base/pretrained models well-calibrated on EU-AI-Act ordinal compliance, i.e. is their supervised temperature fit `T*_pre ≈ 1`? If not, the premise is task-invalid — learn this for ~$10 on the cheap model first.
- **Q2 — Post-training effect:** for each post-trained model, how much temperature softening (`τ_oc`) is needed to recover calibration **while retaining the post-training accuracy gain**?
- **Q3 — Transfer legitimacy:** is `τ_oc` **stable across model families**? Tight clustering ⇒ overconfidence is roughly model-agnostic in temperature units ⇒ a transferred constant for the closed evaluators is defensible, and we learn its value. Scatter ⇒ it is not.
- **Q4 — Closed-side sanity check:** when the open-derived constant is applied to the closed pair's (**Claude + GPT**) JUDEX outputs on AIReg, does the **Murphy Reliability** term improve **without destroying Resolution or RPS**? *(Needs a Claude+GPT AIReg run; the legacy `stage9-gemini-gpt-medium` run is Gemini/GPT.)*

### Load-bearing caveat (the accuracy gate)
If the open models are *also* inaccurate on AIReg, their `T*` will peg too and the study is uninformative — same failure as the closed evaluators. **Phase 0 is a free accuracy pre-check that gates all spend.**

### Exploratory extension (out of core scope; free once base elicitations exist)
- **Decorrelated dispersion:** the base models are also *decorrelated, well-calibrated* references, so adding their distributions to the GT-free dispersion replicate pool could fix the dispersion-≈-identity failure (`dispersion_calibration_recovery` in `judex-evaluator`, §4.7). This is a **different production mechanism** (a live reference, not a transferred constant), so it is deliberately outside Study A's four-question core — pursue only if Q3 shows `τ_oc` is unstable (i.e. a single constant won't do).

---

## 1. Architecture

```
  ┌──────────────────────── local macOS (Claude Code) ────────────────────────┐
  │  judex-calibration repo · orchestration · analysis · reuses judex-evaluator │
  │     calibration.py (fit_temperature, apply_temperature,                     │
  │     fit_dispersion_temperature) + experiments.py (murphy_decomposition,     │
  │     dispersion_calibration_recovery) + canonical AIReg GT (ground_truth.py) │
  └───────────────┬───────────────────────────────────────┬───────────────────┘
                  │ HTTPS (SSH tunnel)                      │ HTTPS (API keys, Keychain)
                  ▼                                         ▼
   ┌─────────────────────────────┐          ┌─────────────────────────────────┐
   │ RENTED GPU box (vLLM)        │          │ Post-trained variants via API    │
   │  serves ONE base model at a  │          │  OpenRouter / first-party        │
   │  time, OpenAI-compatible     │          │  (verbalized dist + logprobs     │
   │  /v1/completions w/ logprobs │          │   where available)               │
   │  → token-sliced 5-way dist   │          │                                  │
   └─────────────────────────────┘          └─────────────────────────────────┘
```

- **Base (pre-trained) leg:** served by vLLM on a rented box, one model per session (these don't co-reside). Distributions obtained by **token-slicing logits over the 5 compliance-level tokens** — base models can't verbalize a distribution, so we read logprobs directly.
- **Post (post-trained) leg:** API from the Mac, exactly as the existing JUDEX rater path. Where the provider exposes `logprobs` (OpenRouter: qwen3.5-35b-a3b, llama-4-maverick, deepseek-v4-pro, kimi-k2-thinking — confirmed), we also collect a token-sliced post distribution so pre vs post is measured in the *same* channel.
- **Analysis** runs entirely local, reusing the merged `judex-evaluator` calibration tooling.

---

## 2. Model inventory & serving feasibility

> **VERIFY at download time** — exact param counts, architectures, and HF repo availability for the 2026 checkpoints below are best-estimate; confirm with `hf` and the model card before provisioning. Sizes drive everything.

| Family | Base repo (HF) | Post (API) | ~Total / active | bf16 weights | Min GPU (bf16) | Tier |
|---|---|---|---|---|---|---|
| Qwen | `Qwen/Qwen3.5-35B-A3B-Base` | `Qwen/Qwen3.5-35B-A3B` (OpenRouter) | 35B / 3B MoE | ~70 GB | 1×H100-80 | **cheap** |
| Gemma ⁷ | `google/gemma-4-26B-A4B` | `google/gemma-4-26B-A4B-it` (HF→vLLM) | 25.2B / 3.8B MoE | ~50 GB | 1×H100-80 | **cheap** |
| Llama | `meta-llama/Llama-4-Maverick-17B-128E` | `…-Instruct` (OpenRouter→DeepInfra) | 400B / 17B MoE | ~800 GB | 8×H100-80 | mid |
| GLM | `zai-org/GLM-4.5-Base` | `zai-org/GLM-4.5` (OpenRouter→Z.AI) | ~355B / 32B MoE | ~710 GB | 8×H200 | mid |
| DeepSeek | `deepseek-ai/DeepSeek-V4-Pro-Base` | `deepseek-ai/DeepSeek-V4-Pro` (native) | ~1.6T / 49B MoE | ~3.2 TB | multi-node (≫16×H100) | giant |
| Mistral | `mistralai/Mistral-Large-3-675B-Base-2512` | `…-Instruct-2512` (native) | ~675B / 41B MoE (+2.5B vision) | ~1.35 TB | 16×H100 / 8×B200 | giant |
| Kimi | `moonshotai/Kimi-K2-Base` | `moonshotai/Kimi-K2-Thinking` (OpenRouter→Novita) | ~1T / 32B MoE | ~2 TB | 16×H200 / 24×H100 | giant |

⁷ **Gemma is the seventh model (added 2026-07-02).** It became eligible when the collaborative-evaluation
pair switched back to **Anthropic + GPT**, freeing Google/Gemini from the reserved-evaluator role. `gemma-4-26B-A4B`
is a granular MoE (25.2B total / 3.8B active, 8-of-128 experts + 1 shared), Apache-2.0, **ungated**, 256K ctx, ~50 GB
bf16 → single 80 GB GPU (cheapest live leg alongside Qwen). Corrections folded in: **Mistral-Large-3 is MoE**
(675B / 41B active + 2.5B vision — *not* dense), and **DeepSeek-V4-Pro is ~1.6 T / 49B active** (the earlier
671B/37B were V3.1's numbers) — its serving/cost below is under-provisioned and must be re-derived.

**Serving notes**
- **vast.ai hosts no base models** — its "Models" marketplace is instruct-only. So we do NOT use it; we rent a **generic GPU + `vllm/vllm-openai` and pull every repo (base *and* post) from HF via `--model`**. This makes the **HF download the dominant cost**: filter offers for `inet_down` + `disk_space`, gate the HF token/license, reuse one box across a family's two legs, and for the giants use a **persistent volume** (`HF_HOME`/`--download-dir`) so 0.7–2 TB isn't re-pulled per launch. Full walkthrough: `docs/vast_quickstart.md`.
- MoE: use vLLM `--enable-expert-parallel` with `--tensor-parallel-size`/`--pipeline-parallel-size`. Giants need **2 nodes** (pipeline-parallel) or B200-class single node.
- **Quantization confound:** fp8 halves the cluster but perturbs exactly the logits we measure. Calibration is logit-shape-sensitive. **Prefer bf16 for the base leg.** If fp8 is unavoidable for the giants, run the §4.4 token-slice-validity check to bound the quant effect before trusting `T*_pre`.
- Inference is *tiny* (120 cells × a few short forward passes). **GPU cost is dominated by download + load time, not inference** — so minimize wall-clock on the box (pre-stage weights to a persistent volume, serve, run, tear down).

---

## 3. Cost estimates

**Design change vs the old table:** vast hosts no base, and the same-channel design serves **both**
variants on vast vLLM — so each family now pays **two HF downloads** (base + post), and **download
wall-clock dominates** (inference is minutes). The old "~$750 base + ~$50 post-API" no longer holds.

Assumptions: bf16 = 2 bytes/param; each variant pulled once; **one box reused for both legs**; HF pull
≈ **400 MB/s** (`HF_HUB_ENABLE_HF_TRANSFER=1` on a high-`inet_down` offer — **±2× is the biggest
swing**); vast spot ≈ **$2/H100-GPU-hr, $2.75/H200-GPU-hr** (on-demand clouds ~30% higher); GPU count =
min bf16 fit (≥675B, ~1.6T and 1T do **not** fit 16×H100 → H200 / multi-node).

| Family | bf16 weights (base+post) | GPUs | wall-clock (both legs) | cost (vast spot) |
|---|---|---|---|---|
| Qwen 35B-A3B | 0.14 TB | 1×H100 | ~1 h | **~$5** |
| Gemma 26B-A4B | 0.10 TB | 1×H100 | ~1 h | **~$8** |
| Llama-4 400B | 1.6 TB | 8×H100 | ~2.5 h | ~$40 |
| GLM-4.5 355B | 1.4 TB | 8×H200 | ~2.5 h | ~$50 |
| DeepSeek V4-Pro ~1.6T | ~6.4 TB | multi-node | ~8 h | **~$400+ (re-derive)** |
| Mistral 675B | 2.7 TB | 16×H200 | ~3.7 h | ~$165 |
| Kimi-K2 1T | 4.0 TB | 16×H200 | ~4.7 h | ~$210 |
| **Both-legs subtotal (7 models)** | | | | **~$880** |
| **+ ×1.4 buffer** (failed offers, slow CDN, re-runs) | | | | **~$1,230** |

*Deltas vs the old 6-model table: **+Gemma ~$8** (cheap, single-GPU), and **DeepSeek re-costed** — V4-Pro is
~1.6 T (not 671B), so its ~$165 was a large under-estimate; the giant tier needs a fresh derivation.*

**Range: ~$650 (vast spot + fast CDN, no re-runs) → ~$1,500 (on-demand rates + slow CDN + buffer);
plan ~$900–1,100.** Download throughput and spot pricing are the two big swings.

**Cheaper alternative (~$550–800):** keep the **post leg on OpenRouter** (verbalized, no GPU) and
GPU-serve only the **7 base** variants from HF (≈$490 +buffer ~$690, + post API ~$50). Halves the
giant downloads but reintroduces the **cross-channel confound** (base token-slice vs post verbalized).

**Minimum-viable: ~$50–90.** Phase 0 (free, done) → Qwen both legs (~$5–10) → Llama both legs
(~$40–80); validate pipeline + accuracy gate before committing to the giants.

Notes: **fp8** would ~halve the giants' download + VRAM (≈ −$250) but perturbs the logits we measure —
bf16 only. A **persistent HF-cache volume** does *not* help a single clean run (base≠post, each pulled
once); it only saves re-downloads on re-runs (~$0.10/GB-mo; a 4 TB Kimi cache ≈ $400/mo).

---

## 3a. Methodology updates (2026-06-27, user decisions)

These supersede the corresponding defaults below:
1. **Live post runs use reasoning ON** (the deployed behavior). Reasoning is disabled only for cheap smoke tests. So the post leg is *not* the cheap OpenRouter verbalized path for the live study.
2. **Token-slicing for BOTH variants** — base *and* post served on **vast.ai vLLM** with logit access, so pre/post are compared in the **same channel** (removes the cross-channel confound; the `qwen3.5-35b-a3b` OpenRouter post leg stays only as a verbalized smoke/fallback, since OpenRouter routes give no logprobs for it).
3. **Base models also reason** — give the base a few-shot **chain-of-thought** scaffold before the answer, so the pre/post comparison is matched on reasoning condition (isolating post-training, not reasoning-vs-not). Token-slice the answer logits *after* the reasoning span in both legs (requires a forced `</think>`/`Answer:` scaffold + logprobs at the answer position).
4. Implication: `elicit_base.py` and the post token-slice path both target vast.ai vLLM (reasoning + answer-position logit read), not OpenRouter, for live runs.

## 4. Methodology

### 4.1 The 120 evaluation cells and GT
- Cells: 24 examinees × 5 Articles (9,10,12,14,15), `item_label` like `"Art 9 / Scenario A | Use 1"`.
- GT: loaded by `aireg.load_cells()` through the evaluator's **manifest-verified** `synthesize_aireg_bench_ground_truth(judex-ground-truth/data/distributional_labels/, LABELS)` — the current canonical `mgmfrm_anchored_projection` (cumulative-consistency) bundle, git-tracked and reproducible on a fresh clone. **As re-materialized 2026-07-09** it is freethresh-thresholded, **readout-tempered at τ = 0.675**, and **continuous** (grid snap off) — so the GT we fit temperatures against is itself a tempered readout, and it validates as `unavailable` on two hard-failure families (R-hat and prior_predictive). Full provenance in the `aireg.py` docstring; pinned by `tests/test_integration_seams.py::GtVintageGuardTests`. The item_label→document join comes from `judex-corpus/step3_4/ground_truth.json` and evidence from `judex-corpus/step5/documents/`. **Do NOT read GT from a run's `metrics_report.json`** — `runs/` is gitignored and a run embeds the GT that was canonical at run time (stale after any re-fit; this bit us twice — see the remediation doc's 2026-07-18 addendum). **AIReg is the VALIDATION set; never used as ICL.**

### 4.2 Base-model elicitation — MCQA token-slicing (vLLM)
Base models need **few-shot** framing to follow the answer format. **Draw few-shot examples from the exemplar/calibration corpus** (`judex-corpus/leaf_exemplars`) — disjoint from AIReg, so the firewall holds.

Prompt skeleton (per cell):
```
[few-shot: 3–5 (excerpt, Article criterion, single-letter answer) examples from the exemplar corpus]
Excerpt: <AIReg cell excerpt>
Article <n> criterion: <criterion text>
Compliance level (A=very_low, B=low, C=moderate, D=high, E=very_high). Answer:
```
Read logits over the five answer tokens. **Pick single-token labels** (verify per tokenizer; `A`–`E` or `1`–`5` are safe). Two methods:

- **Primary (1 call/cell):** `/v1/completions` with `max_tokens=1, logprobs=20, temperature=0`, then read `choices[0].logprobs.top_logprobs[0]`, extract the five label tokens, softmax → 5-way distribution.
- **Robust fallback (5 calls/cell)** if a label misses the top-20: append each candidate, `echo=True, max_tokens=0, prompt_logprobs=0`, read the appended token's logprob; softmax the five.

```python
# vLLM OpenAI-compatible client (base served at http://<box>:8000)
import openai, math
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")
r = client.completions.create(model=MODEL, prompt=prompt, max_tokens=1,
                              logprobs=20, temperature=0)
top = r.choices[0].logprobs.top_logprobs[0]          # {token: logprob}
lp  = [top.get(t, -50.0) for t in (" A"," B"," C"," D"," E")]   # verify tokenization
m   = max(lp); ex = [math.exp(x-m) for x in lp]; Z = sum(ex)
dist = [e/Z for e in ex]                              # Type-C-comparable 5-way
```
Serve: `vllm serve <base_repo> --dtype bfloat16 --tensor-parallel-size N [--enable-expert-parallel] --port 8000`.

### 4.3 Post-model elicitation
- Verbalized 5-way distribution via the existing JUDEX rater contract (API from Mac).
- Where `logprobs` exists (OpenRouter: qwen3.5-35b-a3b, llama-4-maverick, deepseek-v4-pro, kimi-k2-thinking; gemma-4-26b-a4b is served on vLLM so it has logprobs natively), also collect a token-sliced post distribution (same letters) so pre vs post is measured in the **same channel** — removes the verbalized-vs-logit confound.

### 4.4 Token-slice validity check (the original "Phase 1")
For the post models that expose **both** verbalized and logprob channels, compare `INV_SOFTMAX(verbalized)` vs token-sliced logits (W1/KL). This empirically tests whether the two channels agree — the assumption the whole base-vs-post comparison rests on. Run it before trusting cross-channel comparisons.

### 4.5 Per-family fits (reuse `judex.calibration`, i.e. `judex-evaluator/src/judex/calibration.py`)
For each family, against AIReg GT, fit **both** objectives (we proved neither is a free lunch under low accuracy, so report both):
- `T*_pre`  — `fit_temperature(pre_dist, gt)` (RPS-min) and the reliability-min variant.
- `T*_post` — same for the post distributions.
- `τ_oc` — the clean post-training overconfidence temperature: the T that maps **post → pre calibration** (or `T*_post` directly if Q1 confirms `T*_pre ≈ 1`).
- Always alongside: **argmax accuracy** per model (the gate) and the **Murphy decomposition** (is the addressable error Reliability or Resolution?).

### 4.6 Cross-family stability & decision (Q3 + Q4)
- Plot/serialize the seven `(T*_pre, T*_post, τ_oc, accuracy)`.
- **Document-clustered bootstrap** (reuse the pattern; resample the 24 docs) CIs on `τ_oc` and on the cross-family spread.
- **Decision rule:** if `τ_oc` clusters tightly *and* the bases clear the accuracy gate, adopt `median(τ_oc)` as the transferred constant for the closed evaluators. `study_a.calibration_block()` emits a drop-in `pipeline.yaml → calibration` block (`mode: temperature`; written to `runs/<run>/pipeline_calibration_block.json`). **Seam note:** the merged `pipeline.yaml` calibration seam is **global** (applied at the Phase-1 gleaning site to whichever families run, no `family_id`). At evaluation time those families are the two **closed** evaluators (now **Claude + GPT**) — the open annotator raters run only at *construction* and are never calibrated here — so for the intended case (two closed families + one clustered constant) pasting the block into the top-level `calibration` key is **adequate**. **Family-scoping is an optional refinement**, needed only if τ_oc doesn't cluster (per-family T), if an arm runs a different evaluator pair, or to move the correction to Phase-3; it is specified in `docs/integration_remediation_2026_07_01.md` (a separate, approved evaluator change, relevant only at this Phase-4 decision). Else, report negative. **§4.8 (E6) extends this Q4 check into the rescoped core paper's demonstration leg** — run it off the same Phase-4 artifacts.

### 4.7 Decorrelated dispersion (exploratory extension — see §0)
Add the base-model distributions to the dispersion replicate pool for the closed evaluators (a *decorrelated, well-calibrated* reference, fixing the correlated-overconfidence blindness) and re-run `dispersion_calibration_recovery` on the existing `stage9-gemini-gpt-medium` / `phase23-deference-fix-native` runs. **Pitfall:** do not match the evaluator's *width* to a base model's width (a well-calibrated weak model is appropriately wide; copying it over-widens). Use base disagreement only as *added dispersion*.

### 4.8 E6 — the distributional-utility demonstration leg (2026-07-17 addendum)

**Context.** The rescoped judex-core paper carries an experiment battery proving
the material distinction between discrete and distributional labels (umbrella
`spec/analysis_2026_07_17_distributional_utility_experiment_battery.md`; this
is its **E6**). Q4 already checks that the transferred constant improves Murphy
Reliability on the closed pair without destroying Resolution/RPS; E6 extends
that check into the paper-grade exhibit. **No new machinery** — E6 is analysis
on Q4's artifacts, plus two pre-registered exhibits (with one exception: E6.4's
E2 half has nothing to re-score — see "open dependencies" below). The closed pair is
currently **Claude Sonnet 4.6 (medium effort) + GPT 5.4 (medium reasoning)**
— evaluator families `anthropic_claude_medium` + `openai_gpt_standard` —
(2026-07-17; Haiku 4.5 / GPT-5.4-mini are ruled out as insufficiently capable
for the evaluation task).

> **CORRECTION (2026-07-18): no existing run realizes that pair at 120 cells.**
> An earlier draft of this section named `stage9-sweep-sonnet-gpt-v2` as "the
> Claude+GPT AIReg run Q4 needs". It is not. Verified against the run's own
> artifacts: `cross_family_evaluation_slice.json` gives families
> `anthropic_claude` + `openai_gpt` (the bare ids — no `reasoning_effort`), and
> `provider_usage_summary.json` `by_model` shows its GPT seat is
> **`gpt-5.4-mini`** ($10.50) — one of the two models this very paragraph rules
> out. It also fails the vintage test independently: `created_at`
> 2026-06-21, `stop_reason: "error"`, no `run_identity` block, and it predates
> contract 0.2.0 (2026-07-05) and the corpus-v2 binding (2026-07-14). The run
> that *does* use the right pair, `stage9-claude-gpt-medium`, covers 3 docs /
> 15 items, not 120. **Q4 and E6 therefore require a fresh on-pair sweep**
> (~$13.6/doc measured × 24 ≈ **$290–330**), and any Q4 result must state which
> config bundle it ran under (`--config-dir configs_v2exemplars` for corpus-v2
> exemplars; `configs/` is still the evaluator default and is v1).

- **E6.1 — Pre-correction measurement.** On the closed pair's live AIReg run:
  Murphy REL/RES/UNC, entropy deficit vs GT (mean ΔH), and coverage@90 of the
  point band. This is the "overconfidence is measurable only distributionally"
  half of the exhibit.
- **E6.2 — Correction mechanism = the Phase-4 transferred `median(τ_oc)`,
  full stop.** Not a supervised fit on the closed pair (measured failure: T
  pegs ~19 and flattens to the marginal under the accuracy deficit — §0), not
  DACA (abandoned — §0), not GT-free dispersion (fits identity — §0). E6
  inherits Study A's Q1–Q3 gates unchanged; if Q3 scatters and Phase 4 reports
  negative, E6 reports the same negative (publish-the-null discipline).
- **E6.3 — The discrete-blindness exhibit.** Temperature scaling is
  argmax-preserving, so argmax accuracy and quadratic-weighted κ vs GT are
  **bit-identical pre/post correction** — verify mechanically and report as
  the invariance row — while REL, coverage, and ΔH move. Every discrete-label
  metric is provably blind to the entire intervention; this is the paper's
  "invisible quality axis" demonstration.
- **E6.4 — Downstream endpoint deltas.** Re-score the battery's E2 routing
  signals (prediction entropy, pair W1, Δ_res) and E3 decision-cost endpoints
  pre/post τ_oc, under the battery's endpoint discipline (outcome endpoints
  adjudicate; RPS/REL/coverage are diagnostics only). Doc-clustered bootstrap
  (resample the 24 docs, §4.6 pattern) for every CI.
- **E6.5 — Optional secondary (accuracy-gated).** The peg-at-19 failure was
  measured on the Gemini/GPT-era runs (44–52% argmax). If the **current** pair
  clears an argmax-accuracy floor on AIReg, a held-out-split supervised fit
  may no longer peg: report fitted-vs-transferred **convergence** as bonus
  validation of the transfer (firewall per risk 6: split/CV only; never
  tune-on-test for reported numbers).

**Cost.** One fresh closed-pair sweep, **~$290–330** (the "$0 if
`stage9-sweep-sonnet-gpt-v2` is vintage-valid" branch is closed — see the
correction above; that run is off-pair, not merely stale). **Gates to
pre-register before running:** the E6.3 invariance row must be exact; the
E6.1→post REL improvement must clear a doc-clustered CI; E6.4 deltas are
reported win-or-null.

**Two open dependencies to settle before E6 runs** (both sit outside this repo,
so they are flagged, not patched here):
1. **E6.4 assumes E2 artifacts exist.** On `judex-ground-truth` `develop` the
   Tier-0 battery ships `run_e{3,4,5,5b,8}.py` and `e{3,4,5,5b,8}_results.json`
   — **E2 is registered, never executed**, so there are no routing signals to
   "re-score". E3's decision-cost engine *does* exist and consumes
   temperature-calibrated credences unchanged, so the E3 half of E6.4 is ready
   and the E2 half is a build, not a re-score.
2. **The paper specifies a different correction mechanism.** `judex-paper`'s
   rescoped core paper still describes E6 step (ii) as *"supervised,
   agreement-filtered temperature scaling"* citing DACA, and scopes its
   calibration claims to that setting. This guide's transferred-`median(τ_oc)`
   mechanism is the *correct, later* one — the umbrella battery spec records
   "MECHANISM CORRECTED 2026-07-17 against the Study A guide" and adopts it —
   but the paper was not updated. **Study A is right and the paper lags**;
   §subsec:calibration needs a paper-side rewrite before E6's output can land
   in it. Escalate rather than silently reconciling in either direction.
3. **Pin the A–E orientation before E6.4.** §4.2 slices tokens A–E; the E3 cost
   engine is orientation-sensitive (`grade 1 = very_low`, so `a > l` is false
   clearance). State the mapping explicitly in the Q4 artifacts or the cost
   deltas can silently invert.

---

## 5. Repo structure (`judex-calibration`)

```
judex-calibration/
├── README.md
├── pyproject.toml                 # depends on judex-evaluator (editable sibling)
├── .gitignore                     # runs/, weights/, .venv/, *.nc
├── configs/
│   ├── models.yaml                # the §2 inventory (base repo, post id, serving args, fewshot_k)
│   └── providers.yaml             # post-trained API providers + Keychain keys
├── src/judex_calibration/
│   ├── elicit_base.py             # vLLM token-slicing (§4.2); per-cell fewshot callable
│   ├── elicit_post.py             # API verbalized + logprob (§4.3)
│   ├── fewshot.py                 # k-shot loader from the corpus store (§4.2)
│   ├── aireg.py                   # 120 cells + CANONICAL manifest-verified GT (git-tracked sources)
│   ├── study_a.py                 # Q1/Q2 fits, τ_oc; Q3 cross-family; Q4 closed_side_check; calibration_block()
│   └── decorrelated_dispersion.py # (planned) §4.7 extension wiring into judex-evaluator
├── scripts/
│   ├── phase0_accuracy_precheck.py   # FREE gate (canonical GT argmax vs AIReg LLM annotations)
│   ├── run_qwen_phase1.py            # per-family driver: elicit base/post → Study A report + calib block
│   ├── provision_vast.sh             # vast.ai provision/poll/teardown (Mode C)
│   └── serve_vllm_vastai.md          # the four serving modes reference
├── docs/
│   ├── study_a_implementation_guide.md
│   ├── vast_quickstart.md            # end-to-end account→data→shutdown walkthrough
│   └── integration_remediation_2026_07_01.md   # this pass's audit + seam spec
├── runs/                          # per-run artifacts (gitignored bulk; runs/phase0 kept)
└── tests/                         # test_elicit_base.py (offline, monkeypatched)
```

**Dependency on `judex-evaluator`:** `pip install -e ../judex-evaluator` so `from judex.calibration import fit_temperature, fit_dispersion_temperature, dispersion_calibration_recovery` and `from judex.experiments import murphy_decomposition` are reused verbatim — no fork of the math.

**Umbrella wiring (do when ready, not yet):**
```
cd judex && git submodule add ./judex-calibration judex-calibration   # or the remote URL once pushed
```
Keep it a sibling submodule like the other three. Per project convention, `develop` is the integration branch — branch off `develop`, never create `integration`.

---

## 6. Claude Code integration & workflow

Claude Code on the Mac is the orchestrator. The remote GPU box is treated as an external resource it drives over SSH/HTTP.

> **Faster option — run Claude Code *on* the vast box.** Install Claude Code on the rented instance
> and let it orchestrate the whole run there (serve vLLM → elicit → swap base/post → analyse →
> return results → teardown), troubleshooting OOM/context/logprobs autonomously. Full runbook (install,
> headless auth, and a paste-ready orchestration brief): `docs/vast_claude_code_orchestration.md`.

### 6.1 Permissions (`judex-calibration/.claude/settings.local.json`)
Allowlist the recurring read-only/local calls to cut prompts: `hf`, `ssh <box>`, `curl http://localhost:8000/*`, `vllm` (on the box via ssh), `python scripts/*`, and the Keychain pattern `security find-generic-password -s *-api-key -w`. (See `/fewer-permission-prompts`.)

### 6.2 Serving = an HTTP endpoint you call from the Mac
Inference is OpenAI-compatible HTTP from the Mac — **not** the box's CLI, and **SSH is optional**. The four serving modes (A serverless, B one-click template, C custom on-demand `vllm serve` — *primary*, D offline in-process) are in `scripts/serve_vllm_vastai.md`; the base leg requires **Mode C** (base weights + `/v1/completions` logprobs + bf16). The single `vllm serve` launch can be the instance's `--onstart-cmd`, so the box boots already serving; Claude Code then polls `curl http://<host>:<port>/v1/models` and elicits.

### 6.3 Per-family runbook (see `scripts/serve_vllm_vastai.md` for full commands)
1. Provision an on-demand box that boots serving the **base** repo (Mode C `--onstart-cmd`, bf16).
2. Reach it **directly** at the public `http://<host>:<port>` (or an optional SSH tunnel); poll `/v1/models`.
3. `run_qwen_phase1.py --base-url … --base-model <base_repo>` → base distributions (token-slice).
4. Relaunch the box on the **post** repo; `run_qwen_phase1.py --post-url … --post-model <post_repo>`.
5. `run_qwen_phase1.py --analyze-only` → per-family `T*_pre/T*_post/τ_oc/accuracy/murphy` + Q4 closed-side.
6. **Tear down the box** (cost is wall-clock).

### 6.4 Secrets
Post-model API keys stay in macOS Keychain (`security find-generic-password -s <provider>-api-key -w`, exported in-shell, never printed). HF token likewise (`hf login` on the box, or `HF_TOKEN` from Keychain).

### 6.5 Memory / handoff
Record run-ids, fitted `τ_oc`, the Q1–Q4 verdicts, and any negative results in a `judex-calibration` project memory + a `spec/handoff_*.md`, consistent with the existing JUDEX cadence.

---

## 7. Phased plan with decision gates

| Phase | Action | Cost | Gate to proceed |
|---|---|---|---|
| **0a** | **Local plumbing smoke** (optional, free) — on the **Mac (M1 Pro/Metal)** serve one small **int4/Q4** stand-in (e.g. Qwen3-4B) via **llama.cpp / llama-cpp-python** (or a local CUDA box via vLLM) and run `run_qwen_phase1.py --limit 6 --no-reason` end to end. Proves serve→token-slice→analysis→calibration-block before any spend. See `docs/local_smoke_quickstart.md`. | $0 | Pipeline emits `study_a_report.json` + `pipeline_calibration_block.json`. **DISCARD every number — τ_oc is MEANINGLESS** under int4 + a stand-in model + `--limit` + `--no-reason` (the run self-flags `smoke`); the gate checks only that the pipeline runs. |
| **0** | **Free accuracy pre-check** — parse the 10 existing AIReg-Bench LLM annotations vs human GT; compute argmax accuracy. | $0 | If even frontier models (o3/gpt5/sonnet/gemini-pro) score ~low, the accuracy gate is structural → **fix accuracy/elicitation first; do NOT spend.** |
| **1** | **The two-family cheap trial, leg 1: Qwen 35B** (base+post, **bf16, all 120 cells, reasoning ON — PAID**; the first *real* measurement, distinct from the free int4 `--limit`/`--no-reason` Mac smoke in 0a whose numbers are discarded); run §4.4 validity check on a post model with logprobs. `--family qwen --out runs/qwen`. | ~$5 | Pipeline green; token-slicing valid; Qwen clears accuracy gate. |
| **1b** | **Trial leg 2: Gemma 26B-A4B** — the other cheap, single-GPU pair (ungated Apache-2.0; Google lineage, so the trial spans two distinct lineages; post leg served plain — no reasoning parser). Run back-to-back with Phase 1 as ONE trial (quickstart §12), then **merge**: `--merge qwen=runs/qwen gemma=runs/gemma` → the first two-family `τ_oc` spread before any mid/giant spend. | ~$8 | Gemma clears the accuracy gate; `τ_oc` finite on both; the 2-family spread plausibly clustered. |
| **2** | Add **Llama-4-Maverick** (mid). Three-family `τ_oc` + cross-family check. | ~$60 | `τ_oc` plausible & accuracy adequate on ≥3 families. |
| **3** | Commit to the **three giants + GLM** (bf16, multi-node). Full seven-family Q1/Q2/Q3. | ~$700+ | — |
| **4** | Decision: adopt `median(τ_oc)` transferred constant (paste the emitted block into `judex-evaluator/configs/pipeline.yaml` → `calibration`) or report negative. **Refuse adoption if `tau_oc_any_saturated`** — a pegged τ_oc is a boundary artefact, not a fit. Optional, gated *on* Q3 being negative: the §4.7 decorrelated-dispersion extension. | $0 | — |
| **4b** | **E6 — the distributional-utility demonstration leg** (§4.8): fresh on-pair closed sweep, then E6.1–E6.5 off those artifacts. Not $0 — no existing 120-cell run uses the current pair. | ~$290–330 | Q1–Q3 gates passed; E6 gates pre-registered before the sweep runs. |

---

## 8. Risks & honest caveats

1. **Accuracy gate may moot the study** — if the open models are also inaccurate on EU-AI-Act compliance, every `T*` pegs (as it did for Gemini/GPT). Phase 0 + Phase 1 are designed to fail cheap.
2. **fp8 quantization confound** — fp8 corrupts the logits we measure; prefer bf16 on the base leg, or bound the effect via §4.4.
3. **Model availability/sizes** — the 2026 base checkpoints and exact architectures must be verified at download; sizing/cost shifts if they differ.
4. **Transfer to closed evaluators is an assumption** — tested only indirectly (cross-family clustering + applying `median(τ_oc)` to the closed pair (Claude/GPT) on AIReg and checking the Murphy Reliability drop). More grounded than DACA's, not a proof.
5. **Base-model prompt sensitivity** — base models are format-fragile; few-shot count/wording affects the token-sliced distribution. Hold the few-shot block fixed across families; treat it as part of the measurement instrument.
6. **Calibration/validation firewall** — AIReg is the validation set. A single transferred scalar T for *production* (future docs) is legitimate; for *reporting AIReg numbers* fit T on a held-out split / CV to avoid tuning-on-test.
7. **The post panel and the corpus store are now aligned (2026-07-03)** — the `judex-corpus` few-shot exemplar store is the **7-rater** build (Gemma collected as the seventh seat via OpenRouter/Novita bf16), and the GT leaf/dimension labels were re-fit on the same 7-rater panel. The Study A panel and the corpus annotation panel are the same seven families; any *future* panel change still requires a deliberate re-pin plus a corpus rebuild kept separate from this study. **Update 2026-07-14 — the adopted vintage is corpus v2:** the same 7-seat panel re-annotated the Grok-4.5 re-authored (`ambiguity_structured`) excerpts under contract 0.2.0 — **1757 rows (1449 leaf + 308 dimension)** in `judex_leaf_exemplar_construction_v2/`; `fewshot.DIMENSION_STORE` points there. v2 dimension exemplars are **~14× longer** (mean ~9.7k chars), so the live prompt budget was re-measured (`scripts/measure_prompt_budget.py`): worst-case prompt ≈ **18.3k tokens**, required context **≈20.4k** at `--budget 2048` ⇒ `--max-model-len` **≥ 24576, standard pin 32768** (all SEVEN families measured and fit — llama completed 2026-07-14 after its HF gate access was granted). The v2 few-shot block is part of the measurement instrument (risk 5) — the SAME per-Article k=4 blocks are held fixed across all seven families.

---

## 9. Immediate next step

Run **Phase 0** (free) — it costs nothing, reuses data already on disk, and decides whether any GPU spend is justified. Everything downstream is gated on it.
