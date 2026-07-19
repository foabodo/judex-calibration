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

**What the open pre/post pairs uniquely add:** the pre-trained variant is (assumed) well-calibrated; the post-trained variant is overconfident. Running *both* on AIReg-Bench (which has independent human GT) lets us measure the **clean post-training overconfidence**, holding base capability fixed — the one signal neither the internal dispersion nor the accuracy-contaminated supervised fit can give. The base models also serve as a **decorrelated reference** to fix the dispersion-≈-identity failure — **confirmed 2026-07-19 (§4.7):** the 3-base pool breaks the identity fit and recovers the supervised ordering GT-free.

### Scientific questions (answered in order)

Study A is **not** the generic JUDEX calibration arm. That arm fits temperatures against GT for models we control. Study A tests whether a **portable constant** derived from open hybrid pairs is scientifically defensible for **closed** models (**Anthropic + GPT**) whose base variants are inaccessible.

- **Q1 — Premise validation:** are base/pretrained models well-calibrated on EU-AI-Act ordinal compliance, i.e. is their supervised temperature fit `T*_pre ≈ 1`? If not, the premise is task-invalid — learn this for ~$10 on the cheap model first.
- **Q2 — Post-training effect:** for each post-trained model, how much temperature softening (`τ_oc`) is needed to recover calibration **while retaining the post-training accuracy gain**?
- **Q3 — Transfer legitimacy:** is `τ_oc` **stable across model families**? Tight clustering ⇒ overconfidence is roughly model-agnostic in temperature units ⇒ a transferred constant for the closed evaluators is defensible, and we learn its value. Scatter ⇒ it is not.
- **Q4 — Closed-side sanity check:** when the open-derived constant is applied to the closed pair's (**Claude + GPT**) JUDEX outputs on AIReg, does the **Murphy Reliability** term improve **without destroying Resolution or RPS**? *(Needs a Claude+GPT AIReg run; the legacy `stage9-gemini-gpt-medium` run is Gemini/GPT.)*

### Load-bearing caveat (the accuracy gate)
If the open models are *also* inaccurate on AIReg, their `T*` will peg too and the study is uninformative — same failure as the closed evaluators. **Phase 0 is a free accuracy pre-check that gates all spend.**

**Corrected gate definition (2026-07-19).** The original criterion — argmax accuracy above the
0.20 uniform-chance floor — is **misspecified**: the AIReg GT argmax marginal is imbalanced
(level 2 = 33.3%, level 5 = 27.5% of the 120 cells; computed from the canonical `gt_argmax`), so
label-blind strategies clear 0.20 easily — marginal-matched guessing scores **0.249** and
always-predict-level-2 scores **0.333**. The k=4 few-shot exemplars expose prior-like label
information, so the majority-class floor is the fair null. **Resolution is PRIMARY (adopted 2026-07-19, user decision).** Discrete labels are noisy
summaries of the annotators' distributional credences (the core paper's founding hypothesis), and
argmax scores a model against that noisiest compression — *before* the monotone recalibration this
study exists to apply. Murphy **resolution** is the right capacity measure because it is
(a) **chance-proof** — no label-blind predictor has any; (b) **recalibration-invariant** — it
measures exactly the signal a fitted temperature can recover; and (c) empirically the only
headline metric that predicts post-correction quality. Worked proof (Gemma-4-31B, k=5): the post
leg *loses* to its base on argmax (0.417 vs 0.425) and on raw W1 (1.229 vs 0.790) but wins on
resolution (0.0513 vs 0.0457) — and after applying its own fitted temperature its mean W1 (0.759)
**beats the base** (0.790). Argmax and raw W1 conflate discrimination with calibration; resolution
does not. (Caveat: same-cells tempering is in-sample; the honest out-of-sample version is the
transferred cross-family constant, which is Q3/Q4's job. Near-zero resolution edges need a
bootstrap — finite-sample resolution is positively biased.)

A leg **clears the gate** iff:
1. **Resolution (primary):** Murphy **resolution > 0** with margin (bootstrap if near zero) —
   equivalently, post-recalibration RPS beats the **climatology baseline** (always predict the GT
   mean distribution; RPS = the uncertainty term, **0.0681**).
2. **Argmax (secondary diagnostic, reported never gating alone):** accuracy vs the
   **majority-class floor 0.333** (binomial SE ≈ 0.043 at n=120) screens for degenerate
   predictors and preserves comparability with AIReg-Bench conventions. Significantly below the
   floor (as the Gemma-26B-A4B base: 0.167, z = −3.87, resolution ≈ noise — FAILED, τ_oc pegged
   artefactually) corroborates a resolution failure; at-floor with real resolution still passes
   (the k=4 Qwen base did: acc 0.350 ≈ floor, resolution 0.0332).

**Measurement-noise band (measured 2026-07-19, sequential-vs-concurrent replicate on identical
hardware):** greedy 2048-token CoT chains are chaotic under bf16 numeric jitter (vLLM kernels are
not batch-invariant; hardware/version changes perturb identically), so **per-cell distributions
are not reproducible** (mode agreement 78/120 between replicates) while **aggregates are**:
argmax within binomial noise, fitted T within ~±10–15% ⇒ **single-measurement τ_oc carries
≈ ±20%**. Interpret τ_oc differences within that band as noise (e.g. Qwen's k=4→k=5 shift
1.28→1.60); the cross-family Q3 spread (3× at the cheap tier) sits far outside it.

### Exploratory extension (out of core scope; free once base elicitations exist)
- **Decorrelated dispersion (EXECUTED 2026-07-19 — POSITIVE):** the pursue-condition fired (Q3-negative, §4.4), and the extension ran as a $0 analysis on the trial's gate-passing base legs — results and E6 role now recorded in §4.7; full record + reproducing harness: umbrella `spec/analysis_2026_07_19_dispersion_pool_47.md`. Still a **different production mechanism** (a live decorrelated reference, not a transferred constant), still outside the four-question core; `mode: dispersion` remains unpromoted pending the fresh on-pair sweep (§4.8).

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
| Gemma ⁷ | `google/gemma-4-31B` | `google/gemma-4-31B-it` (HF→vLLM) | 31.3B dense | ~63 GB | 1×H200-141 (or 2×A100-SXM4 TP2) | **cheap** |
| Llama | `meta-llama/Llama-4-Maverick-17B-128E` | `…-Instruct` (OpenRouter→DeepInfra) | 400B / 17B MoE | ~800 GB | 8×H200-141 (TP8+EP; 8×H100=640 GB does NOT fit — fixed 2026-07-19) | mid |
| GLM | `zai-org/GLM-4.5-Base` | `zai-org/GLM-4.5` (OpenRouter→Z.AI) | ~355B / 32B MoE | ~710 GB | 8×H200 | mid |
| DeepSeek | `deepseek-ai/DeepSeek-V4-Pro-Base` | `deepseek-ai/DeepSeek-V4-Pro` (native) | ~1.6T / 49B MoE | ~3.2 TB | multi-node (≫16×H100) | giant |
| Mistral | `mistralai/Mistral-Large-3-675B-Base-2512` | `…-Instruct-2512` (native) | ~675B / 41B MoE (+2.5B vision) | ~1.35 TB | 16×H100 / 8×B200 | giant |
| Kimi | `moonshotai/Kimi-K2-Base` | `moonshotai/Kimi-K2-Thinking` (OpenRouter→Novita) | ~1T / 32B MoE | ~2 TB | 16×H200 / 24×H100 | giant |

⁷ **Gemma is the seventh model (added 2026-07-02; checkpoint swapped 2026-07-19).** It became eligible
when the collaborative-evaluation pair switched back to **Anthropic + GPT**, freeing Google/Gemini from
the reserved-evaluator role. The original checkpoint, the MoE `gemma-4-26B-A4B`, ran the full cheap-trial
leg on 2026-07-19 and **failed the accuracy gate** (base argmax 0.167 — significantly below even the
0.333 majority-class floor, z = −3.87, per the corrected gate definition in §"Load-bearing caveat";
resolution ≈ noise; τ_oc pegged at the
20.0 bound) — replaced by the **dense `gemma-4-31B`** (+`-it`), 31.3B bf16 ≈ 63 GB, ungated (verified via
HF API 2026-07-19), sized like Qwen (1×H200-class, not an 80 GB card). The GT **annotator** seat remains
`gemma-4-26B-A4B-it` — the seat-7 corpus collection is history and does not move. Corrections folded in: **Mistral-Large-3 is MoE**
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

**RE-DERIVED 2026-07-19 at 0.85 gpu-util** (the protocol pin; usable VRAM = N×card×0.85, need =
weights + 8×measured-KV for `--workers 8`). Measured k=5 prompt budgets (2026-07-19,
`scripts/measure_prompt_budget.py`, per-family tokenizer): Llama required ctx **21,949**
(KV 4.0 GB/seq), GLM **22,279** (KV 7.8 GB/seq) — both inside the 32768 pin; Llama's HF gate
re-verified under the current token. Market snapshot 2026-07-19: exactly ONE qualifying 8×H200
offer ($33.69/hr, rel 0.999, 5.9 Gbps, 15 TB disk) — **8×H200 availability is thin; expect to
wait for an offer or relax terms, and re-survey before scheduling.** Wall-clock model per
mid-tier family: download ~20–40 min/leg (~0.7 GB/s) + load ~10 min + elicitation ~10–25 min/leg
at workers=8; two-box base∥post halves calendar time at equal cost.

| Family | bf16 weights (base+post) | GPUs (@0.85) | wall-clock (both legs) | cost (vast spot) |
|---|---|---|---|---|
| Qwen 35B-A3B | 0.14 TB | 1×H200 | ~1 h | **~$5–8** (RAN: ≈$8) |
| Gemma 4-31B | 0.13 TB | 1×H200 (or 2×A100-SXM4) | ~1.5 h | **~$8–12** (RAN: ≈$10) |
| Llama-4 400B | 1.6 TB | **8×H200** (need 840 GB of 958; 8×H100 does NOT fit) | ~2–2.5 h | ~$60–90 |
| GLM-4.5 355B | 1.4 TB | **8×H200** (need 798 GB of 958) | ~2–2.5 h | ~$55–85 |
| Mistral 675B | 2.7 TB | **12×H200 ⇒ 2×8-node MULTI-NODE** (1394 GB > any single node; 8×B200@179 GB = 1217 GB also short) | re-derive | re-derive |
| Kimi-K2 1T | 4.0 TB | **18×H200 ⇒ 3-node MULTI-NODE** (2048 GB) | re-derive | re-derive |
| DeepSeek V4-Pro ~1.6T | ~6.4 TB | **28×H200 ⇒ 4-node MULTI-NODE** (3256 GB) | re-derive | re-derive |
| **Phase-2 subtotal (Llama+GLM, the tiebreaker set)** | | | | **~$115–175 (+~25% failure margin ⇒ ~$145–220)** |

⚠ **Every giant is now multi-node at bf16** — no single vast node (incl. 8×B200) holds ≥675B.
vLLM multi-node (pipeline-parallel across boxes) on vast is **UNVERIFIED for this workload**;
Phase 3 needs its own feasibility study (networking, NCCL across hosts, vast cluster support)
**before** any giant rental — which is gated behind the Phase-2 stopping rule anyway.

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
- Where `logprobs` exists (OpenRouter: qwen3.5-35b-a3b, llama-4-maverick, deepseek-v4-pro, kimi-k2-thinking; gemma-4-31b is served on vLLM so it has logprobs natively), also collect a token-sliced post distribution (same letters) so pre vs post is measured in the **same channel** — removes the verbalized-vs-logit confound.

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
- **Decision rule:** if `τ_oc` clusters tightly *and* the bases clear the accuracy gate, adopt `median(τ_oc)` as the transferred constant for the closed evaluators. `study_a.calibration_block()` emits a drop-in `pipeline.yaml → calibration` block (`mode: temperature`; written to `runs/<run>/pipeline_calibration_block.json`). **Seam note:** the merged `pipeline.yaml` calibration seam is **global** (applied at the Phase-1 gleaning site to whichever families run, no `family_id`). At evaluation time those families are the two **closed** evaluators (now **Claude + GPT**) — the open annotator raters run only at *construction* and are never calibrated here — so for the intended case (two closed families + one clustered constant) pasting the block into the top-level `calibration` key is **adequate**. **Family-scoping is an optional refinement**, needed only if τ_oc doesn't cluster (per-family T), if an arm runs a different evaluator pair, or to move the correction to Phase-3; it is specified in `docs/integration_remediation_2026_07_01.md` (a separate, approved evaluator change, relevant only at this Phase-4 decision). Else, report negative. **§4.8 (E6) extends this Q4 check into the rescoped core paper's demonstration leg** — run it off the same Phase-4 artifacts. **OUTCOME (2026-07-19): Q3-NEGATIVE — the adoption path is CLOSED.** Gate-passing τ_oc = {qwen 1.60, llama31 1.86, gemma31 4.88} (GLM 8.84 soft — base marginal on the resolution gate): max/min 3.05 > the adopted ≤2 stopping rule ⇒ `median(τ_oc)` is never adopted and any emitted `pipeline_calibration_block.json` is do-not-paste. Q4 survives in **range-robust** form: on the off-pair legacy run, *every* τ in the open-panel range improves Reliability without Resolution/RPS damage — benefit band (1.00, 22.8] ⊇ [1.60, 8.84], doc-clustered bootstrap coverage 0.946 (`spec/analysis_2026_07_19_q4_range_robustness.md`, umbrella; sweep instrument `scripts/q4_range_robustness.py`, merged to develop). E6's mechanism is therefore the supervised T\* + pre-registered sensitivity band — see §4.8 (updated).

### 4.7 Decorrelated dispersion (exploratory extension — see §0)
Add the base-model distributions to the dispersion replicate pool for the closed evaluators (a *decorrelated, well-calibrated* reference, fixing the correlated-overconfidence blindness) and re-run `dispersion_calibration_recovery` on the existing `stage9-gemini-gpt-medium` / `phase23-deference-fix-native` runs. **Pitfall:** do not match the evaluator's *width* to a base model's width (a well-calibrated weak model is appropriately wide; copying it over-widens). Use base disagreement only as *added dispersion*. **OUTCOME (2026-07-19): RAN, POSITIVE CONTROL** — the 3-base pool recovers 66–97% of the supervised RPS improvement on both legacy closed runs and reproduces the supervised family ordering exactly on the four open post variants (Kendall +1.0); magnitude stays basin-scale, entropy matching blind to location error. Per the paper (develop `f7b5706`) it enters E6 as the **GT-free triangulator** — concurrence = an unsaturated fit landing inside the on-pair benefit band; divergence is published, not reconciled. `mode: dispersion` remains NOT promoted (needs the fresh on-pair sweep).

### 4.8 E6 — the distributional-utility demonstration leg (2026-07-17 addendum)

**Context.** The rescoped judex-core paper carries an experiment battery proving
the material distinction between discrete and distributional labels (umbrella
`spec/analysis_2026_07_17_distributional_utility_experiment_battery.md`; this
is its **E6**). Q4 already checks that the closed-pair correction improves
Murphy Reliability without destroying Resolution/RPS — per the measured Q3
negative (§4.4 outcome) that correction is now the supervised T\* + sensitivity
band of E6.2, not a transferred constant; E6 extends that check into the
paper-grade exhibit. **No new machinery** — E6 is analysis
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
- **E6.2 — Correction mechanism = the accuracy-gated held-out supervised T\*
  on the closed pair, wrapped in the pre-registered sensitivity band
  [1.60, 8.84].** (REWRITTEN 2026-07-19 — supersedes the transferred
  `median(τ_oc)` mechanism, which died with Q3: measured spread 3.05× > the
  ≤2 rule. The paper carries the same mechanism: judex-paper develop
  `f7b5706`.) The supervised fit is admissible *because* it is
  accuracy-gated — the peg-at-19 failure was the Gemini-era accuracy
  deficit's artifact, and the off-pair fit now lands unsaturated (3.5585).
  Not DACA (abandoned — §0); not τ_DACA (FAILED validation 2026-07-19:
  filter attenuation growing with the estimand + reference
  non-exchangeability; retired to a published negative,
  `spec/analysis_2026_07_19_tau_daca_triangulation.md`); GT-free dispersion
  enters only as the §4.7 decorrelated-pool **triangulator** (basin-scale
  read-out; concurrence = an **unsaturated** fit landing inside the on-pair
  band — a pegged fit is a boundary artefact, never concurrence). Band criteria
  frozen before the run: Reliability strictly improves, RPS no worse,
  Resolution within 10% of uncalibrated; grid + doc-clustered bootstrap
  exactly per `scripts/q4_range_robustness.py` (merged — the frozen
  instrument). Off-pair evidence: benefit band (1.00, 22.8] ⊇ [1.60, 8.84],
  coverage 0.946 (`spec/analysis_2026_07_19_q4_range_robustness.md`) — the
  method transfers, the numbers do not. If the on-pair band fails to cover
  the panel range, E6 reports that negative (publish-the-null discipline).
- **E6.3 — The discrete-blindness exhibit.** Temperature scaling is
  argmax-preserving, so argmax accuracy and quadratic-weighted κ vs GT are
  **bit-identical pre/post correction on tie-free items** — the invariance
  row is pre-registered **up to exact top-two ties** (2026-07-19: exactly
  tied top-two credences flip on floating-point tie-break through the logit
  round-trip — 2–3/120 cells with 0.35/0.35 ties on the off-pair run, even
  at T=1; score the row on tie-free items, report tied items separately) —
  verify mechanically while REL, coverage, and ΔH move. Every discrete-label
  metric is provably blind to the entire intervention; this is the paper's
  "invisible quality axis" demonstration.
- **E6.4 — Downstream endpoint deltas.** Re-score the battery's E2 routing
  signals (prediction entropy, pair W1, Δ_res) and E3 decision-cost endpoints
  pre/post the E6.2 correction (supervised T\*, band-swept — no τ_oc constant
  exists to apply), under the battery's endpoint discipline (outcome endpoints
  adjudicate; RPS/REL/coverage are diagnostics only). Doc-clustered bootstrap
  (resample the 24 docs, §4.6 pattern) for every CI.
- **E6.5 — SUPERSEDED (2026-07-19): absorbed into E6.2.** The held-out
  supervised fit this row proposed as "optional secondary" is now the
  *primary* mechanism (E6.2), and fitted-vs-transferred convergence is moot —
  there is no transferred constant. Its replacement triangulation row is the
  §4.7 decorrelated-dispersion GT-free fit, scored at basin scale as
  concurrence-inside-the-band. The firewall carries over unchanged (risk 6:
  split/CV only; never tune-on-test for reported numbers).

**Cost.** One fresh closed-pair sweep, **~$290–330** (the "$0 if
`stage9-sweep-sonnet-gpt-v2` is vintage-valid" branch is closed — see the
correction above; that run is off-pair, not merely stale). **Gates to
pre-register before running:** the E6.3 invariance row must be exact **on
tie-free items** (tie-aware rule — see E6.3); the E6.1→post REL improvement
must clear a doc-clustered CI; the **on-pair benefit band must cover the
panel range [1.60, 8.84]** (criteria + grid + bootstrap frozen per E6.2);
E6.4 deltas are reported win-or-null.

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
   **RESOLVED 2026-07-18 (user-directed, not silent):** the paper-side rewrite
   is committed on `judex-paper` branch `claude-calibration-mechanism-rewrite`
   (`2df3e9e`, unmerged): §subsec:calibration Stage 2 now specifies transferred
   `median(τ_oc)` as primary under the four gates (accuracy floor, `T*_pre ≈ 1`
   premise, cross-family clustering, no saturated fit), with the held-out
   supervised fit and an **agreement-filtered RPS alignment to the base
   references (τ_DACA — DACA's filter, top-1 objective replaced by the ordinal
   Brier/RPS)** demoted to triangulating estimators; §protocol E6 step (ii)
   updated to match, with an explicit gate-failure null clause. The τ_DACA
   estimator is implemented in this repo (`study_a.fit_tau_daca` /
   `daca_triangulation`, branch `claude-tau-daca-triangulation`) — free
   analysis on Q4 + base-leg artifacts, no new inference.
   **CLOSED 2026-07-19: the paper and this guide now specify the SAME
   mechanism.** The measured outcomes ran the null clause: Q3-negative
   (spread 3.05×) killed the transferred constant, and τ_DACA failed
   validation (retired, published negative). The paper-side chain is MERGED
   to `judex-paper` develop `f7b5706` (`64fa438` null-clause invocation →
   E6 = supervised T\* + pre-registered sensitivity band; `dda0bec` §4.7
   pool as E6's GT-free triangulator; `f7b5706` tie-aware invariance gate +
   band composition caveat). §4.8 above was rewritten to match — this
   dependency is settled, in the *opposite* direction from the 07-18 note:
   the transferred-`median(τ_oc)` mechanism this guide once defended is the
   one that died.
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
│   └── decorrelated_dispersion.py # (unbuilt) §4.7 evaluator wiring — the 2026-07-19 analysis ran from the umbrella spec/ harness instead; build only if mode:dispersion is ever promoted
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
| **1** | **The two-family cheap trial, leg 1: Qwen 35B** (base+post, **bf16, all 120 cells, reasoning ON — PAID**; the first *real* measurement, distinct from the free int4 `--limit`/`--no-reason` Mac smoke in 0a whose numbers are discarded); run §4.4 validity check on a post model with logprobs. `--family qwen --out runs/qwen`. | ~$5 | Pipeline green; token-slicing valid; Qwen clears the **corrected** accuracy gate (majority-floor + resolution legs, §"Load-bearing caveat") — RAN 2026-07-19: passes on resolution (0.0332), argmax 0.350 ≈ the 0.333 floor. |
| **1b** | **Trial leg 2: Gemma 4-31B** — the other cheap pair (ungated; Google lineage, so the trial spans two distinct lineages; post leg served plain — no reasoning parser). **History:** the original 26B-A4B ran this leg 2026-07-19 and FAILED the gate (base argmax 0.167 < chance; τ_oc pegged at 20 — see `runs/gemma` @26B), forcing the swap to the dense 31B; re-collect in a fresh dir. Run back-to-back with Phase 1 as ONE trial (quickstart §12), then **merge**: `--merge qwen=runs/qwen gemma=runs/gemma31` → the first two-family `τ_oc` spread before any mid/giant spend. | ~$8–12 | **RAN 2026-07-19, k=5 canonical (`runs/trial_cheap_k5`): both families PASS the corrected gate** (Qwen 0.458/res 0.0456 → τ_oc 1.60; Gemma-31B 0.425/res 0.0457 → τ_oc 4.88, its post +res but T≈12). **Q3 at 2 families: NOT clustered — spread 3.28 (3×), far outside the ±20% τ_oc noise band.** k=4 runs archived (`runs/trial_cheap_k4`) as the protocol ablation: the k=5 E-exemplar fix lifted argmax +8–17pp on every leg of both families. |
| **2** | Add **Llama-4-Maverick + GLM-4.5** (both mid-tier, same 8×H200-class hardware — GLM at 355B ≈ 710 GB bf16 rides the Llama box class). **The Q3 TIEBREAKER PHASE** — the cheap tier split 3× (Qwen 1.60 / Gemma 4.88). **ADOPTED stopping rule (user, 2026-07-19): the gate decision is taken over the FOUR families {Qwen, Gemma, Llama, GLM} — adopt `median(τ_oc)` only if max/min ≤ 2 across the gate-passing set; if the 4-family spread exceeds it, report Q3-negative and SKIP the giants** (saves ~$500+). GLM is in the tiebreaker set (not deferred to Phase 3) per the user's psychometric rationale: on a separate analysis GLM sits ~halfway between Llama-4 and Gemma-4-26B on rating consistency and ~halfway between Llama-4 and Qwen-3.5 on strictness/leniency, so its τ_oc position is maximally informative about clustering. ⚠ **Sizing bug**: the §2 table's "8×H100-80" (640 GB) does NOT hold 400B bf16 ≈ 800 GB at 0.85 util — needs 8×H200 (or 16×H100); re-derive the whole mid/giant VRAM/cost table at 0.85 before renting. | ~$145–220 both families (re-derived 2026-07-19 incl. failure margin; see the §2 cost table) | Corrected gate passed per family; the 4-family spread decides: within rule → Phase 3; outside → Q3-negative, stop. **OUTCOME (2026-07-19 evening): RAN — Q3-NEGATIVE, stop.** Maverick's base FAILED the gate (pegged, reference-degenerate) → user-directed swap Llama-4 → **Llama-3.1-405B** (dense; base PASSES 0.392 / res 0.0393) → τ_oc **1.86**; GLM-4.5 τ_oc **8.84** but base only marginal on the resolution gate (soft point; its post is the best judge at 0.675). Gate-passing spread {1.60, 1.86, 4.88}: **max/min 3.05 > 2** ⇒ giants SKIPPED (see the sibling handoff `spec/handoff_2026_07_19_gemma26_to_31b_replacement.md` for both swaps). |
| **3** | Commit to the **three giants** (DeepSeek V4-Pro, Mistral-Large-3, Kimi-K2; bf16, multi-node; GLM moved up to Phase 2 as part of the tiebreaker set, 2026-07-19). Full seven-family Q1/Q2/Q3. **Reached ONLY if the Phase-2 stopping rule passes.** **OUTCOME (2026-07-19): NOT REACHED — the Phase-2 stopping rule failed (Q3-negative); the ~$600+ is saved.** | ~$600+ (re-derive at 0.85) | — |
| **4** | Decision: adopt `median(τ_oc)` transferred constant (paste the emitted block into `judex-evaluator/configs/pipeline.yaml` → `calibration`) or report negative. **Refuse adoption if `tau_oc_any_saturated`** — a pegged τ_oc is a boundary artefact, not a fit. Optional, gated *on* Q3 being negative: the §4.7 decorrelated-dispersion extension. **OUTCOME (2026-07-19): NEGATIVE — nothing pasted, nothing adoptable.** The Q3-negative branch's live outputs both RAN: the §4.7 extension (positive control → E6's GT-free triangulator) and the Q4 range-robustness sweep (benefit band ⊇ the panel range, coverage 0.946) — together they re-frame E6 as supervised T\* + sensitivity band (§4.6/§4.8 outcomes). | $0 | — |
| **4b** | **E6 — the distributional-utility demonstration leg** (§4.8): fresh on-pair closed sweep, then E6.1–E6.4 + the §4.7 pool-triangulator read-out off those artifacts (E6.5 absorbed into E6.2). Not $0 — no existing 120-cell run uses the current pair. | ~$290–330 | Q1–Q3 answered (Q3 negative — E6.2 carries the supervised T\* + band mechanism); E6 gates incl. the tie-aware invariance rule and the triangulator read-out pre-registered before the sweep runs. |

---

## 8. Risks & honest caveats

1. **Accuracy gate may moot the study** — if the open models are also inaccurate on EU-AI-Act compliance, every `T*` pegs (as it did for Gemini/GPT). Phase 0 + Phase 1 are designed to fail cheap.
2. **fp8 quantization confound** — fp8 corrupts the logits we measure; prefer bf16 on the base leg, or bound the effect via §4.4.
3. **Model availability/sizes** — the 2026 base checkpoints and exact architectures must be verified at download; sizing/cost shifts if they differ.
4. **Transfer to closed evaluators is an assumption** — tested only indirectly (cross-family clustering + applying `median(τ_oc)` to the closed pair (Claude/GPT) on AIReg and checking the Murphy Reliability drop). More grounded than DACA's, not a proof. **Update 2026-07-19: the point-constant reading is dead** (Q3-negative); the assumption survives only in *band* form — the off-pair range-robustness (benefit band ⊇ the open-panel range, coverage 0.946) is the indirect evidence, and it is re-tested on-pair as a pre-registered E6 gate (§4.8). Its edges are a property of the measured pair's miscalibration: the method transfers, the numbers never do.
5. **Base-model prompt sensitivity** — base models are format-fragile; few-shot count/wording affects the token-sliced distribution. Hold the few-shot block fixed across families; treat it as part of the measurement instrument.
6. **Calibration/validation firewall** — AIReg is the validation set. A single transferred scalar T for *production* (future docs) is legitimate; for *reporting AIReg numbers* fit T on a held-out split / CV to avoid tuning-on-test.
7. **The post panel and the corpus store are now aligned (2026-07-03)** — the `judex-corpus` few-shot exemplar store is the **7-rater** build (Gemma collected as the seventh seat via OpenRouter/Novita bf16), and the GT leaf/dimension labels were re-fit on the same 7-rater panel. The Study A panel and the corpus annotation panel are the same seven families; any *future* panel change still requires a deliberate re-pin plus a corpus rebuild kept separate from this study. **Update 2026-07-14 — the adopted vintage is corpus v2:** the same 7-seat panel re-annotated the Grok-4.5 re-authored (`ambiguity_structured`) excerpts under contract 0.2.0 — **1757 rows (1449 leaf + 308 dimension)** in `judex_leaf_exemplar_construction_v2/`; `fewshot.DIMENSION_STORE` points there. v2 dimension exemplars are **~14× longer** (mean ~9.7k chars), so the live prompt budget was re-measured (`scripts/measure_prompt_budget.py`): worst-case prompt ≈ **18.3k tokens**, required context **≈20.4k** at `--budget 2048` ⇒ `--max-model-len` **≥ 24576, standard pin 32768** (all SEVEN families measured and fit — llama completed 2026-07-14 after its HF gate access was granted). The v2 few-shot block is part of the measurement instrument (risk 5) — the SAME per-Article k=4 blocks are held fixed across all seven families.

---

## 9. Immediate next step

**(Updated 2026-07-19 — Phases 0–2 have RUN; Q3 is NEGATIVE; Phase 3 is skipped; Phase 4 reported negative.)**
The one remaining paid leg is **Phase 4b: the E6 on-pair sweep** (~$290–330, user-gated): a fresh
Claude+GPT 120-cell run under `--config-dir configs_v2exemplars`, then E6.1–E6.4 off its artifacts
with the §4.8 gates exactly as pre-registered (supervised T\* + sensitivity band; tie-aware
invariance row; band-coverage endpoint; `scripts/q4_range_robustness.py` as the frozen sweep
instrument). Free precursors already done: the §4.7 pool positive control and the off-pair
range-robustness sweep. Before the paid run, the only open hygiene item is the
`calibration_block()` clustering guard (so a non-clustered `median(τ_oc)` can never emit an
adoptable-looking block).
