# Study A — Open Pre/Post-Pair Calibration: Implementation Guide

**Repo:** `judex/judex-calibration` (new) · **Driver:** Claude Code on local macOS · **Compute:** rented remote GPU (vLLM)
**Status:** plan / runbook. No spend until Phase 0 gate passes.

---

## 0. Why this exists (carry-over from the calibration thread)

We need **overconfidence-correcting recalibration for the closed JUDEX evaluators (Gemini, GPT) on future out-of-sample documents with no ground truth**. Three GT-free routes were exhausted:

- **DACA** — wrong objective (top-1 ECE), argmax-agreement filter discards ambiguous cells, and no open base sibling for Gemini/GPT. Abandoned.
- **Internal permutation-gleaning dispersion** — merged to `judex-evaluator` develop as infra (`mode: noop`), but on the live runs it fits **T ≈ 1.0 (identity)** because the two evaluators are *correlated and confidently-wrong-in-agreement*, so their own re-draws don't reveal the overconfidence.
- **Supervised temperature scaling** — works (Murphy Reliability 0.028 → 0.0045) but T pegs at ~19 and is wildly per-doc unstable, because the accuracy deficit (44–52% argmax) makes RPS-min *and* Reliability-min both flatten to the marginal. **Objective swap does not help in isolation** — verified.

**What the open pre/post pairs uniquely add:** the pre-trained variant is (assumed) well-calibrated; the post-trained variant is overconfident. Running *both* on AIReg-Bench (which has independent human GT) lets us measure the **clean post-training overconfidence**, holding base capability fixed — the one signal neither the internal dispersion nor the accuracy-contaminated supervised fit can give. The base models also serve as a **decorrelated reference** to fix the dispersion-≈-identity failure.

### Scientific questions (answered in order)

Study A is **not** the generic JUDEX calibration arm. That arm fits temperatures against GT for models we control. Study A tests whether a **portable constant** derived from open hybrid pairs is scientifically defensible for **closed** models (Gemini/GPT) whose base variants are inaccessible.

- **Q1 — Premise validation:** are base/pretrained models well-calibrated on EU-AI-Act ordinal compliance, i.e. is their supervised temperature fit `T*_pre ≈ 1`? If not, the premise is task-invalid — learn this for ~$10 on the cheap model first.
- **Q2 — Post-training effect:** for each post-trained model, how much temperature softening (`τ_oc`) is needed to recover calibration **while retaining the post-training accuracy gain**?
- **Q3 — Transfer legitimacy:** is `τ_oc` **stable across model families**? Tight clustering ⇒ overconfidence is roughly model-agnostic in temperature units ⇒ a transferred constant for the closed evaluators is defensible, and we learn its value. Scatter ⇒ it is not.
- **Q4 — Closed-side sanity check:** when the open-derived constant is applied to Gemini/GPT JUDEX outputs on AIReg, does the **Murphy Reliability** term improve **without destroying Resolution or RPS**?

### Load-bearing caveat (the accuracy gate)
If the open models are *also* inaccurate on AIReg, their `T*` will peg too and the study is uninformative — same failure as the closed evaluators. **Phase 0 is a free accuracy pre-check that gates all spend.**

### Exploratory extension (out of core scope; free once base elicitations exist)
- **Decorrelated dispersion:** the base models are also *decorrelated, well-calibrated* references, so adding their distributions to the GT-free dispersion replicate pool could fix the dispersion-≈-identity failure (`dispersion_calibration_recovery` in `judex-evaluator`, §4.7). This is a **different production mechanism** (a live reference, not a transferred constant), so it is deliberately outside Study A's four-question core — pursue only if Q3 shows `τ_oc` is unstable (i.e. a single constant won't do).

---

## 1. Architecture

```
  ┌──────────────────────── local macOS (Claude Code) ────────────────────────┐
  │  judex-calibration repo · orchestration · analysis · reuses judex-evaluator │
  │     calibration.py (fit_temperature, fit_dispersion_temperature,            │
  │     dispersion_calibration_recovery, murphy_decomposition) + AIReg GT       │
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
- **Post (post-trained) leg:** API from the Mac, exactly as the existing JUDEX rater path. Where the provider exposes `logprobs` (OpenRouter: qwen3-235b-thinking, llama-4-maverick, deepseek-v4-pro, kimi-k2-thinking — confirmed), we also collect a token-sliced post distribution so pre vs post is measured in the *same* channel.
- **Analysis** runs entirely local, reusing the merged `judex-evaluator` calibration tooling.

---

## 2. Model inventory & serving feasibility

> **VERIFY at download time** — exact param counts, architectures, and HF repo availability for the 2026 checkpoints below are best-estimate; confirm with `hf` and the model card before provisioning. Sizes drive everything.

| Family | Base repo (HF) | Post (API) | ~Total / active | bf16 weights | Min GPU (bf16) | Tier |
|---|---|---|---|---|---|---|
| Qwen | `Qwen/Qwen3.5-35B-A3B-Base` | `Qwen/Qwen3.5-35B-A3B` (OpenRouter) | 35B / 3B MoE | ~70 GB | 1×H100-80 | **cheap** |
| Llama | `meta-llama/Llama-4-Maverick-17B-128E` | `…-Instruct` (OpenRouter→DeepInfra) | 400B / 17B MoE | ~800 GB | 8×H100-80 | mid |
| GLM | `zai-org/GLM-4.5-Base` | `zai-org/GLM-4.5` (OpenRouter→Z.AI) | ~355B / 32B MoE | ~710 GB | 8×H200 | mid |
| DeepSeek | `deepseek-ai/DeepSeek-V4-Pro-Base` | `deepseek-ai/DeepSeek-V4-Pro` (native) | ~671B / 37B MoE | ~1.3 TB | 16×H100 / 8×B200 | giant |
| Mistral | `mistralai/Mistral-Large-3-675B-Base-2512` | `…-Instruct-2512` (native) | ~675B | ~1.35 TB | 16×H100 / 8×B200 | giant |
| Kimi | `moonshotai/Kimi-K2-Base` | `moonshotai/Kimi-K2-Thinking` (OpenRouter→Novita) | ~1T / 32B MoE | ~2 TB | 16×H200 / 24×H100 | giant |

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
min bf16 fit (671–675B and 1T do **not** fit 16×H100 → H200).

| Family | bf16 weights (base+post) | GPUs | wall-clock (both legs) | cost (vast spot) |
|---|---|---|---|---|
| Qwen 35B-A3B | 0.14 TB | 1×H100 | ~1 h | **~$5** |
| Llama-4 400B | 1.6 TB | 8×H100 | ~2.5 h | ~$40 |
| GLM-4.5 355B | 1.4 TB | 8×H200 | ~2.5 h | ~$50 |
| DeepSeek 671B | 2.7 TB | 16×H200 | ~3.7 h | ~$165 |
| Mistral 675B | 2.7 TB | 16×H200 | ~3.7 h | ~$165 |
| Kimi-K2 1T | 4.0 TB | 16×H200 | ~4.7 h | ~$210 |
| **Both-legs subtotal** | | | | **~$635** |
| **+ ×1.4 buffer** (failed offers, slow CDN, re-runs) | | | | **~$890** |

**Range: ~$650 (vast spot + fast CDN, no re-runs) → ~$1,500 (on-demand rates + slow CDN + buffer);
plan ~$900–1,100.** Download throughput and spot pricing are the two big swings.

**Cheaper alternative (~$550–800):** keep the **post leg on OpenRouter** (verbalized, no GPU) and
GPU-serve only the **6 base** variants from HF (≈$480 +buffer ~$670, + post API ~$50). Halves the
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
- GT: `judex-evaluator/data/external/aireg_bench/judex_annotations/distributional_annotations.csv` (3 human raters, distributional `p_1..p_5`) → reconcile to the item-level GT distribution exactly as `judex-evaluator` already does (reuse its loader). **AIReg is the VALIDATION set; never used as ICL.**

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
- Where `logprobs` exists (OpenRouter: qwen3-235b-thinking, llama-4-maverick, deepseek-v4-pro, kimi-k2-thinking), also collect a token-sliced post distribution (same letters) so pre vs post is measured in the **same channel** — removes the verbalized-vs-logit confound.

### 4.4 Token-slice validity check (the original "Phase 1")
For the post models that expose **both** verbalized and logprob channels, compare `INV_SOFTMAX(verbalized)` vs token-sliced logits (W1/KL). This empirically tests whether the two channels agree — the assumption the whole base-vs-post comparison rests on. Run it before trusting cross-channel comparisons.

### 4.5 Per-family fits (reuse `judex-evaluator/calibration.py`)
For each family, against AIReg GT, fit **both** objectives (we proved neither is a free lunch under low accuracy, so report both):
- `T*_pre`  — `fit_temperature(pre_dist, gt)` (RPS-min) and the reliability-min variant.
- `T*_post` — same for the post distributions.
- `τ_oc` — the clean post-training overconfidence temperature: the T that maps **post → pre calibration** (or `T*_post` directly if Q1 confirms `T*_pre ≈ 1`).
- Always alongside: **argmax accuracy** per model (the gate) and the **Murphy decomposition** (is the addressable error Reliability or Resolution?).

### 4.6 Cross-family stability & decision (Q3 + Q4)
- Plot/serialize the six `(T*_pre, T*_post, τ_oc, accuracy)`.
- **Document-clustered bootstrap** (reuse the pattern; resample the 24 docs) CIs on `τ_oc` and on the cross-family spread.
- **Decision rule:** if `τ_oc` clusters tightly *and* the bases clear the accuracy gate, adopt `median(τ_oc)` as the transferred constant for the closed evaluators (config flip in `judex-evaluator/pipeline.yaml`, the seam already merged). Else, report negative.

### 4.7 Decorrelated dispersion (exploratory extension — see §0)
Add the base-model distributions to the dispersion replicate pool for the closed evaluators (a *decorrelated, well-calibrated* reference, fixing the correlated-overconfidence blindness) and re-run `dispersion_calibration_recovery` on the existing `stage9-gemini-gpt-medium` / `phase23-deference-fix-native` runs. **Pitfall:** do not match the evaluator's *width* to a base model's width (a well-calibrated weak model is appropriately wide; copying it over-widens). Use base disagreement only as *added dispersion*.

---

## 5. Repo structure (`judex-calibration`)

```
judex-calibration/
├── README.md
├── pyproject.toml                 # depends on judex-evaluator (editable sibling)
├── .gitignore                     # runs/, weights/, .venv/, *.nc
├── configs/
│   ├── models.yaml                # the §2 inventory (base repo, post id, serving args)
│   └── serving.yaml               # GPU profiles, vllm flags, tunnel config
├── src/judex_calibration/
│   ├── serving.py                 # provision/serve helpers, health checks
│   ├── elicit_base.py             # vLLM token-slicing (§4.2)
│   ├── elicit_post.py             # API verbalized + logprob (§4.3)
│   ├── aireg.py                   # AIReg cell + GT loader (reuse judex-evaluator)
│   ├── study_a.py                 # Q1/Q2 per-family fits, τ_oc; Q3 cross-family stability; Q4 closed_side_check
│   └── decorrelated_dispersion.py # exploratory-extension wiring into judex-evaluator
├── scripts/
│   ├── phase0_accuracy_precheck.py   # FREE gate (existing AIReg LLM annotations)
│   ├── run_family.sh                 # one family end-to-end (serve→elicit→teardown)
│   └── analyze.py                    # aggregate + report
├── runs/                          # per-run artifacts (gitignored bulk, manifests kept)
└── tests/
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
| **0** | **Free accuracy pre-check** — parse the 10 existing AIReg-Bench LLM annotations vs human GT; compute argmax accuracy. | $0 | If even frontier models (o3/gpt5/sonnet/gemini-pro) score ~low, the accuracy gate is structural → **fix accuracy/elicitation first; do NOT spend.** |
| **1** | Stand up repo + pipeline on **Qwen 35B** (base, bf16) end-to-end; run §4.4 validity check on a post model with logprobs. | ~$5 | Pipeline green; token-slicing valid; Qwen clears accuracy gate. |
| **2** | Add **Llama-4-Maverick** (mid). Two-family `τ_oc` + cross-family check. | ~$60 | `τ_oc` plausible & accuracy adequate on ≥2 families. |
| **3** | Commit to the **three giants + GLM** (bf16, multi-node). Full six-family Q1/Q2/Q3. | ~$700 | — |
| **4** | Decision: adopt `median(τ_oc)` transferred constant (config flip in `judex-evaluator`) or report negative; run Q3 decorrelated-dispersion. | $0 | — |

---

## 8. Risks & honest caveats

1. **Accuracy gate may moot the study** — if the open models are also inaccurate on EU-AI-Act compliance, every `T*` pegs (as it did for Gemini/GPT). Phase 0 + Phase 1 are designed to fail cheap.
2. **fp8 quantization confound** — fp8 corrupts the logits we measure; prefer bf16 on the base leg, or bound the effect via §4.4.
3. **Model availability/sizes** — the 2026 base checkpoints and exact architectures must be verified at download; sizing/cost shifts if they differ.
4. **Transfer to closed evaluators is an assumption** — tested only indirectly (cross-family clustering + applying `median(τ_oc)` to Gemini/GPT on AIReg and checking the Murphy Reliability drop). More grounded than DACA's, not a proof.
5. **Base-model prompt sensitivity** — base models are format-fragile; few-shot count/wording affects the token-sliced distribution. Hold the few-shot block fixed across families; treat it as part of the measurement instrument.
6. **Calibration/validation firewall** — AIReg is the validation set. A single transferred scalar T for *production* (future docs) is legitimate; for *reporting AIReg numbers* fit T on a held-out split / CV to avoid tuning-on-test.
7. **The post panel will change** — per the user, adopting these six replaces some current JUDEX post-trained raters. Re-pin the exemplar/annotator panel deliberately; keep the calibration corpus rebuild separate from this study.

---

## 9. Immediate next step

Run **Phase 0** (free) — it costs nothing, reuses data already on disk, and decides whether any GPU spend is justified. Everything downstream is gated on it.
