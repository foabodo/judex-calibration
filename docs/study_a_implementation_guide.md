# Study A — Open Pre/Post-Pair Calibration: Implementation Guide

**Repo:** `judex/judex-calibration` (new) · **Driver:** Claude Code on local macOS · **Compute:** rented remote GPU (vLLM)
**Status:** plan / runbook. No spend until Phase 0 gate passes.

---

## 0. Why this exists (carry-over from the calibration thread)

We need **overconfidence-correcting recalibration for the closed JUDEX evaluators (Claude, GPT) on future out-of-sample documents with no ground truth**. Three GT-free routes were exhausted:

- **DACA** — wrong objective (top-1 ECE), argmax-agreement filter discards ambiguous cells, and no open base sibling for Claude/GPT. Abandoned.
- **Internal permutation-gleaning dispersion** — merged to `judex-evaluator` develop as infra (`mode: noop`), but on the live runs it fits **T ≈ 1.0 (identity)** because the two evaluators are *correlated and confidently-wrong-in-agreement*, so their own re-draws don't reveal the overconfidence.
- **Supervised temperature scaling** — works (Murphy Reliability 0.028 → 0.0045) but T pegs at ~19 and is wildly per-doc unstable, because the accuracy deficit (44–52% argmax) makes RPS-min *and* Reliability-min both flatten to the marginal. **Objective swap does not help in isolation** — verified.

**What the open pre/post pairs uniquely add:** the pre-trained variant is (assumed) well-calibrated; the post-trained variant is overconfident. Running *both* on AIReg-Bench (which has independent human GT) lets us measure the **clean post-training overconfidence**, holding base capability fixed — the one signal neither the internal dispersion nor the accuracy-contaminated supervised fit can give. The base models also serve as a **decorrelated reference** to fix the dispersion-≈-identity failure.

### Scientific questions
- **Q1 (premise check):** Is `T*_pre ≈ 1` per family on this task? If the bases are *not* well-calibrated on EU-AI-Act compliance, the whole premise is task-invalid — better to learn this for ~$10 on the cheap model first.
- **Q2 (transferability):** Measure the clean post-training overconfidence temperature `τ_oc` per family. Do the six `τ_oc` **cluster**? Tight cluster ⇒ overconfidence is roughly model-agnostic in temperature units ⇒ a **transferred constant T for the closed evaluators is justified, and we learn its value**. Scatter ⇒ it isn't.
- **Q3 (decorrelated dispersion):** Does adding base-model distributions to the dispersion replicate pool make `dispersion_calibration_recovery` (already in `judex-evaluator`) recover the supervised optimum for the closed evaluators?

### Load-bearing caveat (the accuracy gate)
If the open models are *also* inaccurate on AIReg, their `T*` will peg too and the study is uninformative — same failure as the closed evaluators. **Phase 0 is a free accuracy pre-check that gates all spend.**

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

> **VERIFY at download time** — exact param counts, architectures, and HF repo availability for the 2026 checkpoints below are best-estimate; confirm with `huggingface-cli` and the model card before provisioning. Sizes drive everything.

| Family | Base repo (HF) | Post (API) | ~Total / active | bf16 weights | Min GPU (bf16) | Tier |
|---|---|---|---|---|---|---|
| Qwen | `Qwen/Qwen3.5-35B-A3B-Base` | `Qwen/Qwen3.5-35B-A3B` (OpenRouter) | 35B / 3B MoE | ~70 GB | 1×H100-80 | **cheap** |
| Llama | `meta-llama/Llama-4-Maverick-17B-128E` | `…-Instruct` (OpenRouter→DeepInfra) | 400B / 17B MoE | ~800 GB | 8×H100-80 | mid |
| GLM | `zai-org/GLM-4.5-Base` | `zai-org/GLM-4.5` (OpenRouter→Z.AI) | ~355B / 32B MoE | ~710 GB | 8×H200 | mid |
| DeepSeek | `deepseek-ai/DeepSeek-V4-Pro-Base` | `deepseek-ai/DeepSeek-V4-Pro` (native) | ~671B / 37B MoE | ~1.3 TB | 16×H100 / 8×B200 | giant |
| Mistral | `mistralai/Mistral-Large-3-675B-Base-2512` | `…-Instruct-2512` (native) | ~675B | ~1.35 TB | 16×H100 / 8×B200 | giant |
| Kimi | `moonshotai/Kimi-K2-Base` | `moonshotai/Kimi-K2-Thinking` (OpenRouter→Novita) | ~1T / 32B MoE | ~2 TB | 16×H200 / 24×H100 | giant |

**Serving notes**
- MoE: use vLLM `--enable-expert-parallel` with `--tensor-parallel-size`/`--pipeline-parallel-size`. Giants need **2 nodes** (pipeline-parallel) or B200-class single node.
- **Quantization confound:** fp8 halves the cluster but perturbs exactly the logits we measure. Calibration is logit-shape-sensitive. **Prefer bf16 for the base leg.** If fp8 is unavoidable for the giants, run the §4.4 token-slice-validity check to bound the quant effect before trusting `T*_pre`.
- Inference is *tiny* (120 cells × a few short forward passes). **GPU cost is dominated by download + load time, not inference** — so minimize wall-clock on the box (pre-stage weights to a persistent volume, serve, run, tear down).

---

## 3. Cost estimates

GPU spot rates (2026, rough): H100-80 ≈ $2.5/GPU-hr, H200 ≈ $3.5, B200 ≈ $5.5. Per-model wall-clock = weight download + load + 120-cell run (+buffer).

| Family | GPUs × hrs (bf16) | Est. base-serving cost |
|---|---|---|
| Qwen 35B | 1×H100 × 2h | **~$5** |
| Llama 400B | 8×H100 × 3h | ~$60 |
| GLM 355B | 8×H200 × 3h | ~$85 |
| DeepSeek 671B | 16×H100 × 4h | ~$160 |
| Mistral 675B | 16×H100 × 4h | ~$160 |
| Kimi 1T | 16×H200 × 5h | ~$280 |
| **Base total (bf16)** | | **~$750** |
| fp8 on the 3 giants instead | | ~$450–500 (with quant-confound risk) |
| Post API (6 models × 120 cells, single-pass) | | ~$20–80 (reasoning models cost more) |
| **Study A total** | | **~$800–1,200** (×1.5 buffer for re-runs → up to ~$1.5k) |

**Minimum-viable path: ~$70.** Run **Phase 0 (free)** then **Qwen ($5) + Llama ($60)** end-to-end first. If those two families validate the pipeline *and* clear the accuracy gate, commit to the giants; if not, stop having spent ~$70 instead of ~$1k.

Storage: HF download egress is free; use the GPU provider's NVMe scratch (usually included). A persistent volume to cache the giants between sessions is optional (~$0.10/GB-mo — a 2 TB cache ≈ $200/mo; only if re-running).

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

### 4.6 Cross-family stability & decision (Q2)
- Plot/serialize the six `(T*_pre, T*_post, τ_oc, accuracy)`.
- **Document-clustered bootstrap** (reuse the pattern; resample the 24 docs) CIs on `τ_oc` and on the cross-family spread.
- **Decision rule:** if `τ_oc` clusters tightly *and* the bases clear the accuracy gate, adopt `median(τ_oc)` as the transferred constant for the closed evaluators (config flip in `judex-evaluator/pipeline.yaml`, the seam already merged). Else, report negative.

### 4.7 Decorrelated dispersion (Q3)
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
│   ├── study_a.py                 # per-family fits, τ_oc, cross-family stability
│   └── decorrelated_dispersion.py # Q3 wiring into judex-evaluator
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
Allowlist the recurring read-only/local calls to cut prompts: `huggingface-cli`, `ssh <box>`, `curl http://localhost:8000/*`, `vllm` (on the box via ssh), `python scripts/*`, and the Keychain pattern `security find-generic-password -s *-api-key -w`. (See `/fewer-permission-prompts`.)

### 6.2 Background serving pattern
`vllm serve` is long-running. Launch it on the box and keep Claude Code free:
- Start serving in the background (Claude Code `Bash` with `run_in_background: true`, command = `ssh <box> 'vllm serve … '`), poll `curl …/health` until ready, then elicit.
- Or a tmux session on the box; Claude Code polls health and streams the run.

### 6.3 Per-family runbook (`scripts/run_family.sh <family>`)
1. Provision box (or reuse), `huggingface-cli download <base_repo>` to NVMe.
2. `vllm serve` (bf16) in background; wait for `/health`.
3. Open SSH tunnel `localhost:8000 → box:8000`.
4. `python -m judex_calibration.elicit_base --family <f> --out runs/<run-id>/`.
5. `python -m judex_calibration.elicit_post --family <f>` (API from Mac; keys via Keychain).
6. `python -m judex_calibration.study_a --family <f>` → per-family `T*_pre/T*_post/τ_oc/accuracy/murphy`.
7. **Tear down the box** (cost is wall-clock).

### 6.4 Secrets
Post-model API keys stay in macOS Keychain (`security find-generic-password -s <provider>-api-key -w`, exported in-shell, never printed). HF token likewise (`huggingface-cli login` on the box, or `HF_TOKEN` from Keychain).

### 6.5 Memory / handoff
Record run-ids, fitted `τ_oc`, the Q1/Q2/Q3 verdicts, and any negative results in a `judex-calibration` project memory + a `spec/handoff_*.md`, consistent with the existing JUDEX cadence.

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

1. **Accuracy gate may moot the study** — if the open models are also inaccurate on EU-AI-Act compliance, every `T*` pegs (as it did for Claude+GPT). Phase 0 + Phase 1 are designed to fail cheap.
2. **fp8 quantization confound** — fp8 corrupts the logits we measure; prefer bf16 on the base leg, or bound the effect via §4.4.
3. **Model availability/sizes** — the 2026 base checkpoints and exact architectures must be verified at download; sizing/cost shifts if they differ.
4. **Transfer to closed evaluators is an assumption** — tested only indirectly (cross-family clustering + applying `median(τ_oc)` to Claude/GPT on AIReg and checking the Murphy Reliability drop). More grounded than DACA's, not a proof.
5. **Base-model prompt sensitivity** — base models are format-fragile; few-shot count/wording affects the token-sliced distribution. Hold the few-shot block fixed across families; treat it as part of the measurement instrument.
6. **Calibration/validation firewall** — AIReg is the validation set. A single transferred scalar T for *production* (future docs) is legitimate; for *reporting AIReg numbers* fit T on a held-out split / CV to avoid tuning-on-test.
7. **The post panel will change** — per the user, adopting these six replaces some current JUDEX post-trained raters. Re-pin the exemplar/annotator panel deliberately; keep the calibration corpus rebuild separate from this study.

---

## 9. Immediate next step

Run **Phase 0** (free) — it costs nothing, reuses data already on disk, and decides whether any GPU spend is justified. Everything downstream is gated on it.
