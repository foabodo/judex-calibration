# JUDEX — project context (auto-loaded by Claude Code)

Distilled project knowledge so a Claude Code instance — on the user's Mac **or** on a rented vast.ai
GPU box — starts with the same understanding. Full detail lives in the repo docs (pointers below).

## Study A workflows — never cross the recipes
The same driver (`scripts/run_qwen_phase1.py`) runs all of them; the flags/host/model decide which.
Only the VAST LIVE recipe produces the study's numbers; the LOCAL PILOT (third recipe, added
2026-07-14) produces labeled pilot evidence about the phenomenon, never the panel constant:
- **MAC SMOKE** — free plumbing test: one small **int4/Q4** stand-in (e.g. Qwen3-4B) on **llama.cpp /
  Metal**, `--limit N --no-reason`. Validates the machinery only — **τ_oc is MEANINGLESS; discard it,
  never paste its calibration block.** One run dir per family (`--family qwen|gemma|... --out
  runs/<family>`); `--merge qwen=runs/qwen gemma=runs/gemma --out runs/merged` gives a real
  multi-lineage smoke (e.g. Qwen3-4B + Gemma-3-4B — genuinely different families, so `tau_oc_summary`
  has an actual spread, not one family repeated). (`docs/local_smoke_quickstart.md`)
- **VAST LIVE** — the real, paid experiment: the **seven panel** base+post pairs on **vLLM**, **bf16**,
  **all 120 cells**, **reasoning ON** (omit `--limit/--no-reason`). Its τ_oc **is** the study output.
  (`docs/vast_quickstart.md` · `docs/vast_claude_code_orchestration.md`)
- **LOCAL fp16 SCIENCE PILOT** — free 4B-pair pilot of the *phenomenon* (not the panel constant):
  **fp16 ggufs** (never int4), all 120 cells, reasoning ON, native `llama-server`, family tags
  `gemma3-4b`/`qwen3-4b` (never the panel keys). Not smoke-flagged — but its calibration block is
  **never integrated**, and its τ_oc is pilot evidence only (fp16/Metal channel; 4B accuracy-gate
  caveats). (`docs/local_smoke_quickstart.md` §2·Mac-D · `scripts/run_local_f16_pilot.sh`)

Runs that set `--limit`/`--no-reason` (or analyse <120 cells) are auto-flagged `smoke` in
`study_a_report.json` and the calibration block, and print `[SMOKE] … do NOT paste into pipeline.yaml`.

## What JUDEX is
Distributional LLM-as-judge for EU AI Act technical-file compliance. The evaluator emits a full
probability distribution over a **5-level ordinal compliance scale** (a Type-C credence), scored
against a distributional ground truth with optimal-transport metrics (Wasserstein / RPS). Four repos,
git submodules of the `judex` umbrella:
- `judex-corpus` — 24-PDF corpus + the **7-rater** leaf/dimension exemplar store, **corpus-v2
  vintage** (adopted 2026-07-14): `leaf_exemplars/judex_leaf_exemplar_construction_v2/`, **1757
  rows: 1449 leaf + 308 dimension** — every v1 excerpt re-authored 1:1 by Grok 4.5 in the
  `ambiguity_structured` style and re-annotated from scratch by the same 7-seat panel under
  contract 0.2.0. The **dimension store is the few-shot source** (`fewshot.DIMENSION_STORE`); v2
  dimension exemplar texts are ~13× longer than v1 (mean 9.7k vs 756 chars), which re-sized the live
  prompt budget (see the vast section below). The v1 tree (`judex_leaf_exemplar_construction/`)
  is the archived baseline — never rebuild or edit it. Store probabilities are on the **0.05
  elicitation grid**; the AIReg GT below is **continuous** — the two sides of the instrument sit on
  different supports (harmless for the base leg, which emits only a letter A–E).
- `judex-ground-truth` — Bayesian MG-MFRM labels; the canonical AIReg-Bench ground truth.
  **Re-materialized 2026-07-09** (after this repo's original verification): freethresh thresholds,
  **readout temperature τ = 0.675**, grid snap OFF (continuous). Moved the labels by **W1 mean
  0.194** with **0/120 mode flips** — argmax-derived numbers are stable, every distribution-fitting
  number is not. Full provenance + the R-hat/validation caveats: `aireg.py` module docstring;
  pinned against drift by `tests/test_integration_seams.py::GtVintageGuardTests`.
- `judex-evaluator` — the runtime + scoring + calibration tooling (Study A reuses it verbatim).
  Two config bundles exist: `configs/` (v1 exemplars, **the default**) and `configs_v2exemplars/`
  (corpus-v2 exemplars, selected per run with `--config-dir configs_v2exemplars`). Study A's
  few-shot is corpus-**v2**, so a Q4/E6 closed-pair run must pass that flag or the closed leg is
  framed on v1 exemplars while the open legs are framed on v2.
- `judex-calibration` — **Studies A + B** (this repo): open pre/post-pair temperature
  calibration, measured in two channels (Study A token-logit, Study B verbalized).

## Current state — verbalized-first (2026-07-21)
The calibration contribution is rebuilt on **Study B** (the verbalized/production channel);
Study A is retained as the **channel-selection negative result** that motivated it. Adoption
panel **{qwen 1.281, gemma31 1.025, glm 1.025, maverick 1.380}** (gemma31 = chat-template
collection; llama31 measured but excluded pre-hoc, numerically inert): clustering ratio
**1.346 ≤ 2 PASSES** (vs 3.05 in the logit channel), transferred judge temperature
**T_J = 1.153**, sensitivity band **[1.03, 1.38]**. Governing protocol =
**`docs/e6_onpair_decision_protocol_r3.md` (r3, ADOPTED 2026-07-21)** — r2 retired with its
channel, retained unedited as the logit-era record; never reuse its constants (band
[1.60, 4.88], anchor 2.794, λ 0.35). Architecture + R0 evidence:
`docs/verbalized_calibration_plan.md` + `docs/verbalized_r0_analyses_2026_07_21.md`
(+ `scripts/verbalized_reframe_r0.py`). Post-run driver for the sweep:
`scripts/e6_r3_arms.py` (arms 1/2/3a/3b; arm 4 = `study_b.analyze_confidence` on extracted
per-seat confidence). Study B data: `runs/study_b_*` (gitignored, this Mac);
results of record `docs/study_b_results_2026_07_20.md`. Both papers carry the contribution
(synthesis-validation v6 `4fc1e94`; core v3 `4eca63d`). Next paid step = the on-pair E6
sweep (~$445, user-gated; umbrella `spec/runbook_2026_07_20_e6_onpair_sweep.md`).

## Study A (what this repo does — historical record; see "Current state" above)
Measure the *clean post-training overconfidence temperature* by running the **base (pre)** and
**instruct (post)** variants of open models on the 120 AIReg-Bench cells (independent human GT), to
decide whether a **transferred constant temperature** can correct the **closed** JUDEX evaluators
(**Anthropic + GPT** — switched back from Gemini/GPT 2026-07-02, which freed Google to join the
annotators). Questions: **Q1** is the base calibrated (`T*_pre ≈ 1`)? **Q2** post overconfidence
`τ_oc`? **Q3** is `τ_oc` stable across families? **Q4** does applying it to the closed pair (Claude/GPT)
on AIReg improve
Murphy **reliability** without hurting resolution/RPS?
**Q3 OUTCOME (2026-07-19): NEGATIVE — no transferable constant.** Gate-passing τ_oc
{qwen 1.60, llama31 1.86, gemma31 4.88} (GLM's τ_oc EXCISED 2026-07-20 — its base leg fails
the resolution-primary gate, so the value is a bad-reference artifact; umbrella
`spec/amendment_2026_07_20_band_glm_excision.md`): max/min 3.05 > the
adopted ≤2 rule ⇒ `median(τ_oc)` is NEVER adopted, the emitted calibration block is
do-not-paste, and the giants phase is SKIPPED per the stopping rule.
**E6 (guide §4.8):** Q4 extends into the rescoped judex-core paper's
distributional-utility demonstration — pre/post overconfidence measurement, the
argmax-invariance ("discrete metrics are blind") exhibit (**tie-aware**: exact top-two
ties flip on fp tie-break, score the row on tie-free items), and downstream
routing/decision-cost deltas. **Mechanism SUPERSEDED 2026-07-21**: the logit-era
supervised-T\*-plus-band mechanism (band [1.60, 4.88], amendment
`spec/amendment_2026_07_20_band_glm_excision.md`; benefit band (1.00, 22.8], coverage
0.965) is retired with its channel — the correction is now the transferred verbalized
constant T_J = 1.153 under protocol r3 (see "Current state"). τ_DACA failed validation
in the logit channel (published negative; retried in-channel report-only per r3); the
decorrelated-dispersion pool survives re-based to verbalized bases (r3 F6).
On-pair sweep decision logic = **`docs/e6_onpair_decision_protocol_r3.md` (ADOPTED
2026-07-21)** — one source of truth; do not restate the ladder elsewhere. r2
(`docs/e6_onpair_decision_protocol.md`) is the retained logit-era record.
Current closed pair: **Sonnet 4.6 (medium effort) + GPT 5.4 (medium reasoning)**
(evaluator families `anthropic_claude_medium` + `openai_gpt_standard`);
Haiku 4.5 / GPT-5.4-mini ruled out (insufficiently capable on the task).
**No existing 120-cell run realizes that pair** (verified 2026-07-18): `stage9-sweep-sonnet-gpt-v2`
is `anthropic_claude` + `openai_gpt` — its GPT seat is **gpt-5.4-mini**, a ruled-out model, and it
predates contract 0.2.0 and corpus-v2. `stage9-claude-gpt-medium` *is* the right pair but covers
3 docs / 15 items. So **Q4 and E6 need a fresh on-pair sweep (~$445, re-quoted
2026-07-20 after the preflight cache measurement)** — not $0.

Pipeline (this repo, `src/judex_calibration/`):
- `aireg.load_cells()` → 120 cells with the **canonical, manifest-verified** GT, reproducible from the
  git-tracked `judex-ground-truth` bundle. **Never read GT from a `runs/` metrics_report** — those are
  gitignored and can be stale.
- `fewshot.build_fewshot_by_criterion()` → k=4 per-Article few-shot from the corpus store,
  **firewall-disjoint** from AIReg.
- `elicit_base` → `/v1/completions` token-slice over A–E; **server-agnostic** logprobs parsing
  (vLLM, llama.cpp, and chat shapes) so both hosts work — but **llama.cpp/Metal (Mac) is the SMOKE
  plumbing only (result discarded); vLLM on vast (bf16) is the real measurement.**
- `study_a` → Q1–Q4 + `calibration_block()` (a drop-in `judex-evaluator/configs/pipeline.yaml →
  calibration` block, `mode: temperature` — note the `configs/` segment). The evaluator drops
  `provenance` under that mode, so keep `pipeline_calibration_block.json` as the audit trail.
- Driver: `scripts/run_qwen_phase1.py` — same script, flags pick the workflow. **SMOKE:** `--limit N
  --no-reason` (fast, result discarded). **LIVE:** omit both (all 120 cells, reasoning ON). `--out`,
  `--base-url/--post-url`, `--base-model/--post-model`, `--analyze-only` re-scores an existing `--out`.
  Legs **checkpoint per cell** (atomic write) and **resume** by skipping cached cells — crash
  recovery ONLY: a `<leg>.meta.json` sidecar pins (model, reason, budget) and a mismatched resume
  hard-errors; use a fresh `--out` for a different experiment.

## The 7-model panel (`configs/models.yaml`)
deepseek-v4-pro · mistral-large-2512 · qwen3.5-35b-a3b · llama-4-maverick · glm-4.5 · kimi-k2-thinking
· **gemma-4-31B** (dense; swapped 2026-07-19 from the MoE 26B-A4B after it failed the cheap-trial
accuracy gate — base argmax 0.167 < chance, τ_oc pegged at 20; the GT *annotator* seat stays
gemma-4-26B-A4B-it) — each a base+post pair. **Google is now included** (added 2026-07-02): the
collaborative-evaluation pair switched back to **Anthropic + GPT**, so Gemini is no longer a reserved
evaluator and the annotator↔evaluator firewall no longer bars Google. The firewall now only bars
**Anthropic and OpenAI** models from the annotator panel (they are the reserved evaluators).

## Environment (machine-dependent)
- **On the user's Mac:** use the **`judex-arm` conda env**
  (`/Users/fabodo/anaconda3/envs/judex-arm/bin/python`). The user does **not** use venvs. Secrets come
  from the macOS Keychain (`security find-generic-password -s <name>-api-key -w`).
- **On a rented vast box:** use the repo **`.venv`** (created by `scripts/provision_claude_code.sh`);
  no Keychain — secrets arrive via env vars. vLLM serves on `127.0.0.1:8000`.

## Running the vast experiment (see `docs/vast_claude_code_orchestration.md` §5 for the full brief)
- **bf16** for the real run — *not* fp8/int4 (quantization perturbs the very logits the study measures).
  int4/Q4 is fine **only for the throwaway Mac smoke (llama.cpp), never on the rented GPU** — on vast,
  always bf16.
- `--gpu-memory-utilization 0.85` — the standard pin (changed from 0.92 on 2026-07-19): the driver's
  echo fallback triggers an fp32 log-softmax over the full vocab × the ~18.3k-token prompt (~2.5 GB
  transient at a 262k vocab) which OOM-killed the engine at 0.92; 0.85 costs only KV the sequential
  driver never uses.
- **Multi-GPU TP is the sanctioned budget fallback, SXM/NVLink pairs ONLY** — PCIE pairs hang at NCCL
  init; a c10d rendezvous timeout means a bad host (re-rent a different machine_id, don't debug).
  Prefer a single ≥141 GB card when the price gap is ≲$1/hr — failed multi-GPU inits cost more than
  the card premium. `vastai create instance` can leave a box created-but-stopped: nudge
  `vastai start` once (provision_vast.sh does this).
- **k=5 few-shot is the protocol** (2026-07-19, user directive; models.yaml `fewshot_k: 5`): one
  exemplar per compliance level — k=4's stratified round-robin silently omitted `very_high` (E)
  from every prompt and cost 8–17pp argmax on every trial leg. `--fewshot-k` overrides per run and
  is pinned in the leg meta sidecar (mismatched resume hard-errors).
- **`--workers 8` is the panel standard** (2026-07-19): concurrent cells let vLLM batch — a leg
  drops from ~45 to ~2–10 min. Per-cell distributions are chaotic under bf16 CoT jitter with or
  without batching; aggregates hold within ~±10–15% per-leg T ⇒ **single-measurement τ_oc ≈ ±20%**
  (the documented noise band — see the guide's "Load-bearing caveat").
- **Gate is resolution-primary** (2026-07-19): Murphy resolution > 0 with margin is the capacity
  criterion (chance-proof, recalibration-invariant); argmax vs the 0.333 majority-class floor is a
  reported diagnostic only (the 0.20 uniform floor was misspecified — GT argmax marginal is
  imbalanced).
- `--max-model-len 32768` — the standard pin; **≈ 23.3k required** under corpus v2 + k=5 few-shot
  (measured 2026-07-19; re-measure per family tokenizer before provisioning). Measured
  2026-07-14 (`scripts/measure_prompt_budget.py`, each family's own tokenizer): worst-case live
  prompt ≈ **18.3k tokens** (k=4 v2 few-shot ≈7k + evidence ≈10k + criterion/scaffold), plus the
  2048-token CoT `--budget` ⇒ **≈20.4k required**; 32768 keeps headroom for a raised budget on
  thinking post legs. KV at this length is small next to weights (1.3–6.9 GB/seq bf16 across the
  panel), so the pre-v2 box sizing stands. Logprobs must be enabled.
- Family serving quirks: **Mistral-Large-3** ships Mistral-native format (`params.json`, no
  `config.json`) — serve with `--config-format mistral --load-format mistral --tokenizer-mode
  mistral` (its HF `tokenizer.json` also over-counts vs the tekken tokenizer vLLM actually uses).
  **Llama-4-Maverick is HF-gated** — access for the `hf-token` account was granted + verified
  2026-07-14 (measured: max prompt 17,270 tok, required 19,355, ceilings 262k/1M — fits; the
  gate is per-account, so re-check before provisioning under a different token).
- **Keep vLLM alive across commands** — run it **detached** (its own `tmux` window / `nohup` + PID),
  never as a tracked background task (Claude Code kills those ~5 s after a `-p` run ends).
- `runs/` is gitignored → persist `study_a_report.json` + `pipeline_calibration_block.json` (scp, or a
  force-added `vast-run-*` branch) **before teardown**. **Do not destroy the box yourself** — the user
  holds the kill-switch.

## Integrity invariants (do not break)
- AIReg-Bench is the **validation** set — never inject it as few-shot (few-shot is corpus-only, disjoint).
- The canonical AIReg GT bundle currently validates as **`unavailable`** on **two** hard-failure
  families — `prior_predictive` (not computed) **and `convergence_rhat`** (max R-hat 1.0123 > 1.01;
  ESS passes at 558). Safe to *use*, but must not be described as a "validated benchmark".
- **Temperatures are fit on Study A's own range** `T_BOUNDS = (0.25, 20.0)`, passed explicitly to
  every evaluator fitter. Do not drop the argument: `judex.calibration`'s default is `(0.25, 4.0)`,
  which silently censored `T_rps` at 4.0 while `T_rel`/`τ_oc` ran to 20 (every fp16-pilot leg pegged
  there). Any temperature on a boundary is a peg, not a fit — `study_a.saturated()` flags it and the
  report/calibration block carry `*_saturated`; a saturated `τ_oc` must never be adopted.
- Numbers fit against the pre-2026-07-09 GT are **not** carryable. Worked example: the recorded
  sanity fit `T_rps 2.4434` on `stage9-gemini-gpt-medium` re-derives to **3.5585** on the current
  bundle (+46%; verified bound-independent). Re-derive, never carry forward.
- `develop` is the integration branch in every repo (never create `integration`).

## Docs
`docs/study_a_implementation_guide.md` (the plan) · `docs/vast_quickstart.md` (provision a box) ·
`docs/vast_claude_code_orchestration.md` (run Claude Code on the box) ·
`docs/local_smoke_quickstart.md` (free Mac plumbing check with one small int4 model — machinery only; τ_oc meaningless).
