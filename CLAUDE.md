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
  dimension exemplar texts are ~14× longer than v1 (mean ~9.7k chars), which re-sized the live
  prompt budget (see the vast section below). The v1 tree (`judex_leaf_exemplar_construction/`)
  is the archived baseline — never rebuild or edit it.
- `judex-ground-truth` — Bayesian MG-MFRM labels; the canonical AIReg-Bench ground truth.
- `judex-evaluator` — the runtime + scoring + calibration tooling (Study A reuses it verbatim).
- `judex-calibration` — **Study A** (this repo): open pre/post-pair temperature calibration.

## Study A (what this repo does)
Measure the *clean post-training overconfidence temperature* by running the **base (pre)** and
**instruct (post)** variants of open models on the 120 AIReg-Bench cells (independent human GT), to
decide whether a **transferred constant temperature** can correct the **closed** JUDEX evaluators
(**Anthropic + GPT** — switched back from Gemini/GPT 2026-07-02, which freed Google to join the
annotators). Questions: **Q1** is the base calibrated (`T*_pre ≈ 1`)? **Q2** post overconfidence
`τ_oc`? **Q3** is `τ_oc` stable across families? **Q4** does applying it to the closed pair (Claude/GPT)
on AIReg improve
Murphy **reliability** without hurting resolution/RPS?

Pipeline (this repo, `src/judex_calibration/`):
- `aireg.load_cells()` → 120 cells with the **canonical, manifest-verified** GT, reproducible from the
  git-tracked `judex-ground-truth` bundle. **Never read GT from a `runs/` metrics_report** — those are
  gitignored and can be stale.
- `fewshot.build_fewshot_by_criterion()` → k=4 per-Article few-shot from the corpus store,
  **firewall-disjoint** from AIReg.
- `elicit_base` → `/v1/completions` token-slice over A–E; **server-agnostic** logprobs parsing
  (vLLM, llama.cpp, and chat shapes) so both hosts work — but **llama.cpp/Metal (Mac) is the SMOKE
  plumbing only (result discarded); vLLM on vast (bf16) is the real measurement.**
- `study_a` → Q1–Q4 + `calibration_block()` (a drop-in `pipeline.yaml → calibration` block,
  `mode: temperature`).
- Driver: `scripts/run_qwen_phase1.py` — same script, flags pick the workflow. **SMOKE:** `--limit N
  --no-reason` (fast, result discarded). **LIVE:** omit both (all 120 cells, reasoning ON). `--out`,
  `--base-url/--post-url`, `--base-model/--post-model`, `--analyze-only` re-scores an existing `--out`.
  Legs **checkpoint per cell** (atomic write) and **resume** by skipping cached cells — crash
  recovery ONLY: a `<leg>.meta.json` sidecar pins (model, reason, budget) and a mismatched resume
  hard-errors; use a fresh `--out` for a different experiment.

## The 7-model panel (`configs/models.yaml`)
deepseek-v4-pro · mistral-large-2512 · qwen3.5-35b-a3b · llama-4-maverick · glm-4.5 · kimi-k2-thinking
· **gemma-4-26B-A4B** — each a base+post pair. **Google is now included** (added 2026-07-02): the
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
- `--max-model-len 32768` — the standard pin; **≥ 24576 is required** under corpus v2. Measured
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
- The canonical AIReg GT bundle currently validates as **`unavailable`** (prior_predictive not computed):
  safe to *use*, but must not be described as a "validated benchmark".
- `develop` is the integration branch in every repo (never create `integration`).

## Docs
`docs/study_a_implementation_guide.md` (the plan) · `docs/vast_quickstart.md` (provision a box) ·
`docs/vast_claude_code_orchestration.md` (run Claude Code on the box) ·
`docs/local_smoke_quickstart.md` (free Mac plumbing check with one small int4 model — machinery only; τ_oc meaningless).
