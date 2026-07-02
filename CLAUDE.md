# JUDEX — project context (auto-loaded by Claude Code)

Distilled project knowledge so a Claude Code instance — on the user's Mac **or** on a rented vast.ai
GPU box — starts with the same understanding. Full detail lives in the repo docs (pointers below).

## What JUDEX is
Distributional LLM-as-judge for EU AI Act technical-file compliance. The evaluator emits a full
probability distribution over a **5-level ordinal compliance scale** (a Type-C credence), scored
against a distributional ground truth with optimal-transport metrics (Wasserstein / RPS). Four repos,
git submodules of the `judex` umbrella:
- `judex-corpus` — 24-PDF corpus + the **6-rater** leaf/dimension exemplar store (the few-shot source).
- `judex-ground-truth` — Bayesian MG-MFRM labels; the canonical AIReg-Bench ground truth.
- `judex-evaluator` — the runtime + scoring + calibration tooling (Study A reuses it verbatim).
- `judex-calibration` — **Study A** (this repo): open pre/post-pair temperature calibration.

## Study A (what this repo does)
Measure the *clean post-training overconfidence temperature* by running the **base (pre)** and
**instruct (post)** variants of open models on the 120 AIReg-Bench cells (independent human GT), to
decide whether a **transferred constant temperature** can correct the **closed** JUDEX evaluators
(Gemini/GPT). Questions: **Q1** is the base calibrated (`T*_pre ≈ 1`)? **Q2** post overconfidence
`τ_oc`? **Q3** is `τ_oc` stable across families? **Q4** does applying it to Gemini/GPT on AIReg improve
Murphy **reliability** without hurting resolution/RPS?

Pipeline (this repo, `src/judex_calibration/`):
- `aireg.load_cells()` → 120 cells with the **canonical, manifest-verified** GT, reproducible from the
  git-tracked `judex-ground-truth` bundle. **Never read GT from a `runs/` metrics_report** — those are
  gitignored and can be stale.
- `fewshot.build_fewshot_by_criterion()` → k=4 per-Article few-shot from the corpus store,
  **firewall-disjoint** from AIReg.
- `elicit_base` → vLLM `/v1/completions` token-slice over A–E; **server-agnostic** logprobs parsing
  (vLLM, llama.cpp, and chat shapes) — so a local Metal server (Mac) and vLLM (vast) both work.
- `study_a` → Q1–Q4 + `calibration_block()` (a drop-in `pipeline.yaml → calibration` block,
  `mode: temperature`).
- Driver: `scripts/run_qwen_phase1.py` (`--base-url/--post-url --base-model/--post-model --out
  --limit N --no-reason --analyze-only`).

## The 6-model panel (`configs/models.yaml`)
deepseek-v4-pro · mistral-large-2512 · qwen3.5-35b-a3b · llama-4-maverick · glm-4.5 · kimi-k2-thinking
— each a base+post pair. **Google is deliberately excluded** (Gemini is a reserved *evaluator* — the
annotator↔evaluator firewall). Never add a Google model to the panel.

## Environment (machine-dependent)
- **On the user's Mac:** use the **`judex-arm` conda env**
  (`/Users/fabodo/anaconda3/envs/judex-arm/bin/python`). The user does **not** use venvs. Secrets come
  from the macOS Keychain (`security find-generic-password -s <name>-api-key -w`).
- **On a rented vast box:** use the repo **`.venv`** (created by `scripts/provision_claude_code.sh`);
  no Keychain — secrets arrive via env vars. vLLM serves on `127.0.0.1:8000`.

## Running the vast experiment (see `docs/vast_claude_code_orchestration.md` §5 for the full brief)
- **bf16** for the real run — *not* fp8/int4 (quantization perturbs the very logits the study measures;
  int4 is fine only for a throwaway plumbing smoke).
- `--max-model-len ≥ 16384` (AIReg prompts are ~14k tokens); logprobs must be enabled.
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
`docs/local_smoke_quickstart.md` (validate the pipeline on a small model first).
