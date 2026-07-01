# judex-calibration ↔ role-repo integration remediation (2026-07-01)

**Context.** After `judex-corpus` / `judex-ground-truth` / `judex-evaluator` were stabilized
(develop `dab1db2` / `fcf17a9` / `cb71d05`), this pass identified and closed the development
gaps between `judex-calibration` (Study A) and the three repos it depends on, so the vLLM
experiments run against the *current* develop state and results integrate back cleanly.
Scope: `judex-calibration` code/docs + umbrella acknowledgment. No paid experiments were run.

## Integration audit — 5 points

| # | Point | Status found | Action |
|---|---|---|---|
| 1 | Evaluator calibration API (`study_a.py` imports) | **intact** | none (verified: 18/18 `MetricItemResult` fields, all 5 symbols, smoke ran 120 cells) |
| 2 | AIReg GT source (`aireg.py`) | **stale + unreproducible** | **rewritten** to canonical, manifest-verified, git-tracked GT |
| 3 | Corpus few-shot | **declared but not wired** (toy stub) | **new `fewshot.py`** loader; `run_qwen_phase1` wired |
| 4 | Result-integration seam | **partial** (global seam only) | study_a **emits a drop-in block**; evaluator family-scoping **specified** (§ Seam) |
| 5 | Umbrella acknowledgment | **docs omission** | umbrella README/Makefile updated |

## What was rectified (code)

### #2 — `aireg.py`: canonical, reproducible GT (the material fix)
- **Before:** GT (and evidence, and the item→doc join) were read from `judex-evaluator/runs/stage9-gemini-gpt-medium/metrics_report.json` — a **gitignored** run (absent on a fresh clone / the remote box) whose embedded GT predated the 2026-06-30 cumulative-consistency re-fit.
- **After:** `load_cells()` synthesizes the 120-cell GT through the evaluator's **manifest-verified** `synthesize_aireg_bench_ground_truth(judex-ground-truth/data/distributional_labels/, LABELS)`; the item_label→doc join is reconstructed from git-tracked `judex-corpus/step3_4/ground_truth.json`; evidence from git-tracked `judex-corpus/step5/documents/<doc>/{application.md,data.md}`; criterion text from `judex-evaluator/configs/rubric.yaml`.
- **Verified:** new GT == canonical barycenter to **1.4e-17**; **differs from the stale run by mean L1 0.567** (argmax identical 120/120 — the staleness was silent). The end-to-end effect is real: fitting `T*` against canonical vs stale GT gives **T_rps 2.44 vs 4.0** on the Gemini/GPT smoke. 120/120 cells load on the current tracked artifacts alone.

### #3 — `fewshot.py`: real k-shot from the corpus store
- **Before:** `run_qwen_phase1.py` hardcoded one toy `DEFAULT_FEWSHOT` (article_10 / answer A) for *every* cell, despite `models.yaml` declaring `fewshot_source: judex-corpus/leaf_exemplars, fewshot_k: 4`.
- **After:** `build_fewshot_by_criterion(cells)` draws `k` (from `models.yaml`) exemplars **per Article** from the git-tracked corpus `dimension_exemplar_store.json`, mirroring the evaluator's stratified `fixed_set` selection (round-robin over levels in scale order, least-used rater at each step), and renders each as `(excerpt, Article criterion, single-letter A–E answer)`. `elicit_base.run_variant` now accepts a per-cell `fewshot` callable.
- **Verified:** 5 per-Article blocks at k=4, ordinal ramp across distinct raters; **firewall disjoint** from AIReg (`source_item_label ∩ AIReg item_label = ∅`); deterministic.

### #4a — `study_a.calibration_block()`: drop-in evaluator config
- `study_a.calibration_block(report)` turns a `cross_family()` result into a ready-to-merge `pipeline.yaml → calibration` block (`{mode: "temperature", temperature: median(τ_oc), provenance: {...}}`), mirroring `judex.experiments.dispersion_calibration_config`'s shape. It uses `mode: "temperature"` (a fixed transferred constant is a supervised-derived scalar) — **not** `mode: "dispersion"`, which would stamp a false `gt_free_dispersion_fit` provenance. `run_qwen_phase1.py` writes it to `runs/<run>/pipeline_calibration_block.json`.

### Portability
- Removed the hardcoded `/Users/fabodo/...` absolute paths in `aireg.py`, `study_a.py`, `phase0_accuracy_precheck.py`; all now resolve the sibling repos relative to the repo root. `phase0` reads GT from the canonical git-tracked barycenter (stdlib-only, reproducible; argmax unchanged so its numbers are stable — proxy ceiling 0.658).

**Tests:** `tests/` 3 passed; `study_a` smoke, `fewshot` smoke, `phase0`, and the `calibration_block` emitter all green under `../judex-evaluator/.venv`.

## Seam — result integration back into the evaluator (spec; NOT implemented here)

The guide §4.6 assumed a "config flip in `judex-evaluator/pipeline.yaml`, the seam already merged."
The seam **exists** but is **global**: `pipeline.yaml → calibration = {mode, temperature}` is consumed by
`judex.calibration.calibrate_distribution`, whose **only** live call site is
`judex.evaluation.permutation_glean` (Phase-1 gleaning), passed the whole `configs` bundle with **no
`family_id`** — so a naive flip applies the constant to **every** family (base/annotator raters *and*
the closed Gemini/GPT evaluators). Study A's τ_oc is meant for the **closed evaluators only**.

**To adopt `median(τ_oc)` without over-correcting the open raters, the evaluator needs (in a
separate, approved change — it re-touches the stabilized evaluator, and is only needed at Study A
Phase 4):**
1. **Family-scoped calibration.** Allow `calibration` to carry per-family overrides, e.g.
   `calibration: {mode: noop, by_family: {openai_gpt: {mode: temperature, temperature: T}, google_gemini_*: {...}}}`, resolved inside `calibrate_distribution(distribution, config, family_id=...)`.
2. **Plumb `family_id`** into `calibrate_distribution`. `permutation_glean` has no `family_id` in scope
   today; `evaluate_single_family` (`evaluation.py`) does and threads it via `usage_context`, so pass it
   down into `permutation_glean` → `calibrate_distribution`.
3. **Decide the application point.** The sole call site is the Phase-1 gleaning seam. A post-training
   *overconfidence* correction on the **reconciled final** distribution arguably belongs at **Phase-3
   synthesis** (`cross_family._call_phase3_synthesis_with_retries`), which is currently **uncalibrated**.
   Choose Phase-1-only vs Phase-3 (or both) deliberately.

Until that lands, `pipeline_calibration_block.json` is the correct **artifact**; do not paste it into the
global `calibration` key (it would recalibrate the open families too).

**Decorrelated-dispersion extension (§4.7).** `judex.experiments.dispersion_calibration_recovery(items,
replicates_by_item)` exists and is runnable today against the `stage9-gemini-gpt-medium` /
`phase23-deference-fix-native` runs. Adding the base-model distributions to the replicate pool needs a
small `decorrelated_dispersion.py` wrapper (reserved in the repo plan) that unions the base per-cell
distributions into `replicates_by_item` before the call — the evaluator function itself needs no change.

## API-location note (doc accuracy)
`dispersion_calibration_recovery` is exported from **`judex.experiments`** (not `judex.calibration`);
`fit_dispersion_temperature` is in `judex.calibration`; `murphy_decomposition` is in `judex.experiments`.

## Pins
All four repos on `develop`, pins == tips: corpus `dab1db2`, ground-truth `fcf17a9`, evaluator
`cb71d05`, calibration (this change, to be committed). `judex-calibration` references siblings by
relative path only (no stale SHA pins) — correct.
