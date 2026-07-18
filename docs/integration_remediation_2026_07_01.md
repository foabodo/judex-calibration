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

The guide §4.6 assumed adopting the constant is just "a config flip in `judex-evaluator/pipeline.yaml`,
the seam already merged." The seam **exists** but is **global**: `pipeline.yaml → calibration =
{mode, temperature}` is consumed by `judex.calibration.calibrate_distribution`, whose only live call
site is the Phase-1 gleaning loop in `judex.evaluation` (`evaluation.py` ~L331), passed the whole
`configs` bundle with **no `family_id`** — so one value is applied identically to every family that runs.

**Correction (what "every family" actually means here).** At *evaluation* time the only families that
reach `calibrate_distribution` are the **cross-family evaluator pair** — the two **closed** evaluators
JUDEX judges with (Gemini Pro + GPT in production; some arms run an Anthropic + OpenAI pair instead).
The six **open annotator raters run at construction time only** (they produce the corpus exemplars) and
are **never** in the evaluation calibration path — so a global flip does **not** touch the open raters.
An earlier draft of this note said it would; that was inaccurate.

**So for the simplest intended case — two closed evaluator families and one well-clustered constant —
the existing GLOBAL seam is adequate:** set `calibration: {mode: temperature, temperature: median(τ_oc)}`
and both closed evaluators are softened, which is the intent. `pipeline_calibration_block.json` drops
straight in.

**Family-scoping is an OPTIONAL refinement, not a blocker.** It earns its keep only under one of three
triggers (each a separate, approved evaluator change — it re-touches the stabilized evaluator, and is
relevant only at Study A Phase 4):
1. **τ_oc does not cluster (Q3 negative).** A single global scalar can't express Gemini ≠ GPT; you'd want
   per-family temperatures. If Q3 clusters tightly, one global constant is exactly right — no scoping needed.
2. **A run uses a different evaluator pair.** Because some arms swap the pair (e.g. Anthropic + GPT), a
   global constant softens whatever pair that run uses; scope it if you want the transferred constant
   pinned to specific closed families regardless of the run's pair.
3. **Application point.** The sole call site is Phase-1 gleaning (per-family, pre-reconciliation). A
   post-training *overconfidence* correction on the **reconciled final** distribution arguably belongs at
   **Phase-3 synthesis** (`cross_family._call_phase3_synthesis_with_retries`), currently **uncalibrated**.
   This is orthogonal to scoping — decide Phase-1-only vs Phase-3 (or both) deliberately.

**If you do scope it**, the mechanism is: allow `calibration` to carry per-family overrides, e.g.
`calibration: {mode: noop, by_family: {openai_gpt: {mode: temperature, temperature: T}, google_gemini_*: {...}}}`,
resolved inside `calibrate_distribution(distribution, config, family_id=...)`; plumb `family_id` down —
`permutation_glean` has none in scope today, but `evaluate_single_family` does and threads it via
`usage_context`, so pass it into `permutation_glean → calibrate_distribution`.

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

---

## Addendum — 2026-07-03 re-verification at the 7-rater refresh tips

The three sibling repos moved to corpus `aa0c928` / ground-truth `bccccea` / evaluator `16b39a5`
(7-rater annotator panel: Gemma seat-7 collected, exemplar stores rebuilt 672→**784** rows
(644 leaf + 140 dimension), GT leaf/dim labels re-fit, AIReg-Bench GT bundle **rebuilt** at
4000 draws/8000 tune nutpie — argmax-identical, W1 ≈ 0.0008 vs the previous bundle, but new
hashes/trace). Every seam above was re-verified live against those tips:

- **#2 `aireg.load_cells()`**: 120/120 cells, manifest verification passes against the NEW
  bundle; GT == canonical barycenter CSV to max |diff| **1.39e-17**. The supervised-fit sanity
  number re-derives to **T_rps 2.4434** on the Gemini/GPT run (recorded 2.44 — unchanged, as
  expected given W1 ≈ 0.0008 label movement).
- **#3 `fewshot.py`**: dimension store now 140 rows / **7 raters** (all rows carry the 7-field
  contract incl. `compliance_distribution`/`confidence_distribution`/`findings`); k=4 selection
  yields 4 distinct raters per Article with an ordinal ramp; `compliance_1to5 == argmax+1`
  invariant holds 140/140; firewall `source_item_label ∩ AIReg item_label = ∅` re-confirmed.
- **`phase0`**: proxy ceiling **0.658** (gemini-pro) unchanged under the refreshed bundle —
  the Phase 0 gate conclusion stands.
- **Panel note**: "the six open annotator raters" in the seam discussion above is now **seven**
  (Gemma added); the construction-time-only argument is unchanged.
- Seam-verification tests added: `tests/test_integration_seams.py`.

---

## Addendum — 2026-07-18 re-verification at the corpus-v2 / Tier-0 tips

**Pins.** corpus `9541c5b` · ground-truth `b2e4fe3` · evaluator `2b6322b` · paper `7d0ed54` ·
calibration `d5f5c90`. All on `develop`. The two prior pin lists (body §Pins, and the 2026-07-03
addendum) are **historical** — do not read them as current.

**Everything above this line is a dated record.** Where it conflicts with this addendum, this
addendum wins. The three body statements most likely to mislead a reader today:

| Body says | Now |
|---|---|
| "a config flip in `judex-evaluator/pipeline.yaml`" (§Seam) | the file is **`judex-evaluator/configs/pipeline.yaml`**; there is no top-level `pipeline.yaml`. A second bundle `configs_v2exemplars/pipeline.yaml` also exists |
| "Gemini Pro + GPT in production" (§Seam correction) | the pair switched to **Anthropic + GPT** on 2026-07-02. The `by_family: {openai_gpt, google_gemini_*}` scoping sketch is illustrative syntax only |
| "the six open annotator raters" (§Seam) | **seven** (already corrected in the 2026-07-03 addendum) |
| "`murphy_decomposition` is in `judex.experiments`" (§API-location note) | **defined** in `judex.metrics_report` since 2026-07-18; re-exported from `judex.experiments`, so the import still works — non-breaking |
| "`evaluation.py` ~L331" (§Seam) | `evaluation.py:340` |
| dimension store "140 rows", exemplar stores "784 rows (644 leaf + 140 dimension)" (2026-07-03 addendum) | corpus-v2: **1757 rows = 1449 leaf + 308 dimension**, 44 Article excerpts × 7 raters, 8-field contract 0.2.0 |

### 1. The AIReg GT changed formulation on 2026-07-09 — and we missed it (the material finding)

`aireg.py` described the bundle as "last re-fit 2026-07-03 at 4000 draws/8000 tune". That was
true at GT `5716e36`. It was invalidated four days later, by `89f40b7` + `3c2ebdb`:

- thresholds **pooled → freethresh**; readout temperature **τ = 1 → 0.65 → 0.675** (re-pinned by
  user decision); 0.05 grid snap **off globally** → labels are now **continuous** (verified 0/600
  probabilities on the grid).
- Movement vs the previously shipped object: **W1 mean 0.194** (median 0.184, max 0.417), mean
  entropy 1.065 → 1.252 nats, **0/120 mode flips** (KL-anchor construction guarantee).

This is exactly the silent staleness the body's §#2 rewrite exists to prevent — the mechanism
worked (we *load* the current bundle every time), but the *description* froze, and with it the
carried numbers. Concretely, re-derived 2026-07-18 on `stage9-gemini-gpt-medium`:

> **`T_rps` 2.4434 → 3.5585 (+46%)**, verified bound-independent (identical under both (0.25, 4.0)
> and (0.25, 20.0)). The 2026-07-03 addendum's "unchanged, as expected given W1 ≈ 0.0008" reasoning
> does not survive a W1 of 0.194. **Any T\*/τ_oc number recorded before 2026-07-09 must be
> re-derived, not carried forward.**

Argmax-derived numbers are stable, as the 0/120 mode flips imply: the Phase-0 proxy ceiling
**0.658 re-derives exactly**, and `runs/phase0/accuracy_precheck.json` reproduces byte-identically.

**Sampler provenance** (trace attrs in `airegbench_mgmfrm_cumulative_consistency_idata.nc` are
authoritative): **draws 2000 / tune 4000**, 4 chains, target_accept 0.99, seed 42, nutpie 0.16.8,
0 divergences — compliant with the 2026-07-14 convention. *Footgun:* the sidecar
`validation_diagnostics_v4.json` records `draws 1000 / tune 2000` — argparse defaults stamped over
the cached-trace path by `build_airegbench_canonical_sources.py`. Trust the trace, not the sidecar.
**Validation is `unavailable` on TWO hard-failure families**, not one: `prior_predictive` (not
computed) *and* `convergence_rhat` (max 1.0123 > 1.01). ESS passes (558).

**Guard added:** `tests/test_integration_seams.py::GtVintageGuardTests` now pins the *formulation*
(status, model, counts, `thresholds=free`, `readout_tau=0.675`, grid-snap off, continuity, both
hard-failure families) — not just the shape. A future re-materialization fails loudly here.

### 2. `T_rps` was censored by an inherited evaluator default (code fix)

`study_a.score_variant` called `fit_temperature(pairs)` with no `bounds`, silently inheriting
`judex.calibration.DEFAULT_TEMPERATURE_BOUNDS = (0.25, 4.0)` — while `T_rel` and `τ_oc`, both
searched on the module's own `GRID`, ran to 20. The three temperatures were not on one scale, and
a pegged `T_rps` read as a fit. Every fp16-pilot leg shows it: `T_rps = 4.0` on all four legs
(`runs/pilot_f16_*`), alongside `T_rel` 7.6–20.0 and a shipped `tau_oc_median = 4.877` that
*exceeds the evaluator's own declared bound*.

Fixed: `study_a.T_BOUNDS = (0.25, 20.0)` is now passed explicitly to every fitter (so an
evaluator-side default change cannot move our numbers), and `study_a.saturated()` flags any
temperature on a boundary. Reports and the calibration block carry `T_rps_saturated`,
`T_rel_saturated`, `tau_oc_saturated`, `tau_oc_any_saturated`, `tau_oc_saturated_families`;
`run_qwen_phase1.py` prints a `[PEGGED]` refusal banner. The 2.4434 → 3.5585 move above is
*not* attributable to this change — it is purely the GT.

**fp16 4B pilot re-derived 2026-07-18** (`--analyze-only` / `--merge` over the cached
`pre.json`/`post.json` — free, no re-elicitation; both families are full 120-cell reasoning-ON
legs from 2026-07-14, i.e. already on the post-07-09 GT, so only the bound bug applied):

| family | leg | argmax | `T_rps` was | `T_rps` now | `T_rel` |
|---|---|---|---|---|---|
| gemma3-4b | pre | 0.150 | 4.00 | **20.00** (saturated) | 20.00 (saturated) |
| gemma3-4b | post | 0.242 | 4.00 | 18.59 | 16.01 |
| qwen3-4b | pre | 0.158 | 4.00 | 15.24 | 12.81 |
| qwen3-4b | post | 0.317 | 4.00 | 6.34 | 7.62 |

Every recorded `4.00` was the censor, not a fit — the true optima are 1.6×–5× higher. **`τ_oc` is
unchanged** (4.8772 / 4.2040) because `fit_tau_oc` always searched `GRID`, so the pilot's headline
output and the "accuracy gate binds at 4B" conclusion both stand — they are simply better
supported now: gemma3-4b's pre leg pegs on *both* objectives, which is the §0 flatten-to-the-marginal
failure made visible rather than hidden behind a 4.0.

That comparison exposed a second gap, now closed: **a finite `τ_oc` can still be meaningless if the
`pre` leg it aligns to is itself pegged.** gemma3-4b is exactly that — `τ_oc` 4.877 sits mid-range
and reads like a measurement while its reference could not be fit at all. New
`tau_oc_reference_degenerate` / `tau_oc_degenerate_reference_families` flags catch it, and the
merged pilot now trips `[PEGGED] … DEGENERATE REFERENCE on ['gemma3-4b']` where it previously
printed a clean `[integrate]` banner inviting a paste into `pipeline.yaml`. Q3 must exclude such
families. The pre-fix reports are preserved in this session's scratchpad.

### 3. The emitted calibration block loses its provenance on paste

`judex.calibration.calibrate_distribution` reads `provenance` **only** on the `dispersion` branch.
`mode: temperature` routes to `calibrate_invert_softmax(...)`, which records
`method: "invert_softmax"` and drops the dict — including the `smoke` flag that
`run_qwen_phase1.py` stamps to stop a plumbing value being mistaken for real. The block is still
*accepted verbatim*; the audit trail simply does not survive. Body §#4a's reasoning for choosing
`temperature` over `dispersion` (avoiding a false `gt_free_dispersion_fit` stamp) stands — the
cost was just undocumented. Mitigation is documentation-only on our side (this is an
evaluator-side change): keep `pipeline_calibration_block.json` beside the run as the audit trail.
Noted in `study_a.calibration_block`'s docstring and the vast doc's §6.

### 4. Corpus-v2 seams re-verified (all green)

`fewshot.py` is bound to `judex_leaf_exemplar_construction_v2/`. Re-verified against the v2 store:
**308 dimension rows** (44 excerpts × 7 raters, matching `EXPECTED_RATERS` exactly), leaf **1449**,
total **1757**; all eight fields `fewshot.py` reads present on 308/308; `compliance_1to5 ==
argmax(probabilities)+1` holds **308/308 with 0 ties**; k=4 selection yields 4 distinct raters per
Article with an ordinal ramp; firewall `source_item_label ∩ AIReg item_label = ∅` re-confirmed.

`fewshot.select_rows` was diffed against the evaluator's `judex.exemplars._stratified_fixed_set`
across k ∈ {1,2,3,4,5,6,8,10,12,20,63} × all 5 Articles: **identical selection and order in every
case**. One latent hazard to know about: the evaluator derives level order from the row's
positional `probabilities` array, falling back to the `compliance_distribution` dict, which is
keyed **alphabetically**. Today every row has `probabilities`, so the two agree; a future store row
without it would make the evaluator round-robin alphabetically while we stay in scale order.

Two observations, neither an error, both worth a deliberate decision:
- **k=4 never shows level 5.** Round-robin at k=4 yields compliance levels [1,2,3,4] for all five
  Articles — the base leg never sees a `very_high` / answer-E exemplar.
- **Grid asymmetry.** Corpus exemplar probabilities are 100% on the 0.05 elicitation grid; the
  AIReg GT is continuous (since 2026-07-09). Harmless for the base leg — `render_block` emits only
  a letter — but the instrument's two sides sit on different supports.

### 5. Evaluator-side facts that moved

- **`configs_v2exemplars/`** is a full parallel bundle whose only difference is the exemplar pool
  (corpus-v2). **`configs/` is still the evaluator default and is still v1.** Study A's few-shot is
  v2, so a Q4/E6 closed-pair run must pass `--config-dir configs_v2exemplars` or the closed leg is
  framed on v1 exemplars while the open legs are on v2. `rubric.yaml` and `output_contract.yaml`
  are byte-identical across the bundles, so `aireg.py` reading `configs/rubric.yaml` is safe.
- **`stage9-audit` CLI** (`judex.stage9_audit`, added 2026-07-18) ships per-stage Murphy
  decomposition, declared-vs-computed revision discipline, and paired bootstrap+sign-test
  comparison against a baseline run. E6.1/E6.4 hand-specify much of this; prefer reusing it. Caveat:
  its bootstrap resamples **cells**, not documents, so it is not a drop-in for the doc-clustered CI
  §4.6 requires — it is the base to build on, not the answer.
- **No 120-cell run realizes the current closed pair.** `stage9-sweep-sonnet-gpt-v2` is
  `anthropic_claude` + `openai_gpt` with a **`gpt-5.4-mini`** GPT seat (a ruled-out model), created
  2026-06-21, `stop_reason: "error"`, no `run_identity` block, pre-0.2.0 and pre-corpus-v2.
  `stage9-claude-gpt-medium` is the right pair at 3 docs / 15 items. Guide §4.8 corrected: E6 costs
  **~$290–330**, not $0. Measured per-doc for this pair is $13.61.

### 6. Fresh-clone / provisioning exposure (open — needs a push, not a code change)

`judex-calibration` pins nothing; it reads siblings by relative path. Correct by design — but the
*umbrella* pins them, and `scripts/provision_claude_code.sh` clones the umbrella
(`-b calibration-integration`) and runs `git submodule update`, so a box gets the **remote**
umbrella's pins. As of 2026-07-18 the local umbrella is **14 commits ahead of
`origin/calibration-integration`**, so a box provisioned today would receive:

| submodule | box gets | vs local develop |
|---|---|---|
| judex-calibration | `e84e009` (07-14) | **1 behind — no E6 leg at all** |
| judex-ground-truth | `cc664f8` (07-16) | 4 behind (no Tier-0 battery / E3) |
| judex-evaluator | `a8fc216` (07-14) | 3 behind |
| judex-paper | `8fdec2d` (07-16) | 8 behind |
| judex-corpus | `9541c5b` | current |

Mitigated for the evaluator specifically: `calibration.py`, `exemplars.py`, `ground_truth.py`,
`core/`, `configs/rubric.yaml` are **byte-identical** between `a8fc216` and `2b6322b`, so every
seam above holds at both. It is *not* mitigated for judex-calibration itself. **Push the umbrella
(and the 3 unpushed evaluator + 3 unpushed paper commits) before provisioning**; the script now
prints the pins it actually checked out so a stale box is visible at provision time rather than
mid-run.

### 7. Known non-issues (checked, no action)

`synthesize_aireg_bench_ground_truth(data_dir, category_labels)` signature and return shape
unchanged; all seven `study_a.py` evaluator imports resolve; `mode: temperature` is still a valid
evaluator calibration mode; `mgmfrm_anchored_projection` is still THE canonical AIReg family (the
new `data/distributional_labels/corpus_v2/` family is corpus leaf/dimension reconstruction labels
and does not shadow it — the evaluator loader never inspects subdirectories); every sibling path
referenced anywhere in this repo exists; the measured prompt budget (worst prompt ≈18.3k tok,
≈20.4k required, `--max-model-len ≥ 24576`) re-renders byte-stable, since corpus has not moved.

**Tests:** `tests/` **31 passed** (was 27; +4 `GtVintageGuardTests`).
