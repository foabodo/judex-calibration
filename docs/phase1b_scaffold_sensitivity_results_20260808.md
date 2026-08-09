# Phase 1b — k=5 scaffold sensitivity on the full adoption panel (record)

Session 2026-08-09 03:2x–03:4x UTC. **No elicitation ran. Total spend $0.00.** Every
artifact below is in the working tree or under gitignored `runs/`; no git commits.

Scope: the Phase-1b leg of `spec/plan_2026_08_08_verbalized_arm_gap_closure.md` §3 —
extending the k=5 coverage-preserving scaffold-sensitivity study from the two Phase-1a
families to the remaining adoption-panel families **glm** and **maverick**, so the full
panel {qwen, gemma31, glm, maverick} is measured under perturbation and the published
panel-level constants can be recomputed under each variant.

Triggered by memo `spec/memo_2026_08_08_phase0_prespecified_reads.md` §A.6, on the
Phase-1a outcome **BOUNDED-LARGE** (B = 0.2971) recorded in
`docs/phase1a_scaffold_sensitivity_results_20260808.md`.

> **Outcome of this session: BLOCKED ON FUNDS, before any spend.** The vast.ai account is
> credit-only with **$47.62** available. The cheapest live 8×H200 offer is **$31.58/hr**,
> and Phase 1b needs four sequential 8×H200 rentals (glm pre, glm post, maverick pre,
> maverick post). The available credit buys **1.51 hours of a single box, total, for the
> entire campaign** — less than one family's pair. No box was created. Full pricing
> evidence in §3.
>
> Everything that Phase 1b could achieve at $0 was done instead: the analysis instrument
> was extended to the full panel and to panel-level recomputation and is verified against
> the published constants (§2), and the pre-provisioning prompt-budget gate for both 1b
> families was discharged offline (§4). The campaign is execution-ready and needs only
> funds.

---

## 1. What was NOT collected

| Run dir | Family | Legs | Variant | Status |
|---|---|---|---|---|
| `runs/study_b_glm_k5v1` | GLM-4.5 355B-A32B | pre + post | V1 `alt_set` | NOT CREATED |
| `runs/study_b_glm_k5v2` | GLM-4.5 355B-A32B | pre + post | V2 `rev_order` | NOT CREATED |
| `runs/study_b_maverick_k5v1` | Llama-4-Maverick 17B-128E | pre + post | V1 `alt_set` | NOT CREATED |
| `runs/study_b_maverick_k5v2` | Llama-4-Maverick 17B-128E | pre + post | V2 `rev_order` | NOT CREATED |

Consequently **no B-Q1 gate outcome, no Murphy resolution, no per-leg delta and no panel
recomputation exists for glm or maverick under any perturbation.** The baseline run dirs
`runs/study_b_glm` and `runs/study_b_maverick` were read only, and are unmodified.

### Serving plan that these dirs would have used (verified, not executed)

Read off the baseline sidecars `runs/study_b_{glm,maverick}/*.meta.json` and
`scripts/provision_vast.sh`:

| Family | pre model | post model | transport |
|---|---|---|---|
| glm | `zai-org/GLM-4.5-Base` | `zai-org/GLM-4.5` | self-hosted vLLM raw `/v1/completions`, bf16 |
| maverick | `meta-llama/Llama-4-Maverick-17B-128E` | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | self-hosted vLLM raw `/v1/completions`, bf16 |

Both families were clean on raw completions on **both** legs in the shipped campaign (GLM
96.7 / 92.5, Maverick 98.3 / 98.3 contract-complete), so — unlike gemma31 in Phase 1a —
their shipped baseline dirs are **already serving-matched** to the variants. No chat
endpoint and no matched-transport baseline re-collection is needed for 1b: the delta
reference is the shipped baseline dir itself, on both sides bf16 raw completions.

`scripts/provision_vast.sh` was verified against its real interface rather than a
remembered invocation: it applies `--enable-expert-parallel` to `zai-org/GLM-*` **and** to
`meta-llama/Llama-4-Maverick-*` (both MoE, lines 66–67, applied to base and post alike),
and it attaches no reasoning parser to either family's post leg (GLM is explicitly cased
as "raw `/v1/completions` driver — no chat reasoning parser needed"; Maverick falls to the
catch-all and is served plain). Both match the baseline legs. Serving pins would have been
bf16, `--max-model-len 32768`, `--gpu-memory-utilization 0.85`, `--tensor-parallel-size 8`,
`--workers 8`, budget 2048, temperature 0, 120 cells.

Hardware constraint confirmed, not assumed: GLM-4.5 is 355 B parameters ≈ **710 GB** bf16
and Maverick is 400 B ≈ **803 GB** bf16, both above 8×H100-80's 640 GB. 8×H200 (1128 GB)
is the floor. Never PCIE pairs.

---

## 2. The analysis instrument — extended, verified, and ready

`scripts/absolute_estimand_k5variants.py` (the labeled Phase-1a variant copy; the frozen
`scripts/absolute_vs_ratio_estimand.py` was **not** touched) now covers all four families
× {baseline, V1, V2} and gained the panel-level read that is the whole reason Phase 1b
exists.

### 2.1 Family map extended

`glm` → `study_b_glm` / `study_b_glm_k5v1` / `study_b_glm_k5v2`;
`maverick` → `study_b_maverick` / `study_b_maverick_k5v1` / `study_b_maverick_k5v2`.
Missing variant dirs are reported as `not available` per family, never silently skipped.

### 2.2 Baselines recomputed from the run dirs (not trusted from the task brief)

| Family | reference dir | τ_v | T_abs(pre) | T_abs(post) | argmax pre | argmax post |
|---|---|---:|---:|---:|---:|---:|
| qwen | study_b_qwen | 1.281052 | 3.1901 | 2.6365 | 0.4818 | 0.6017 |
| gemma31 (published, OpenRouter post) | study_b_gemma31_api | 1.025179 | 2.6285 | 2.2417 | 0.4417 | 0.6780 |
| gemma31 (matched transport) | study_b_gemma31_vllmchat_base | 1.025179 | 2.6285 | 2.2431 | 0.4417 | 0.6417 |
| **glm** | study_b_glm | **1.025179** | 2.5524 | **2.4639** | 0.5431 | 0.5948 |
| **maverick** | study_b_maverick | **1.379820** | 4.9408 | **2.0351** | 0.3250 | 0.5000 |

The brief's stated values (glm τ_v 1.0252 / T_abs 2.464; maverick 1.3798 / 2.035) are
confirmed to floating-point identity: glm τ_v = 1.0251785151221313, T_abs(post) =
2.4639456555374104; maverick τ_v = 1.379820350674421, T_abs(post) = 2.0350699223438102.

### 2.3 Panel-level recomputation added

For each scaffold the four-family panel is rebuilt from the legs elicited under that
scaffold, and every published panel quantity is recomputed on it:

- **τ_v side** — the value set, band [min, max], max/min **cluster ratio** against the ≤2
  rule, the **interpolated panel median** (the T_J analog) and study_a's upper-median
  convention, and **LOFO** max |ln(transferred / own)| under the exact
  `verbalized_reframe_r0.lofo_transfer` convention (transferred = median of the other
  three).
- **T_abs(post) side** — the same six quantities (the T_A analog is the interpolated
  median; the published ≤0.191 is this side's LOFO).
- Per-panel B-Q1 gate status for all eight legs.

A panel is computed **only** when all four families are present under that scaffold. Every
statistic here — ratio, band, median, LOFO — is defined over the whole panel, so a
three-family "panel" would be a different quantity wearing the same name; the driver
reports it unavailable and names the missing families instead.

Variant panels are differenced against the **matched-transport** baseline panel (the
like-for-like denominator, per the Phase-1a design change); the published-pairing panel is
carried alongside as the reproduction check.

### 2.4 Verification — the published panel constants fall out exactly

Driver output, `panel constants of record: ALL reproduce exactly`:

| Published quantity | Value of record | Recomputed |
|---|---:|---:|
| τ_v cluster ratio (r3 F1, printed 1.346) | 1.345931786826454 | identical |
| τ_v band (printed [1.03, 1.38]) | [1.0251785151221313, 1.379820350674421] | identical |
| τ_v interpolated median = **T_J** (printed 1.153) | 1.1531152939244511 | identical |
| τ_v LOFO max \|ln\| | 0.29708655150331403 | identical |
| T_abs cluster ratio (rA1 A3, printed 1.296) | 1.295554049625498 | identical |
| T_abs band | [2.0350699223438102, 2.6365430791635713] | identical |
| T_abs interpolated median = **T_A** (printed 2.353) | 2.3528040605496594 | identical |
| T_abs LOFO max \|ln\| (**the published ≤0.191**) | 0.1912338118671258 | identical |

Both LOFO worst cases are attained on **maverick** — which is precisely why the two
missing 1b families are the load-bearing ones for a panel-level read.

Noted in passing, not a finding: the τ_v LOFO worst case (0.29709) and the Phase-1a bound
B (0.29709, qwen/rev_order) are the *same number*. Neither is a coincidence nor a
relationship — both are exactly **4 steps of the 60-point log grid** on [0.25, 20.0]
(step ln 1.07710 = 0.074272), and τ_v is an exhaustive argmin on that grid. Two distinct
movements landing four grid steps apart collide on the same value. Do not read it as a
correspondence between scaffold sensitivity and panel sensitivity.

### 2.5 Matched-transport baseline panel (the delta denominator, once 1b data exists)

| Estimand | values | ratio (≤2) | band | median | LOFO max |
|---|---|---:|---|---:|---:|
| τ_v | qwen 1.2811, gemma31 1.0252, glm 1.0252, maverick 1.3798 | 1.3459 ✓ | [1.0252, 1.3798] | 1.1531 | 0.2971 (maverick) |
| T_abs | qwen 2.6365, gemma31 2.2431, glm 2.4639, maverick 2.0351 | 1.2956 ✓ | [2.0351, 2.6365] | 2.3535 | 0.1912 (maverick) |

It differs from the published panel in one cell only (gemma31 T_abs 2.2431 vs 2.2417,
Δln +0.0006) and the τ_v panel is bit-identical, so the median moves 2.3528 → 2.3535 and
no ratio, band edge or LOFO value changes at four decimals.

---

## 3. Why nothing was elicited — the funding blocker, with evidence

### 3.1 Account state (read 2026-08-09 ~03:30 UTC)

| Field | Value |
|---|---|
| `credit` | **$47.6227** |
| `balance` | 0 |
| `billing_creditonly` | 1 |
| `autobill_amount` / `autobill_threshold` | 5.0 / 5.0 |
| `vastai show instances` | `[]` (empty at session start) |

The account is **credit-only**: instances are stopped when credit is exhausted, and the
$5 autobill top-up is an order of magnitude below one hour of the required hardware.
$47.6227 is exactly the closing credit of the Phase-1a ledger — nothing was added since.

### 3.2 Live offer snapshot (captured 2026-08-09 03:34:03 UTC)

Query: `gpu_name=H200 num_gpus=8 static_ip=true direct_port_count>1 inet_down>1000
inet_down_cost<0.05 reliability>0.98 disk_space>1100 cuda_vers>=12.4 rentable=true`,
`--order dph`. **Three offers exist worldwide:**

| Offer | Machine | $/hr on-demand | $/hr min bid | inet_down (Mbps) | disk (GB) |
|---|---|---:|---:|---:|---:|
| 47225585 | 38592 | **31.582** | 29.47 | 7,240 | 8,742 |
| 19317221 | 32379 | 32.633 | 30.53 | 2,710 | 24,871 |
| 20654524 | 37735 | 34.738 | 31.53 | 7,699 | 5,208 |

Relaxing to any 8-GPU ≥140 GB configuration adds only 8×B200 at $40.00–45.00/hr. The
interruptible bid floor ($29.47) is 7% below on-demand and carries interruption risk on a
job whose cost is dominated by an unrepeatable download — not a saving worth taking.

**$31.58/hr is the floor.**

### 3.3 The projection

Four sequential rentals are required (pre and post are different checkpoints = different
downloads; the vast account has no SSH keys registered, so box reuse is unavailable —
discovered in Phase 1a). Both variants run on each box while it is warm, which is the
batching win, but it does not reduce the number of rentals.

| Per-box wall clock | Cost/box @ $31.58 | 4-box campaign |
|---:|---:|---:|
| 0.75 h | $23.69 | $94.75 |
| 1.00 h | $31.58 | **$126.33** |
| 1.50 h | $47.37 | **$189.49** |
| 2.00 h | $63.16 | $252.66 |

The realistic band is 1.0–1.5 h/box → **$126–190**, consistent with both plan anchors
(§1f: glm $60–90, maverick $70–110 for both legs) and with the original Study-B campaign
(≈$280–370 for four 8×H200 model-pairs). It sits inside the approved $140–240 envelope and
under the $250 cap. Download components at the offers' link rates: GLM 710 GB ≈ 8–30 min,
Maverick 803 GB ≈ 9–34 min, before container pull, 8-way model load and two legs.

**Against $47.6227 of credit that is 1.51 hours of one box for the whole campaign** — not
one family's pair, which needs two boxes. The approved envelope cannot be drawn on because
the account cannot fund it.

### 3.4 Why no partial spend was made

A partial run was considered and rejected on the merits, not only on cost:

1. **The 1b deliverable is panel-level.** The cluster ratio, band, interpolated median and
   LOFO bound are all defined over the four-family panel. Perturbing three of four
   families yields **no** panel recomputation under either variant — the exact question 1b
   exists to answer stays unanswered.
2. **No affordable subset is even a family.** One family = two boxes ≈ $63–95 > $47.62. A
   pre leg alone yields no τ_v, no pairing and no delta.
3. **Credit exhaustion mid-download destroys the spend.** Under credit-only billing the
   box stops; a 710–803 GB download that does not finish produces nothing and is not
   resumable on a new rental.

So the correct action was zero spend, and the plan's own rule was applied in its extreme
form: the projection exceeds what is available, so report rather than start.

---

## 4. Pre-provisioning gate discharged at $0 — prompt budget

`scripts/scaffold_variant_probe.py` gained the four 1b tokenizers
(`zai-org/GLM-4.5{,-Base}`, `meta-llama/Llama-4-Maverick-17B-128E{,-Instruct}`), all four
already in the local HF cache. The probe runs strictly offline (`HF_HUB_OFFLINE=1`) and
downloads nothing. Worst-case live raw prompt + JSON scaffold + 2048 CoT budget + 1, per
family-leg × variant, against the 32768 `--max-model-len` pin:

| Family / leg | scaffold | few-shot max | prompt max | required | fits 32768 |
|---|---|---:|---:|---:|:--:|
| glm / pre + post | baseline | 13,760 | 24,229 | 26,281 | YES |
| glm / pre + post | **alt_set** | 14,024 | 24,495 | **26,547** | YES |
| glm / pre + post | rev_order | 13,760 | 24,229 | 26,281 | YES |
| maverick / pre + post | baseline | 13,648 | 23,851 | 25,904 | YES |
| maverick / pre + post | **alt_set** | 13,832 | 24,037 | 26,090 | YES |
| maverick / pre + post | rev_order | 13,648 | 23,851 | 25,904 | YES |

Worst case across all of Phase 1b is **26,547 tokens** (glm, `alt_set`), leaving **6,221
tokens of headroom** under the pin. `alt_set` is the larger scaffold on both families, as
in Phase 1a, and `rev_order` is token-identical to baseline by construction (same rows,
reordered). **No serving change is needed for either family** — the standard 32768 pin
holds. Artifact: `runs/scaffold_variant_probe_1b.json`.

---

## 5. Panel recomputation — status

| Panel | Availability |
|---|---|
| baseline, published pairing | AVAILABLE — reproduces every published constant exactly (§2.4) |
| baseline, matched transport | AVAILABLE (§2.5) |
| **V1 `alt_set` panel** | **NOT AVAILABLE** — missing glm, maverick; present qwen, gemma31 |
| **V2 `rev_order` panel** | **NOT AVAILABLE** — missing glm, maverick; present qwen, gemma31 |

The two tables the brief asks for (V1 panel, V2 panel) **cannot be produced**, and no
partial substitute is offered: two families out of four is not a panel. Which published
panel quantities move under perturbation, and whether the ≤2 rule and the band survive,
therefore remain **open** — unchanged from before this session.

What is now known is the shape of the answer's sensitivity. The τ_v LOFO worst case and
the T_abs LOFO worst case are both attained on **maverick**, and glm sits at the τ_v band's
lower edge (1.0252, tied with gemma31). The two unmeasured families are the band's lower
edge and the LOFO extremum, i.e. exactly the two cells whose movement would decide the
cluster ratio and the bound. That is a statement about which measurements matter, not a
prediction of their outcome.

---

## 6. Pooled bound and the memo row — UNCHANGED from Phase 1a

With no 1b data, the pooled bound over all measured families × variants is still the
Phase-1a value:

- **B = max over legs and variants of { |Δln τ_v|, |Δln T_abs(post)| } = 0.2971**,
  attained at qwen / `rev_order` / Δln τ_v (2 families × 2 variants = 4 pairs).
- vs one τ_v grid step (0.0743): **4.00 steps**; vs the LOFO worst case (0.191): **1.56×**.
- max |Δargmax| = **16.67 pp** (gemma31 / `rev_order`, post leg).
- Gate-restricted recomputation (dropping qwen / `rev_order`, the one pair with a failing
  leg): **B = 0.2228**, still > 0.191.

**Memo §A.5 row: BOUNDED-LARGE** (B > 0.191), on both the full and the gate-restricted
computation. Quoting the memo's row verbatim, which is what this licenses in v15:

> The scaffold materially shapes the reported estimands. The stability observation (cluster
> ratio 1.296, LOFO ≤ 0.191) is re-read as conditional on the scaffold, the
> transferred-constant claim is restated as scaffold-conditional, and the bound B is
> reported as the headline of the sensitivity appendix. This weakens the paper but is
> strictly better than the same fact surfacing in review; v14's existing hedges already
> anticipate it.

The memo's §6-caveat stands and is now sharper: **the bound rests on two families**, which
memo §A.6 already called "too thin" for a load-bearing bound — and 1b, the remedy it
prescribed, did not run. Any v15 sentence built on B must disclose that it is a two-family
bound over four (family, variant) pairs, one of which carries a failing contract gate.

---

## 7. Cost ledger

No instance was created. No API leg was run.

| Line | Amount |
|---|---:|
| vast.ai | **$0.000** |
| OpenRouter | **$0.000** |
| **Phase 1b total** | **$0.000** |

Against the approved envelope $140–240 and the hard cap $250: **$0 spent, 0% of the cap.**
Vast credit unchanged at **$47.6227** (identical to the Phase-1a closing balance).

**Fleet teardown verified:** `vastai show instances` returns `[]` — verified at session
start and at session end. Nothing was ever provisioned, so nothing was left running.

Per-leg attribution, the convention Phase 1a introduced to fill the project's disclosed
per-leg-billing gap, has no rows to add this session.

---

## 8. Anomalies and honest caveats

1. **The blocker is funds, not feasibility.** Nothing technical stands in the way: the
   serving path is verified, the provisioning script already handles both families'
   expert-parallel MoE requirement, the prompt budget fits with 6.2k tokens of headroom,
   the analysis instrument is written and verified end to end, and the baseline dirs are
   already serving-matched so no extra baseline re-collection is needed (unlike gemma31 in
   1a). The campaign needs approximately **$126–190** of vast credit and nothing else.
2. **Topping up the account is outside what this task may do.** Adding funds is a payment
   action and is left to the user.
3. **No estimate of the 1b outcome is offered.** No glm or maverick perturbed leg exists;
   the panel question is open, and §5's observation about which cells matter is structural,
   not predictive.
4. **The published panel constants were re-derived, not assumed.** All eight reproduce to
   floating-point identity from the baseline run dirs (§2.4), which also validates the new
   panel code path against known answers before it is ever pointed at new data.
5. **No baseline run dir was written to.** `runs/study_b_glm`, `runs/study_b_maverick`,
   `runs/study_b_qwen`, `runs/study_b_gemma31*` are unmodified. The frozen drivers
   (`absolute_vs_ratio_estimand.py`, `absolute_mechanism_rA1.py`, `e6_r3_arms.py`,
   `q4_range_robustness.py`, `range_robustness_verbalized.py`) were not edited.
6. **The offer snapshot is a point-in-time read.** vast pricing moves; the $31.58/hr floor
   and the three-offer supply were true at 03:34 UTC on 2026-08-09 and should be re-read
   before any funded attempt.

---

## 9. To resume

1. Fund the vast account to **≥ $200** (campaign $126–190 plus headroom for one dead
   rental — Phase 1a hit exactly that failure mode and vast billed $0 for it).
2. Four sequential rentals, never more than two alive at once, `--order dph`, 8×H200,
   `VAST_DISK` ≥ 1100 for maverick / ≥ 950 for glm, both variants elicited on each box
   while warm:
   `glm pre → glm post → maverick pre → maverick post`, fresh out-dirs
   `runs/study_b_{glm,maverick}_k5v{1,2}` (the resume guard hard-errors on variant
   mismatch).
3. `--analyze` per dir; check B-Q1 ≥ 0.90 and positive Murphy resolution; record failures
   rather than resampling; use checkpoint/resume only for transients.
4. `python scripts/absolute_estimand_k5variants.py` — the per-leg deltas, the pooled bound
   and **both variant panel tables** then compute with no further code changes.
5. Update this document in place with the leg inventory, the two panel tables, the pooled
   bound and the invoice-exact ledger.

---

## 10. Files

Modified (working tree, uncommitted):
- `scripts/absolute_estimand_k5variants.py` — glm + maverick added to the family map and
  to `BASELINE_OF_RECORD`; `ADOPTION_PANEL`, `PANEL_OF_RECORD`, `PANEL_BASELINE_DIRS`;
  `_estimand_panel`, `panel_from_measurements`, `panel_delta`, `panel_of_record_check`;
  panel wiring and console block in `main`. Still a labeled variant read, not the rA1
  driver.
- `scripts/scaffold_variant_probe.py` — the four glm/maverick tokenizers added to
  `TOKENIZE_REPOS`.

Created:
- `docs/phase1b_scaffold_sensitivity_results_20260808.md` — this document.
- `runs/scaffold_variant_probe_1b.json` (gitignored) — the offline prompt-budget read.

Rewritten (gitignored): `runs/estimand_k5variants/estimand_k5variants.json` — now carries
the four-family map and the panel blocks; the Phase-1a per-leg numbers in it are unchanged.

Unchanged: every frozen driver and every baseline run dir.
