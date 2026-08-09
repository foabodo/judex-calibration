# Phase 1b — k=5 scaffold sensitivity on the full adoption panel (record)

Campaign run 2026-08-09 06:05–08:08 UTC. Elicitation only; no git commits; every artifact
below is in the working tree or under gitignored `runs/`.

Scope: the Phase-1b leg of `spec/plan_2026_08_08_verbalized_arm_gap_closure.md` §3 —
extending the k=5 coverage-preserving scaffold-sensitivity study from the two Phase-1a
families to the remaining adoption-panel families **glm** and **maverick**, so the full
panel {qwen, gemma31, glm, maverick} is measured under perturbation and the published
panel-level constants can be recomputed under each variant. Read under the rules fixed in
advance by `spec/memo_2026_08_08_phase0_prespecified_reads.md` §A, on the Phase-1a outcome
**BOUNDED-LARGE** (B = 0.2971) recorded in
`docs/phase1a_scaffold_sensitivity_results_20260808.md`.

> **Outcome: three of four rentals succeeded; six of eight legs collected.** GLM is
> complete on both variants and both legs. Maverick has both **pre** legs but **neither
> post leg**: the fourth rental was stopped by vast's credit-only billing at 749 of ~803 GB
> of weight download, and could not be restarted. **The panel-level recomputation — the
> deliverable Phase 1b exists for — therefore remains unavailable under both variants**,
> because every panel statistic is defined over all four families. The pooled bound B is
> unchanged at 0.2971 and the memo row is still **BOUNDED-LARGE**. Total spend **$132.84**
> against a $250 cap. Fleet empty.
>
> *Note on the first attempt:* an earlier session recorded this campaign as blocked on a
> $47.62 credit balance and made no spend. That blocker record was superseded by the
> clarification that the account carries automatic top-up; this document replaces it with
> the executed campaign. The top-up is real — a $80 credit landed at 06:04:53 UTC as the
> first box came up — but it is **not unlimited in rate**, which is what §6 is about.

---

## 1. Per-leg inventory and B-Q1 gates

All legs: bf16 self-hosted vLLM, raw `/v1/completions` on **both** legs of **both**
families, `--max-model-len 32768`, `--gpu-memory-utilization 0.85`,
`--tensor-parallel-size 8`, `--enable-expert-parallel`, `--workers 8`, budget 2048,
temperature 0, 120 cells, `fewshot_k` 5. Gate = contract_complete ≥ 0.90.

| Run dir | leg | variant | model | n | parse | contract_complete | gate | argmax | Murphy resolution |
|---|---|---|---|---:|---:|---:|:--:|---:|---:|
| study_b_glm (baseline) | pre | baseline | GLM-4.5-Base | 120 | 0.9667 | 0.9667 | PASS | 0.5431 | 0.04651 |
| study_b_glm (baseline) | post | baseline | GLM-4.5 | 120 | 0.9667 | 0.9250 | PASS | 0.5948 | 0.05558 |
| study_b_glm_k5v1 | pre | alt_set | GLM-4.5-Base | 120 | 0.9917 | 0.9917 | PASS | 0.4874 | 0.02987 |
| study_b_glm_k5v1 | post | alt_set | GLM-4.5 | 120 | 0.9417 | **0.8833** | **FAIL** | 0.6637 | 0.06049 |
| study_b_glm_k5v2 | pre | rev_order | GLM-4.5-Base | 120 | 0.9667 | 0.9583 | PASS | 0.3793 | 0.04273 |
| study_b_glm_k5v2 | post | rev_order | GLM-4.5 | 120 | 0.9500 | 0.9000 | PASS | 0.6228 | 0.05886 |
| study_b_maverick (baseline) | pre | baseline | Maverick-17B-128E | 120 | 1.0000 | 0.9833 | PASS | 0.3250 | 0.03818 |
| study_b_maverick (baseline) | post | baseline | Maverick-…-Instruct | 120 | 1.0000 | 0.9833 | PASS | 0.5000 | 0.05435 |
| study_b_maverick_k5v1 | pre | alt_set | Maverick-17B-128E | 120 | 0.9667 | 0.9667 | PASS | 0.3621 | 0.01840 |
| study_b_maverick_k5v1 | post | alt_set | — | — | — | — | **NOT COLLECTED** | — | — |
| study_b_maverick_k5v2 | pre | rev_order | Maverick-17B-128E | 120 | 0.9667 | 0.9667 | PASS | 0.3017 | 0.02224 |
| study_b_maverick_k5v2 | post | rev_order | — | — | — | — | **NOT COLLECTED** | — | — |

Notes.

- **One gate failure**: glm V1 post, contract_complete 0.8833. Recorded, not repaired — a
  parse failure is an observation in this channel and is never resampled. Its effect on the
  verdict is tested in §3.2. glm V2 post lands exactly on the threshold (0.9000, PASS).
- **Every collected leg has strictly positive Murphy resolution**, so no leg is a degenerate
  reference. The weakest is maverick V1 pre (0.01840) — as in Phase 1a, the alternate
  exemplar draw costs a base leg more than the order permutation does.
- **No repetition collapse and no serving pathology anywhere.** Both families were clean on
  raw completions on both legs in the shipped campaign (GLM 96.7/92.5, Maverick 98.3/98.3),
  and remain so under perturbation; unlike gemma31 in Phase 1a, neither family needed a chat
  endpoint or a matched-transport baseline re-collection. The delta reference is the shipped
  baseline dir itself, bf16 raw completions on both sides.
- **Two transients, both recovered by design.** glm V2 pre died at cell 85 and glm V2 post
  at cell 67, both with `ConnectionResetError`; both resumed from their per-cell checkpoints
  to 120/120 under identical leg config. Crash recovery, not re-elicitation.
- Baseline run dirs `runs/study_b_glm` and `runs/study_b_maverick` were read only and are
  unmodified.

---

## 2. The pre-specified read (memo §A.2)

τ_v is an exhaustive argmin over a 60-point logarithmic grid on [0.25, 20.0]; one grid step
is |Δln| = 0.0743, so a smaller movement is *not resolvable* and is reported as no observed
movement. T_abs is a golden-section refine, so its deltas are continuous.

Baselines recomputed from the run dirs, reproducing the brief's values to floating-point
identity (glm τ_v = 1.0251785151221313, T_abs(post) = 2.4639456555374104; maverick
τ_v = 1.379820350674421, T_abs(post) = 2.0350699223438102):

| Family | reference dir | τ_v | T_abs(pre) | T_abs(post) | argmax pre | argmax post |
|---|---|---:|---:|---:|---:|---:|
| glm | study_b_glm | 1.025179 | 2.5524 | 2.4639 | 0.5431 | 0.5948 |
| maverick | study_b_maverick | 1.379820 | 4.9408 | 2.0351 | 0.3250 | 0.5000 |

### 2.1 Levels

| family / variant | argmax(pre) | argmax(post) | τ_v | T_abs(pre) | T_abs(post) |
|---|---:|---:|---:|---:|---:|
| glm / baseline | 0.5431 | 0.5948 | 1.0252 | 2.5524 | 2.4639 |
| glm / alt_set | 0.4874 | 0.6637 | 1.0252 | 4.0231 | 2.1535 |
| glm / rev_order | 0.3793 | 0.6228 | 1.0252 | 2.9179 | 2.1694 |
| maverick / baseline | 0.3250 | 0.5000 | 1.3798 | 4.9408 | 2.0351 |
| maverick / alt_set | 0.3621 | — | — | 8.1188 | — |
| maverick / rev_order | 0.3017 | — | — | 5.2894 | — |

No τ_v is saturated; no T_abs is pegged at a search bound.

### 2.2 The three deltas — glm (complete)

| family / variant | Δargmax(pre), pp | Δargmax(post), pp | Δln τ_v | grid steps | Δln T_abs(post) | Δln T_abs(pre) |
|---|---:|---:|---:|---:|---:|---:|
| glm / alt_set | −5.57 | +6.89 | **0.0000** | 0.00 | **−0.1347** | +0.4550 |
| glm / rev_order | −16.38 | +2.80 | **0.0000** | 0.00 | **−0.1273** | +0.1338 |

### 2.3 maverick — no delta of record

Both maverick variant dirs hold a **pre leg only**. τ_v is fitted post→pre and T_abs of
record is taken on the post leg, so **neither of the two estimands that enter B exists for
maverick under either variant**, and no maverick row belongs in the delta table. The
pre-side numbers in §2.1 are printed as a partial diagnostic and are explicitly *not* a
variant result: they cannot be compared with the glm/qwen/gemma31 rows, which are pair
quantities.

For completeness, the pre-side movements are Δargmax(pre) +3.71 pp / −2.33 pp and
Δln T_abs(pre) +0.4966 / +0.0682 for alt_set / rev_order respectively.

---

## 3. Bound criterion and the memo row

### 3.1 Pooled over Phase 1a + Phase 1b

Six complete (family, variant) pairs now exist — qwen×2, gemma31×2, glm×2 — out of the eight
the campaign aimed at.

| family / variant | Δln τ_v | Δln T_abs(post) | max\|Δargmax\|, pp |
|---|---:|---:|---:|
| qwen / alt_set | −0.2228 | +0.0138 | 5.86 |
| qwen / rev_order | **−0.2971** | −0.1643 | 3.38 |
| gemma31 / alt_set | 0.0000 | +0.0040 | 11.67 |
| gemma31 / rev_order | 0.0000 | −0.0579 | **16.67** |
| glm / alt_set | 0.0000 | −0.1347 | 6.89 |
| glm / rev_order | 0.0000 | −0.1273 | 16.38 |
| maverick / alt_set | — | — | — |
| maverick / rev_order | — | — | — |

- **max \|Δln τ_v\| = 0.2971** (qwen / rev_order) — 4.00 grid steps.
- **max \|Δln T_abs(post)\| = 0.1643** (qwen / rev_order).
- **B = max over legs and variants of { \|Δln τ_v\|, \|Δln T_abs\| } = 0.2971**, attained at
  qwen / rev_order / Δln τ_v — **unchanged by Phase 1b**.
- vs one τ_v grid step (0.0743): **4.00 steps**; vs the LOFO worst case (0.191): **1.56×**.
- max \|Δargmax\| = **16.67 pp** (gemma31 / rev_order, post leg), with glm / rev_order's pre
  leg a close second at 16.38 pp.

### 3.2 Robustness to the gate failures

Recomputing B over the four pairs whose *both* legs pass the B-Q1 gate — i.e. dropping
qwen / rev_order and glm / alt_set — gives **B = 0.2228** (qwen / alt_set / Δln τ_v), max
\|Δargmax\| 16.67 pp. Still > 0.191, so the row is **bounded-large either way**. The verdict
does not rest on either defective leg.

### 3.3 The memo row

Memo §A.5 defines: bounded-small = B < 0.0743 **and** max\|Δargmax\| below ~3 pp;
bounded-comparable-to-LOFO = 0.0743 ≤ B ≤ 0.191; bounded-large = B > 0.191.

> **Outcome: BOUNDED-LARGE**, on both the full and the gate-restricted computation.

Quoting memo §A.5's bounded-large row verbatim, which is what this outcome licenses in v15:

> The scaffold materially shapes the reported estimands. The stability observation (cluster
> ratio 1.296, LOFO ≤ 0.191) is re-read as conditional on the scaffold, the
> transferred-constant claim is restated as scaffold-conditional, and the bound B is
> reported as the headline of the sensitivity appendix. This weakens the paper but is
> strictly better than the same fact surfacing in review; v14's existing hedges already
> anticipate it.

Also fixed in advance and unchanged: the **scaffold-free boundary** is stated once — a
zero-shot arm is not available for base models, because the exemplars carry the contract
format and bases without them fail the parse gate. The answerable question is sensitivity
*within* the family of valid coverage-complete scaffolds. And the k=4 token-slice trial is
not cited as evidence in v15.

### 3.4 Secondary reads (memo §A.3: reported, not part of B)

- **glm's τ_v does not move at all**, under either perturbation, exactly as gemma31's did
  not. Three of the four panel families now show τ_v pinned to its baseline grid point under
  coverage-preserving perturbation; qwen is the sole family whose ratio estimand moves, and
  it moves 3–4 steps in the *same* (downward) direction under both variants.
- **glm's absolute estimand does move**, and in the same direction under both variants:
  T_abs(post) falls by 0.13 in logs (−0.1347 alt_set, −0.1273 rev_order), roughly 1.8 grid
  steps' worth. glm is the first family where the T_abs side moves materially while the τ_v
  side is inert — the mirror image of qwen. The pattern from Phase 1a holds and sharpens:
  **which estimand is sensitive is family-specific, not scaffold-specific.**
- **T_abs(pre) inflates under `alt_set` on every family that has been measured**: gemma31
  +0.7209, maverick +0.4966, glm +0.4550, qwen +0.0118. Three of four move by roughly half a
  log unit in the same direction. This is excluded from B by the memo's definition (which
  takes T_abs on the *post* leg) and is reported as an observation: the disjoint exemplar
  draw consistently makes base legs more in need of absolute correction. It is the single
  largest movement class in the whole 1a+1b set.
- **Δargmax against the stated expectation (memo §A.4).** The expectation recorded before
  any data was that coverage-preserving deltas would be materially smaller than the
  coverage-violating k=4 E-hole deltas (+8.3/+10.8/+10.8/+17.5 pp). On argmax they are still
  not: glm / rev_order moves the base leg −16.38 pp, alongside gemma31's 16.67 pp. The memo
  pre-committed to the reading of exactly this case — the instrument is sensitive to the
  scaffold rather than only to category coverage, which is a finding in its own right.

---

## 4. Panel-level recomputation — STILL NOT AVAILABLE

| Panel | Availability |
|---|---|
| baseline, published pairing | AVAILABLE — reproduces every published constant exactly (§4.1) |
| baseline, matched transport | AVAILABLE (§4.2) |
| **V1 `alt_set` panel** | **NOT AVAILABLE** — missing maverick; present qwen, gemma31, glm |
| **V2 `rev_order` panel** | **NOT AVAILABLE** — missing maverick; present qwen, gemma31, glm |

The two panel tables the campaign was run to produce **cannot be produced**, and no partial
substitute is offered. Every panel statistic — the max/min cluster ratio against the ≤2 rule,
the band, the interpolated median (the T_J / T_A analog) and the LOFO bound — is defined over
the whole four-family panel; three families out of four is a different quantity wearing the
same name. The driver reports it unavailable and names the missing family rather than
computing a subset panel.

The shortfall is maximally unlucky. **Both LOFO worst cases are attained on maverick** — it
is the τ_v band's upper edge (1.3798) and the T_abs band's lower edge (2.0351) — so maverick
is precisely the family whose movement would decide both the cluster ratio and the bound.
This is a statement about which measurement is missing, not a prediction of its outcome, and
no estimate of the 1b panel result is offered.

### 4.1 Verification — the published panel constants fall out exactly

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

This validates the panel code path against known answers; it is unchanged by this campaign
and is restated because it is the check that licenses the driver to be pointed at new data.

Noted in passing, not a finding: the τ_v LOFO worst case (0.29709) and the bound B (0.29709,
qwen/rev_order) are the *same number*. Neither a coincidence nor a relationship — both are
exactly **4 steps of the 60-point log grid** on [0.25, 20.0] (step ln 1.07710 = 0.074272),
and τ_v is an exhaustive argmin on that grid. Two distinct movements landing four grid steps
apart collide on the same value. Do not read it as a correspondence between scaffold
sensitivity and panel sensitivity.

### 4.2 Matched-transport baseline panel (the delta denominator, once maverick post exists)

| Estimand | values | ratio (≤2) | band | median | LOFO max |
|---|---|---:|---|---:|---:|
| τ_v | qwen 1.2811, gemma31 1.0252, glm 1.0252, maverick 1.3798 | 1.3459 ✓ | [1.0252, 1.3798] | 1.1531 | 0.2971 (maverick) |
| T_abs | qwen 2.6365, gemma31 2.2431, glm 2.4639, maverick 2.0351 | 1.2956 ✓ | [2.0351, 2.6365] | 2.3535 | 0.1912 (maverick) |

It differs from the published panel in one cell only (gemma31 T_abs 2.2431 vs 2.2417,
Δln +0.0006) and the τ_v panel is bit-identical, so the median moves 2.3528 → 2.3535 and no
ratio, band edge or LOFO value changes at four decimals.

### 4.3 What can be said about the ≤2 rule and the band without maverick

Only this, and it is stated as arithmetic rather than as a result: under both variants the
three measured families' τ_v values are unchanged from baseline except qwen's (1.2811 →
1.0252 under alt_set, → 0.9518 under rev_order), and glm's T_abs(post) falls to 2.1535 /
2.1694. **Whether the ≤2 rule and the band survive is not determined by these three cells**,
because maverick holds both bands' extreme edges in the baseline panel and its perturbed
value is unmeasured. The honest answer to "which published quantities move, by how much, and
does the ≤2 rule survive" is: **open — the campaign did not reach the measurement that
decides it.**

---

## 5. Cost ledger

Per-instance figures are vast's own invoice rows (GPU + storage + download + upload), not
estimates. Per-leg attribution divides an instance's total by the legs it served; download
and load time are inside those totals and are not separated out.

| Instance | Role | Offer / machine | GPU | $/hr | GPU-h billed | GPU | storage | download | upload | Charge | Legs served | $/leg |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 47246194 | glm PRE | 47225585 / 38592 | 8×H200 | 31.5789 | 0.705 | 22.267 | 0.390 | 12.027 | 0.076 | **$34.760** | glm k5v1 pre, k5v2 pre | $17.380 |
| 47247945 | glm POST | 47173263 / 54969 | 8×H200 | 31.5789 | 0.971 | 30.678 | 0.558 | 12.027 | 0.104 | **$43.367** | glm k5v1 post, k5v2 post | $21.684 |
| 47249569 | maverick PRE | 47247896 / 38592 | 8×H200 | 32.1053 | 0.566 | 18.180 | 0.357 | 13.396 | 0.084 | **$32.017** | maverick k5v1 pre, k5v2 pre | $16.009 |
| 47251690 | maverick POST (failed) | 47248814 / 54969 | 8×H200 | 32.1053 | 0.281 | 9.012 | 0.369 | 13.261 | 0.057 | **$22.699** | none | — |

- Download is charged at **$0.019/GB** on these hosts and is a material line: $12–13.4 per
  box, ~30% of the failed box's total. The 1a-era assumption that bandwidth is negligible
  does not carry to 8×H200 hosts.
- 47251690 is a **dead rental that vast did bill** — unlike Phase 1a's 47229101, which never
  served and cost $0. It reached 749 of ~803 GB of weights and produced no leg. $22.699 is
  the honest cost of the failure.

| Line | Amount |
|---|---:|
| vast total (invoice sum, 4 instances) | **$132.843** |
| OpenRouter | **$0.000** |
| **Phase 1b total** | **$132.843** |

Against the expected band $126–190 and the hard cap $250: **inside the band, 53.1% of the
cap.** Total billed GPU time 2.523 h across four instances (2.242 h if the failed box is
excluded). Vast credit moved 47.6227 → 4.7805 with $90.00 of top-ups in between
($80.00 at 06:04:53, $5.00 at 07:46:33, $5.00 at 08:01:51), reconciling with the invoice sum
to under a cent.

**Fleet teardown verified:** `vastai show instances` returns `[]`, confirmed at 08:07:52 UTC
after the last destroy. Never more than two boxes were alive at once.

---

## 6. Why the fourth rental failed — the operating constraint, stated precisely

The account's automatic top-up is real and it fired three times during this campaign, so the
displayed balance was correctly treated as a replenishing float rather than a spending wall.
But the **replenishment rate**, not the balance, is the binding constraint: `autobill_amount`
and `autobill_threshold` are both **$5.00**, and an 8×H200 burns **$32.11/hr ≈ $0.54/min**.
A $5 top-up buys roughly nine minutes of box; a full maverick-post cycle needs about
thirty-five (≈25 min of 803 GB download, ~6 min of 8-way load, ~5 min of legs).

The observed sequence on 47251690:

| Time (UTC) | Event |
|---|---|
| 07:33 | instance created, weights pulling |
| 07:46:57 | credit hits $0 → instance **stopped** at ~748 GB; $5 top-up lands, restarts |
| 07:51:41 | credit hits $0 again → instance **stopped** at 749 GB |
| ~08:02 | $5 top-up lands; single sanctioned restart attempted |
| 08:02 | vast returns *"Required resources are currently unavailable, state change queued"* — the machine's GPUs were re-let during the stop |
| 08:07:52 | destroyed; fleet empty |

Per the campaign rule (*wait ~2 minutes, retry once; if it still fails, tear down and report
— do not loop*) the box was destroyed after exactly one retry. No second maverick-post
rental was attempted: at $4.78 of credit a fresh box would repeat the same stop-thrash from
a **zero-progress download**, guaranteeing another ~$13 of bandwidth plus GPU time for no
leg. Adding funds is a payment action and is not something this task performs.

This is a different blocker from the one the superseded record described. It is not "the
balance is too small"; it is "the top-up increment is an order of magnitude below the
hardware's burn rate, so a long unrepeatable download cannot be carried across it."

### 6.1 What would close the gap

One 8×H200 rental serving `meta-llama/Llama-4-Maverick-17B-128E-Instruct`, running both
variant post legs while warm into `runs/study_b_maverick_k5v{1,2}` (the pre legs are already
on disk and the resume guard hard-errors on variant mismatch, so the dirs are safe to
target). Budget **$35–45** at current prices, ~35 minutes. The prerequisite is a credit
float of **≥ $50 available at once** — not a larger cumulative spend, but a larger single
increment than $5. Then `python scripts/absolute_estimand_k5variants.py` emits both panel
tables and the panel deltas with no further code changes.

---

## 7. Pre-provisioning gate — prompt budget (discharged, unchanged)

`scripts/scaffold_variant_probe.py` carries the four 1b tokenizers
(`zai-org/GLM-4.5{,-Base}`, `meta-llama/Llama-4-Maverick-17B-128E{,-Instruct}`). Worst-case
live raw prompt + JSON scaffold + 2048 CoT budget + 1, per family-leg × variant, against the
32768 `--max-model-len` pin:

| Family / leg | scaffold | few-shot max | prompt max | required | fits 32768 |
|---|---|---:|---:|---:|:--:|
| glm / pre + post | baseline | 13,760 | 24,229 | 26,281 | YES |
| glm / pre + post | **alt_set** | 14,024 | 24,495 | **26,547** | YES |
| glm / pre + post | rev_order | 13,760 | 24,229 | 26,281 | YES |
| maverick / pre + post | baseline | 13,648 | 23,851 | 25,904 | YES |
| maverick / pre + post | **alt_set** | 13,832 | 24,037 | 26,090 | YES |
| maverick / pre + post | rev_order | 13,648 | 23,851 | 25,904 | YES |

Worst case 26,547 tokens (glm, `alt_set`), 6,221 tokens of headroom. Confirmed in the live
run: no leg hit a context error. Artifact: `runs/scaffold_variant_probe_1b.json`.

---

## 8. Anomalies and honest caveats

1. **The campaign is 6/8 legs, and the missing 2 are the load-bearing ones.** maverick holds
   both LOFO worst cases and both band extremes; without its post legs neither variant panel
   exists. The headline Phase-1b deliverable is not delivered.
2. **glm V1 post fails the B-Q1 contract gate** (0.8833 < 0.90). Kept and reported; the bound
   is bounded-large with or without it (§3.2).
3. **glm V2 post sits exactly on the gate** (0.9000). It passes by the stated rule; it is
   flagged because a single additional parse failure would have flipped it.
4. **Two transient crashes, recovered by checkpoint/resume** (glm V2 pre at cell 85, glm V2
   post at cell 67), identical in kind to Phase 1a's. Crash recovery, never re-elicitation.
5. **One billed dead rental**, $22.699 for zero legs (§5, §6). Phase 1a's dead rental was
   free; this one was not, and the difference is that this one ran long enough to pull 749 GB.
6. **The bound B is unchanged by this campaign** and still rests on qwen. Adding glm moved
   neither the maximum nor the memo row; it did add a third family whose τ_v is inert and a
   first family whose T_abs is not. B now rests on three families over six (family, variant)
   pairs, two of which carry a failing contract leg — better than the two-family estimate
   memo §A.6 called "too thin", but still short of the four-family panel the memo prescribed.
7. **No estimate of the maverick or panel outcome is offered.** §4.3's observation about
   which cells decide the answer is structural, not predictive.
8. **maverick's pre-side numbers are not variant results** and must not be tabulated
   alongside the pair deltas. Both dirs carry a README saying so.
9. **No baseline run dir was written to.** `runs/study_b_glm`, `runs/study_b_maverick`,
   `runs/study_b_qwen`, `runs/study_b_gemma31*` are unmodified. The frozen drivers
   (`absolute_vs_ratio_estimand.py`, `absolute_mechanism_rA1.py`, `e6_r3_arms.py`,
   `q4_range_robustness.py`, `range_robustness_verbalized.py`) were not edited.
10. **Prices moved during the campaign.** The cheapest 8×H200 offer was $31.582/hr at the
    start and $32.105/hr for the later boxes, and only three such offers exist worldwide, so
    the second and third rentals came off different machines than the first. Re-read offers
    before any funded follow-up.

---

## 9. To resume

1. Fund a credit float of **≥ $50 available in one increment** (the constraint is the $5
   top-up granularity, not the cumulative spend — §6).
2. One rental: `scripts/provision_vast.sh up meta-llama/Llama-4-Maverick-17B-128E-Instruct
   post` with `VAST_OFFER_QUERY` for 8×H200, `VAST_DISK=1150`, `VLLM_TP=8`, `--order dph`,
   on-demand only.
3. Both post legs on that one warm box, into the existing dirs:
   `--out runs/study_b_maverick_k5v1 --leg post --scaffold-variant alt_set` and
   `--out runs/study_b_maverick_k5v2 --leg post --scaffold-variant rev_order`,
   `--workers 8 --budget 2048`.
4. `--analyze` per dir; check B-Q1 ≥ 0.90 and positive Murphy resolution; record failures
   rather than resampling; checkpoint/resume for transients only.
5. `python scripts/absolute_estimand_k5variants.py` — the maverick deltas, **both variant
   panel tables** and the panel deltas then compute with no further code changes.
6. Update §2.3, §3 and §4 of this document with the completed read.

---

## 10. Files

Modified (working tree, uncommitted):
- `docs/phase1b_scaffold_sensitivity_results_20260808.md` — this document; the previous
  blocked-on-funds record is replaced by the executed campaign.

Unchanged from the prior session (already committed at `07ed19b`, the branch head this
campaign ran on; no commit was made by this session):
- `scripts/absolute_estimand_k5variants.py` — the four-family map, `ADOPTION_PANEL`,
  `PANEL_OF_RECORD`, `PANEL_BASELINE_DIRS`, `_estimand_panel`, `panel_from_measurements`,
  `panel_delta`, `panel_of_record_check`. **No edit was needed this session** — the run dirs
  it already expects are exactly the ones the campaign created.
- `scripts/scaffold_variant_probe.py` — the four glm/maverick tokenizers.

New run dirs (gitignored): `runs/study_b_glm_k5v1`, `runs/study_b_glm_k5v2` (complete
pairs), `runs/study_b_maverick_k5v1`, `runs/study_b_maverick_k5v2` (pre legs only), each
with a `README.md` recording provenance, instance ids, gates and standing.

Rewritten (gitignored): `runs/estimand_k5variants/estimand_k5variants.json` — now carries
the glm variant measurements and deltas; the Phase-1a per-leg numbers in it are unchanged
and the panel blocks still report both variant panels unavailable.

Unchanged: every frozen driver and every baseline run dir.
