# Phase 1b — k=5 scaffold sensitivity on the full adoption panel (record)

Campaign run 2026-08-09 06:05–08:08 UTC, closed 2026-08-09 12:09–12:44 UTC by a fifth
rental. Elicitation only; no git commits; every artifact below is in the working tree or
under gitignored `runs/`.

Scope: the Phase-1b leg of `spec/plan_2026_08_08_verbalized_arm_gap_closure.md` §3 —
extending the k=5 coverage-preserving scaffold-sensitivity study from the two Phase-1a
families to the remaining adoption-panel families **glm** and **maverick**, so the full
panel {qwen, gemma31, glm, maverick} is measured under perturbation and the published
panel-level constants can be recomputed under each variant. Read under the rules fixed in
advance by `spec/memo_2026_08_08_phase0_prespecified_reads.md` §A, on the Phase-1a outcome
**BOUNDED-LARGE** (B = 0.2971) recorded in
`docs/phase1a_scaffold_sensitivity_results_20260808.md`.

> **Outcome: COMPLETE. Eight of eight (family, variant) pairs collected; both panel tables
> exist.** GLM and maverick are both complete on both variants and both legs, so the full
> adoption panel {qwen, gemma31, glm, maverick} is now measured under each perturbation and
> **the panel-level recomputation — the deliverable Phase 1b exists for — is delivered**
> (§4). Every maverick leg passes the B-Q1 gate with positive Murphy resolution.
>
> The headline: **maverick's τ_v collapses from 1.3798 to 1.0252 under *both* variants**
> — exactly 4 grid steps, the same magnitude as qwen/rev_order — so the pooled bound
> **B = 0.2971 is unchanged in value but no longer rests on a single family or on a
> gate-failing leg**, and the gate-restricted bound rises from 0.2228 to **0.2971**. The
> memo row is **BOUNDED-LARGE** on both computations. At panel level the τ_v panel
> **collapses to a single point** under alt_set (all four families at 1.0252, ratio 1.000,
> LOFO 0.000) and to [0.9518, 1.0252] under rev_order: the ≤2 clustering rule survives
> comfortably under both variants, the **published τ_v band [1.03, 1.38] does not**, and
> the transferred constant **T_J moves 1.1531 → 1.0252 under both** (−0.1176 in logs,
> 1.58 grid steps). The T_abs LOFO bound ≤0.191 survives under both (0.1820 / 0.0856).
>
> Closure spend **$32.57** against a $60 closure cap; **campaign total $165.41** against
> the $250 cap. Fleet empty.
>
> *Campaign history, retained as record.* An earlier session recorded this campaign as
> blocked on a $47.62 credit balance and made no spend. That blocker record was superseded
> by the clarification that the account carries automatic top-up; this document replaced it
> with the executed campaign. The top-up is real — an $80 credit landed at 06:04:53 UTC as
> the first box came up — but it is **not unlimited in rate**, which is what §6 is about,
> and it is what killed the fourth rental at 749 of ~803 GB. The fifth rental (§5.1) closed
> the gap from a $44.75 standing float, and cost less than projected because vast's machine
> 54969 still held the Maverick-Instruct weights the dead fourth rental had pulled.

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
| study_b_maverick_k5v1 | post | alt_set | Maverick-…-Instruct | 120 | 1.0000 | **1.0000** | PASS | 0.5417 | 0.05006 |
| study_b_maverick_k5v2 | pre | rev_order | Maverick-17B-128E | 120 | 0.9667 | 0.9667 | PASS | 0.3017 | 0.02224 |
| study_b_maverick_k5v2 | post | rev_order | Maverick-…-Instruct | 120 | 1.0000 | 0.9917 | PASS | 0.4750 | 0.05667 |

Notes.

- **One gate failure**: glm V1 post, contract_complete 0.8833. Recorded, not repaired — a
  parse failure is an observation in this channel and is never resampled. Its effect on the
  verdict is tested in §3.2. glm V2 post lands exactly on the threshold (0.9000, PASS).
- **Every leg has strictly positive Murphy resolution**, so no leg is a degenerate
  reference. The weakest is maverick V1 pre (0.01840) — as in Phase 1a, the alternate
  exemplar draw costs a base leg more than the order permutation does.
- **Both maverick post legs are the cleanest legs in the campaign**: V1 post is the only
  leg anywhere at contract_complete 1.0000, V2 post at 0.9917. Both were collected
  concurrently on one warm box, 120/120 on the first pass, with no transient and no resume.
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
| maverick / alt_set | 0.3621 | 0.5417 | **1.0252** | 8.1188 | 2.2282 |
| maverick / rev_order | 0.3017 | 0.4750 | **1.0252** | 5.2894 | 2.3632 |

No τ_v is saturated; no T_abs is pegged at a search bound.

### 2.2 The three deltas — glm and maverick

| family / variant | Δargmax(pre), pp | Δargmax(post), pp | Δln τ_v | grid steps | Δln T_abs(post) | Δln T_abs(pre) |
|---|---:|---:|---:|---:|---:|---:|
| glm / alt_set | −5.57 | +6.89 | **0.0000** | 0.00 | **−0.1347** | +0.4550 |
| glm / rev_order | −16.38 | +2.80 | **0.0000** | 0.00 | **−0.1273** | +0.1338 |
| maverick / alt_set | +3.71 | +4.17 | **−0.2971** | 4.00 | **+0.0907** | +0.4966 |
| maverick / rev_order | −2.33 | −2.50 | **−0.2971** | 4.00 | **+0.1495** | +0.0682 |

### 2.3 maverick — the read

maverick is the second family (after qwen) whose ratio estimand moves resolvably, and the
first whose τ_v moves by the **same amount under both perturbations**: 1.3798 → 1.0252,
Δln = −0.29709, exactly **4.00 grid steps** of the 60-point log grid, under alt_set and
rev_order alike. It lands on 1.0252 — the grid point gemma31 and glm already occupy — so
under either variant the panel's ratio estimand is the same value for three families
(alt_set: all four).

Two structural remarks, both arithmetic rather than interpretation:

- The movement is **downward**, the same direction as qwen's under both of its variants.
  Every resolvable τ_v movement observed anywhere in Phase 1a + 1b is downward; none is
  upward, and no family's τ_v rises under any perturbation.
- maverick's absolute estimand moves in the **opposite** direction from glm's:
  T_abs(post) rises (+0.0907 alt_set, +0.1495 rev_order) where glm's fell (−0.1347,
  −0.1273). Since maverick holds the T_abs band's *lower* edge and glm sits mid-band, the
  two movements compress the T_abs panel from both sides — which is what §4 measures.

maverick is the only family in the campaign whose τ_v and T_abs(post) **both** move beyond
noise under both variants.

---

## 3. Bound criterion and the memo row

### 3.1 Pooled over Phase 1a + Phase 1b

All **eight** (family, variant) pairs the campaign aimed at now exist — qwen×2, gemma31×2,
glm×2, maverick×2 — over the full adoption panel.

| family / variant | Δln τ_v | Δln T_abs(post) | max\|Δargmax\|, pp |
|---|---:|---:|---:|
| qwen / alt_set | −0.2228 | +0.0138 | 5.86 |
| qwen / rev_order | **−0.2971** | **−0.1643** | 3.38 |
| gemma31 / alt_set | 0.0000 | +0.0040 | 11.67 |
| gemma31 / rev_order | 0.0000 | −0.0579 | **16.67** |
| glm / alt_set | 0.0000 | −0.1347 | 6.89 |
| glm / rev_order | 0.0000 | −0.1273 | 16.38 |
| maverick / alt_set | **−0.2971** | +0.0907 | 4.17 |
| maverick / rev_order | **−0.2971** | +0.1495 | 2.50 |

- **max \|Δln τ_v\| = 0.2971** — a **three-way tie** at exactly 4.00 grid steps
  (qwen / rev_order, maverick / alt_set, maverick / rev_order). The tie is a quantization
  artefact of the 60-point log grid, not a coincidence of magnitudes: τ_v is an exhaustive
  argmin on that grid and three distinct movements each landed four steps down.
- **max \|Δln T_abs(post)\| = 0.1643** (qwen / rev_order) — unchanged by Phase 1b;
  maverick's +0.1495 is the second largest and glm's −0.1347 the third.
- **B = max over legs and variants of { \|Δln τ_v\|, \|Δln T_abs\| } = 0.2971**.
  Its *value* is unchanged by the closure, but its *standing* is not: it was attained by
  one family on one variant, and is now attained by **two families on three of the eight
  pairs**.
- vs one τ_v grid step (0.0743): **4.00 steps**; vs the LOFO worst case (0.191): **1.56×**.
- max \|Δargmax\| = **16.67 pp** (gemma31 / rev_order, post leg), with glm / rev_order's pre
  leg a close second at 16.38 pp. maverick's argmax movements are the *smallest* in the
  campaign (4.17 / 2.50 pp) even though its τ_v movement is the largest — a direct instance
  of the argmax-invariance the paper's utility exhibit is built on.

### 3.2 Robustness to the gate failures

Recomputing B over the six pairs whose *both* legs pass the B-Q1 gate — i.e. dropping
qwen / rev_order and glm / alt_set — now gives **B = 0.2971** (maverick / rev_order /
Δln τ_v; maverick / alt_set ties it), max \|Δargmax\| 16.67 pp. Before the closure this
gate-restricted bound was 0.2228; **the closure raises it to the full-set value**, so the
full and gate-restricted computations now agree exactly. The verdict was already
bounded-large either way; it is now bounded-large at the *same number* either way, and the
bound no longer depends on any leg that fails the contract gate.

### 3.3 The memo row

Memo §A.5 defines: bounded-small = B < 0.0743 **and** max\|Δargmax\| below ~3 pp;
bounded-comparable-to-LOFO = 0.0743 ≤ B ≤ 0.191; bounded-large = B > 0.191.

> **Outcome: BOUNDED-LARGE**, on both the full and the gate-restricted computation, at
> B = 0.2971 in both cases, over all eight (family, variant) pairs of the full adoption
> panel. This is the final Phase-1b read; no measurement remains outstanding.

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
  not. With maverick measured, the final split is **two families inert** (gemma31, glm —
  both already sitting at 1.0252) and **two families that move, both downward** (qwen 3–4
  steps, maverick 4 steps under both variants). The families whose τ_v is inert are exactly
  the two already at the panel's lower grid point; the families that move are the two above
  it, and both move down onto it. Under alt_set every family ends at 1.0252.
- **glm's absolute estimand does move**, and in the same direction under both variants:
  T_abs(post) falls by 0.13 in logs (−0.1347 alt_set, −0.1273 rev_order), roughly 1.8 grid
  steps' worth. glm is the only family where the T_abs side moves materially while the τ_v
  side is inert — the mirror image of gemma31. The pattern from Phase 1a holds and
  sharpens: **which estimand is sensitive is family-specific, not scaffold-specific.**
  maverick is the one family where *both* sides move.
- **The two T_abs movements have opposite signs and both compress the panel.** glm's
  T_abs(post) falls from mid-band; maverick's rises off the band's lower edge (+0.0907,
  +0.1495). qwen, the band's upper edge, falls under rev_order (−0.1643) and is flat under
  alt_set (+0.0138). The net effect at panel level is a narrower T_abs band under both
  variants (§4.3) — the movements are not independent noise around the published values;
  they run inward.
- **T_abs(pre) inflates under `alt_set` on every family measured**: gemma31 +0.7209,
  maverick +0.4966, glm +0.4550, qwen +0.0118. Three of four move by roughly half a log
  unit in the same direction. This is excluded from B by the memo's definition (which takes
  T_abs on the *post* leg) and is reported as an observation: the disjoint exemplar draw
  consistently makes base legs more in need of absolute correction. It is the single
  largest movement class in the whole 1a+1b set.
- **Δargmax against the stated expectation (memo §A.4).** The expectation recorded before
  any data was that coverage-preserving deltas would be materially smaller than the
  coverage-violating k=4 E-hole deltas (+8.3/+10.8/+10.8/+17.5 pp). On argmax they are still
  not: glm / rev_order moves the base leg −16.38 pp, alongside gemma31's 16.67 pp. The memo
  pre-committed to the reading of exactly this case — the instrument is sensitive to the
  scaffold rather than only to category coverage, which is a finding in its own right.
  maverick is the counter-example inside the same set: its argmax barely moves (4.17 /
  2.50 pp, the two smallest movements in the campaign) while its τ_v moves the maximum
  observed. Argmax movement and estimand movement are not proxies for one another in
  either direction.

---

## 4. Panel-level recomputation — DELIVERED

| Panel | Availability |
|---|---|
| baseline, published pairing | AVAILABLE — reproduces every published constant exactly (§4.1) |
| baseline, matched transport | AVAILABLE (§4.2) — the delta denominator |
| **V1 `alt_set` panel** | **AVAILABLE** (§4.3) — all four families |
| **V2 `rev_order` panel** | **AVAILABLE** (§4.4) — all four families |

Every panel statistic — the max/min cluster ratio against the ≤2 rule, the band, the
interpolated median (the T_J / T_A analog) and the LOFO transfer bound — is defined over the
whole four-family panel, so no subset panel was ever computed or reported. With maverick's
post legs collected, both variant panels are complete and are reported below against the
matched-transport baseline (§4.2), which is the like-for-like denominator.

Note on the two edges: **both baseline LOFO worst cases are attained on maverick** — it is
the τ_v band's upper edge (1.3798) and the T_abs band's lower edge (2.0351) — so maverick
was the family whose movement would decide both the cluster ratio and the bound. It moved,
on both estimands, under both variants. That is why §4.3 and §4.4 look the way they do.

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

Noted in passing, not a finding: the τ_v LOFO worst case (0.29709) and the bound B (0.29709)
are the *same number*. Neither a coincidence nor a relationship — both are exactly **4 steps
of the 60-point log grid** on [0.25, 20.0] (step ln 1.07710 = 0.074272), and τ_v is an
exhaustive argmin on that grid. Distinct movements landing four grid steps apart collide on
the same value; with the closure collected, **three** such movements now do (qwen/rev_order,
maverick/alt_set, maverick/rev_order). Do not read the collision as a correspondence between
scaffold sensitivity and panel sensitivity. The mechanical reason it recurs is that maverick
is 4 steps above the 1.0252 grid point in the baseline panel, which is both what makes it
the LOFO worst case and what its τ_v movement traverses.

### 4.2 Matched-transport baseline panel (the delta denominator)

| Estimand | values | ratio (≤2) | band | median | LOFO max |
|---|---|---:|---|---:|---:|
| τ_v | qwen 1.2811, gemma31 1.0252, glm 1.0252, maverick 1.3798 | 1.3459 ✓ | [1.0252, 1.3798] | 1.1531 | 0.2971 (maverick) |
| T_abs | qwen 2.6365, gemma31 2.2431, glm 2.4639, maverick 2.0351 | 1.2956 ✓ | [2.0351, 2.6365] | 2.3535 | 0.1912 (maverick) |

It differs from the published panel in one cell only (gemma31 T_abs 2.2431 vs 2.2417,
Δln +0.0006) and the τ_v panel is bit-identical, so the median moves 2.3528 → 2.3535 and no
ratio, band edge or LOFO value changes at four decimals.

### 4.3 THE V1 PANEL — `alt_set` (disjoint alternate exemplar set)

Four-family panel rebuilt entirely from legs elicited under V1; compared against §4.2.

| Estimand | values | ratio (≤2) | band | median | LOFO max \|ln\| |
|---|---|---:|---|---:|---:|
| τ_v | qwen 1.0252, gemma31 1.0252, glm 1.0252, maverick 1.0252 | **1.0000** ✓ | **[1.0252, 1.0252]** | **1.0252** | **0.0000** (gemma31) |
| T_abs | qwen 2.6731, gemma31 2.2521, glm 2.1535, maverick 2.2282 | **1.2413** ✓ | [2.1535, 2.6731] | 2.2402 | **0.1820** (qwen) |

Movement of every published quantity, against the matched-transport baseline:

| Published quantity | baseline | V1 `alt_set` | movement |
|---|---:|---:|---|
| τ_v cluster ratio (published 1.346) | 1.3459 | **1.0000** | −0.3459 |
| τ_v ≤2 rule | PASS | **PASS** | **survives** |
| τ_v band (published [1.03, 1.38]) | [1.0252, 1.3798] | **[1.0252, 1.0252]** | low Δln 0.0000, high Δln **−0.2971** |
| τ_v interpolated median = **T_J** (published 1.153) | 1.1531 | **1.0252** | Δln **−0.1176** = 1.58 grid steps |
| τ_v LOFO max \|ln\| | 0.2971 | **0.0000** | −0.2971 |
| T_abs cluster ratio (published 1.296) | 1.2956 | **1.2413** | −0.0543 |
| T_abs ≤2 rule | PASS | **PASS** | **survives** |
| T_abs band (published [2.0351, 2.6365]) | [2.0351, 2.6365] | **[2.1535, 2.6731]** | low Δln **+0.0566**, high Δln **+0.0138** |
| T_abs interpolated median = **T_A** (published 2.353) | 2.3535 | **2.2402** | Δln **−0.0494** |
| T_abs LOFO max \|ln\| (**published ≤0.191**) | 0.1912 | **0.1820** | −0.0092, **bound survives** |

Verdicts, stated plainly:

- **The ≤2 clustering rule SURVIVES**, on both estimands, by a wider margin than in the
  published panel (τ_v 1.3459 → 1.0000; T_abs 1.2956 → 1.2413). The perturbation makes the
  panel *more* clustered, not less.
- **The published τ_v band DOES NOT SURVIVE as a band.** It collapses to the single point
  1.0252: all four families land on the same grid point, so the interval is degenerate. It
  is contained inside the published [1.03, 1.38], but "the band is reproduced" would be
  false — the band's whole content is the spread, and the spread is zero.
- **The published T_abs band does not survive either**, in the opposite way: its lower edge
  rises (2.0351 → 2.1535, +0.0566) and its upper edge rises *past* the published upper edge
  (2.6365 → 2.6731, +0.0138), so the V1 band is not contained in the published one.
- **The LOFO bound SURVIVES.** T_abs LOFO 0.1820 ≤ 0.191; τ_v LOFO falls to exactly 0.0000,
  because a degenerate panel transfers perfectly by construction.
- **T_J moves.** 1.1531 → 1.0252, 1.58 grid steps down. Cross-reference to the record, not
  an interpretation: 1.0252 is the value that **fails F4** on the executed on-pair sweep
  (`spec/results_2026_07_21_r1_onpair_sweep.md`: arm 1 passes at T_J = 1.153, band lower
  edge 1.0252 fails). Under V1 the panel's transferred constant is that value.

Gate standing of this panel: `all legs pass B-Q1: False` — **glm V1 post is 0.8833** and is
the sole failing leg. Every other V1 leg passes; maverick V1 post is 1.0000, the cleanest
leg in the campaign.

### 4.4 THE V2 PANEL — `rev_order` (reversed presentation order)

| Estimand | values | ratio (≤2) | band | median | LOFO max \|ln\| |
|---|---|---:|---|---:|---:|
| τ_v | qwen 0.9518, gemma31 1.0252, glm 1.0252, maverick 1.0252 | **1.0771** ✓ | **[0.9518, 1.0252]** | **1.0252** | **0.0743** (qwen) |
| T_abs | qwen 2.2370, gemma31 2.1170, glm 2.1694, maverick 2.3632 | **1.1163** ✓ | [2.1170, 2.3632] | 2.2032 | **0.0856** (maverick) |

| Published quantity | baseline | V2 `rev_order` | movement |
|---|---:|---:|---|
| τ_v cluster ratio (published 1.346) | 1.3459 | **1.0771** | −0.2688 |
| τ_v ≤2 rule | PASS | **PASS** | **survives** |
| τ_v band (published [1.03, 1.38]) | [1.0252, 1.3798] | **[0.9518, 1.0252]** | low Δln **−0.0743**, high Δln **−0.2971** |
| τ_v interpolated median = **T_J** (published 1.153) | 1.1531 | **1.0252** | Δln **−0.1176** = 1.58 grid steps |
| τ_v LOFO max \|ln\| | 0.2971 | **0.0743** | −0.2228 |
| T_abs cluster ratio (published 1.296) | 1.2956 | **1.1163** | −0.1793 |
| T_abs ≤2 rule | PASS | **PASS** | **survives** |
| T_abs band (published [2.0351, 2.6365]) | [2.0351, 2.6365] | **[2.1170, 2.3632]** | low Δln **+0.0394**, high Δln **−0.1095** |
| T_abs interpolated median = **T_A** (published 2.353) | 2.3535 | **2.2032** | Δln **−0.0660** |
| T_abs LOFO max \|ln\| (**published ≤0.191**) | 0.1912 | **0.0856** | −0.1057, **bound survives** |

Verdicts, stated plainly:

- **The ≤2 clustering rule SURVIVES**, on both estimands, again by a wider margin than
  published (τ_v 1.0771, T_abs 1.1163).
- **The published τ_v band DOES NOT SURVIVE.** The panel's τ_v band is [0.9518, 1.0252]:
  its upper edge falls onto the published *lower* edge, and its lower edge falls **one grid
  step below the published band entirely** (0.9518 < 1.0252). The V2 band and the published
  band share exactly one point.
- **The published T_abs band is not reproduced but is strictly contained**: [2.1170, 2.3632]
  ⊂ [2.0351, 2.6365]. Both edges move inward; the panel narrows.
- **The LOFO bound SURVIVES**, comfortably: T_abs 0.0856 ≤ 0.191, τ_v 0.0743 (exactly one
  grid step, qwen).
- **T_J moves to 1.0252**, identically to V1 — because under both variants three of four
  families sit on that grid point and the interpolated median takes it.

Gate standing of this panel: `all legs pass B-Q1: False` — **qwen V2 post is 0.8000** and is
the sole failing leg. glm V2 post sits exactly on the threshold (0.9000, PASS); maverick V2
post is 0.9917.

### 4.5 What the two panels say together

Stated as arithmetic, without extrapolation beyond the memo's table:

1. **The ≤2 clustering rule is the robust part of the published stability story.** It holds
   under both perturbations, on both estimands, with more margin than the published panel
   has. Nothing in this campaign threatens it.
2. **The band is the fragile part.** Under both variants the τ_v panel collapses toward the
   1.0252 grid point; the published [1.03, 1.38] interval is not recovered under either.
   Under V2 the panel exits the published band at the bottom.
3. **The transferred constant T_J moves the same way under both variants**: 1.1531 →
   1.0252, 1.58 grid steps. The direction and magnitude are variant-independent even though
   the underlying per-family movements are not.
4. **T_A moves less than T_J**, and in the same direction: 2.3535 → 2.2402 (V1) / 2.2032
   (V2), Δln −0.0494 / −0.0660, both under one τ_v grid step of movement.
5. **The published LOFO bound ≤0.191 is not breached by either variant panel** (0.1820,
   0.0856). Panel-composition sensitivity does *not* get worse under scaffold perturbation;
   it gets better, because the panels are more clustered. The perturbation's cost is
   located in the *level* of the transferred constant, not in its transferability.

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
| 47268347 | maverick POST (closure) | 47253385 / 54969 | 8×H200 | 31.5789 | 0.592 | 18.703 | 0.378 | 13.396 | 0.094 | **$32.571** | maverick k5v1 post, k5v2 post | $16.286 |

- Download is charged at **$0.019/GB** on these hosts and is a material line: $12–13.4 per
  box, ~30% of the failed box's total and 44% of the closure box's. The 1a-era assumption
  that bandwidth is negligible does not carry to 8×H200 hosts.
- 47251690 is a **dead rental that vast did bill** — unlike Phase 1a's 47229101, which never
  served and cost $0. It reached 749 of ~803 GB of weights and produced no leg. $22.699 is
  the honest cost of the failure.
- The failure was **not wholly wasted**: the closure rental landed on the same machine
  (54969) and its weight pull completed in about 19 minutes to the same 749 GB mark, which
  is why the closure box billed 0.592 GPU-h rather than the ~0.8 h projected.
- **Read the invoice after teardown, not before.** The closure box's GPU line settled at
  0.592 h / $18.703 after destroy; a mid-run read of the same row showed 0.527 h / $16.640.
  Only the post-teardown figure is the invoice, and it is the one tabulated above.

### 5.1 The closure rental (2026-08-09, 12:09–12:44 UTC)

| Step | Time (UTC) | Note |
|---|---|---|
| offer search | 12:08 | 3 live 8×H200 on-demand offers worldwide; cheapest by `--order dph` = 47253385, machine 54969, $31.5789/hr, well under the $36/hr stop-and-report threshold |
| instance created | 12:09:54 | `VAST_DISK=1150`, `VLLM_TP=8`, `--enable-expert-parallel`, bf16, `--max-model-len 32768`, `--gpu-memory-utilization 0.85` |
| weights pulled | 12:11 – 12:28 | 749 GB; credit $44.75 → $21.47 |
| 8-way load | 12:28 – 12:35 | 55 safetensors shards; EP 16/128 experts per rank |
| endpoint ready | 12:36:45 | `GET /v1/models 200` |
| both post legs | 12:37 – 12:41 | run **concurrently** on the one warm box, `--workers 8` each; 120/120 both, no transient, no resume |
| `--analyze` ×2 | 12:41 – 12:43 | both gates PASS |
| destroyed | 12:44 | `vastai show instances` → `[]` |

Running both legs at once against a single warm 8×H200 box is the change that made the
closure cheap: the elicitation phase cost about **4 minutes of GPU total** for two legs,
against ~5 min *per leg* sequentially on the earlier boxes. Nothing about the legs changed —
same driver, same flags, same 120 cells — only that they shared the batching window.

| Line | Amount |
|---|---:|
| vast, first campaign (invoice sum, 4 instances) | $132.843 |
| vast, closure rental (47268347) | $32.571 |
| OpenRouter | $0.000 |
| **Phase 1b total (5 instances)** | **$165.414** |

Against the closure's own hard cap of **$60**: the closure spent **$32.571, 54.3% of it**.
Against the Phase-1b expected band $126–190 and the hard cap $250: **$165.414 is inside the
band, 66.2% of the cap.** Total billed GPU time **3.115 h** across five instances (2.834 h
excluding the dead box).

Credit reconciliation for the closure: $44.7524 → $12.1818, a movement of **$32.5706**
against a settled invoice sum of $32.571 — to a hundredth of a cent. **No auto top-up fired
during the closure**: the balance never approached the $5 threshold, which is exactly the
condition §6 said was needed.

**Fleet teardown verified:** `vastai show instances` returns `[]`, confirmed at 08:07:52 UTC
after the first campaign's last destroy and again at 12:44 UTC after the closure destroy.
Never more than two boxes were alive at once; at the close, none.

---

## 6. Why the fourth rental failed — the operating constraint, stated precisely

*Retained verbatim as the campaign record. The constraint it describes is real and was the
sole reason the first campaign stopped at 6/8 legs; §6.2 records how the closure satisfied
it.*

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

### 6.1 What would close the gap (the plan, as written before the closure)

One 8×H200 rental serving `meta-llama/Llama-4-Maverick-17B-128E-Instruct`, running both
variant post legs while warm into `runs/study_b_maverick_k5v{1,2}` (the pre legs are already
on disk and the resume guard hard-errors on variant mismatch, so the dirs are safe to
target). Budget **$35–45** at current prices, ~35 minutes. The prerequisite is a credit
float of **≥ $50 available at once** — not a larger cumulative spend, but a larger single
increment than $5. Then `python scripts/absolute_estimand_k5variants.py` emits both panel
tables and the panel deltas with no further code changes.

### 6.2 How the closure discharged it — DONE

Executed 2026-08-09 12:09–12:44 UTC exactly as §6.1 specified, from a standing float of
**$44.75** (below the $50 stated prerequisite, but the diagnosis was the right one: what
mattered was that the float exceeded the download's cost, not that it hit a round number).
Ledger in §5.1. Three things went better than the plan predicted:

1. **The float never came near the $5 threshold.** It stood at $14.29 at teardown and
   $12.18 after final settlement, so the $5-increment top-up mechanism was never exercised
   and the stop-thrash of §6 could not recur. The constraint §6 identified is confirmed by its absence: with a float larger than
   the download, the same hardware, the same model and the same machine completed without
   incident.
2. **The weight pull was faster and cheaper than modelled** (~19 min, $13.40) because
   machine 54969 still held the shards the dead fourth rental had pulled.
3. **Both legs ran concurrently on one warm box**, so elicitation cost ~4 GPU-minutes total
   instead of ~10 sequential. Total $32.57 against the $35–45 projection.

No retry, no transient, no resume, and one rental — the §6.1 plan executed once and worked.

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
   exists. **CLOSED 2026-08-09 by the fifth rental (§5.1, §6.2); the campaign is 8/8 pairs
   and the deliverable is delivered (§4).** Retained as the record of the first campaign.
2. **glm V1 post fails the B-Q1 contract gate** (0.8833 < 0.90). Kept and reported; the bound
   is bounded-large with or without it (§3.2). It is the sole failing leg in the V1 panel.
3. **qwen V2 post fails the gate too** (0.8000, Phase 1a) and is the sole failing leg in the
   V2 panel. So **each variant panel carries exactly one gate-failing leg**, and they are
   different families. Neither panel's headline movement is produced by its failing leg:
   under V1 the τ_v collapse is driven by qwen and maverick, and glm's τ_v does not move at
   all; under V2 the collapse is driven by maverick, whose legs both pass.
4. **glm V2 post sits exactly on the gate** (0.9000). It passes by the stated rule; it is
   flagged because a single additional parse failure would have flipped it.
5. **Two transient crashes, recovered by checkpoint/resume** (glm V2 pre at cell 85, glm V2
   post at cell 67), identical in kind to Phase 1a's. Crash recovery, never re-elicitation.
   The closure rental had none.
6. **One billed dead rental**, $22.699 for zero legs (§5, §6). Phase 1a's dead rental was
   free; this one was not, and the difference is that this one ran long enough to pull 749 GB.
7. **B's value is unchanged by this campaign; its standing is not.** B = 0.2971 both before
   and after, but at the close it is attained on **three** of eight (family, variant) pairs
   by **two** families (qwen/rev_order, maverick/alt_set, maverick/rev_order), and the
   gate-restricted computation now returns the same 0.2971 rather than 0.2228. The bound no
   longer depends on any leg that fails the contract gate, and it is computed over the full
   four-family panel the memo prescribed rather than the "too thin" subset of §A.6.
8. **The three-way tie at 0.2971 is grid quantization, not a physical coincidence.** τ_v is
   an exhaustive argmin on a 60-point log grid; three movements of exactly 4 steps produce
   the identical log-delta. Do not report it as three independent measurements agreeing.
9. **Both variant τ_v panels are degenerate or near-degenerate**, so their LOFO values
   (0.0000 and 0.0743) are small *by construction* — a panel whose members share a grid
   point transfers perfectly. The T_abs LOFO values (0.1820, 0.0856) carry the real
   information about transferability under perturbation.
10. **maverick's pre-side numbers are now part of complete pairs.** The earlier caveat that
    they were a partial diagnostic no longer applies; both dirs' READMEs were rewritten.
11. **No baseline run dir was written to.** `runs/study_b_glm`, `runs/study_b_maverick`,
    `runs/study_b_qwen`, `runs/study_b_gemma31*` are unmodified. The frozen drivers
    (`absolute_vs_ratio_estimand.py`, `absolute_mechanism_rA1.py`, `e6_r3_arms.py`,
    `q4_range_robustness.py`, `range_robustness_verbalized.py`) were not edited, and
    `scripts/absolute_estimand_k5variants.py` needed no edit either.
12. **Prices moved during the campaign.** The cheapest 8×H200 offer was $31.582/hr at the
    start, $32.105/hr for the middle boxes, and $31.582/hr again for the closure, and only
    three such offers exist worldwide, so the rentals came off different machines. Re-read
    offers before any funded follow-up.
13. **The closure ran both post legs concurrently on one box.** The legs are independent
    (different dirs, different scaffold variants, per-cell checkpoints) and nothing about
    the elicitation contract changed, but they shared a vLLM batching window, which no
    earlier leg in this campaign did. Recorded in case any future comparison is sensitive
    to serving concurrency.

---

## 9. Resume steps — ALL DISCHARGED

The six steps this section carried are done, in order, on 2026-08-09:

| # | Step | Status |
|---|---|---|
| 1 | credit float ≥ the download's cost, available at once | DONE — $44.75 standing float; $12.18 after final settlement; no top-up fired |
| 2 | one 8×H200 on-demand rental, `--order dph`, `VAST_DISK=1150`, `VLLM_TP=8` | DONE — instance 47268347, offer 47253385, machine 54969, $31.5789/hr |
| 3 | both post legs on the warm box into the existing dirs, `--workers 8 --budget 2048` | DONE — 120/120 each, run concurrently, resume guard accepted both same-variant post legs |
| 4 | `--analyze` per dir; B-Q1 ≥ 0.90 and positive Murphy resolution | DONE — V1 post 1.0000 / res 0.05006; V2 post 0.9917 / res 0.05667; both PASS |
| 5 | `python scripts/absolute_estimand_k5variants.py` | DONE — no code change needed; both panel tables and both panel deltas emitted |
| 6 | update §2.3, §3 and §4 with the completed read | DONE — this revision |

**Nothing is outstanding in Phase 1b.** The open work is downstream: carrying the
BOUNDED-LARGE row and the two panel tables into Paper B v15's Limitations and sensitivity
appendix, per memo §A.5. That is a writing action on the paper repo, not an elicitation.

---

## 10. Files

Modified (working tree, uncommitted; branch `claude-verbalized-gap-closure`, no commit made
by either session):
- `docs/phase1b_scaffold_sensitivity_results_20260808.md` — this document. The
  blocked-on-funds record was replaced by the executed campaign; the
  panel-unavailability record is now replaced by the actual panel tables (§4.3, §4.4,
  §4.5), with the campaign history — including the top-up-rate incident (§6) — kept intact.

Rewritten (gitignored):
- `runs/study_b_maverick_k5v1/README.md`, `runs/study_b_maverick_k5v2/README.md` — the
  `INCOMPLETE PAIR — pre leg only` marking is removed; each now records both legs, both
  instance ids, gates, Murphy resolutions, estimands and deltas.
- `runs/estimand_k5variants/estimand_k5variants.json` — now carries all eight
  (family, variant) measurements and deltas, both variant panels as `available: true`, and
  both panel-delta blocks. The Phase-1a per-leg numbers in it are unchanged.

New leg artifacts (gitignored): `runs/study_b_maverick_k5v1/post_verbalized.json` +
`.meta.json`, `runs/study_b_maverick_k5v2/post_verbalized.json` + `.meta.json`, and the
rescored `study_b_report.json` in each of those two dirs.

Unchanged:
- `scripts/absolute_estimand_k5variants.py` — **no edit was needed in either session**; the
  run dirs it already expects are exactly the ones the campaign created.
- `scripts/scaffold_variant_probe.py`.
- Every frozen driver (`absolute_vs_ratio_estimand.py`, `absolute_mechanism_rA1.py`,
  `e6_r3_arms.py`, `q4_range_robustness.py`, `range_robustness_verbalized.py`) and every
  baseline run dir (`runs/study_b_glm`, `runs/study_b_maverick`, `runs/study_b_qwen`,
  `runs/study_b_gemma31*`).
