# Phase 1a — k=5 coverage-preserving scaffold sensitivity (results)

Campaign run 2026-08-09 01:53–03:01 UTC. Elicitation only; no git commits; every artifact
below is in the working tree or under gitignored `runs/`.

Scope: the Phase-1a leg of `spec/plan_2026_08_08_verbalized_arm_gap_closure.md` §3, read
under the rules fixed in advance by `spec/memo_2026_08_08_phase0_prespecified_reads.md` §A.
Two coverage-preserving perturbations of the shipped k=5 exemplar scaffold, elicited on the
verbalized channel over the same 120 AIReg-Bench cells, on two families:

- **V1 `alt_set`** — same one-per-level walk, the shipped set's `source_item_label`s passed
  to `exclude`, giving a fully disjoint draw. Coverage-complete.
- **V2 `rev_order`** — the shipped rows rendered `very_high → very_low`. Same rows, same
  coverage, different position.

Neither reproduces the deprecated k=4 E-hole: both carry one exemplar per ordinal level.

---

## 0. Design change adopted mid-campaign (2026-08-09)

The original plan paired gemma31's bf16 raw-completions **pre** leg with an **OpenRouter**
chat **post** leg, mirroring the shipped `runs/study_b_gemma31_api` pairing. A directive
arriving during the first API leg required **quantization-matched serving**: the post twin
must be served at the same floating-point precision as the base leg, self-hosted by
preference.

What changed:

1. The OpenRouter V1 post leg was **stopped at 74 of 120 cells** (all 74 parse_ok) and the
   V2 API leg was never started. The partial artifact is kept as a labeled transport annex
   (`runs/study_b_gemma31_api_k5v1/README.md`), not as an input to any read.
2. gemma31 was re-collected end to end on our own boxes at bf16 — raw `/v1/completions`
   pre, vLLM `/v1/chat/completions` post. The chat template is a precondition, not a
   preference: gemma-4 post collapses under greedy raw continuation (30.8%
   contract-complete, the recorded pathology in `runs/study_b_gemma31/`).
3. Because the shipped gemma31 post came from a different provider stack, a **matched
   transport baseline** post leg was collected on the same box at the same dtype under the
   shipped scaffold: `runs/study_b_gemma31_vllmchat_base`. **That is the delta reference
   for gemma31** — not the OpenRouter-era τ_v 1.0252 / T_abs 2.2417.

Consequence for the read: every gemma31 delta below is same-box, same-dtype,
same-transport on both sides, with the scaffold as the only difference. The qwen arm needed
no change; both of its legs were already bf16 vLLM raw completions on our own box.

### Code written for this (working tree, uncommitted)

`--base-url` on `scripts/run_study_b_api_leg.py` and
`src/judex_calibration/elicit_api_verbalized.py`: when set, the chat instrument targets a
self-hosted OpenAI-compatible endpoint — no Keychain read, no `Authorization` header, no
OpenRouter provider-preference block, channel `verbalized_vllm_chat`, and the sidecar
records the served model id, `dtype: bfloat16`, and the endpoint host. When absent the
OpenRouter path is byte-identical, asserted in a test down to the request body's field
order. `endpoint_host` is provenance, not leg identity, and is excluded from the resume
guard (`UNGUARDED_META`) so a re-provisioned box can still resume a crashed leg; scaffold
and transport mismatches still hard-error. Suite: **116 passed, 5 subtests** with the
judex-arm python (was 108 + 5 before this pass; +8 new tests in
`tests/test_scaffold_variants.py::SelfHostedChatTests`).

Prompt budget was re-measured for the chat path with gemma-4-31b-it's own tokenizer before
any spend: worst-case chat prompt **25,302 tokens** (`alt_set`) + 4,096 output = 29,398,
inside the 32,768 pin with 3,370 tokens of headroom. No serving change needed.

---

## 1. Per-leg inventory and B-Q1 gates

All legs: bf16, `--max-model-len 32768`, `--gpu-memory-utilization 0.85`, `--workers 8`,
budget 2048, 120 cells, temperature 0. Gate = contract_complete ≥ 0.90.

| Run dir | leg | variant | channel | n | parse | contract_complete | gate | argmax | Murphy resolution |
|---|---|---|---|---:|---:|---:|:--:|---:|---:|
| study_b_qwen (baseline) | pre | baseline | verbalized | 120 | 0.9167 | 0.9167 | PASS | 0.4818 | 0.03366 |
| study_b_qwen (baseline) | post | baseline | verbalized | 120 | 0.9833 | 0.9583 | PASS | 0.6017 | 0.05732 |
| study_b_qwen_k5v1 | pre | alt_set | verbalized | 120 | 0.9833 | 0.9833 | PASS | 0.4237 | 0.04209 |
| study_b_qwen_k5v1 | post | alt_set | verbalized | 120 | 0.9667 | 0.9000 | PASS | 0.5431 | 0.05564 |
| study_b_qwen_k5v2 | pre | rev_order | verbalized | 120 | 0.9500 | 0.9500 | PASS | 0.4737 | 0.04135 |
| study_b_qwen_k5v2 | post | rev_order | verbalized | 120 | 0.8917 | **0.8000** | **FAIL** | 0.6355 | 0.05716 |
| study_b_gemma31_vllmchat_base | pre | baseline | verbalized | 120 | 1.0000 | 1.0000 | PASS | 0.4417 | 0.04037 |
| study_b_gemma31_vllmchat_base | post | baseline | verbalized_vllm_chat | 120 | 1.0000 | 0.9583 | PASS | 0.6417 | 0.05565 |
| study_b_gemma31_k5v1 | pre | alt_set | verbalized | 120 | 1.0000 | 0.9917 | PASS | 0.3250 | 0.02601 |
| study_b_gemma31_k5v1 | post | alt_set | verbalized_vllm_chat | 120 | 1.0000 | 0.9917 | PASS | 0.7167 | 0.05696 |
| study_b_gemma31_k5v2 | pre | rev_order | verbalized | 120 | 0.9917 | 0.9917 | PASS | 0.4034 | 0.03500 |
| study_b_gemma31_k5v2 | post | rev_order | verbalized_vllm_chat | 120 | 1.0000 | 0.9917 | PASS | 0.4750 | 0.05461 |

Notes.

- **One gate failure**: qwen V2 post, contract_complete 0.80. Recorded, not repaired — a
  parse failure is an observation in this channel and is never resampled. Its effect on the
  verdict is tested in §3.
- **No repetition collapse anywhere.** The historical gemma31 post pathology does not recur
  on the chat endpoint (parse rate 1.000 on all three vllmchat post legs), and no pre leg on
  either family shows an all-or-nothing failure signature. Every leg has strictly positive
  Murphy resolution, so no leg is a degenerate reference.
- gemma31 V1's **pre** leg is the weakest capability read in the set (argmax 0.3250,
  resolution 0.02601) — the alternate exemplar draw costs that base leg more than any other
  perturbation here costs any leg.

---

## 2. The pre-specified read (memo §A.2)

τ_v is an exhaustive argmin over a 60-point logarithmic grid on [0.25, 20.0]; one grid step
is |Δln| = 0.0743, so a smaller movement is *not resolvable* and is reported as no observed
movement. T_abs is a golden-section refine, so its deltas are continuous.

Baselines (recomputed from the run dirs, reproducing exactly):

| Family | reference dir | τ_v | T_abs(pre) | T_abs(post) |
|---|---|---:|---:|---:|
| qwen | study_b_qwen | 1.281052 | 3.1901 | 2.6365 |
| gemma31 | study_b_gemma31_vllmchat_base | 1.025179 | 2.6285 | 2.2431 |

The qwen row is the memo's value of record (τ_v 1.281052072726771, T_abs post
2.6365430791635713) — reproduced to floating-point identity. The gemma31 memo values
(1.0251785151221313 / 2.2416624655619084) belong to the OpenRouter pairing; they are
reproduced exactly too, and are printed by the driver as a legacy cross-check, but they are
not the denominator for these deltas (§0).

### Levels

| family / variant | argmax(pre) | argmax(post) | τ_v | T_abs(pre) | T_abs(post) |
|---|---:|---:|---:|---:|---:|
| qwen / baseline | 0.4818 | 0.6017 | 1.2811 | 3.1901 | 2.6365 |
| qwen / alt_set | 0.4237 | 0.5431 | 1.0252 | 3.2281 | 2.6731 |
| qwen / rev_order | 0.4737 | 0.6355 | 0.9518 | 3.2058 | 2.2370 |
| gemma31 / baseline | 0.4417 | 0.6417 | 1.0252 | 2.6285 | 2.2431 |
| gemma31 / alt_set | 0.3250 | 0.7167 | 1.0252 | 5.4052 | 2.2521 |
| gemma31 / rev_order | 0.4034 | 0.4750 | 1.0252 | 3.0580 | 2.1170 |

No τ_v is saturated; no T_abs is pegged at a search bound.

### The three deltas

| family / variant | Δargmax(pre), pp | Δargmax(post), pp | Δln τ_v | grid steps | Δln T_abs(post) | Δln T_abs(pre) |
|---|---:|---:|---:|---:|---:|---:|
| qwen / alt_set | −5.81 | −5.86 | **−0.2228** | 3.00 | +0.0138 | +0.0118 |
| qwen / rev_order | −0.81 | +3.38 | **−0.2971** | 4.00 | −0.1643 | +0.0049 |
| gemma31 / alt_set | −11.67 | +7.50 | **0.0000** | 0.00 | +0.0040 | +0.7209 |
| gemma31 / rev_order | −3.83 | −16.67 | **0.0000** | 0.00 | −0.0579 | +0.1513 |

---

## 3. Bound criterion and the memo row

**B = max over legs and variants of { |Δln τ_v|, |Δln T_abs| } = 0.2971**, attained at
qwen / rev_order / Δln τ_v.

- vs one τ_v grid step (0.0743): **4.00 steps**
- vs the leave-one-family-out worst case (0.191): **1.56×**
- max |Δargmax| = **16.67 pp** (gemma31 / rev_order, post leg)

Memo §A.5 defines: bounded-small = B < 0.0743 **and** max|Δargmax| below ~3 pp;
bounded-comparable-to-LOFO = 0.0743 ≤ B ≤ 0.191; bounded-large = B > 0.191.

> **Outcome: BOUNDED-LARGE.**

**Robustness to the gate failure.** Recomputing B over the three (family, variant) pairs
whose *both* legs pass the B-Q1 gate — i.e. dropping qwen / rev_order — gives
**B = 0.2228** (qwen / alt_set / Δln τ_v), max |Δargmax| 16.67 pp. Still > 0.191, so the
row is **bounded-large either way**. The verdict does not rest on the one defective leg.

Quoting memo §A.5's bounded-large row verbatim, which is what this outcome licenses in v15:

> The scaffold materially shapes the reported estimands. The stability observation (cluster
> ratio 1.296, LOFO ≤ 0.191) is re-read as conditional on the scaffold, the
> transferred-constant claim is restated as scaffold-conditional, and the bound B is
> reported as the headline of the sensitivity appendix. This weakens the paper but is
> strictly better than the same fact surfacing in review; v14's existing hedges already
> anticipate it.

Also fixed in advance and unchanged by this outcome: the **scaffold-free boundary** is
stated once — a zero-shot arm is not available for base models, because the exemplars carry
the contract format and bases without them fail the parse gate. The answerable question is
sensitivity *within* the family of valid coverage-complete scaffolds. And the k=4
token-slice trial is not cited as evidence in v15.

### Secondary reads (memo §A.3: reported, not part of B)

- **Sign pattern.** On qwen, V1 and V2 move τ_v the **same way** — both down, by 3 and 4
  grid steps. That is not two independent perturbations disagreeing; it is a consistent
  downward pull on the ratio estimand under any coverage-preserving change of the scaffold.
  On gemma31 both variants leave τ_v exactly where the baseline puts it.
- **The two families disagree sharply**, and the disagreement is structured. qwen's τ_v
  moves 3–4 grid steps while its argmax barely moves (≤5.9 pp); gemma31's τ_v does not move
  at all while its argmax moves up to 16.7 pp. The estimand that is sensitive on one family
  is the inert one on the other.
- **Δargmax against the stated expectation (memo §A.4).** The expectation recorded before
  the data was that coverage-preserving deltas would be materially smaller than the
  coverage-violating k=4 E-hole deltas (+8.3/+10.8/+10.8/+17.5 pp on four token-slice legs).
  On argmax they are **not**: 16.67 pp on gemma31 / rev_order post and 11.67 pp on
  gemma31 / alt_set pre sit inside the E-hole range. The memo pre-committed to the reading
  of exactly this case: it is not that the variants were badly built, but that the
  instrument is sensitive to the scaffold rather than only to category coverage — a finding
  in its own right.
- **T_abs(pre) is the largest single movement in the set**: gemma31 / alt_set, Δln +0.7209
  (2.6285 → 5.4052). It is excluded from B by the memo's definition, which takes T_abs on
  the *post* leg. It is reported here because it is the same event as that leg's argmax
  collapse to 0.3250: the alternate draw makes gemma31's base leg both less accurate and
  much more in need of absolute correction.

### The 1a → 1b rule (memo §A.6)

1b (glm + maverick under V1, V2 optional) **is triggered**: the rule runs it whenever 1a
lands bounded-comparable or bounded-large, because the bound is then load-bearing and a
two-family estimate of it is too thin. Not started here; it is a spend decision above this
task's envelope.

---

## 4. Transport diagnostic (report-only, NOT scaffold)

Free by-product of the design change: the **same** scaffold and the **same** weights served
two ways — our bf16 vLLM chat endpoint (`study_b_gemma31_vllmchat_base`) vs OpenRouter's
16-bit-pinned pool (`study_b_gemma31_api`), 118 shared cells.

| Quantity | self-hosted vLLM chat | OpenRouter chat | delta |
|---|---:|---:|---:|
| contract_complete | 0.9583 | 0.9333 | +2.50 pp |
| parse rate | 1.0000 | 0.9833 | +1.67 pp |
| argmax accuracy | 0.6417 | 0.6780 | −3.63 pp |
| T_abs(post) | 2.2431 | 2.2417 | Δln **+0.0006** |
| τ_v (vs the same pre leg) | 1.025179 | 1.025179 | 0 (identical grid point) |
| mean predictive entropy | 0.5205 | 0.5182 | +0.0023 |

Per-cell: mean L1 between the two post distributions **0.1666** (max 1.3734), cell-wise
argmax agreement **0.8814**.

Read: at the aggregate level the two transports are near-indistinguishable — the ratio
estimand lands on the identical grid point and the absolute estimand differs by 0.0006 in
logs, a hundredth of one τ_v grid step. Per cell they are not identical (12% of cells flip
argmax), which is the ordinary bf16/decoding jitter this channel has always carried. This is
a **transport** result and is excluded from B; it is reported because it retroactively
supports the shipped gemma31 pairing rather than undermining it, and because it is the
cleanest same-scaffold two-stack comparison the project has.

---

## 5. Cost ledger

Per-instance figures are vast's own invoice rows (GPU + storage + bandwidth), not
estimates. Per-leg attribution divides an instance's total by the legs it served; download
and load time are inside those totals and are not separated out.

| Instance | Role | Offer / machine | GPU | $/hr | GPU-h billed | Charge | Legs served | $/leg |
|---|---|---|---|---:|---:|---:|---|---:|
| 47229082 | qwen PRE | 40323387 / 131920 | 1×H200 | 3.9342 | 0.346 | $1.371 | qwen k5v1 pre, k5v2 pre | $0.686 |
| 47230297 | qwen POST | 40323388 / 131920 | 1×H200 | 3.9342 | 0.330 | $1.312 | qwen k5v1 post, k5v2 post | $0.656 |
| 47229639 | gemma31 PRE | 39353724 / 131697 | 1×H200 | 3.9342 | 0.406 | $1.753 | gemma31 k5v1 pre, k5v2 pre | $0.877 |
| 47231394 | gemma31 POST | 40323382 / 131920 | 1×H200 | 3.9342 | 0.606 | $2.494 | gemma31 vllmchat baseline post, k5v1 post, k5v2 post | $0.831 |
| 47229101 | gemma31 PRE (aborted) | 39605084 / 132116 | 1×H200 | 3.9342 | — | **$0.000** | none | — |
| — | OpenRouter annex | google/gemma-4-31b-it | — | — | — | $0.174 | 74 partial cells, k5v1 API post | — |

- 47229639 includes a **$0.141 bandwidth charge** (54.3 GB egress at $0.003/GB) — the only
  instance whose host billed for the HF pull.
- 47229101 sat `actual_status=offline` with no port mapping for ~25 minutes, never served,
  and was destroyed and replaced on a different machine. vast billed **$0.00** for it. That
  is why 47229639 exists.

| Line | Amount |
|---|---:|
| vast total (invoice sum, 5 instances) | **$6.930** |
| OpenRouter total | **$0.174** |
| **Campaign total** | **$7.104** |

Against the approved envelope $22–42 and the hard cap $60: **came in at ~32% of the low end
of the envelope, 12% of the cap.** Total billed GPU time 1.688 h across four working
instances. Vast credit moved 54.5523 → 47.6227 ($6.930, reconciling with the invoice sum to
the cent). OpenRouter usage moved 142.182267 → 142.355857.

**Fleet teardown verified:** `vastai show instances` returns `[]`.

Per-leg cost records were a disclosed gap in this project (`plan …gap_closure.md` §1f: "no
per-leg GPU billing exists"). This table fills it for the eight Phase-1a legs plus the
matched-transport baseline.

---

## 6. Anomalies and honest caveats

1. **qwen V2 post fails the B-Q1 contract gate** (0.80 < 0.90). Kept and reported; the
   bound is bounded-large with or without it (§3).
2. **One transient crash, recovered by design.** qwen V1 post died at cell 61 with
   `ConnectionResetError` and was resumed from its per-cell checkpoint to 120/120 under the
   identical leg config. Crash recovery, not re-elicitation; the sidecar guard admitted the
   resume because nothing about the leg had changed.
3. **One dead rental.** 47229101 never served and cost nothing (§5).
4. **The OpenRouter V1 leg is partial (74/120) and is not an input to anything.** Its dir
   carries a README saying so. The API V2 twin was never started.
5. **gemma31's deltas are cross-mode by construction** (raw-completions pre against
   chat-template post) — unavoidable, since gemma-4 post cannot be collected on raw
   continuation at all. Baseline and both variants share that pairing exactly, at the same
   dtype on the same box, so the scaffold delta does not absorb a transport difference. The
   §4 diagnostic bounds what such a difference would even look like.
6. **Two families is thin for a bound this load-bearing.** That is precisely what memo §A.6
   anticipates, and why the outcome triggers 1b.
7. **τ_v's grid quantization cuts both ways here.** qwen's movements are 3 and 4 clean grid
   steps, comfortably resolvable. gemma31's are exactly zero — which means "no observed
   movement", not "no movement": anything under 0.0743 in logs is invisible to this
   instrument, and gemma31's baseline τ_v sits only one grid step above where qwen's
   rev_order variant landed.
8. **No baseline run dir was written to.** `runs/study_b_qwen`, `runs/study_b_gemma31`, and
   `runs/study_b_gemma31_api` are unmodified by this pass; `runs/study_b_gemma31_api_k5v1`
   gained only a README. The frozen `scripts/absolute_vs_ratio_estimand.py` was not edited.

---

## 7. Files

Created (working tree, uncommitted):
- `scripts/absolute_estimand_k5variants.py` — the labeled Phase-1a variant copy of the
  frozen `absolute_vs_ratio_estimand.py`; family→dir map covers qwen + gemma31 only, adds
  the three pre-specified deltas, the bound B (with a gate-restricted recomputation), the
  legacy cross-check and the transport diagnostic. Documented inside as a variant read, not
  the rA1 driver.
- `docs/phase1a_scaffold_sensitivity_results_20260808.md` — this document.
- `runs/study_b_{qwen,gemma31}_k5v{1,2}/README.md`,
  `runs/study_b_gemma31_vllmchat_base/README.md`,
  `runs/study_b_gemma31_api_k5v1/README.md` — provenance and standing per dir.

Modified (working tree, uncommitted):
- `src/judex_calibration/elicit_api_verbalized.py` — `base_url` self-hosted chat path,
  `SELF_HOSTED_CHANNEL`/`SELF_HOSTED_DTYPE`/`UNGUARDED_META`, `chat_endpoint()`.
- `scripts/run_study_b_api_leg.py` — `--base-url`.
- `tests/test_scaffold_variants.py` — `SelfHostedChatTests` (+8 tests); the existing
  OpenRouter mock's signature gained the new keyword.

New run dirs (gitignored): `runs/study_b_qwen_k5v1`, `runs/study_b_qwen_k5v2`,
`runs/study_b_gemma31_k5v1`, `runs/study_b_gemma31_k5v2`,
`runs/study_b_gemma31_vllmchat_base`, `runs/study_b_gemma31_api_k5v1` (annex),
`runs/estimand_k5variants/estimand_k5variants.json` (the machine-readable read).

Unchanged: every frozen driver (`e6_r3_arms.py`, `absolute_vs_ratio_estimand.py`,
`absolute_mechanism_rA1.py`, `q4_range_robustness.py`, `range_robustness_verbalized.py`)
and every baseline run dir.
