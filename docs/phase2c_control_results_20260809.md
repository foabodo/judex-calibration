# Phase 2c — MMLU Defined-Answer Control: Results (2026-08-09)

Status: **EXECUTED 8/8 legs. E1 UNRESOLVED under the pre-specified gate; E2 directionally
unanimous but only 1/4 CI-significant.** Campaign spend **$109.74** against a $140–165
estimate and a $220 hard cap. Fleet empty.

Design of record: `docs/phase2_defined_answer_control_design.md` (D1–D8 confirmed).
Harness: branch `claude-verbalized-gap-closure` @ `42bbac4` (Phase 2b). No commits were
made by this campaign; `runs/control_mmlu_*` is gitignored.

Provenance pins (identical across all eight legs):

| pin | value |
|---|---|
| contract | `mmlu_control_v1` (D3: 4-option nominal, confidence kept, findings dropped) |
| slice_sha256 | `31822879ab5e960557ecf2a1f4fb7ccc56a6110458d577a8d3cf788fedf88c7e` (120 items, 6 subjects × 20) |
| store_sha256 | `c4174d4587618c2aa54a71620fbcb4d2514db611d586fa093ad0ee039c3fa3f2` |
| scaffold | `baseline`, k=4 (one exemplar per option letter), budget 2048, `--workers 8` |
| serving | bf16, `--max-model-len 32768`, `--gpu-memory-utilization 0.85`, on-demand vast.ai |
| gate | `parse_ok ≥ 0.90` (D3, re-derived onto the tier that IS the same object across tasks) |
| bootstrap | B=2000, seed 20260721; item-primary (D7), subject-clustered as sensitivity |

---

## 1. Headline

**The control did not fail to run — it ran, and the base checkpoints failed to answer.**
All eight legs elicited all 120 items. The harness produced scored reports, bootstraps and
both pre-specified reads on every family. But **zero of the four base (pre) legs cleared the
parse gate**, because all four base checkpoints degenerately skip the reasoning span on a
large, *subject-correlated* subset of MMLU items. E1 is defined over gate-passing base legs;
there are none, so **E1 is unresolved as pre-specified**.

Three of the four post legs cleared the gate (qwen post did not).

| | gate PASS | gate FAIL |
|---|---|---|
| pre (base) | — | qwen 86.7%, maverick 79.2%, glm 50.0%, gemma31 36.7% |
| post | gemma31 100%, maverick 97.5%, glm 90.8% | qwen 81.7% |

This is a *finding about the harness on defined answers*, not a machinery defect: the same
two-stage flow, the same stop list, and the same parser produce parse rates of 0.917–1.000
on the AIReg task for these very checkpoints (§4).

---

## 2. Per-family results

MMLU control, K=4, chance floor **0.25** (D8 — stated, not corrected for). Metrics are
computed over parsed records only; `n` is the parsed count out of 120.

| family | leg | n | parse | gate | acc | mean conf | gap | ECE10 | AECE5 | MCE | AUROC |
|---|---|---:|---:|:--:|---:|---:|---:|---:|---:|---:|---:|
| qwen | pre | 104 | 86.67% | FAIL | 0.865 | 0.965 | +0.099 | 0.0995 | 0.0995 | 0.101 | 0.576 |
| qwen | post | 98 | 81.67% | FAIL | 0.867 | 0.952 | +0.085 | 0.0847 | 0.0934 | 0.343 | 0.792 |
| gemma31 | pre | 44 | 36.67% | FAIL | 0.682 | 0.972 | +0.290 | 0.2902 | 0.2902 | 0.396 | 0.531 |
| gemma31 | post | 120 | 100.00% | **PASS** | 0.950 | 0.960 | +0.010 | 0.0339 | 0.0387 | 0.800 | 0.857 |
| glm | pre | 60 | 50.00% | FAIL | 0.900 | 0.960 | +0.060 | 0.0789 | 0.0896 | 0.140 | 0.471 |
| glm | post | 109 | 90.83% | **PASS** | 0.927 | 0.963 | +0.037 | 0.0527 | 0.0643 | 0.204 | 0.488 |
| maverick | pre | 95 | 79.17% | FAIL | 0.842 | 0.978 | +0.136 | 0.1378 | 0.1416 | 0.138 | 0.525 |
| maverick | post | 117 | 97.50% | **PASS** | 0.889 | 0.935 | +0.046 | 0.0745 | 0.0675 | 0.250 | 0.584 |

ECE10 uncertainty (D7: item bootstrap primary, subject clustered = sensitivity only):

| family | leg | item-boot [lo, hi] | subject-clustered [lo, hi] |
|---|---|---|---|
| qwen | pre | [0.045, 0.169] | [0.039, 0.170] (6) |
| qwen | post | [0.027, 0.152] | [0.018, 0.160] (6) |
| gemma31 | pre | [0.161, 0.447] | [0.184, 0.385] (**5** — see §6) |
| gemma31 | post | [0.011, 0.071] | [0.016, 0.072] (6) |
| glm | pre | [0.015, 0.162] | [0.009, 0.189] (6) |
| glm | post | [0.014, 0.107] | [0.026, 0.087] (6) |
| maverick | pre | [0.065, 0.213] | [0.031, 0.250] (6) |
| maverick | post | [0.039, 0.140] | [0.041, 0.142] (6) |

Grid-argmin temperatures (in-sample binned argmins are optimistically biased; reported, not
adopted — none saturated against `T_BOUNDS = (0.25, 20.0)`):

| family | leg | T_ece10 | ECE10 @ T | T_aece5 | AECE5 @ T |
|---|---|---:|---:|---:|---:|
| qwen | pre | 1.6008 | 0.0214 | 1.3798 | 0.0382 |
| qwen | post | 1.1894 | 0.0520 | 1.3798 | 0.0762 |
| gemma31 | pre | 2.6923 | 0.0179 | 2.3207 | 0.0753 |
| gemma31 | post | 0.9518 | 0.0158 | 1.1042 | 0.0381 |
| glm | pre | 1.1042 | 0.0450 | 1.3798 | 0.0621 |
| glm | post | 1.3798 | 0.0413 | 1.2811 | 0.0539 |
| maverick | pre | 1.8571 | 0.0101 | 2.0003 | 0.0644 |
| maverick | post | 1.3798 | 0.0535 | 1.3798 | 0.0525 |

---

## 3. The pre-specified reads

### 3.1 E1 (LEVEL) — UNRESOLVED as pre-specified

The E1 interpretation table (design §3) is defined over **gate-passing base legs**. There are
none. The estimand is therefore **not delivered**, and no row of the table is claimed.

For completeness, and **fenced as descriptive-only**, the base ECE10 values on the parsed
subsets, with the row each *would* land in had it cleared the gate:

| family | base ECE10 | parse | row it would land in |
|---|---:|---:|---|
| glm | 0.0789 | 50.0% | ≲0.10 — DACA-like |
| qwen | 0.0995 | 86.7% | ≲0.10 — DACA-like |
| maverick | 0.1378 | 79.2% | 0.10–0.20 — intermediate |
| gemma31 | 0.2902 | 36.7% | ≳0.20 — AIReg-like |

Descriptive range **0.079–0.290**; only gemma31 falls inside the AIReg v14 floor-clearing
range (0.175–0.398, median 0.260), and it does so on the smallest and most selected subset.

**Why these four numbers cannot be promoted into E1.** The parsed subset is not a random
subsample: it is exactly the set of items on which the base did *not* degenerately skip
reasoning, and that set is subject-correlated (§4). The surviving items are systematically
the ones the model engaged with, so both accuracy and ECE on that subset are biased in an
uncontrolled direction. Reading 0.079 or 0.0995 as "the MMLU base ECE under this harness"
would be reading a survivorship artifact. The honest statement is that **the control as
specified did not measure E1**, and the reason it did not is itself the result.

### 3.2 E2 (DIRECTION) — post improves ECE in 4/4 families; 1/4 CI-significant

`boot_paired_ece_delta`, B=2000, seed 20260721, item-primary.

| family | ECE10 pre | ECE10 post | delta (post−pre) | direction | item-boot CI | excludes 0 |
|---|---:|---:|---:|---|---|:--:|
| gemma31 | 0.2902 | 0.0339 | **−0.2563** | post improves | [−0.3771, −0.1120] | **yes** |
| maverick | 0.1378 | 0.0745 | −0.0633 | post improves | [−0.1187, +0.0146] | no |
| glm | 0.0789 | 0.0527 | −0.0262 | post improves | [−0.0643, +0.0622] | no |
| qwen | 0.0995 | 0.0847 | −0.0148 | post improves | [−0.1027, +0.0474] | no |

**Read.** The *direction* is unanimous and matches AIReg (post improves 6/6 there,
CI-significant in 4). It is **opposite to DACA's MMLU direction** (post degrades 3–7×). On
the pre-specified E2 logic this is the "generalizes across tasks under this harness" branch,
and the DACA-direction contrast becomes an **elicitation-mode contrast rather than a task
contrast**.

**Weight to place on it: limited.** Every pair has a gate-failing pre leg, and pre and post
are scored on *different, non-randomly-selected* item subsets. The one CI-significant family
(gemma31) is the one whose pre leg is most heavily selected (44/120), so its −0.2563 is the
least trustworthy of the four, not the most. The three families whose pre legs survived
best (qwen 86.7%, maverick 79.2%, glm 50.0%) all show small deltas whose CIs include zero.

### 3.3 Chance-floor caveat (D8)

MMLU control ECE sits over a **0.25** chance floor (K=4); the AIReg values it is compared
against sit over **0.20** (K=5). DACA's MMLU numbers are at K=4, so the control is better
matched to the DACA comparand than the paper's existing AIReg-vs-DACA contrast. Floors are
stated; no numeric equality is claimed.

### 3.4 The capability question — bases are weak judges but competent test-takers

Base argmax accuracy against the chance floor, MMLU vs AIReg (AIReg values are the verbalized
pre legs as recorded in `runs/base_calibration_premise/base_calibration_premise.json`):

| family | MMLU base acc (floor 0.25) | above floor | AIReg base acc (floor 0.20) | above floor |
|---|---:|---:|---:|---:|
| glm | 0.900 | +0.650 | 0.5431 | +0.343 |
| qwen | 0.865 | +0.615 | 0.4818 | +0.282 |
| maverick | 0.842 | +0.592 | 0.3250 | +0.125 |
| gemma31 | 0.682 | +0.432 | 0.4417 | +0.242 |

Every base clears the MMLU chance floor by a wide margin — 0.68–0.90 argmax accuracy on the
items it answered — while the same checkpoints sat at 0.33–0.54 on AIReg. **Bases that are
weak judges on AIReg are competent answerers on MMLU.** (Subject to the same survivorship
caveat: this is accuracy on the parsed subset.) This matters for the reading: whatever the
AIReg base miscalibration is, it is not simply "these checkpoints cannot do the task."

### 3.5 DACA comparison table

DACA Fig. 1 (arXiv:2505.16690v2), MMLU, max-prob over A–D, **scaffold-free logit slice**;
ours is the verbalized contract + k=4 few-shot scaffold. Qualitative comparand only.

| source | elicitation | base ECE | post ECE | direction |
|---|---|---|---|---|
| DACA Fig. 1 (4 models) | logit slice, no scaffold, K=4 | 0.034–0.069 | 0.190–0.242 | post **degrades** 3–7× |
| **This control (4 families)** | verbalized + k=4 scaffold, K=4 | 0.079–0.290 *(descriptive; no leg gate-passing)* | **0.034–0.085** | post **improves** 4/4 |
| AIReg v14 (six bases) | verbalized + k=5 scaffold, K=5 | 0.175–0.443 (floor-clearing 0.175–0.398, median 0.260) | — | post improves 6/6 |

**The DACA-gap read.** Our gate-passing post legs (0.034–0.075) land *inside DACA's base
range* and 3–7× below DACA's post range. Three of our four descriptive base legs
(0.079–0.138) sit far closer to DACA's base band than to the AIReg base band. On defined
answers, then, **the verbalized harness does not reproduce AIReg-scale miscalibration** —
which is evidence against "the verbalized elicitation mode inflates ECE per se", the
adversarial reading the ≳0.20 row of the E1 table was written to adjudicate. It is *not*
evidence for the ≲0.10 row, because no base leg cleared the gate.

### 3.6 D6 descriptive τ (TVD objective) — fenced, never adopted

Permutation-invariant TVD grid-argmin aligning post → pre. **Different task, different
reference regime; must not enter the frozen r3/rA1 cluster/band/LOFO constants.** None
saturated.

| family | τ_TVD |
|---|---:|
| qwen | 1.0252 |
| glm | 1.0252 |
| gemma31 | 0.9518 |
| maverick | 0.5659 |

---

## 4. The base-leg reasoning-span collapse (diagnostics)

### 4.1 Mechanism

Two distinct failure modes, both deterministic (temperature 0 throughout — no leg was
retry-burned, and a retry would reproduce the same records):

**(a) Degenerate empty reasoning span — the dominant mode.** The base checkpoint, given the
control prompt ending in `Reasoning:`, emits nothing and jumps straight to the JSON scaffold
token, then emits EOS. Stage 1 stops on `JSON:` with a 1-character span; stage 2 receives
`…Reasoning:\n\nJSON:` and returns **zero characters**. Recorded as `no_json_object` with
`reasoning_chars ≤ 1` and `raw_chars == 0`.

Direct probe against the live gemma31 base box, `clinical_knowledge` item, greedy:

- stage 1 with the harness stop list → output `'\n'`, `stop_reason='JSON:'`
- stage 1 with **no** stop list → output `'\nJSON:'`, then EOS
- stage 2 as the harness sends it → `''`, `finish_reason='stop'`
- stage 2 with **no** stop list → `''` (so the stop list is not the cause)

Empty-generation counts among failures: gemma31 pre 62/76, glm pre 53/60, maverick pre
18/25, qwen pre 0/16.

**(b) JSON syntax invalidity — qwen's mode.** Zero empty generations; the model writes JSON
but writes it invalidly, chiefly LaTeX backslashes inside justification strings
(`json_decode: Invalid \escape`) and unquoted keys (`Expecting property name enclosed in
double quotes`). This is the only mode that also afflicts a post leg (qwen post, 81.67%).

### 4.2 It is task-specific, not a serving or machinery fault

Decisive control, run on the **same box, same model, same two-stage flow** as the collapsing
gemma31 pre leg: the *AIReg* prompt produced an 842-character reasoning span, stopping
correctly on `JSON:`. The machinery reasons on AIReg and empties out on MMLU.

Corroborating: AIReg parse rates for these same checkpoints under the same harness —

| family | AIReg pre | AIReg post | MMLU pre | MMLU post |
|---|---:|---:|---:|---:|
| qwen | 0.9167 | 0.9833 | 0.8667 | 0.8167 |
| gemma31 | 1.0000 | 0.4750 | 0.3667 | 1.0000 |
| glm | 0.9667 | 0.9667 | 0.5000 | 0.9083 |
| maverick | 1.0000 | 1.0000 | 0.7917 | 0.9750 |

(The gemma31 AIReg *post* 0.475 is the known OpenRouter-channel figure; the matched
`verbalized_vllm_chat` post leg parsed 1.000, as does its MMLU control twin.)

### 4.3 It is subject-correlated — parsed / total per subject

| subject | qwen pre | gemma31 pre | glm pre | maverick pre | qwen post | glm post | gemma31 post |
|---|---|---|---|---|---|---|---|
| formal_logic | 19/20 | 6/20 | 1/20 | 19/20 | 17/20 | 20/20 | 20/20 |
| **high_school_mathematics** | **10/20** | **3/20** | **1/20** | **2/20** | **7/20** | 15/20 | 20/20 |
| professional_law | 20/20 | 20/20 | 16/20 | 20/20 | 18/20 | 17/20 | 20/20 |
| clinical_knowledge | 20/20 | **0/20** | 14/20 | 18/20 | 19/20 | 20/20 | 20/20 |
| philosophy | 19/20 | 1/20 | 18/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| econometrics | 16/20 | 14/20 | 10/20 | 16/20 | 17/20 | 17/20 | 20/20 |

`high_school_mathematics` is the worst subject for **every** base leg and for qwen post.
`professional_law` — the longest stems — is the best for every base leg. Median stem length
of failing vs parsed items: gemma31 pre 91 vs 307 chars (short stems fail); glm pre 141 vs 82
(the symbolic subjects, whose stems are longer). The two families thus fail on different
covariates but converge on the same subject.

**This is the survivorship mechanism that disqualifies E1** (§3.1): the base legs are scored
on a subject-skewed subsample, weighted away from mathematics and toward long-form law.

---

## 5. Cost ledger — invoice-exact, per instance

vast.ai invoice rows (GPU + storage + download + upload), read **after** teardown. All
rentals on-demand, `--order dph`. One leg per instance; no instance served two legs, and
there were no dead rentals.

| instance | leg | machine | GPU | $/hr | GPU | storage | download | upload | **charge** |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 47283877 | qwen pre | 131919 | 1×H200 | 3.969 | 1.195 | 0.011 | 0.161 | 0.002 | **$1.369** |
| 47284904 | qwen post | 131919 | 1×H200 | 3.969 | 0.898 | 0.008 | 0.160 | 0.001 | **$1.067** |
| 47285755 | gemma31 pre | 131919 | 1×H200 | 3.969 | 0.818 | 0.007 | 0.140 | 0.001 | **$0.966** |
| 47286445 | gemma31 post | 131919 | 1×H200 | 3.969 | 0.855 | 0.008 | 0.139 | 0.001 | **$1.003** |
| 47287255 | glm pre | 54969 | 8×H200 | 32.209 | 10.065 | 0.203 | 11.989 | 0.063 | **$22.320** |
| 47288421 | glm post | 54969 | 8×H200 | 32.736 | 15.137 | 0.301 | 12.008 | 0.088 | **$27.534** |
| 47289855 | maverick pre | 38592 | 8×H200 | 32.209 | 13.995 | 0.279 | 13.376 | 0.080 | **$27.730** |
| 47291167 | maverick post | 38592 | 8×H200 | 32.736 | 14.016 | 0.262 | 13.396 | 0.079 | **$27.753** |
| | | | | | | | | **total** | **$109.742** |

| | |
|---|---:|
| vast, 8 instances (invoice sum) | $109.742 |
| OpenRouter / API | $0.000 |
| **Phase 2c total** | **$109.742** |
| estimate (design §4) | $140–165 |
| hard cap | $220 |
| **spend vs cap** | **49.9%** |

**Credit reconciliation.** $152.1818 → $42.4376, a movement of **$109.7442** against a
settled invoice sum of $109.742 — reconciling to a fifth of a cent. No auto top-up fired:
the float never approached the $5 threshold. Credit was verified ≥ $50 before each of the
four 8×H200 rentals ($147.99, $126.46, $100.75, $72.59).

**Settlement lag is real, and Phase 1b's rule needs strengthening.** "Read the invoice after
teardown" is necessary but not sufficient. The final instance's GPU line read **$12.003**
immediately after destroy, **$13.355** two minutes later, and settled at **$14.016** two
minutes after that — a 17% under-report at the first post-teardown read. Poll the invoice
until two consecutive reads agree before closing a ledger.

**Why it came in 23% under the low estimate.** The control prompts are ~7× smaller than the
AIReg prompts (max 2,263 tokens vs ~18.3k), so each 120-item leg ran in **1–4 minutes**
rather than tens of minutes; GPU time was dominated by weight download and engine init, not
elicitation. Download was again the material line on 8×H200 hosts ($12.0–13.4 per rental at
$0.019/GB), consistent with Phase 1b.

Wall clock: 15:58 → 18:06 UTC, ~2h08m for eight legs and eight rentals.

---

## 6. Gates and anomalies

1. **Gate 3/8 PASS** (gemma31 post, glm post, maverick post). All four base legs FAIL;
   qwen post FAILS on JSON syntax. Recorded, **not retry-burned** — greedy decoding makes
   every failure deterministic.
2. **E1 not delivered.** No gate-passing base leg exists. §3.1 states this rather than
   promoting survivorship-biased subset numbers into the estimand.
3. **gemma31 pre subject bootstrap has 5 clusters, not 6** — `clinical_knowledge` parsed
   0/20, so the subject is absent from the leg entirely. The sensitivity CI for that leg is
   over five subjects.
4. **gemma31 post MCE 0.800 alongside ECE10 0.034.** A single sparse high-confidence bin
   drives the max-deviation statistic; the equal-mass AECE5 (0.0387) and the item bootstrap
   ([0.011, 0.071]) both agree the leg is well calibrated. MCE is a max over bins and is not
   robust at n=120.
5. **glm and gemma31 post AUROC ≈ 0.47–0.49** — confidence carries essentially no
   discriminative signal about correctness on those legs, despite low ECE. Low ECE with
   chance AUROC means well-matched *average* confidence, not useful *per-item* confidence.
6. **`alt_set` remains unavailable** on the control store (`baseline_all_coverage_complete:
   true`, `alt_set_available: false`) — the k=4 robustness variant would need a deeper store.
   Default-off per D4; not exercised.
7. No transient failures: no `ConnectionResetError`, no resume, no checkpoint recovery, no
   dead rentals, no stop-thrash. Every instance was destroyed; `vastai show instances`
   returns `[]`.

---

## 7. Bound discipline

- **B = 0.2971 is not revised**, and no Phase-1 verdict is revised. The defined-answer
  control and the scaffold-sensitivity work remain separate appendices.
- The MMLU τ_TVD values (§3.6) are descriptive and **never** enter the frozen r3/rA1
  cluster/band/LOFO constants.
- No ordinal metric was computed on control records: `score_control`'s import graph excludes
  `study_a`, Murphy and RPS/W1 by construction (guarded by `tests/test_ece_lift.py`).
- The v15 sentence map in design §6 is **not** actionable as written: it presumes a delivered
  E1. What this campaign licenses is narrower — the E2 direction read (§3.2, weakly), the
  DACA-gap read (§3.5), and the base-leg collapse itself (§4), which is a substantive
  scope-limiting result about the harness on defined answers rather than a null.

---

## 8. Artifacts

- `runs/control_mmlu_{qwen,gemma31,glm,maverick}/` — `{pre,post}_control.json`,
  `.meta.json` sidecars, `control_report.json` (gitignored, this Mac).
- `runs/control_scaffold_probe.json` — the pre-spend offline scaffold/token-budget probe.
- Harness: `src/judex_calibration/{mmlu,control_fewshot,elicit_control,ece,score_control}.py`,
  `scripts/run_control_leg.py`, `scripts/run_control_api_leg.py` (branch
  `claude-verbalized-gap-closure` @ `42bbac4`; 204 tests + 58 subtests pass).
