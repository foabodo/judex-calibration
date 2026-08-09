# E1 re-collection campaign under the v2 control scaffold — HALTED AT THE STOP-LOSS (2026-08-09)

Status: **STOP-LOSS FIRED ON LEG 1. `gemma31` base parsed 80.00% against the 0.90 gate —
FAIL. The campaign was halted, the instance destroyed, and legs 2–4 (qwen post, glm base,
maverick base) were NOT rented.** Spend **$0.853** against an est. $55–75 and a $90 hard
cap (0.9%). Fleet empty.

Predecessors: `docs/phase2c_control_results_20260809.md` (the v1-scaffold post-mortem),
`docs/e1_gate_qwen_v2_20260809.md` (the paid GO gate), design
`docs/phase2_defined_answer_control_design.md` (D1–D8, and the pre-specified E1/E2 tables
in §1/§3).

Harness: branch `claude-verbalized-gap-closure` @ `71300d4`, clean working tree. No commits.
`runs/control_mmlu_gemma31_v2/` is gitignored and is a **fresh** dir —
`runs/control_mmlu_gemma31/` (the Phase-2c v1 record) is untouched.

Provenance pins on the collected leg (byte-identical to the qwen gate leg):

| pin | value |
|---|---|
| contract | `mmlu_control_v1` |
| slice_sha256 | `31822879ab5e96…` (120 items, 6 subjects × 20) |
| store_sha256 | `d34d855eac8c…` (`mmlu_control_exemplars_v2`, 333 rows) |
| exemplar_selection | `band_400_800` |
| parsing | `parse_last_control` (last-object scan) |
| scaffold | `baseline`, k=4, budget 2048, `--workers 8`, greedy |
| serving | bf16, `--max-model-len 32768`, `--gpu-memory-utilization 0.85`, TP=1 |
| gate | `parse_ok ≥ 0.90` (D3) |

---

## 1. Stop-loss outcome

The gemma31 base leg was run first precisely because it is the definitive **Mode B**
(empty-reasoning collapse) family and the worst Phase-2c leg — a ~$1 test of whether the v2
scaffold repairs the mode that the qwen gate leg could not test (qwen never had Mode B; see
the gate doc §7).

| | Phase 2c (v1 store, length-blind, first-`{`) | this run (v2 + band + last-object) |
|---|---:|---:|
| elicited | 120/120 | 120/120 |
| parsed | 44 | **96** |
| `parse_ok` | 36.67% | **80.00%** |
| gate (≥ 0.90) | FAIL | **FAIL** (10.0 points short) |
| 5-field `contract_complete` | 36.67% | 80.00% |
| **empty-reasoning failures** | **74** | **19** |

**The remediation works and is not sufficient.** +43.3 points of parse rate on the same 120
items, same checkpoint, same decoding params, and a **74 → 19** collapse in the
empty-reasoning class — a 74% reduction in the exact failure mode the store-v2 span floor
and the band draw were built to attack. It is still 10 points below the gate, so under the
pre-specified rule this leg does not enter E1, and the two 8×H200 legs (~$50–56, the bulk
of the budget) were not justified. Per the mandate: teardown, stop, report.

Greedy throughout; the leg was not retried and a retry would reproduce the same records.

## 2. Failure taxonomy — Phase 2c vs this run

Classes as defined in the Phase-2c post-mortem §4.1 and the gate doc §3.

| class | gemma31 Phase 2c | gemma31 this run | qwen v2 gate leg (ref) |
|---|---:|---:|---:|
| **empty-reasoning collapse** (`no_json_object`, span ≤ 1 char) | **74** | **19** | 0 |
| **escape** (`json_decode: Invalid \escape`) | 2 | 1 | 4 |
| **invalid distribution** | 0 | 2 | 1 |
| **no JSON object** (non-degenerate span) | 0 | 0 | 1 |
| **brace-hijack** (`Expecting property name enclosed in double quotes`) | 0 | **0** | **0** |
| **other `json_decode`** (`Expecting ',' delimiter`) | 0 | 2 | 0 |
| total failures | 76 | **24** | 6 |

Reading the table:

- **Mode B is 79% of the residue** (19 of 24) and remains the binding constraint. Every
  other class is at 1–2 cells.
- **The brace-hijack class is extinct harness-wide** — 0 cells here, 0 in the qwen v2 leg,
  against 6 in the Phase-2c qwen leg. CHANGE 2 (last-object parsing) is confirmed on a
  second family and costs nothing.
- **The escape class did not grow** (2 → 1) despite spans nearly tripling, so the
  elaborateness-into-illegal-escape mechanism the Mac smoke ladder measured did not appear
  on gemma31.
- Two new `Expecting ',' delimiter` cells (both `philosophy`) are the only genuinely new
  signature; at 1.7% of the leg they are not the reason the gate missed.

**Degenerate spans.** 20 cells emitted a ≤ 1-character reasoning span (Phase 2c: 76), and
**1 of the 20 parsed anyway**. The Phase-2c ratio was 2 of 76. So the degenerate span is
still very nearly fatal on gemma31 — unlike qwen, where all 3 degenerate cells parsed. The
v2 scaffold reduced *how often* gemma31 skips reasoning without changing *what happens when
it does*.

### 2.1 Subject concentration — the residue is one subject

Parsed / 20 per subject, with the degenerate-span count:

| subject | Phase 2c parsed | this run parsed | degenerate spans then → now | empty-reasoning failures now |
|---|---:|---:|---|---:|
| professional_law | 20 | **20** | 0 → 0 | 0 |
| formal_logic | 6 | **19** | 16 → 2 | 1 |
| clinical_knowledge | **0** | **18** | 20 → 1 | 1 |
| econometrics | 14 | **18** | 4 → 0 | 0 |
| philosophy | 1 | **16** | 19 → 2 | 2 |
| **high_school_mathematics** | 3 | **5** | 17 → **15** | **15** |

This is the whole story of the leg. **Five of six subjects are repaired**
(16–20/20, from 0–14/20), and `clinical_knowledge` — which parsed **0/20** in Phase 2c and
was absent from that leg's cluster bootstrap entirely — is back at 18/20 with a single
degenerate cell. The failure is now **almost entirely `high_school_mathematics`**: 15 of the
leg's 24 failures and 15 of its 20 degenerate spans sit there, and the subject moved only
3/20 → 5/20.

Arithmetic on the counterfactual: had `high_school_mathematics` merely matched the *worst*
of the other five subjects (16/20), the leg would have parsed 107/120 = 89.2% — still short.
Had it matched the leg's non-math average (18.2/20), 109/120 = 90.8% — a pass. The gate
therefore turns entirely on one subject for this family, exactly as the Phase-2c post-mortem
predicted (`high_school_mathematics` was the worst subject for **every** base leg and for
qwen post).

The mechanism is unchanged and is not reachable by the two remediations shipped: given a
short arithmetic stem, the gemma31 base checkpoint emits nothing after `Reasoning:` and
jumps to the JSON scaffold token, then EOS. Store v2 lengthened what the *exemplars* show;
it does not make a base checkpoint elect to reason about a one-line maths question.

## 3. Reasoning-span distribution vs the exemplar band

Exemplars actually rendered, all six subjects, k=4 (24 exemplars): spans **637–798, median
~723, zero out of band** (per-subject blocks logged: formal_logic [792, 775, 672, 796],
high_school_mathematics [714, 648, 798, 637], professional_law [795, 754, 755, 661],
clinical_knowledge [723, 734, 693, 677], philosophy [708, 770, 647, 784], econometrics
[697, 753, 646, 638]).

Model-produced spans, characters (linear-interpolated quantiles):

| leg | n | p25 | median | p75 | max | mean | in band [400, 800] |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma31 Phase 2c, passing | 44 | 155 | 206 | 243 | 555 | 208 | **2.3%** |
| gemma31 Phase 2c, failing | 76 | 1 | 1 | 1 | 425 | 9 | 1.3% |
| **gemma31 this run, passing** | 96 | **584** | **686** | **782** | 1115 | 684 | **76.0%** |
| gemma31 this run, failing | 24 | 1 | 1 | 1 | 894 | 143 | 16.7% |
| qwen v2 gate leg, passing (ref) | 114 | 616 | 793 | 1144 | 8788 | 1055 | 43.9% |

**CHANGE 1 worked better on gemma31 than on qwen.** The passing-cell median moved
206 → 686 against an exemplar median of ~723, and the in-band fraction went **2.3% → 76.0%**
— the tightest span-to-scaffold match observed anywhere in this programme (qwen: 43.9%).
The interquartile range is 198 characters against qwen's 528, and the maximum span is 1115
characters — gemma31 has essentially no right tail, which is why it has no escape/elaborateness
failures.

**The failure distribution is bimodal, and both modes are at the left edge.** Failing cells
have p25 = median = p75 = 1 character. Unlike qwen — whose failures sat in the *right* tail
(p75 2178, writing too much) — every gemma31 failure but four is a cell that wrote *nothing*.
The two families fail through opposite mechanisms, and the band instrument, which has an
upper bound precisely to catch qwen's mechanism, cannot reach gemma31's.

## 4. E1 — NOT RESOLVED

E1 (design §1) is base-leg top-1 ECE on MMLU **over gate-passing base legs**. After this
campaign the panel stands at **1 of 4 base families gate-passing**:

| family | base leg collected under v2 | parse_ok | gate | in E1? |
|---|---|---:|:--:|:--:|
| qwen | yes (`runs/control_mmlu_qwen_v2`) | 95.00% | **PASS** | yes |
| gemma31 | yes (`runs/control_mmlu_gemma31_v2`) | 80.00% | FAIL | **no** |
| glm | **not collected** (stop-loss) | — | — | no |
| maverick | **not collected** (stop-loss) | — | — | no |

**E1 remains unresolved.** The estimand is a read over the family panel; a single
gate-passing leg does not deliver it, as the gate doc §5 already stated of that same leg
("One leg is not the estimand… a single family delivers no E1 row and none is claimed
here"). No interpretation row is claimed, and the pre-specified v15 consequence is
therefore not triggered. Nothing here is promoted.

### 4.1 The one gate-passing base leg, reported as a leg and not as E1

| | qwen base, v2 scaffold, n=114/120 |
|---|---:|
| accuracy (chance floor 0.25) | 0.9035 (+0.6535) |
| mean confidence | 0.9775 |
| overconfidence gap | +0.0739 |
| **ECE10** | **0.0802** |
| AECE5 | 0.0739 |
| MCE10 | 0.1197 |
| AUROC (conf vs correct) | 0.4718 |
| ECE10 item bootstrap (B=2000, seed 20260721) | [0.0309, 0.1361] |
| ECE10 subject-clustered (6) | [0.0266, 0.1394] |
| grid-argmin `T_ece10` / value | 1.6008 / 0.0234 (not saturated) |

**Which row it would land in, if the panel ever supports the read.** 0.0802 falls in the
**≲ 0.10 (DACA-like)** row, and its item-bootstrap CI [0.031, 0.136] excludes the ≳ 0.20
row entirely and lies wholly below the AIReg v14 floor-clearing band. The row's
pre-specified v15 consequence, quoted verbatim from design §3:

> §6.2 keeps the reversal, now same-harness; Limitations scope *tightens to the judging
> condition* — the strongest honest version of the current text

**That consequence is NOT licensed by this campaign**, because it is conditioned on a
delivered E1 and one family is not the panel. It is recorded here so that the read is
already specified if the three missing legs are ever collected.

### 4.2 The failing base leg, fenced as descriptive-only

| | gemma31 base, v2 scaffold, n=96/120 |
|---|---:|
| accuracy (chance floor 0.25) | 0.6667 (+0.4167) |
| mean confidence | 0.9746 |
| overconfidence gap | +0.3079 |
| ECE10 | 0.3079 |
| AECE5 | 0.3079 |
| MCE10 | 0.8955 |
| AUROC | 0.6035 |
| ECE10 item bootstrap | [0.2204, 0.4013] |
| ECE10 subject-clustered (6, all present) | [0.2171, 0.4287] |
| grid-argmin `T_ece10` / value | 2.4996 / 0.0470 (not saturated) |

Descriptive only; the leg failed the gate. Its ECE10 barely moved from Phase 2c's 0.2902
(→ 0.3079) even though the parsed subset more than doubled (44 → 96), which is worth stating
plainly: **the ECE is not an artifact of the small Phase-2c subset.** Its ECE10 bins show
95 of 96 items in `[0.9, 1.0)` at accuracy 0.674 — one flat block of near-certain answers
that are right two-thirds of the time. Survivorship is much reduced (all six subjects now
present; the Phase-2c leg's 5-cluster bootstrap anomaly is gone) but not eliminated: the
subset is still weighted away from `high_school_mathematics` (5/20).

### 4.3 The two comparands, for the record

- **AIReg v14 printed**: six bases 0.175–0.443; five floor-clearing legs 0.175–0.398, median
  0.260, over a **0.20** chance floor (K=5).
- **DACA scaffold-free logit slice, matched K=4**: base 0.034–0.069, over a **0.25** chance
  floor.
- **This campaign**: qwen base (gate-passing) 0.0802 — above DACA's band, far below AIReg's.
  gemma31 base (gate-failing, descriptive) 0.3079 — inside the AIReg floor-clearing band.

**D8 chance-floor caveat applies and is not corrected for**: MMLU control ECE sits over 0.25,
AIReg over 0.20; DACA is at K=4 and is therefore the better-matched comparand. Floors are
stated; no numeric equality is claimed.

**Capability contrast (design §3.4 sense).** Both bases clear the MMLU chance floor by a
wide margin on the items they answered — qwen 0.904 (+0.654), gemma31 0.667 (+0.417) — while
the same checkpoints sat at **0.33–0.54 over a 0.20 floor** on AIReg (glm 0.543, qwen 0.482,
gemma31 0.442, maverick 0.325). Bases that are weak judges on AIReg remain competent
test-takers on MMLU, and the v2 scaffold did not change that: whatever the AIReg base
miscalibration is, it is not "these checkpoints cannot do the task." (Subject to the parsed-subset
caveat throughout.)

## 5. E2 — scaffold-version accounting

**Stated plainly: no pair in this programme is fully v2.** The campaign brief anticipated
that qwen's post leg would be re-collected as leg 2; the stop-loss fired on leg 1, so it was
not. Every available pair therefore mixes a **v2 pre** leg with a **Phase-2c v1 post** leg,
or is an untouched v1 pair.

| family | pre scaffold | post scaffold | pair status |
|---|---|---|---|
| qwen | **v2** (`control_mmlu_qwen_v2`, 114/120) | v1 (`control_mmlu_qwen`, 98/120) | **MIXED** |
| gemma31 | **v2** (`control_mmlu_gemma31_v2`, 96/120) | v1 (`control_mmlu_gemma31`, 120/120) | **MIXED** |
| glm | v1 | v1 | unchanged Phase-2c v1 pair |
| maverick | v1 | v1 | unchanged Phase-2c v1 pair |

`boot_paired_ece_delta`, B=2000, seed 20260721, item-primary (D7):

| family | pair | ECE10 pre | ECE10 post | delta (post−pre) | direction | item-boot CI | excludes 0 | Phase-2c v1 delta |
|---|---|---:|---:|---:|---|---|:--:|---:|
| qwen | MIXED v2/v1 | 0.0802 | 0.0847 | **+0.0045** | post **degrades** | [−0.0825, +0.0600] | no | −0.0148 (improves) |
| gemma31 | MIXED v2/v1 | 0.3079 | 0.0339 | **−0.2741** | post improves | [−0.3808, −0.1817] | **yes** | −0.2563 |
| glm | v1/v1 (unchanged) | 0.0789 | 0.0527 | −0.0262 | post improves | [−0.0643, +0.0622] | no | — |
| maverick | v1/v1 (unchanged) | 0.1378 | 0.0745 | −0.0633 | post improves | [−0.1187, +0.0146] | no | — |

**The confound, named.** In both mixed rows the pre and post legs were elicited under
*different exemplar stores, different exemplar-selection rules and different parsers*, and
scored on *different, non-randomly-selected item subsets*. A delta across them is not a
clean post−pre contrast: part of any movement is scaffold version, not post-training. This
is on top of the Phase-2c caveat that pre and post legs never share an item subset.

**What moved.** qwen's delta **flipped sign** (−0.0148 → +0.0045) purely because its v2 pre
leg is better calibrated (0.0995 → 0.0802) on a larger subset (104 → 114), while its post leg
is the same v1 record. Both CIs comfortably include zero — the flip is not a finding, it is
a demonstration of how little weight a mixed delta carries. gemma31's delta grew slightly
more negative (−0.2563 → −0.2741) and remains the one CI-significant row, but it is
CI-significant on the pair whose pre leg fails the gate.

**Net E2 read: the Phase-2c direction claim is not strengthened and not overturned.** Of the
four families, three still point to "post improves" and one (qwen) now points marginally the
other way, with three of four CIs including zero. It remains opposite to DACA's MMLU
direction, and the weight to place on it remains limited for exactly the reasons Phase 2c
gave.

**The option not taken** (explicitly NOT this campaign's to run): re-collecting the three
outstanding post legs — qwen post (1×H200) plus glm post and maverick post (8×H200 each) —
would buy full scaffold-version consistency for ~$55. That is a decision for the
orchestrator, and it is separable from the base-leg problem: post legs already cleared the
gate 3/4 under v1, so the value is comparability, not rescue.

### 5.1 D6 descriptive τ (TVD) — fenced, never adopted

Recomputed on the mixed pairs; **different task, different reference regime; must not enter
the frozen r3/rA1 cluster/band/LOFO constants.** Neither is saturated.

| family | τ_TVD (mixed pair) | Phase-2c v1 τ_TVD |
|---|---:|---:|
| qwen | 0.7072 | 1.0252 |
| gemma31 | 1.0252 | 0.9518 |

## 6. Cost ledger — invoice-exact

One rental, on-demand, `--order dph`, destroyed immediately on the gate read.

| instance | leg | machine | GPU | geo | $/hr | GPU | storage | download | upload | **charge** |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 47302711 | gemma31 base (pre) | 144381 | 1×H200 NVL | Japan | 3.333 (listed 3.387) | 0.699 | 0.012 | 0.141 | 0.001 | **$0.853** |

| | |
|---|---:|
| vast, 1 instance (invoice sum) | $0.853 |
| OpenRouter / API | $0.000 |
| **campaign total** | **$0.853** |
| estimate (four legs) | $55–75 |
| hard cap | $90 |
| **spend vs cap** | **0.9%** |

Wall clock ~14 min of billed GPU (0.210 h): ~6 min weight pull (54.3 GB) + engine init,
~7 min elicitation of 120 items at `--workers 8`. Download 54.3 GB at $0.003/GB = $0.141.

**Invoice polled to stability** per the Phase-2c rule. Post-teardown reads:
**$0.787 → $0.823 → $0.853 → $0.853 (STABLE)** — the first read under-reported by **7.7%**,
inside the 10–17% band the rule was written for. Credit reconciliation:
**$66.549616 → $65.696634**, a movement of **$0.852982** against a settled invoice sum of
**$0.853** — reconciling to a fifth of a cent. No auto top-up fired.

Credit was verified ≥ $50 before the rental ($66.55). The two 8×H200 rentals were never
reached; had the campaign continued, credit after legs 1–3 would have stood at roughly $36
— **below the $50 single-increment float rule**, so the maverick leg would have required a
top-up before it could be rented. Flagged for whoever resumes.

Fleet verified after teardown: `vastai show instances` returns `[]`.

## 7. Gates and anomalies

1. **Stop-loss fired as designed.** gemma31 base 80.00% < 0.90, outside the 0.88–0.90
   borderline band. Teardown, halt, report — legs 2–4 not rented, $0.853 spent against a
   ~$55–75 exposure avoided.
2. **No transient failures**: no `ConnectionResetError`, no resume, no checkpoint recovery,
   no dead rentals, no stop-thrash. All 120 cells elicited on the first pass. The one
   instance was destroyed.
3. **All 24 failures are structural**, recorded with diagnostics and **not retry-burned** —
   greedy decoding makes each deterministic.
4. **`clinical_knowledge` is restored to the leg** (0/20 → 18/20), so the Phase-2c
   5-cluster subject-bootstrap anomaly for gemma31 pre does not recur; all six subjects are
   present.
5. **gemma31 MCE10 0.8955 on a 96-item leg**: driven by a single item in the `[0.8, 0.9)`
   bin that is wrong. MCE is a max over bins and is not robust at this n — the same caveat
   Phase 2c recorded for the gemma31 post leg.
6. **qwen base AUROC 0.4718** (below chance) on the v2 leg, against 0.576 on the v1 leg:
   confidence carries no per-item discriminative signal on that leg despite ECE10 0.0802.
   Low ECE with chance AUROC means well-matched *average* confidence, not useful *per-item*
   confidence.
7. **No ordinal metric was computed on control records** — `score_control`'s import graph
   excludes `study_a`, Murphy and RPS/W1 by construction.
8. `alt_set` not exercised (default off per D4).

## 8. Bound discipline

- **B = 0.2971 is not revised**, and no Phase-1 verdict is revised. Nothing in this campaign
  touches the r3 or rA1 frozen constant sets.
- The MMLU τ_TVD values (§5.1) are descriptive and **never** enter the frozen r3/rA1
  cluster/band/LOFO constants.
- **E1 is not delivered**, so no row of the design §3 interpretation table is claimed and
  the v15 sentence map in design §6 remains non-actionable as written.
- What this campaign licenses is narrow and real: the v2 scaffold **substantially repairs**
  the Mode-B collapse (74 → 19 empty-reasoning cells on the definitive Mode-B family, five of
  six subjects restored) and **steers base output onto the exemplar band** (in-band 2.3% →
  76.0%), while leaving a **single-subject residue** (`high_school_mathematics`) that no
  exemplar-side instrument in the current design reaches.

## 9. Artifacts

- `runs/control_mmlu_gemma31_v2/` — `pre_control.json`, `pre_control.meta.json`,
  `control_report.json` (gitignored, this Mac). New dir; the v1 record
  `runs/control_mmlu_gemma31/` is untouched.
- `runs/control_mmlu_qwen_v2/` — the gate leg, unchanged by this campaign.
- Mixed-pair E2 reports were computed out-of-tree (session scratchpad) and deliberately not
  written into any `runs/` dir, since neither is a clean single-scaffold pair.
