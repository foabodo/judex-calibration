# E1 final panel — option (d), the three outstanding legs at the k=4 v2 scaffold (2026-08-09)

Status: **PANEL MEASUREMENT COMPLETE. E1 IS NOT DELIVERED AS A PANEL READ.** The three legs the
recollection campaign's stop-loss left uncollected were run at the frozen k=4 v2 scaffold. All
three **FAIL** the `parse_ok ≥ 0.90` gate: qwen post **85.83%**, glm base **31.67%**, maverick
base **77.50%**. The four-family base panel therefore stands at **1 of 4 gate-passing** (qwen,
95.00%), and the left-edge (empty-reasoning) diagnosis now covers **three of four families**.
Spend **$30.013** against an est. $55–70 and a **$90** hard cap (**33.3%**). Fleet empty.

Predecessors, in the order of the arc: `docs/phase2_defined_answer_control_design.md` (D1–D8 and
the pre-specified E1/E2 reads in §1/§3) → `docs/phase2c_control_results_20260809.md` (the
v1-scaffold collapse) → `docs/e1_gate_qwen_v2_20260809.md` (the paid GO gate) →
`docs/e1_recollection_results_20260809.md` (the stop-loss) →
`docs/e1_k8_iteration_20260809.md` (the conceded density iteration) → **this document**.

Harness: branch `claude-verbalized-gap-closure` @ `062393e`, clean working tree, **no commits**.
Full suite green before any spend: **233 passed + 183 subtests**. The new records
(`runs/control_mmlu_qwen_v2/post_control.json`, `runs/control_mmlu_glm_v2/`,
`runs/control_mmlu_maverick_v2/`) are gitignored; every v1 dir and both gemma31 v2 dirs were
read-only throughout.

---

## 1. The scaffold, pinned — zero content changes

Every identity field below is byte-identical across all six v2 legs in the programme (qwen pre,
gemma31 pre k=4, and the three collected here), and each is carried in the leg meta sidecar with
its own resume-guard clause.

| pin | value |
|---|---|
| contract | `mmlu_control_v1` (D3: 4-option + confidence kept, findings dropped) |
| slice_sha256 | `31822879ab5e96…` — 120 items, 6 subjects × 20 |
| store_sha256 | `d34d855eac8c…` — `mmlu_control_exemplars_v2`, 333 rows |
| exemplar_selection | `band_400_800` |
| fewshot_k | **4** (`CONTROL_FEWSHOT_K`; k=8 was conceded and is not adopted) |
| scaffold_variant | `baseline` (`alt_set` never exercised, D4) |
| parsing | `parse_last_control` (last-object scan) |
| budget / decoding | 2048 tokens, greedy (`temperature: 0` sent explicitly), `--workers 8` |
| serving | bf16, `--max-model-len 32768`, `--gpu-memory-utilization 0.85` |
| gate | `parse_ok ≥ 0.90` (D3 — re-derived onto the tier that IS the same object across tasks) |

Exemplars actually rendered, all six subjects, k=4 (24 exemplars): spans **637–798, median
718.5, 24/24 in band** — the identical block every leg in this campaign saw.

Transports replicate each family's AIReg serving parity (D5): all three legs are raw
`/v1/completions`, bf16, TP=1 for qwen and TP=8 + `--enable-expert-parallel` for the two MoE
giants. The qwen **post** leg was served plain (no `--reasoning-parser`): the parser affects only
`/v1/chat/completions`, and serving it plain keeps the post leg's served configuration identical
to its own v2 pre leg — which is what the E2 pair needs.

---

## 2. Per-leg gate verdicts

| leg | scaffold | elicited | parsed | `parse_ok` | gate ≥ 0.90 | v1-scaffold leg | Δ vs v1 |
|---|---|---:|---:|---:|:--:|---:|---:|
| **qwen post** | v2, k=4 | 120 | 103 | **85.83%** | **FAIL** | 81.67% | **+4.17** |
| **glm base** | v2, k=4 | 120 | 38 | **31.67%** | **FAIL** | 50.00% | **−18.33** |
| **maverick base** | v2, k=4 | 120 | 93 | **77.50%** | **FAIL** | 79.17% | **−1.67** |

Context rows from the rest of the programme, unchanged by this campaign:

| leg | scaffold | `parse_ok` | gate |
|---|---|---:|:--:|
| qwen base | v2, k=4 | 95.00% | **PASS** |
| gemma31 base | v2, k=4 | 80.00% | FAIL |
| gemma31 base | v2, k=8 | 82.50% | FAIL (conceded) |
| gemma31 post | v1 | 100.00% | PASS |
| glm post | v1 | 90.83% | PASS |
| maverick post | v1 | 97.50% | PASS |

All 360 cells were elicited on the first pass. **No transient failures**: no
`ConnectionResetError`, no resume, no checkpoint recovery, no dead rentals, no stop-thrash. Every
failure recorded here is structural, carries its diagnostic, and was **not retry-burned** —
greedy decoding makes each one deterministic, so a retry reproduces the record.

**The headline result of this campaign is that the v2 scaffold's effect is family-dependent in
sign.** Against the v1 scaffold it moved qwen base **+8.33** and gemma31 base **+43.33**, and it
moves glm base **−18.33** and maverick base **−1.67**. A single scaffold change that repairs two
families and damages two others is not a scaffold fix; §3.4 shows it is not even confined to
parseability.

---

## 3. Failure taxonomies

Classes as defined in the Phase-2c post-mortem §4.1 and carried through every subsequent
document. `empty-reasoning collapse` = `no_json_object` with a reasoning span ≤ 1 character (the
LEFT edge); `escape` = `json_decode: Invalid \escape`; `brace-hijack` = `json_decode: Expecting
property name enclosed in double quotes`.

### 3.1 qwen post — the JSON-syntax leg, improved and still short

| class | Phase 2c (v1) | **this run (v2)** | qwen base v2 (ref) |
|---|---:|---:|---:|
| brace-hijack | **12** | **5** | 0 |
| escape | 4 | **7** | 4 |
| no JSON object (non-degenerate span) | 2 | **3** | 1 |
| invalid distribution | 4 | **1** | 1 |
| empty-reasoning collapse | 0 | **1** | 0 |
| total failures | 22 | **17** | 6 |

Per subject, parsed / 20:

| subject | v1 post | **v2 post** | degenerate spans v1 → v2 |
|---|---:|---:|---|
| clinical_knowledge | 19 | **19** | 4 → 0 |
| econometrics | 17 | **17** | 4 → 5 |
| formal_logic | 17 | **16** | 1 → 11 |
| high_school_mathematics | 7 | **12** | 13 → 4 |
| philosophy | 20 | **20** | 7 → 2 |
| professional_law | 18 | **19** | 1 → 0 |

Three things to read.

**The maths repair is real here too** (7/20 → 12/20, degenerate spans 13 → 4) — the same
direction the store-v2 span floor produced on every family with a maths cell. It is worth +4.2
points of leg-level parse rate and is essentially the whole of the leg's +4.17.

**The brace-hijack class is NOT extinct on this checkpoint.** It fell 12 → 5 under last-object
parsing but did not vanish, unlike on the base legs where it went to zero and stayed there across
four families. Under `parse_last_control` the `Expecting property name` signature no longer
implies the Phase-2c mechanism (a *preceding* TeX object hijacking the first-`{` scan) — it means
the **last** object in the emission is itself malformed. The records store only `raw_chars`, not
the raw emission, so the replacement mechanism is **named as an open question, not diagnosed**.
At 5 cells it is 29% of the residue and, with the 7 escape cells, the reason the leg misses the
gate.

**The band instrument is inert on an instruct checkpoint.** 62 of 120 cells emitted an
11-character reasoning span (v1: 46 of 120), and 70 of the 84 short-span cells parsed anyway; the
other 36 cells have a median span of 1598 characters. The post checkpoint does not imitate the
exemplar span at all — it is bimodal between a stub and a long free-form block. Reasoning-span
statistics therefore **do not compare across a pre/post pair**, and "in-band fraction" is not a
meaningful quantity on a post leg (4.9% here). That is a property of the instruct checkpoint, not
a defect of the scaffold.

### 3.2 glm base — the v2 scaffold made this family dramatically worse

| class | Phase 2c (v1) | **this run (v2)** |
|---|---:|---:|
| **empty-reasoning collapse** | **53** | **79** |
| no JSON object (non-degenerate span) | 6 | **1** |
| escape | 1 | **2** |
| brace-hijack | 0 | **0** |
| total failures | 60 | **82** |

Degenerate (≤ 1-char) spans **59 → 79**, and **0 of the 79 parsed** (v1: 6 of 59 did).

| subject | v1 | **v2** | delta | degenerate v1 → v2 |
|---|---:|---:|---:|---|
| clinical_knowledge | 14 | **10** | −4 | 7 → 10 |
| econometrics | 10 | **11** | +1 | 7 → 6 |
| formal_logic | 1 | **4** | +3 | 19 → 16 |
| high_school_mathematics | 1 | **0** | −1 | 19 → **20** |
| philosophy | 18 | **5** | **−13** | 0 → **15** |
| professional_law | 16 | **8** | **−8** | 7 → 12 |

**This is the sharpest negative result of the arc.** The two subjects flagged as glm's v1
left-edge concentration (`formal_logic` and `high_school_mathematics`, 19 empty-reasoning cells
each) barely moved — formal_logic gained three cells, maths went to a **complete 0/20 wipeout**.
What changed instead is that the two subjects glm handled *well* under v1 collapsed: `philosophy`
18 → 5 with its degenerate count going **0 → 15**, and `professional_law` 16 → 8. The v2 store's
longer exemplars did not teach this base checkpoint that reasoning is mandatory; on four of six
subjects they taught it to skip reasoning more often.

The subject bootstrap for this leg runs over **5 clusters, not 6** — `high_school_mathematics`
parsed 0/20 and is absent from the leg entirely, which is exactly the Phase-2c anomaly that
gemma31's own v2 leg had cured.

**And the band steering worked on the cells that did reason**: passing spans median **667**
against an exemplar median of 718.5, **78.9% in band** — the highest in-band fraction anywhere in
this programme. The instrument did what it was built to do; it simply has no purchase on the
decision to reason at all.

### 3.3 maverick base — one subject, and a small net regression

| class | Phase 2c (v1) | **this run (v2)** |
|---|---:|---:|
| **empty-reasoning collapse** | **18** | **25** |
| no JSON object (non-degenerate span) | 3 | **1** |
| escape | 3 | **1** |
| invalid distribution | 1 | **0** |
| brace-hijack | 0 | **0** |
| total failures | 25 | **27** |

Degenerate (≤ 1-char) spans **18 → 25**, and **0 of the 25 parsed** (v1: 0 of 18 — this family's
degenerate span has always been fatal).

| subject | v1 | **v2** | delta | degenerate v1 → v2 |
|---|---:|---:|---:|---|
| clinical_knowledge | 18 | **15** | **−3** | 2 → 5 |
| econometrics | 16 | **18** | **+2** | 0 → 1 |
| formal_logic | 19 | **19** | 0 | 0 → 1 |
| **high_school_mathematics** | **2** | **2** | **0** | 16 → **17** |
| philosophy | 20 | **19** | −1 | 0 → 1 |
| professional_law | 20 | **20** | 0 | 0 → 0 |

**The maths cell did not move at all**, and it is the entire gate story for this family: 18 of
the leg's 27 failures and 17 of its 25 degenerate spans sit in `high_school_mathematics`, which
parsed **2/20** under both scaffolds. Every other subject is at 15–20/20.

Arithmetic on the counterfactual: had maths matched this leg's non-maths average (18.2/20), the
leg would have parsed 109/120 = **90.8%** — a bare pass. Had it matched the *worst* non-maths
subject (15/20), 106/120 = 88.3% — still a fail, inside the 0.88–0.90 borderline band. So
maverick is the second family (with gemma31) whose gate turns entirely on one subject, and the
first where the store-v2 span floor produced **no** movement there whatsoever. Contrast qwen
base, where the same instrument took maths 10/20 → 19/20.

The other five subjects net **−2** (three flat, econometrics +2, clinical −3, philosophy −1),
which is the whole of the leg's −1.67. This family is *not* the glm case: there is no broad
collapse, just an immovable maths cell plus noise.

---

## 4. Reasoning-span distributions vs the exemplar band

Exemplars shown, k=4, all 24: **637–798, median 718.5, 24/24 in band.**

Model-produced spans, characters (linear-interpolated quantiles):

| leg | n | p25 | median | p75 | max | mean | in band [400, 800] |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen base v2, passing | 114 | 615.5 | 793.0 | 1144.2 | 8788 | 1054.9 | 43.9% |
| gemma31 base v2 k=4, passing | 96 | 583.5 | 685.5 | 782.0 | 1115 | 683.9 | 76.0% |
| gemma31 base v2 k=8, passing | 99 | 580.5 | 727.0 | 817.0 | 1254 | 722.9 | 69.7% |
| **glm base v2, passing** | 38 | 588.2 | **667.0** | 747.5 | 1037 | 679.2 | **78.9%** |
| **maverick base v2, passing** | 93 | 599.0 | **683.0** | 805.0 | 1394 | 699.8 | **69.9%** |
| **qwen post v2, passing** | 103 | 11.0 | 11.0 | 910.5 | 9235 | 762.0 | 4.9% |
| gemma31 base v2 k=4, failing | 24 | 1 | 1 | 1 | 894 | 143.1 | 16.7% |
| **glm base v2, failing** | 82 | 0 | 0 | 0 | 606 | 14.6 | 2.4% |
| **maverick base v2, failing** | 27 | 0 | 0 | 0 | 1033 | 38.4 | 0.0% |
| **qwen post v2, failing** | 17 | 1 | 1 | 11 | 5012 | 565.5 | 0.0% |

**CHANGE 1 is confirmed on all four base families and is confirmed irrelevant to the gate.**
Across the panel the passing-cell median now sits at **667–793** characters against an exemplar
median of 718.5 — a miss of at most 75 characters on any family — with in-band fractions of
43.9–78.9%. That is the band instrument working exactly as designed, on four independent
checkpoints. It is also, at this point, unambiguously **orthogonal to whether a leg passes**: glm
has the tightest median match in the programme (667 vs 718.5) and the worst parse rate in the
programme (31.67%).

**Every base-leg failure distribution is pinned at the left edge.** glm's and maverick's failing
cells have p25 = median = p75 = **0** characters; gemma31's sit at 1. qwen base remains the sole
family whose failures live in the right tail. The band has an upper bound to catch that
right-tail mechanism and, by construction, has nothing that reaches the left edge — which is the
whole finding of this arc.

### 4.1 The scaffold does not only change parseability — it changes the answers

Accuracy and ECE10 recomputed on the **items each family parsed under both scaffolds**, so
survivorship is held fixed:

| family | items parsed under both | acc v1 | acc **v2** | Δ acc | ECE10 v1 | ECE10 **v2** |
|---|---:|---:|---:|---:|---:|---:|
| qwen base | 101 | 0.8614 | **0.8911** | **+0.030** | 0.1033 | **0.0929** |
| gemma31 base | 43 | 0.6977 | **0.7674** | **+0.070** | 0.2740 | **0.2147** |
| **glm base** | 30 | 0.8667 | **0.7333** | **−0.133** | 0.1122 | **0.2385** |
| **maverick base** | 87 | 0.8391 | **0.7011** | **−0.138** | 0.1414 | **0.2774** |

On the *same* items, with mean confidence essentially unchanged (0.960–0.979 throughout), the v2
scaffold costs glm and maverick roughly **13–14 accuracy points** and roughly **doubles** their
ECE10, while gaining qwen and gemma31 3–7 points. This is the mechanism behind glm's leg-level
ECE10 moving 0.0789 → 0.2378 and maverick's 0.1378 → 0.3016: not survivorship, **behaviour**.
Longer, in-band reasoning makes two of these four base checkpoints answer worse at unchanged
confidence.

The consequence for the estimand is direct and must be stated: **the v2 scaffold is a different
scaffold, and on half the panel it moves the E1 level itself.** Any cross-scaffold comparison of
control ECE — including every mixed E2 row in §6.2 — inherits this.

---

## 5. E1 — the panel verdict

E1 (design §1) is base-leg top-1 ECE on MMLU under the identical harness, read **over
gate-passing base legs**. The four-family panel is now complete at the k=4 v2 scaffold:

| family | base leg under v2 | `parse_ok` | gate | in E1? | diagnosis if excluded |
|---|---|---:|:--:|:--:|---|
| **qwen** | `runs/control_mmlu_qwen_v2` | **95.00%** | **PASS** | **yes** | — |
| gemma31 | `runs/control_mmlu_gemma31_v2` | 80.00% | FAIL | no | left edge; 19 cells, **one subject** (maths 5/20) |
| glm | `runs/control_mmlu_glm_v2` | 31.67% | FAIL | no | left edge; 79 cells, **four subjects** |
| maverick | `runs/control_mmlu_maverick_v2` | 77.50% | FAIL | no | left edge; 25 cells, **one subject** (maths 2/20) |

**E1 IS NOT DELIVERED.** The estimand is a read over the family panel, and one gate-passing leg
is not a panel — the position both predecessor documents already took of this same leg
(gate doc §5: "One leg is not the estimand… a single family delivers no E1 row and none is
claimed here"; recollection doc §4). The panel *measurement* is now complete, and what it
measures is that **three of four base checkpoints cannot be brought over the data-quality gate by
any instrument in this design.**

**The left-edge diagnosis now covers three of four families, not two.** The campaign brief
anticipated that a glm failure would make it two-of-four with maverick clearing; maverick did not
clear. gemma31, glm and maverick all fail through the same mechanism — the base checkpoint
declines to emit a reasoning span and jumps to the JSON scaffold token — and they fail it with
0-to-1-character spans that are fatal in every case but one across all three legs. E1 is
therefore a **1-of-4 read**, and the honest description of the control is that it is
**gate-limited to a single family**.

### 5.1 The gate-passing set, scored

n = 1 family. Reported in full so the read is legible; **nothing here is promoted to an E1 row.**

| | qwen base, v2 scaffold, n = 114/120 |
|---|---:|
| accuracy (chance floor 0.25) | 0.9035 (**+0.6535**) |
| mean confidence | 0.9775 |
| overconfidence gap | +0.0739 |
| **ECE10** | **0.0802** |
| **AECE5** | **0.0739** |
| MCE10 | 0.1197 |
| AUROC (conf vs correct) | 0.4718 |
| **ECE10 item bootstrap** (B=2000, seed 20260721, D7 primary) | **[0.0309, 0.1361]** (114 items) |
| ECE10 subject-clustered sensitivity | [0.0266, 0.1394] (6 clusters) |
| grid-argmin `T_ece10` / value | 1.6008 / 0.0234 (**not saturated**) |

**Which row it lands in.** ECE10 = 0.0802 falls in the **≲ 0.10 (DACA-like)** row of the design
§3 interpretation table, and its item-bootstrap CI [0.031, 0.136] excludes the ≳ 0.20 row
entirely and lies wholly below the AIReg v14 floor-clearing band (0.175–0.398). The row's read is
"AIReg base miscalibration is task/reference/scaffold-specific; the harness does not manufacture
it on defined answers", and its pre-specified v15 consequence, quoted **verbatim** from design §3:

> §6.2 keeps the reversal, now same-harness; Limitations scope *tightens to the judging
> condition* — the strongest honest version of the current text

**That consequence is NOT licensed by this campaign.** It is conditioned on a delivered E1, and
one family is not the panel. It is recorded here so the read is already specified if the panel is
ever completed by other means.

### 5.2 The three descriptive companions — gate-failing, never promoted

| | gemma31 base (n=96) | **glm base (n=38)** | **maverick base (n=93)** |
|---|---:|---:|---:|
| accuracy (chance 0.25) | 0.6667 (+0.4167) | 0.7368 (+0.4868) | 0.6774 (+0.4274) |
| mean confidence | 0.9746 | 0.9746 | 0.9790 |
| overconfidence gap | +0.3079 | +0.2378 | +0.3016 |
| ECE10 | 0.3079 | **0.2378** | **0.3016** |
| AECE5 | 0.3079 | 0.2378 | 0.3016 |
| MCE10 | 0.8955 | 0.3933 | 0.3955 |
| AUROC | 0.6035 | 0.4661 | 0.4876 |
| ECE10 item bootstrap | [0.2204, 0.4013] | [0.1087, 0.3874] | [0.2072, 0.3988] |
| ECE10 subject-clustered | [0.2171, 0.4287] (6) | [0.0774, 0.3852] (**5**) | [0.1426, 0.4662] (6) |
| grid-argmin `T_ece10` | 2.4996 (not sat.) | 2.6923 (not sat.) | 2.8999 (not sat.) |

Descriptive only. All three legs failed the gate; none enters E1. Two properties are worth
recording because they bear on how a future reader might be tempted to use them:

1. **They are scored on heavily and non-randomly selected subsets** — 38, 93 and 96 of 120 —
   weighted away from `high_school_mathematics` in every case (glm 0/20, maverick 2/20,
   gemma31 5/20). glm's subset excludes an entire subject, so its cluster bootstrap runs over
   five clusters.
2. **Their ECE10 values are partly an artifact of the v2 scaffold itself** (§4.1): on matched
   items glm's ECE10 goes 0.1122 → 0.2385 and maverick's 0.1414 → 0.2774 under the scaffold
   change alone. Reading these three numbers as "the AIReg-like row on three families" would be
   reading a scaffold effect as a task effect.

### 5.3 The comparands, and the D8 caveat

- **AIReg v14 printed**: six bases 0.175–0.443; the five floor-clearing legs 0.175–0.398, median
  0.260, over a **0.20** chance floor (K=5).
- **DACA scaffold-free logit slice, matched K=4**: base 0.034–0.069, over a **0.25** chance floor.
- **This control**: the one gate-passing base leg (qwen) is **0.0802** — above DACA's band, far
  below AIReg's, and outside the AIReg floor-clearing range. The three gate-failing legs sit at
  0.2378–0.3079, i.e. inside the AIReg floor-clearing band, but they are gate-failing **and**
  scaffold-inflated (§4.1) and are not evidence for the ≳ 0.20 row.

**D8 applies and is not corrected for.** The MMLU control's ECE sits over a 0.25 chance floor;
AIReg's over 0.20. DACA is at K=4 and is therefore the better-matched comparand, which is
precisely why the design chose MMLU. Floors are stated; **no numeric equality is claimed.**

**Capability contrast (design §3.4 sense).** Every base in this panel clears the MMLU chance floor
by a wide margin on the items it answered — qwen 0.904 (+0.654), glm 0.737 (+0.487), maverick
0.677 (+0.427), gemma31 0.667 (+0.417) — while the same checkpoints sat at **0.33–0.54 over a
0.20 floor** on AIReg (glm 0.543, qwen 0.482, gemma31 0.442, maverick 0.325). Bases that are weak
judges on AIReg remain competent test-takers on MMLU. Whatever the AIReg base miscalibration is,
it is not "these checkpoints cannot do the task." Subject to the parsed-subset caveat throughout.

### 5.4 The honest scope, in one place

- The **≲ 0.10 (DACA-like)** row is supported by **one family: qwen**. Its consequence sentence is
  quoted above and is **not triggered**.
- **gemma31, glm and maverick are family-level exceptions**, each with the same left-edge
  diagnosis and none with a promotable number. Two of them (gemma31, maverick) miss the gate on a
  **single subject** and would pass at 90.8% if `high_school_mathematics` merely matched their own
  non-maths average; glm misses it across four subjects and would still fail at 58/120 with a
  perfect maths cell.
- The checkpoint-property finding — *whether a base checkpoint elects to reason at all is a
  property of the checkpoint, not reachable by exemplar content* — now rests on **three of four
  families**, and on a content lever whose sign is family-dependent (§2, §4.1). Both exemplar-side
  instruments the design had (the store-v2 span floor and the band draw) and the one sanctioned
  density iteration (k=8) are spent.
- **No promotion of any failing leg, anywhere in this document or downstream.**

---

## 6. E2 — the direction estimand

### 6.1 qwen: the one fully-v2 pair in the programme

`boot_paired_ece_delta`, B = 2000, seed 20260721, item-primary (D7). **This row REPLACES the
mixed-scaffold qwen row printed in `docs/e1_recollection_results_20260809.md` §5.**

| | qwen, **both legs v2, k=4, band, last-object** |
|---|---:|
| ECE10 pre (n=114) | 0.0802 |
| ECE10 post (n=103) | 0.0819 |
| **delta (post − pre)** | **+0.0017** |
| direction | post **degrades** |
| **item bootstrap CI** | **[−0.0461, +0.0752]** (99 paired items) |
| excludes 0 | **no** |
| subject-clustered CI | [−0.0502, +0.0568] (6 clusters) — excludes 0: no |
| intersection-only point delta (n=99) | pre 0.0744 → post 0.0863, **+0.0118** |
| bootstrap median | +0.0141 |
| *superseded* MIXED v2-pre/v1-post row | +0.0045, CI [−0.0825, +0.0600] |
| Phase-2c v1/v1 row | −0.0148, CI [−0.1027, +0.0474] |
| τ_TVD (D6, DESCRIPTIVE ONLY) | 1.0252 |

**Scaffold-version consistency is achieved and it changes nothing.** The clean delta is
**+0.0017** — a hundred-and-sixtieth of gemma31's — with a CI 70× wider than the point estimate.
The pair points marginally to "post degrades" under all three subset conventions (own-subset
+0.0017, intersection-only +0.0118, bootstrap median +0.0141), and none is distinguishable from
zero. Removing the scaffold confound did not resolve the qwen row; it confirmed there was never a
signal in it.

Two caveats this row carries, distinct from the confound the mixed rows carry:

1. **The post leg fails the gate** (85.83% < 0.90). "Clean" here means *single-scaffold*, not
   *gate-passing*. E2 is a within-family direction read and the design does not gate it, but the
   parsed subsets remain non-random and unequal — 114 vs 103, 99 in common.
2. **Pre and post legs never share an item subset** (the standing Phase-2c caveat). The point
   estimate of record is computed on each leg's own subset; the CI on the 99-item intersection.
   Both are printed above rather than one being quietly chosen.

### 6.2 The other three families remain MIXED-scaffold — recomputed, still labelled

Their v2 **post** legs do not exist and were not in this campaign's scope. Each MIXED row pairs a
**v2 pre** leg with a **Phase-2c v1 post** leg; the v1/v1 rows are the untouched Phase-2c pairs,
shown for contrast.

| family | pair | ECE10 pre | ECE10 post | delta | direction | item-boot CI | excl. 0 | paired n |
|---|---|---:|---:|---:|---|---|:--:|---:|
| gemma31 | MIXED v2/v1 | 0.3079 | 0.0339 | **−0.2741** | post improves | [−0.3808, −0.1817] | **yes** | 96 |
| gemma31 | v1/v1 (Phase 2c) | 0.2902 | 0.0339 | −0.2563 | post improves | [−0.3771, −0.1120] | yes | 44 |
| **glm** | **MIXED v2/v1** | 0.2378 | 0.0527 | **−0.1851** | post improves | [−0.2634, −0.0363] | **yes** | **36** |
| glm | v1/v1 (Phase 2c) | 0.0789 | 0.0527 | −0.0262 | post improves | [−0.0643, +0.0622] | no | 56 |
| **maverick** | **MIXED v2/v1** | 0.3016 | 0.0745 | **−0.2271** | post improves | [−0.3180, −0.1126] | **yes** | **91** |
| maverick | v1/v1 (Phase 2c) | 0.1378 | 0.0745 | −0.0633 | post improves | [−0.1187, +0.0146] | no | 93 |

**The confound, named again because this campaign made it worse, not better.** In every mixed row
the pre and post legs were elicited under *different exemplar stores, different exemplar-selection
rules and different parsers*, and scored on *different, non-randomly-selected item subsets*. Part
of any movement is scaffold version, not post-training — and §4.1 now measures that part directly.

**Both new mixed rows flipped from CI-spanning-zero to CI-significant, and the flip is the
confound.** glm's v1/v1 delta was −0.0262 (CI through zero); swapping in the v2 pre leg makes it
−0.1851 and CI-significant, entirely because the v2 pre leg's ECE10 rose 0.0789 → 0.2378 on a
subset that shrank 60 → 38 items. Maverick's went −0.0633 → −0.2271 for the same reason
(0.1378 → 0.3016). On matched items the v2 scaffold alone accounts for ECE10 rises of
+0.126 (glm) and +0.136 (maverick) — i.e. **most of each row's new "improvement" is the
reference leg being degraded by the scaffold, not the post leg being better.** glm's mixed row is
additionally bootstrapped on **36 common items**.

**Net E2 read: the Phase-2c direction claim is not strengthened and not overturned.** The one
family whose pair is scaffold-consistent (qwen) sits at +0.0017 with a CI through zero. The three
mixed families all point to "post improves", now all three CI-significantly, and all three are
confounded in the direction that *inflates* the improvement. It remains opposite to DACA's MMLU
direction, and the weight to place on it remains limited for exactly the reasons Phase 2c gave.

**The option not taken, and it is not this campaign's to run.** Re-collecting the three
outstanding **post** legs under v2 — gemma31 post (1×H200 vllm-chat), glm post and maverick post
(8×H200 each) — would buy full scaffold-version consistency for roughly **$30–35** at the rates
measured here (§7), down from the ~$55 estimated before this campaign: the low-`inet_down_cost`
hosts used here cut the download line from the Phase-2c $12–13 per big rental to **under $0.10**.
That is a decision for the orchestrator. It is separable from the base-leg problem — the post legs
already cleared the gate 3/4 under v1, so the value is comparability, not rescue — and it would
*not* deliver E1, which is gated on the base legs.

### 6.3 D6 descriptive τ (TVD) — fenced, never adopted

Different task, different reference regime; **must not enter the frozen r3/rA1 cluster/band/LOFO
constants.** None is saturated.

| family | pair | τ_TVD |
|---|---|---:|
| qwen | **v2/v2 (clean)** | 1.0252 |
| gemma31 | MIXED v2/v1 | 1.0252 |
| glm | MIXED v2/v1 | 0.9518 |
| maverick | MIXED v2/v1 | **0.3124** |

---

## 7. Cost — invoice-exact, per instance

> **Scope of this ledger, and where the cumulative one lives** (added 2026-08-09, closing the
> reconciliation finding that no source document carried a cumulative arc ledger). This §7 is
> invoice-exact for **this campaign's three rentals only** — $30.013. It is deliberately not a
> roll-up: the E1 arc also spent on the store-v2 re-annotation (corpus repo `cost_ledger.json`,
> $2.589), the qwen gate ($0.888, gate doc §6), the gemma31 stop-loss ($0.853, recollection doc
> §6) and the k=8 iteration ($0.821, k=8 doc §7). **The sole cumulative ledger of record — Part I
> + the E1 arc + the $0 checks, reconciled end to end against the vast credit trail — is §12 of
> `docs/verbalized_gap_closure_campaign_report_2026_08.md`** (E1-arc subtotal **$35.164**, project
> total **$318.88**). Quote cumulative figures from there, never by summing per-campaign ledgers
> by hand.

Three rentals, on-demand, one leg per instance, at most one box alive at a time, each destroyed
immediately on the gate read. Offers were selected on **cheapest expected total charge**, not
cheapest `$/hr`: the Phase-2c ledger showed download dominating the 8×H200 bill ($12–13 per
rental at $0.019/GB), so both big legs were placed on machine 37735, whose `inet_down_cost` is
**$0.00013/GB** at an 8159 Mbps pull. The 716.7 GB GLM and 803.2 GB Maverick downloads billed
**$0.08** and **$0.09**.

| instance | leg | machine | GPU | geo | $/hr billed (listed) | GPU-h | GPU | storage | download | **charge** |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 47307307 | qwen post | 143222 | 1×H200 NVL | Montana, US | 3.000 (3.267) | 0.153 | 0.460 | 0.043 | 0.082 | **$0.585** |
| 47308062 | glm base | 37735 | 8×H200 | US | 34.737 (35.018) | 0.434 | 15.069 | 0.141 | 0.080 | **$15.290** |
| 47310158 | maverick base | 37735 | 8×H200 | US | 34.737 (35.018) | 0.401 | 13.927 | 0.122 | 0.089 | **$14.138** |
| | | | | | | | | | **total** | **$30.013** |

| | |
|---|---:|
| vast, 3 instances (invoice sum) | **$30.013** |
| OpenRouter / API | $0.000 |
| **campaign total** | **$30.013** |
| estimate | $55–70 |
| hard cap | **$90** |
| **spend vs cap** | **33.3%** |

Wall clock 21:57 → 23:07 UTC, ~1h10m for three legs and three rentals. The 8×H200 legs billed 26
and 24 minutes of GPU end to end (image pull + 700–800 GB weight download + 8-way EP/TP init +
`torch.compile` + 120-cell elicitation); elicitation itself was 1–3 minutes on those boxes.

**Invoices polled to stability** per the standing rule (Phase 1b/2c: teardown-time reads
under-report). Per-leg read sequences:

- qwen post: **$0.520 → $0.585 → $0.585 (STABLE)** — first read under by **11.1%**.
- glm base: **$13.924 → $14.979 → $15.290 → $15.290 (STABLE)** — first read under by **8.9%**.
- maverick base: **$14.138 → $14.138 (STABLE at the first pair)**, then **re-polled after a
  further 3 minutes and again at $14.138** — the first leg in the programme with **no**
  under-report at the first post-teardown read, so it was deliberately re-confirmed rather than
  closed on the two-read rule alone.

**Credit reconciliation.** $64.875163 → **$34.861359**, a movement of **$30.013804** against a
settled invoice sum of **$30.013** — reconciling to a tenth of a cent. No auto top-up fired.
Fleet verified after each teardown and at close: `vastai show instances` returns **`[]`**.

### 7.1 The $50 float rule — a documented 2.0% shortfall before the third rental

The standing Phase-1b rule is "ensure ≥ $50 available in a single increment before each 8×H200
rental", because the $5 autobill trickle cannot feed a 700 GB download. Credit at each decision
point:

| decision point | credit | ≥ $50? |
|---|---:|:--:|
| before qwen post (1×H200 — rule does not apply) | $64.875 | — |
| **before glm base (8×H200)** | **$64.291** | **yes** |
| **before maverick base (8×H200)** | **$49.001** | **NO — $1.00 short (2.0%)** |

**The rule is arithmetically unsatisfiable at this credit level.** With $64.88 at the campaign's
start, any first 8×H200 leg costing more than $14.88 puts the second below $50 — and the measured
cost was $15.29. No further top-up was available: the +$25.00 that took credit from $41.55 to
$66.55 landed *before* the recollection campaign and is the "top-up in place" this campaign's
brief refers to. Topping up is not an action this session can take.

The check applied in its place, stated so it can be audited: **runway.** $49.001 at $34.74/hr is
**84.6 minutes** of continuous burn against a measured **26-minute** comparable leg (glm, same
machine, a 12% smaller download) — **3.0× margin** — with the campaign standing at **$15.875 of a
$90 hard cap (17.6%)** and a brief whose only stated stopping condition beyond that cap is the cap
itself. The rental proceeded on that basis and settled at $14.138, leaving $34.861. Recorded here
in full because a 2% deviation from a written rule should be visible rather than buried.

---

## 8. Gates and anomalies

1. **Gate 0/3 in this campaign; 1/4 across the completed base panel.** No gate-passing leg was
   added. Recorded, **not retry-burned** — greedy decoding makes every failure deterministic.
2. **The v2 scaffold's effect is family-dependent in sign** (§2) — the campaign's headline
   finding. +8.33 / +43.33 parse points on qwen and gemma31; −18.33 / −1.67 on glm and maverick.
3. **The scaffold moves the level estimand, not just parseability** (§4.1). On matched items it
   costs glm and maverick ~13–14 accuracy points at unchanged confidence and roughly doubles
   their ECE10. Every cross-scaffold ECE comparison in this programme inherits this.
4. **`high_school_mathematics` is the binding subject on every failing base family** — glm 0/20,
   maverick 2/20, gemma31 5/20 — and the one subject the store-v2 span floor fully repaired on
   qwen (10/20 → 19/20). On maverick it produced **zero** movement.
5. **glm's subject bootstrap runs over 5 clusters, not 6** — maths parsed 0/20 and the subject is
   absent from the leg. Same anomaly class as gemma31's Phase-2c pre leg.
6. **The brace-hijack class is not extinct on the qwen post checkpoint** (12 → 5, not → 0), and
   under last-object parsing its Phase-2c mechanism no longer explains it. The records retain
   only `raw_chars`, so the replacement mechanism is **unresolved and named as such** (§3.1). It
   is extinct on all four base legs.
7. **The band instrument is inert on instruct checkpoints.** 62/120 qwen post cells emit an
   11-character stub span and mostly parse anyway; span statistics are not comparable across a
   pre/post pair (§3.1).
8. **maverick τ_TVD 0.3124** — the lowest in the programme, on a mixed pair. Descriptive under D6
   and never adopted.
9. **maverick's ECE10 rose 0.1378 → 0.3016 on essentially the same subset size** (95 → 93) and a
   near-identical subject composition. This is behaviour, not survivorship, and is the cleanest
   instance of anomaly 3.
10. **No transient failures** anywhere: no `ConnectionResetError`, no resume, no checkpoint
    recovery, no dead rentals, no stop-thrash, no NCCL/rendezvous failures on either 8-way box.
    All 360 cells elicited on the first pass.
11. **No ordinal metric was computed on control records** — `score_control`'s import graph
    excludes `study_a`, Murphy and RPS/W1 by construction, and the guard test for that is in the
    green suite.
12. `alt_set` was not exercised (default off, D4).
13. **$50 float shortfall of $1.00 before the third rental**, documented with its runway
    substitute in §7.1.

---

## 9. Bound discipline

- **B = 0.2971 is not revised**, and **no Phase-1 verdict is revised.** Nothing in this campaign
  touches the r3 or rA1 frozen constant sets.
- The MMLU τ_TVD values (§6.3) are descriptive under D6 and **never** enter the frozen r3/rA1
  cluster/band/LOFO constants.
- **E1 is not delivered.** No row of the design §3 interpretation table is claimed, and the v15
  sentence map in design §6 remains non-actionable as written. The row the single gate-passing
  family *would* land in, and its consequence, are quoted in §5.1 for the record only.
- `CONTROL_FEWSHOT_K` remains **4**. The k=8 iteration was conceded and is not adopted.
- **Failing legs are never promoted.** The gemma31, glm and maverick descriptive tables (§5.2)
  are reported so those legs are legible and for no other purpose.
- What this campaign licenses is narrow and real: the base panel is **complete and gate-limited to
  one family**; the left-edge mechanism is confirmed on **three of four** base checkpoints; the
  exemplar-side lever that repairs it on two families **damages** it on the other two; and the
  qwen E2 pair, now scaffold-consistent, carries **no signal in either direction**.

## 10. Artifacts

- `runs/control_mmlu_qwen_v2/post_control.json` + `.meta.json` — the new post leg. The dir's
  `pre_control.json` (the gate leg) is unchanged; `control_report.json` was regenerated by
  `--analyze` so it carries the pair.
- `runs/control_mmlu_glm_v2/` and `runs/control_mmlu_maverick_v2/` — **fresh** dirs
  (`pre_control.json`, `pre_control.meta.json`, `control_report.json`).
- Untouched and read-only throughout: `runs/control_mmlu_{qwen,gemma31,glm,maverick}/` (the
  Phase-2c v1 record), `runs/control_mmlu_gemma31_v2/`, `runs/control_mmlu_gemma31_v2k8/`.
- The cross-family §4.1 and §6.2 recomputations were run out-of-tree (session scratchpad) and
  deliberately **not** written into any `runs/` dir, since none is a clean single-scaffold pair.
- All `runs/` dirs are gitignored, this Mac. **No commits were made.**
