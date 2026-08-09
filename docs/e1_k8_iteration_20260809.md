# E1 residual, option (b) — the k=8 exemplar-density iteration and its gemma31 micro-gate (2026-08-09)

Status: **(b) CONCEDED. `parse_ok` 82.50% (99/120) against the 0.90 gate — FAIL, and outside
the 0.88–0.90 borderline band. Doubling exemplar density bought +2.5 points and did not close
the gap.** Spend **$0.821** against a $10 hard cap (8.2%). Fleet empty. The exemplar-side levers
are exhausted; the left-edge mechanism is not reachable by scaffold content.

The ONE sanctioned additional scaffold-content iteration after the E1 re-collection campaign
halted at its stop-loss (`docs/e1_recollection_results_20260809.md`). Predecessors:
`docs/e1_gate_qwen_v2_20260809.md` (the paid GO gate), `docs/phase2c_control_results_20260809.md`
(the v1-scaffold post-mortem), design `docs/phase2_defined_answer_control_design.md`.

Harness: branch `claude-verbalized-gap-closure` @ `5327351` + the k=8 selector generalization
below. No commits. `runs/control_mmlu_gemma31_v2k8/` is gitignored and is a **fresh** dir; the
k=4 v2 record `runs/control_mmlu_gemma31_v2/` and the Phase-2c v1 record
`runs/control_mmlu_gemma31/` are both untouched.

---

## 1. The hypothesis, stated so it can fail

The k=4 v2 leg scored `parse_ok` **80.00%** against the 0.90 gate. The residue was one subject:
`high_school_mathematics` supplied **15 of the 24 failures** and **15 of the 20 degenerate
(≤ 1-character) reasoning spans**, and parsed **5/20**. Every failing cell in that subject is a
LEFT-edge span — the base checkpoint emits nothing after `Reasoning:` and jumps straight to the
JSON scaffold token, then EOS. Store v2's span floor lengthened what the exemplars *show*; the
band draw steered the model's own spans onto that show (in-band 2.3% → 76.0%). Neither made the
base checkpoint elect to reason about a one-line symbolic maths question.

**(b)'s hypothesis:** doubling the demonstration density — k=8, two exemplars per answer letter
— gives the base more pattern mass to lock onto before the target item, and the "reasoning span
is non-optional" regularity is carried by eight demonstrations instead of four.

This is a **content lever of the same kind** as the two already shipped, and it is the last one
the design has. If it does not move the maths cell, the left-edge mechanism is beyond
exemplar-side instruments and the concession is the result.

## 2. What changed in the harness ($0, STEP 1)

`k=8` needed no new selection machinery — the round-robin already walks the letter ramp
repeatedly — but the coverage rule and its guards were stated at k=4 and had to be generalized.

**`src/judex_calibration/control_fewshot.py`**

- The coverage rule is now stated as **"every letter covered, `k/4` exemplars per letter"**, of
  which "one per letter" is the k=4 instance. `CONTROL_FEWSHOT_K` is **unchanged at 4** — k=8 is
  a per-leg `--fewshot-k`, not a new protocol constant. The default moves only if the
  orchestrator adopts it.
- New `assert_letter_balanced(subject, rows, k, what)` — a FAIL-LOUD guard applied to **every**
  draw (baseline, rev_order, and both halves of alt_set), asserting `len(rows) == k` and exactly
  `k/4` rows per letter. It also rejects a `k` that is not a positive multiple of 4: a k=5
  control draw would cover one letter twice and three once, re-introducing on the control the
  exact letter imbalance that made k=4 defective on the 5-level AIReg instrument.
- The `alt_set` guard gained a **per-letter count** check on top of the existing letter-SET
  check. At k=4 the two coincide; at k > 4 an alternate draw can match the set while
  re-weighting the letters (3 A's and 1 B at k=8), which is a different scaffold, not a
  source-disjoint twin of the same one.
- `alt_set_feasibility` now reports `starved_letters` **at the requested density** — a letter is
  starved when the source-excluded remainder holds fewer than `k/4` rows, not merely zero.

Nothing else moved: no exemplar edits, no parser edits, no prompt-text edits, no contract
change, no decoding-parameter change, no change to the scored slice. The store pin, the band,
the last-object parser, the stop lists, the K=4 answer contract and the 2048-token budget are
byte-identical to the k=4 v2 leg.

### 2.1 The k=8 draw walks correctly

Deterministic; 8 rows, 2 per letter, no repeated exemplar, for all six subjects. **The k=8 draw
is a strict superset of the k=4 draw** — the walk is deterministic and pass 1 is unchanged, so
the density iteration ADDS four exemplars per subject and substitutes none. That is what makes
this a clean density contrast rather than a second, differently-populated scaffold.

Per-subject rendered spans (characters), k=8, band `[400, 800]`:

| subject | k=4 spans | k=8 spans (added four in **bold**) | min | median | max | in band |
|---|---|---|---:|---:|---:|---:|
| clinical_knowledge | 723, 734, 693, 677 | 723, **696**, 734, **720**, 693, **701**, 677, **687** | 677 | 698 | 734 | 8/8 |
| econometrics | 697, 753, 646, 638 | 697, **637**, 753, **814**, 646, **698**, 638, **691** | 637 | 694 | 814 | **7/8** |
| formal_logic | 792, 775, 672, 796 | 792, **739**, 775, **591**, 672, **630**, 796, **780** | 591 | 757 | 796 | 8/8 |
| high_school_mathematics | 714, 648, 798, 637 | 714, **765**, 648, **760**, 798, **745**, 637, **628** | 628 | 730 | 798 | 8/8 |
| philosophy | 708, 770, 647, 784 | 708, **764**, 770, **721**, 647, **632**, 784, **781** | 632 | 742 | 784 | 8/8 |
| professional_law | 795, 754, 755, 661 | 795, **781**, 754, **742**, 755, **748**, 661, **762** | 661 | 754 | 795 | 8/8 |

All 48: **min 591, median 722, max 814, 47/48 in band.** (k=4, all 24: 637–798, median 718,
24/24 in band.) The single out-of-band pick is `econometrics/B` at **814 characters — 14 over
the ceiling**, and it is the documented nearest-the-band fallback firing exactly where it
should: that is the only (subject, letter) bucket on store v2 with in-band depth 1, so the
second pass has nothing in band left to draw. Pinned in the test suite by bucket name, so a
future store that loses in-band depth fails there rather than silently in a live leg.

The exemplar band is therefore essentially unmoved by the density change: the k=8 median (722)
sits within 4 characters of the k=4 median (718), which matters because the band's whole
function is to steer the model's own span, and a shifted target would confound the density read.

Render order is the repeated letter ramp **A, A, B, B, C, C, D, D** — the sort is stable, so
pass 1's exemplar for a letter precedes pass 2's, and the block is still one monotone sweep over
the letters. Least-used-rater balancing holds: 4–7 distinct rater seats per k=8 block, no seat
taking more than 3 of the 8.

### 2.2 `alt_set` at k=8 — feasible for three of six subjects (REPORTED, not required)

| subject | k=4 | k=8 | starved letter at k=8 |
|---|:--:|:--:|---|
| clinical_knowledge | feasible | **feasible** | — |
| high_school_mathematics | feasible | **feasible** | — |
| professional_law | feasible | **feasible** (6 of 8 rows fall back on span) | — |
| econometrics | feasible | **infeasible** | B |
| formal_logic | feasible | **infeasible** | B |
| philosophy | feasible | **infeasible** | D |

Store v2 is depth **2 in source items** per (subject, letter) bucket, so a k=8 primary that
spends both source items of a letter leaves that letter with nothing under source-label
exclusion, and the guard raises — correctly, and loudly. `alt_set` is default-OFF (D4) and no
leg in this programme has used it, so this is a reported property of the store's depth, not a
blocker: raising the density trades the disjoint-twin variant for demonstration mass, and the
guard makes that trade visible instead of silent.

### 2.3 Token budget (offline, gemma-4-31B's own tokenizer)

| | k=4 | k=8 |
|---|---:|---:|
| few-shot block, min / median / max tokens | 1806 / 2177 / 2605 | 3556 / 4432 / 5155 |
| stage-1 prompt, min / median / **worst** | 1881 / 2334 / **3134** | 3633 / 4572 / **5684** |
| required = worst prompt + 2048 budget + scaffold + 1 | 5186 | **7736** |
| headroom against the 32768 ctx pin | 27582 | **25032** |

Doubling the demonstrations costs ~2.5k tokens of context and leaves 25k of headroom. Not a
constraint; the leg is slightly longer in wall clock only because the prompt is longer.

### 2.4 Test suite

**233 passed + 183 subtests** (k=4 v2 leg: 221 + 116). The added `DensityTests` class states
the coverage rule at **both** densities — letter balance, determinism, rater spread, the
repeated ramp, the k=8 ⊃ k=4 superset property, the band-fallback ledger, the k=8 render, and
the store's backslash/control-character invariant on the four ADDED exemplars per subject — plus
two new fail-loud paths (a non-multiple-of-4 `k`, and a store that starves a bucket at density
2 while being fine at density 1).

## 3. Serving + run

| pin | value | vs the k=4 v2 leg |
|---|---|---|
| model | `google/gemma-4-31B`, bf16, raw `/v1/completions` | same |
| box | 1×H200 NVL, machine 143222, Montana US, on-demand `--order dph`, $3.00/hr | same class |
| vLLM | `--max-model-len 32768 --gpu-memory-utilization 0.85 --tensor-parallel-size 1` | same |
| driver | `run_control_leg.py --out runs/control_mmlu_gemma31_v2k8 --leg pre --fewshot-k 8 --workers 8` | **k 4 → 8** |
| contract | `mmlu_control_v1` | same |
| slice_sha256 | `31822879ab5e96…` (120 items, 6 subjects × 20) | same |
| store_sha256 | `d34d855eac8c…` (`mmlu_control_exemplars_v2`, 333 rows) | same |
| exemplar_selection | `band_400_800` | same |
| parsing | `parse_last_control` (last-object scan) | same |
| scaffold / decoding | `baseline`, budget 2048, greedy (temperature 0) | same |
| gate | `parse_ok ≥ 0.90` (D3) | same |

`runs/control_mmlu_gemma31_v2k8/` is a **fresh** dir. Every provenance pin except `fewshot_k` is
byte-identical to the k=4 v2 leg, so this is a single-variable contrast. All 120 cells elicited
on the first pass; no transient failures, no resume, no retry (greedy — a retry reproduces the
same records).

## 4. Gate verdict — FAIL

| | Phase 2c (v1, length-blind, first-`{`) | k=4 v2 | **k=8 v2 (this run)** |
|---|---:|---:|---:|
| elicited | 120/120 | 120/120 | **120/120** |
| parsed | 44 | 96 | **99** |
| `parse_ok` | 36.67% | 80.00% | **82.50%** |
| gate (≥ 0.90) | FAIL | FAIL | **FAIL** (7.5 points short) |
| 5-field `contract_complete` | 36.67% | 80.00% | **82.50%** |
| degenerate (≤ 1-char) spans | 76 | 20 | **17** |
| of which parsed anyway | 2 | 1 | **0** |

**+2.5 points.** Doubling the demonstration mass moved the leg from 80.00% to 82.50% — real,
reproducible, and roughly a fifth of what the gate needed. It is outside the 0.88–0.90
borderline band, so there is no second reading to report.

The degenerate-span count fell 20 → 17, and **none of the 17 parsed** (k=4: 1 of 20). The
mechanism is unchanged: when this checkpoint elects not to reason, the cell is lost.

## 5. Failure taxonomy

| class | Phase 2c | k=4 v2 | **k=8 v2** |
|---|---:|---:|---:|
| **empty-reasoning collapse** (`no_json_object`, span ≤ 1 char) | **74** | **19** | **17** |
| escape (`json_decode: Invalid \escape`) | 2 | 1 | **2** |
| invalid distribution | 0 | 2 | **0** |
| no JSON object (non-degenerate span) | 0 | 0 | **1** |
| brace-hijack (`Expecting property name…`) | 0 | 0 | **0** |
| other `json_decode` (`Expecting ',' delimiter`) | 0 | 2 | **1** |
| total failures | 76 | 24 | **21** |

Reading it:

- **Mode B is still 81% of the residue** (17 of 21) and is still the binding constraint. Every
  other class sits at 0–2 cells, exactly as at k=4.
- **The brace-hijack class stays extinct** — 0 cells, a third consecutive leg. CHANGE 2 holds.
- **One genuinely new signature**: a single `high_school_mathematics` cell emitted a **5,887-character**
  reasoning span and then failed to produce a JSON object at all. That is the *right*-tail
  (elaborateness) mechanism, which gemma31 had never shown — its k=4 maximum span was 1,115
  characters. Doubling the demonstrations bought a right tail as well as a left-edge repair. At
  one cell it is not the reason the gate missed, but it is the first evidence that further
  density would trade one failure mode for the other rather than removing either.

### 5.1 THE decisive cell — `high_school_mathematics`

| | k=4 v2 | **k=8 v2** |
|---|---:|---:|
| parsed | **5/20** | **10/20** |
| degenerate (≤ 1-char) spans | 15 | **9** |
| 1-char LEFT-edge failures | 15 | **9** |
| right-tail failure (span 5,887, no JSON) | 0 | **1** |

**The hypothesis was directionally right and quantitatively insufficient.** The maths cell
**doubled**, 5/20 → 10/20, and the left-edge count fell by 40%, 15 → 9. This is the largest
movement any instrument in this programme has produced on that subject, and it confirms the
mechanism the iteration targeted: more demonstration mass does make this base checkpoint more
likely to elect to reason about a short symbolic item.

It is also, on its own, worth only +4.2 points of leg-level parse rate — and the leg gained only
+2.5, because the maths gain was partly spent elsewhere (§5.2). Even a *perfect* maths cell
(20/20) would have put this leg at 109/120 = 90.8%, a bare pass; the subject cannot carry the
gate alone, and it did not come close to perfect.

### 5.2 The other five subjects — one regressed, and it disqualifies the run twice over

| subject | k=4 v2 parsed | **k=8 v2 parsed** | delta | degenerate k=4 → k=8 | left-edge failures k=4 → k=8 |
|---|---:|---:|---:|---|---|
| professional_law | 20/20 | **20/20** | 0 | 0 → 0 | 0 → 0 |
| econometrics | 18/20 | **18/20** | 0 | 0 → 0 | 0 → 0 |
| clinical_knowledge | 18/20 | **18/20** | 0 | 1 → 2 | 1 → 2 |
| philosophy | 16/20 | **19/20** | **+3** | 2 → 0 | 2 → 0 |
| high_school_mathematics | 5/20 | **10/20** | **+5** | 15 → 9 | 15 → 9 |
| **formal_logic** | 19/20 | **14/20** | **−5** | 2 → **6** | 1 → **6** |

**`formal_logic` lost five cells, and lost them to the same left-edge mechanism the iteration
was meant to fix**: its degenerate-span count *tripled*, 2 → 6, and all six are failures. This
is not noise — the run is greedy and deterministic — and it is not a parsing artifact; it is the
Mode-B collapse appearing in a subject that had very nearly been repaired at k=4.

That is a **−5 regression against a −2 tolerance**, so the decision rule's second clause fails
independently of the first. Even had `parse_ok` reached 0.90, this leg would not have qualified
k=8 as a uniform scaffold: it moves failure mass between subjects rather than removing it. Two
subjects gained 8 cells, one lost 5, three were unmoved, and the net was +3.

The honest summary of the density lever is therefore **redistribution, not repair**.

## 6. Reasoning-span distribution vs the exemplar band

Exemplars rendered: k=4 spans 637–798, median 718, 24/24 in band; k=8 spans 591–814, median 722,
47/48 in band (§2.1). The target the model is shown is essentially unmoved.

Model-produced spans, characters (linear-interpolated quantiles):

| leg | n | p25 | median | p75 | max | mean | in band [400, 800] |
|---|---:|---:|---:|---:|---:|---:|---:|
| k=4 v2, passing | 96 | 583.5 | 685.5 | 782.0 | 1115 | 683.9 | **76.0%** |
| **k=8 v2, passing** | 99 | 580.5 | **727.0** | 817.0 | **1254** | 722.9 | **69.7%** |
| k=4 v2, failing | 24 | 1 | 1 | 1 | 894 | 143.1 | 16.7% |
| **k=8 v2, failing** | 21 | 1 | 1 | 1 | **5887** | 391.6 | **9.5%** |
| qwen v2 gate leg, passing (ref) | 114 | 616 | 793 | 1144 | 8788 | 1055 | 43.9% |

Two things to read.

**The span-to-scaffold match got tighter, not looser.** The passing median moved 685.5 → **727.0**
against an exemplar median of **722** — a 5-character miss, the closest match observed anywhere in
this programme. The in-band *fraction* fell (76.0% → 69.7%) only because the whole distribution
shifted right past the 800-character ceiling: p75 rose 782 → 817. The band instrument is working
exactly as designed and is now, if anything, slightly *too* tight a target for the density.

**The failure distribution is still pinned at the left edge, with a new right tail.** p25 =
median = p75 = 1 character, unchanged. What changed is the maximum: 894 → **5,887**, the single
runaway cell of §5. So k=8 shifted a little probability mass out of the left edge (20 → 17
degenerate cells) and, for the first time on this family, some into the right tail. The two
failure modes gemma31 and qwen exhibit separately are both now present on gemma31.

## 7. Cost — invoice-exact

One rental, on-demand, `--order dph`, destroyed immediately on the gate read.

| instance | leg | machine | GPU | geo | $/hr | GPU | storage | download | **charge** |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| 47305076 | gemma31 base (pre), k=8 | 143222 | 1×H200 NVL | Montana, US | 3.000 (listed 3.267) | 0.682 | 0.068 | 0.071 | **$0.821** |

| | |
|---|---:|
| vast, 1 instance (invoice sum) | $0.821 |
| OpenRouter / API | $0.000 |
| **iteration total** | **$0.821** |
| estimate | $1–2 |
| hard cap | $10 |
| **spend vs cap** | **8.2%** |

Billed GPU 0.227 h (~13.6 min): ~1.2 min weight pull (58.25 GiB checkpoint, 74 s — a fast host),
~2.5 min engine init + `torch.compile` + CUDA-graph capture, ~10 min elicitation of 120 cells at
`--workers 8`. Download 54.7 GB at $0.001/GB = $0.071.

**Invoice polled to stability** per the standing rule. Post-teardown reads:
**$0.696 → $0.757 → $0.790 → $0.821 → $0.821 → $0.821 (STABLE)** — the first read under-reported
by **15.2%**, inside the 10–17% band the rule was written for and the worst under-report yet
observed. Credit reconciliation: **$65.696634 → $64.875163**, a movement of **$0.821470** against
a settled invoice sum of **$0.821** — reconciling to a twentieth of a cent. No auto top-up fired.

Fleet verified after teardown: `vastai show instances` returns `[]`.

## 8. Decision-rule outcome — **(b) CONCEDED**

The rule, as pre-stated:

> `parse_ok < 0.90` → **(b) CONCEDED**: the left-edge mechanism is beyond content levers;
> report and stop.

`parse_ok` = **0.8250**. The **CONCEDED** branch fires, and it fires cleanly — 82.50% is 7.5
points under the gate and 5.5 points under the bottom of the 0.88–0.90 borderline band, so there
is no second reading and no recommendation beyond the numbers.

The success branch's second clause fails independently: `formal_logic` went 19 → 14, a **−5**
regression against the **−2** tolerance. The run would have been disqualified as a uniform
scaffold even at a passing parse rate.

**Consequences that do NOT follow, stated so they are not assumed:**

- k=8 does **not** become the candidate uniform control scaffold. `CONTROL_FEWSHOT_K` stays
  **4** and no re-gate of the qwen base at k=8 is triggered.
- No E1 row is claimed. E1 still stands at **1 of 4 base families gate-passing** (qwen 95.00%),
  unchanged by this iteration, and the design §3 interpretation table remains unclaimed.
- The gemma31 k=8 leg's descriptive metrics (below) are **not** promoted, exactly as the k=4 v2
  leg's were not.

**What this iteration nonetheless established, and it is not nothing:**

1. **The density hypothesis is real but small.** Demonstration mass does move the left-edge
   collapse — maths 5/20 → 10/20, left-edge failures there 15 → 9 — which is the first
   confirmation of the mechanism the whole exemplar-side programme has been theorising.
2. **The lever redistributes rather than repairs.** +8 cells across two subjects, −5 in a third,
   net +3. A content instrument that moves failure mass between subjects is not a scaffold fix,
   and no amount of further tuning of the same lever changes that shape.
3. **Both failure tails are now live on one family.** The 5,887-character runaway says the next
   increment of density buys right-tail JSON breakage in exchange for left-edge repair. The band
   has an upper bound precisely to catch that mode, and at k=8 the model's p75 has already
   crossed the band ceiling.

Points 2 and 3 together are the substantive reason the concession is a concession and not a
"try k=12": the lever's marginal return is small, its sign is not uniform across subjects, and
its cost has begun to appear in the opposite failure mode.

### 8.1 Descriptive companions — NOT promoted, gate-failing leg

| | k=4 v2 (n=96) | **k=8 v2 (n=99)** |
|---|---:|---:|
| accuracy (chance floor 0.25) | 0.6667 | **0.7576** |
| mean confidence | 0.9746 | 0.9677 |
| overconfidence gap | +0.3079 | **+0.2101** |
| ECE10 | 0.3079 | **0.2101** |
| AECE5 | 0.3079 | 0.2101 |
| MCE10 | 0.8955 | **0.2163** |
| AUROC (conf vs correct) | 0.6035 | 0.5256 |
| ECE10 item bootstrap (B=2000, seed 20260721) | [0.2204, 0.4013] | [0.1308, 0.2991] |
| ECE10 subject-clustered (6, all present) | [0.2171, 0.4287] | [0.1143, 0.3186] |

Descriptive only; the leg failed the gate, and **the two rows are not comparable as calibration
measurements** — they are scored on different, non-randomly-selected 96- and 99-item subsets of
the same 120. The ECE10 drop 0.308 → 0.210 tracks the accuracy rise 0.667 → 0.758 at essentially
unchanged confidence, and the accuracy rise is mostly survivorship: the k=8 subset includes five
more maths items, which are the items this checkpoint gets right when it reasons at all. The
MCE10 collapse (0.896 → 0.216) is the k=4 leg's single-item `[0.8, 0.9)` bin artifact
disappearing, not a calibration improvement. Nothing here is a finding.

## 9. Artifacts and disposition

- `runs/control_mmlu_gemma31_v2k8/` — `pre_control.json`, `pre_control.meta.json`,
  `control_report.json` (gitignored, this Mac). New dir; `runs/control_mmlu_gemma31_v2/` (k=4 v2)
  and `runs/control_mmlu_gemma31/` (Phase-2c v1) are untouched.
- Harness changes (§2) are **kept**: they are a strict generalization and strengthening of the
  selector's fail-loud guards, they leave `CONTROL_FEWSHOT_K = 4` and every k=4 draw
  byte-identical, and the suite pins both densities. Nothing about keeping them adopts k=8.
- **No commits.** Working tree only.

## 10. Bound discipline

- **B = 0.2971 is not revised**, and no Phase-1 verdict is revised. Nothing here touches the r3
  or rA1 frozen constant sets.
- **E1 is not delivered.** No row of the design §3 interpretation table is claimed and the v15
  sentence map in design §6 remains non-actionable.
- This was the **one sanctioned additional scaffold-content iteration**. It is spent. The
  exemplar-side design has no further instrument, and the orchestrator's remaining choices are
  the ones that were on the table before it: **(d) at k=4** (collect the outstanding legs under
  the k=4 v2 scaffold and accept a panel with gemma31 excluded) versus **(a)**.
