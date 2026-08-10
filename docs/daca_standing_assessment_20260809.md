# DACA — Standing Assessment after the MMLU Control (2026-08-09)

Status: assessment of record for the v15 authoring pass. Consolidates the 2026-07-21
premise analysis (`spec/analysis_2026_07_21_base_calibration_premise.md`), the 2026-07-24
objective-contamination audit, and the E1 arc (`docs/e1_final_panel_results_20260809.md`,
campaign report §7a). One open check is in flight and gets appended here on completion
(§5; `docs/tau_daca_scaffold_variant_check_20260809.md`).

DACA = Luo et al., NeurIPS 2025 (arXiv:2505.16690): KL alignment of a post-trained
model to a pre-trained reference on argmax-agreement cells; logits method; verbalized
elicitation is its baseline; Fig. 1 measures base ECE 0.034–0.069 / post 0.190–0.242 on
MMLU via a scaffold-free next-token slice over {A,B,C,D}.

## 1. DACA's claims, disentangled

| claim | kind | standing after our record |
|---|---|---|
| C1 "pre-trained models are well calibrated" | premise (motivational) | **ill-posed as a model-intrinsic property; false in the judging condition; loosely held (n=1) under our verbalized harness on MMLU; untested by us in their own elicitation mode** |
| C2 the agreement-filtered alignment estimator works | method | **corroborated by us, twice, beyond its stated assumptions** |
| C3 post-training degrades calibration (3–7× on MMLU) | direction | **reversed in the judging condition; null on our one clean MMLU pair; unrefuted in their setting** |
| C4 logits > verbalized elicitation | channel | **setting-dependent — vindicated for base-reference elicitation on defined answers (by our own failure); inapplicable to production distributional judging** |
| — truthfulness of their measurements | — | **unchallenged; what falls is the imported generalization, not their numbers** |

## 2. The evidence, item by item

**C2 (method) — the part we keep confirming.**
- Full-scale verbalized τ_DACA: 12/12 valid cross-family fits, median 0.922, inside the
  registered F7 window — computed with our open bases as references, the same bases that
  need T_abs = 2.55–5.53 against the human panel. *The estimator works while the stated
  premise is false*: it needs an informative, exchangeable reference, not a calibrated
  one. Survives the matched-objective audit (matched-W1 median 0.988).
- Corroborated again on the on-pair confirmation sweep (E6 arm 3a, 4 valid fits).

**C4 (channel) — the E1 arc is a vindication of their design choice, in their setting.**
Three of four base checkpoints will not execute a verbalized two-stage contract on short
symbolic defined-answer items (left-edge reasoning skip; robust to span length, exemplar
density, and parsing — $35 of remediation proved it a checkpoint property). DACA's logit
slice sidesteps elicitability entirely. For *their* use case — base models as calibration
references on defined-answer tasks — the logit channel is arguably the only reliable one;
their cross-architecture headline (GPT-4o calibrated by a Gemma base) would have been
impossible with verbalized reference elicitation on 3/4 of our families. No transfer of
that superiority to production distributional judging: deployed judges expose no logits,
the estimand is a full distribution against a dispersed reference, top-1 ECE is nearly a
restatement of the accuracy deficit there, and the repair asymmetry (one temperature
removes ~70% of ECE but ~16% of W1 distance; 66–97% of W1 error is location) makes the
two calibration senses different repair problems. Our own logit-channel program failed in
the judging setting while the verbalized one produced the paper.

**C1 (premise) — three grades of failure, one honest concession.**
1. *Ill-posed (R10, the deepest cut):* measured base ECE on defined answers is
   scaffold-entangled with family-dependent SIGN — one scaffold change moved
   both-scaffold-item accuracy glm −13.3 / maverick −13.8 pts at unchanged confidence
   (~2× ECE) while moving qwen +3.0 / gemma31 +7.0. "Base model X has ECE ≈ y" is a
   property of the (model, elicitation) pair. DACA's Fig. 1 is elicitation-conditional,
   not wrong.
2. *False in the judging condition:* bases need absolute corrections 2.43–5.53 both
   channels; correctness-ECE 0.175–0.443 vs DACA's 0.034–0.069, and the design that
   assumed the premise was falsified by its own gate.
3. *Loosely held where we could measure it at matched K:* qwen base under our verbalized
   harness on MMLU = ECE 0.0802 — qualitatively DACA-like (≲0.10), above their entire
   band (max 0.069), and moving 0.08↔0.11 across our own scaffold versions.
4. *Concession:* we never ran the scaffold-free logit slice on MMLU, so we cannot
   confirm or refute their band in their own mode (~$5–10 if ever wanted; sits awkwardly
   with the P3 posture; not needed for v15 — R10 does the argumentative work).

**C3 (direction):** AIReg = post-training *improves* correctness-ECE 6/6 (CI 4/6).
MMLU = one clean fully-v2 pair (qwen): +0.0017, CI includes 0 — null, not degradation;
other families' MMLU E2 unrescuable in-design (their pre legs fail gates regardless of
post re-collection). DACA's 3–7× degradation stands unrefuted in their setting.

## 3. τ_DACA — role clarity (resolves an apparent contradiction in the record)

Two true statements about two different roles:
- **As an adoption instrument: not useful, never adopted.** Failed outright in the logit
  channel (null clause invoked); sits systematically below T_J (0.922/0.988 vs 1.153);
  objective-dependent spread; agreement filter starves at small n (the 15-item R0
  "non-corroboration" was filter starvation, not reference non-exchangeability). The
  supervised/absolute fits carry the estimand.
- **As a GT-free cross-check in the verbalized channel: corroborates**, at full scale,
  under the F7 window test (≥3 valid fits within [τ*/2, 2τ*]) — a deliberately wide bar
  suited to a diagnostic.

**Transfer conditions for the estimator in our use case** (all from the record):
(i) elicitable references in the channel — true on the long-prose task (pre-leg parse
0.92–1.00), false on MMLU for 3/4 bases; the left-edge contrast is the cleanest
demonstration that long-prose scaffolds are where our verbalized references live;
(ii) reference and target sharing an entropy regime (gaps 0.016–0.092 verbalized; the
token-slice failure was genuinely-sharper references breaking the filter);
(iii) full-scale n for the agreement filter;
(iv) scaffold-conditionality inherited from the references — the base reference legs
move under coverage-preserving perturbation (alt_set inflates T_abs(pre) +0.46–0.72 ln
on three families), so the corroboration was scaffold-conditional in an untested way
until the §5 check.

## 4. v15 posture

Keep v14's "none of this impugns DACA's method," and strengthen it two ways: add the
elicitability concession (their channel choice is right for their setting — our own
failed remediation is the evidence), and upgrade the premise critique from
"fails in our setting" to "ill-posed as a model-intrinsic property" per R10. Cite their
Fig. 1 as elicitation-conditional, never as refuted. The DACA-facing sentences in §6.2
and the control appendix should carry exactly this three-way split: method confirmed,
premise ill-posed, channel choice setting-dependent.

## 5. Scaffold-variant τ_DACA check — [PENDING; appended on completion]

Closes transfer-condition (iv): recompute τ_DACA with V1/V2 variant pre legs as
references ($0, data on disk). Pre-specified: survives iff the F7 bar holds per variant;
exploratory-only (F7 was registered for the baseline scaffold). Result doc:
`docs/tau_daca_scaffold_variant_check_20260809.md`.
