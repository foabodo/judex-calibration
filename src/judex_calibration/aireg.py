"""AIReg-Bench cell loader for Study A — canonical GT, reproducible on a fresh clone.

Yields the 120 evaluation cells (24 examinees x 5 Articles) keyed by ``item_label``,
each with: the CURRENT canonical human-reconciled GT distribution, the Article
criterion text, and the evidence text. Everything is sourced from **git-tracked**
sibling-repo artifacts — no dependency on a gitignored run directory:

  GT distribution : judex-ground-truth/data/distributional_labels/  (the canonical
                    ``mgmfrm_anchored_projection`` bundle — cumulative-consistency
                    model ``cumulative_consistency_fixed_sigma_item_mgmfrm.v1``, loaded
                    through judex-evaluator's manifest-verified
                    ``synthesize_aireg_bench_ground_truth`` so the SHA-256 contract is
                    enforced and we always pick up the current canonical labels across
                    future re-pins).  See WHAT THE GT ACTUALLY IS below — the bundle was
                    re-materialized on 2026-07-09 and its formulation is load-bearing here.
  item_label->doc : judex-corpus/step3_4/ground_truth.json  (``labelled_sections``).
  criterion text  : judex-evaluator/configs/rubric.yaml.
  evidence text   : judex-corpus/step5/documents/<doc_id>/{application.md,data.md}.

Why not read a run's ``metrics_report.json`` (the previous approach): ``runs/`` is
gitignored (absent on a fresh clone / the remote box), and a run embeds the GT that
was canonical *at run time* — the pre-2026-06-30 hyperprior GT, whose 120 probability
vectors differ from the current cumulative-consistency GT (argmax is identical, so the
staleness is silent). A distribution-fitting calibration study must use the current
canonical GT; hence we synthesize it from the tracked bundle every load.

WHAT THE GT ACTUALLY IS (re-verified 2026-07-24 against judex-ground-truth ``5ee7786``;
every figure below was recomputed from the artifacts for this pass, not carried over).
The reference is the AIReg-Bench human panel reconciled by the cumulative-consistency
MG-MFRM over **three rating SLOTS per cell** (``rater_id`` 0/1/2; the fit's design is
``n_raters = 3``). [Corrected 2026-07-25: the slots are filled from a pool of SIX human
annotators under a use-case block design — one fixed triple per use case — per
``judex-corpus/aireg-bench/human_annotations_disaggregated.xlsx``; the fitted rater facets
describe slots, not people, and the reference is RECONSTRUCTED from categorical votes, not
elicited.] It is **not** the 7-seat LLM exemplar panel in judex-corpus — that
panel is few-shot material only and is firewall-disjoint from this reference.

The bundle was **re-materialized on 2026-07-09** (``89f40b7`` + ``3c2ebdb``) — after this
module's original 2026-07-03 verification — with three coupled formulation changes:

  * thresholds **pooled -> freethresh** (``thresholds: "free"``). Verifiable structurally from
    the trace: the posterior carries ``thresholds_r_loc`` / ``thresholds_r_log_gap`` (the
    freethresh-only latents) and none of the pooled-only ``threshold_location`` /
    ``global_log_spacing`` / ``sigma_threshold_shape`` / ``z_threshold_spacing``;
  * a **readout temperature tau = 0.675** (1 -> 0.65 -> 0.675, re-pinned by user decision),
    tempering P(Y<=k) on the READOUT ONLY — the fit, severities, theta and lens geometry are
    unchanged, and every identified threshold functional is exactly tau-invariant;
  * the 0.05 grid snap **turned off globally** — the labels are now **continuous** (verified:
    0/600 barycenter and 0/1800 per-rater probabilities land on the 0.05 grid at tol 1e-6,
    while the archived pre-07-09 barycenter is 600/600 on-grid), because the object they are
    scored against (the evaluator's Phase-4 reconciled barycenter) is continuous and never
    re-snapped.

Movement vs the previously shipped object — ``variants/airegbench_raw_pooled_snapped_2026_07_09/``,
byte-identical to the blob at ``5716e36`` (the 2026-07-03 bundle), so it really is the prior
canonical: **W1 mean 0.178** (median 0.176, max 0.414), mean entropy **1.065 -> 1.237 nats**
(dH +0.172), and **0/120 mode flips** (a KL-anchor construction guarantee).  W1 here is the
project's ``label_variants.w1_ordinal``: L1 distance between ordinal CDFs on the equally
spaced 1..5 support (POT and the closed form agree to <1e-9).  So this is precisely the silent
staleness the paragraph above warns about, a second time and **~210x** larger than the
2026-07-03 refresh (``d4d2baa -> 5716e36``, W1 mean 0.00083, max 0.05 — one grid step on a
single cell): argmax-derived numbers are stable — ``phase0_accuracy_precheck`` reads only the
barycenter argmax and mode flips are 0, so its proxy ceiling 0.658 is invariant by construction
— but **every distribution-fitting number moved**; any T*/tau_oc figure recorded before
2026-07-09 must be re-derived, not carried forward.
  CAUTION: the movement figures previously recorded here (W1 mean 0.194 / median 0.184 /
  max 0.417, entropy -> 1.252 nats, ~240x) are the **tau = 0.65** intermediate ``89f40b7``,
  reproduced exactly against that blob; they were never recomputed after the tau = 0.675
  re-pin.  The archived comparand's own ``PROVENANCE.txt`` carries the same lag — its
  "superseded-by" line still says ``tau=0.65``.  Distrust both.

Reference properties of the shipped bundle, recomputed and matching the canonical manifest's
own ``invariants``: mean max-prob 0.4607, mean normalized entropy 0.7686, 0/120 cells above
0.9, argmax majority-class share 0.3333 (= the A2 argmax floor).

Two consequences worth stating explicitly, because Study A's estimands are temperatures:
  1. The GT we fit against **is itself a tempered readout** (tau = 0.675). Study A's T* and
     tau_oc are measured against that object; they are not comparable to temperatures fit
     against the pre-2026-07-09 (tau = 1, snapped, pooled) labels.
  2. Corpus few-shot exemplars ARE on the 0.05 grid (elicitation rule A7; verified 1260/1260
     leaf+dimension barycenter probabilities on-grid) while this GT is continuous. Harmless
     for the base leg — ``fewshot.render_block`` emits only a letter A-E — but the two sides
     of the instrument are on different supports; do not assume otherwise.

Sampler, from the trace's own attrs (``airegbench_mgmfrm_cumulative_consistency_idata.nc``,
authoritative): **draws 2000 / tune 4000** (the ``posterior`` draw dimension, and the
``tuning_steps`` entry of ``posterior.attrs``), 4 chains, nutpie 0.16.8, 0 divergences —
compliant with the 2026-07-14 convention (raise *tune* to remediate convergence; draws are a
precision knob capped at 2000).
``target_accept`` and ``seed`` are **NOT recorded in the trace** (its attrs carry only
created_at / arviz_version / inference_library{,_version} / sampling_time / tuning_steps); the
0.99 and 42 that used to be asserted here come from the sidecar, which is exactly the file the
next paragraph says not to trust for sampler settings — treat them as unverified.  FOOTGUN: the
sidecar ``validation_diagnostics_v4.json`` records ``draws 1000 / tune 2000``; those are
``build_airegbench_canonical_sources.py`` argparse defaults stamped over the cached-trace
path, not what was run. Trust the trace attrs.

Validation status is ``unavailable`` with **two** hard-failure families, not one:
``convergence_rhat`` (max R-hat 1.01229 > 1.01 threshold — a live, shipped property, not a
resolved footnote) and ``prior_predictive`` (not computed). ESS passes (bulk min 557.9, tail
min 579.3, both >= 400). Safe to *use*; must never be described as a validated benchmark.

EVIDENCE SOURCE CHOICE: this uses the JUDEX TechOps document (what the JUDEX evaluators
saw), assembled from the git-tracked corpus step5 markdown, not the raw isolated AIReg
excerpt the human GT raters scored. That keeps all annotators on the same, already-joined
evidence. (~1% char difference vs a run's assembled ``document.json`` — assembled headers
only; for the exact render use ``judex-corpus/step8/pdfs/<doc_id>.pdf``.) Documented so
the choice is explicit and reversible.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# --- sibling-repo layout (this repo is a sibling checkout of the other three) ---
# JUDEX_UMBRELLA overrides the parent-dir default so a git worktree of this repo (which
# lives under <umbrella>/worktrees/, not the umbrella root) still finds the siblings.
_CAL_ROOT = Path(__file__).resolve().parents[2]          # .../judex-calibration
_UMBRELLA = Path(os.environ.get("JUDEX_UMBRELLA") or _CAL_ROOT.parent)  # .../judex
EVAL = _UMBRELLA / "judex-evaluator"
GT_LABELS_DIR = _UMBRELLA / "judex-ground-truth" / "data" / "distributional_labels"
CORPUS = _UMBRELLA / "judex-corpus"
RUBRIC = EVAL / "configs" / "rubric.yaml"
JOIN = CORPUS / "step3_4" / "ground_truth.json"
DOCS = CORPUS / "step5" / "documents"

# Fixed ordinal scale (positional very_low..very_high) shared with elicit_base/post.
LABELS: Tuple[str, ...] = ("very_low", "low", "moderate", "high", "very_high")

# judex-evaluator provides the manifest-verified canonical GT synthesizer. Prefer an
# editable install (`pip install -e ../judex-evaluator`); fall back to the sibling src.
_EVAL_SRC = str(EVAL / "src")
if _EVAL_SRC not in sys.path:
    sys.path.insert(0, _EVAL_SRC)
from judex.ground_truth import synthesize_aireg_bench_ground_truth  # noqa: E402


@dataclass(frozen=True)
class Cell:
    item_id: str
    item_label: str
    document_id: str
    article: str
    criterion_id: str
    criterion_text: str
    evidence_text: str
    gt_probs: tuple          # current canonical reconciled GT distribution (very_low..very_high)
    gt_labels: tuple
    gt_argmax: int           # 0..4


def _load_rubric_text() -> Dict[str, str]:
    """criterion_id -> 'Article title: sub-criterion titles' from rubric.yaml."""
    txt = RUBRIC.read_text()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        import yaml  # configs are JSON-compatible YAML
        data = yaml.safe_load(txt)
    out: Dict[str, str] = {}

    def walk(node):
        if isinstance(node, dict):
            cid, title = node.get("id"), node.get("title")
            if isinstance(cid, str) and isinstance(title, str):
                kids = node.get("children") or node.get("sub_criteria") or node.get("subcriteria") or []
                sub = [k.get("title") for k in kids if isinstance(k, dict) and k.get("title")]
                out[cid] = title + ((": " + "; ".join(sub)) if sub else "")
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return out


def _item_label_to_doc() -> Dict[str, dict]:
    """{item_label: {doc_id, article}} from the git-tracked corpus join.

    Reconstructs the exact item_label -> document_id join the JUDEX runs used
    (verified 0 mismatches vs runs/stage9-gemini-gpt-medium). We do NOT parse the
    document from item_label: 'Scenario A/B/C' does not discriminate the three docs
    within a Use, so the explicit corpus join is authoritative.
    """
    data = json.loads(JOIN.read_text())
    out: Dict[str, dict] = {}
    for doc in data.get("documents", []):
        for section in doc.get("labelled_sections", []):
            label = section.get("item_label")
            if label:
                out[label] = {"doc_id": section["doc_id"], "article": section.get("article")}
    return out


def _evidence_text(doc_id: str) -> str:
    """Assemble the TechOps evidence for a document from git-tracked corpus step5 markdown."""
    doc_dir = DOCS / doc_id
    parts: List[str] = []
    for name in ("application.md", "data.md"):
        path = doc_dir / name
        if path.exists():
            parts.append(path.read_text())
    return "\n\n".join(parts)


def load_cells() -> List[Cell]:
    """Build the 120 AIReg cells against the CURRENT canonical GT (manifest-verified)."""
    synthesis = synthesize_aireg_bench_ground_truth(GT_LABELS_DIR, LABELS)
    rubric = _load_rubric_text()
    join = _item_label_to_doc()
    ev_cache: Dict[str, str] = {}
    cells: List[Cell] = []
    for item in synthesis.items:
        label = item.item_label
        j = join.get(label)
        if j is None:
            raise KeyError(f"No corpus item_label->doc join for canonical GT cell {label!r}.")
        doc_id = j["doc_id"]
        article = str(j.get("article", ""))
        criterion_id = f"article_{article}"
        if doc_id not in ev_cache:
            ev_cache[doc_id] = _evidence_text(doc_id)
        gt = item.ground_truth.to_dict()
        probs = tuple(gt["probabilities"])
        labels = tuple(gt["labels"])
        argmax = max(range(len(probs)), key=lambda i: probs[i])
        cells.append(Cell(
            item_id=item.item_id, item_label=label, document_id=doc_id,
            article=article, criterion_id=criterion_id,
            criterion_text=rubric.get(criterion_id, criterion_id),
            evidence_text=ev_cache[doc_id],
            gt_probs=probs, gt_labels=labels, gt_argmax=argmax,
        ))
    return cells


if __name__ == "__main__":
    cells = load_cells()
    print(f"loaded {len(cells)} cells (canonical mgmfrm_anchored_projection GT, manifest-verified)")
    c = cells[0]
    print("sample:", c.item_label, "| article", c.article, "| criterion:", c.criterion_text[:70])
    print("  evidence chars:", len(c.evidence_text), "| gt_argmax:", c.gt_labels[c.gt_argmax],
          "| gt:", [round(p, 2) for p in c.gt_probs])
