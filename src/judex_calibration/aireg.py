"""AIReg-Bench cell loader for Study A — canonical GT, reproducible on a fresh clone.

Yields the 120 evaluation cells (24 examinees x 5 Articles) keyed by ``item_label``,
each with: the CURRENT canonical human-reconciled GT distribution, the Article
criterion text, and the evidence text. Everything is sourced from **git-tracked**
sibling-repo artifacts — no dependency on a gitignored run directory:

  GT distribution : judex-ground-truth/data/distributional_labels/  (the canonical
                    ``mgmfrm_anchored_projection`` bundle — cumulative-consistency
                    fit, last re-fit 2026-07-03 at 4000 draws/8000 tune (nutpie; the
                    1000/2000 settings failed the R-hat gate) — loaded through
                    judex-evaluator's manifest-verified
                    ``synthesize_aireg_bench_ground_truth`` so the SHA-256 contract is
                    enforced and we always pick up the current canonical labels across
                    future re-pins).
  item_label->doc : judex-corpus/step3_4/ground_truth.json  (``labelled_sections``).
  criterion text  : judex-evaluator/configs/rubric.yaml.
  evidence text   : judex-corpus/step5/documents/<doc_id>/{application.md,data.md}.

Why not read a run's ``metrics_report.json`` (the previous approach): ``runs/`` is
gitignored (absent on a fresh clone / the remote box), and a run embeds the GT that
was canonical *at run time* — the pre-2026-06-30 hyperprior GT, whose 120 probability
vectors differ from the current cumulative-consistency GT (argmax is identical, so the
staleness is silent). A distribution-fitting calibration study must use the current
canonical GT; hence we synthesize it from the tracked bundle every load.

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
