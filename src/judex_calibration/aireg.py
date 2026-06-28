"""AIReg-Bench cell loader for Study A.

Yields the 120 evaluation cells (24 examinees x 5 Articles) keyed by ``item_label``,
each with: the human-reconciled GT distribution, the Article criterion text, and the
evidence text. The AIReg->item_label join is already solved upstream, so we read it
back from a JUDEX run's ``metrics_report.json`` (GT + document_id/criterion_id) plus
the per-document ``document.json`` (evidence) and the evaluator ``rubric.yaml``
(criterion text).

EVIDENCE SOURCE CHOICE: this uses the JUDEX TechOps document (what the JUDEX
evaluators saw), not the raw isolated AIReg excerpt the human GT raters scored.
That keeps all annotators on the same, already-joined evidence; swapping to raw
excerpts would require the corpus step5 (ArticleN/UseN/SystemM) join. Documented
so the choice is explicit and reversible.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List

EVAL = Path("/Users/fabodo/Downloads/Projects/judex/judex-evaluator")
DEFAULT_RUN = EVAL / "runs" / "stage9-gemini-gpt-medium"
RUBRIC = EVAL / "configs" / "rubric.yaml"
LABELS = ("very_low", "low", "moderate", "high", "very_high")


@dataclass(frozen=True)
class Cell:
    item_id: str
    item_label: str
    document_id: str
    article: str
    criterion_id: str
    criterion_text: str
    evidence_text: str
    gt_probs: tuple          # reconciled human GT distribution (very_low..very_high)
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


def _evidence_text(run: Path, document_id: str) -> str:
    doc = json.loads((run / "documents" / document_id / "document.json").read_text())
    return "\n".join(p.get("text", "") for p in doc.get("pages", []))


def load_cells(run: Path = DEFAULT_RUN) -> List[Cell]:
    metrics = json.loads((run / "metrics_report.json").read_text())
    rubric = _load_rubric_text()
    ev_cache: Dict[str, str] = {}
    cells: List[Cell] = []
    for it in metrics["items"]:
        md = it["metadata"]
        doc_id = md["document_id"]
        if doc_id not in ev_cache:
            ev_cache[doc_id] = _evidence_text(run, doc_id)
        cid = md.get("criterion_id", "")
        gt = it["ground_truth"]
        cells.append(Cell(
            item_id=it["item_id"], item_label=it["item_label"], document_id=doc_id,
            article=str(md.get("article", "")), criterion_id=cid,
            criterion_text=rubric.get(cid, cid),
            evidence_text=ev_cache[doc_id],
            gt_probs=tuple(gt["probabilities"]), gt_labels=tuple(gt["labels"]),
            gt_argmax=it["ground_truth_argmax_index"],
        ))
    return cells


if __name__ == "__main__":
    cells = load_cells()
    print(f"loaded {len(cells)} cells")
    c = cells[0]
    print("sample:", c.item_label, "| article", c.article, "| criterion:", c.criterion_text[:70])
    print("  evidence chars:", len(c.evidence_text), "| gt_argmax:", c.gt_labels[c.gt_argmax],
          "| gt:", [round(p, 2) for p in c.gt_probs])
