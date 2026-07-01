"""Few-shot block loader for the base/post token-slice channel.

Draws ``k`` exemplars per Article criterion from the git-tracked corpus-native
exemplar store (``judex-corpus/leaf_exemplars`` — the ``fewshot_source`` declared in
``configs/models.yaml``) and renders each as an (excerpt, Article criterion,
single-letter A-E answer) triplet for the base-model MCQA scaffold. These are the
same corpus-native exemplars the JUDEX evaluators are given, so the base leg is
framed on the same instrument.

Selection mirrors the evaluator's stratified fixed_set (``judex.exemplars
._stratified_fixed_set``): round-robin over compliance levels in scale order,
picking the least-used rater at each step, so a small ``k`` still spans levels (and
raters). Deterministic; ties break on stored order.

FIREWALL: store exemplars are synthetic JUDEX corpus excerpts (``source_item_label``
like ``article_10_excerpt_0``), disjoint from the 120 AIReg cells (``Art N / Scenario
X | Use Y``) — so few-shot from here never leaks the AIReg validation set. An optional
``exclude`` set adds defense-in-depth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# --- sibling-repo layout (this repo is a sibling checkout of judex-corpus) ---
_CAL_ROOT = Path(__file__).resolve().parents[2]          # .../judex-calibration
_UMBRELLA = _CAL_ROOT.parent                             # .../judex
CORPUS = _UMBRELLA / "judex-corpus"
# The Article-level (dimension) store is keyed by criterion_id (article_9..article_15),
# matching aireg.Cell.criterion_id. The sibling exemplar_store.json is leaf-keyed only.
DIMENSION_STORE = (
    CORPUS / "leaf_exemplars" / "judex_leaf_exemplar_construction"
    / "exemplar_store" / "dimension_exemplar_store.json"
)
MODELS_YAML = _CAL_ROOT / "configs" / "models.yaml"

LEVELS = ("very_low", "low", "moderate", "high", "very_high")
LETTERS = ("A", "B", "C", "D", "E")

_STORE_CACHE: Optional[Dict[str, list]] = None


def _load_store() -> Dict[str, list]:
    global _STORE_CACHE
    if _STORE_CACHE is None:
        data = json.loads(DIMENSION_STORE.read_text())
        _STORE_CACHE = data.get("store", data)
    return _STORE_CACHE


def default_k() -> int:
    """fewshot_k from configs/models.yaml (elicitation.base), default 4."""
    try:
        import yaml
        cfg = yaml.safe_load(MODELS_YAML.read_text())
        return int(cfg["elicitation"]["base"]["fewshot_k"])
    except Exception:
        return 4


def _letter(row: dict) -> str:
    # compliance_1to5 == argmax(probabilities)+1 (verified identical across the store)
    return LETTERS[int(row["compliance_1to5"]) - 1]


def select_rows(criterion_id: str, k: int, exclude: Optional[Sequence[str]] = None) -> List[dict]:
    """Stratified fixed_set over levels x raters, mirroring judex.exemplars."""
    excluded = set(exclude or ())
    rows = [r for r in _load_store().get(criterion_id, []) if r.get("source_item_label") not in excluded]
    by_level: Dict[str, list] = {lv: [] for lv in LEVELS}
    for r in rows:
        by_level.setdefault(r.get("compliance_level"), []).append(r)
    rater_use: Dict[str, int] = {}
    picked: List[dict] = []
    used_ids = set()
    while len(picked) < k:
        progressed = False
        for lv in LEVELS:
            if len(picked) >= k:
                break
            cands = [r for r in by_level.get(lv, []) if r.get("id") not in used_ids]
            if not cands:
                continue
            # least-used rater; ties keep stored order (stable sort)
            cands.sort(key=lambda r: rater_use.get(r.get("rater_model"), 0))
            choice = cands[0]
            picked.append(choice)
            used_ids.add(choice.get("id"))
            rater_use[choice.get("rater_model")] = rater_use.get(choice.get("rater_model"), 0) + 1
            progressed = True
        if not progressed:
            break
    return picked


def render_block(row: dict, criterion_text: str) -> str:
    """One (excerpt, criterion, single-letter answer) few-shot example."""
    justification = (row.get("compliance_justification") or "").strip()
    return (
        f"Example.\nEvidence:\n{row['text']}\n\n"
        f"Criterion:\n{criterion_text}\n\n"
        f"Reasoning: {justification}\n"
        f"Answer: {_letter(row)}\n\n"
    )


def build_fewshot(criterion_id: str, criterion_text: str, k: Optional[int] = None,
                  exclude: Optional[Sequence[str]] = None) -> str:
    """Assemble the k-example few-shot prefix for one Article criterion.

    Examples are ordered very_low -> very_high (A -> E) for a clean ordinal ramp.
    """
    if k is None:
        k = default_k()
    rows = select_rows(criterion_id, k, exclude=exclude)
    rows.sort(key=lambda r: int(r["compliance_1to5"]))
    return "".join(render_block(r, criterion_text) for r in rows)


def build_fewshot_by_criterion(cells, k: Optional[int] = None) -> Dict[str, str]:
    """{criterion_id: few-shot string} for every Article criterion present in ``cells``."""
    text_by_crit: Dict[str, str] = {}
    for c in cells:
        text_by_crit.setdefault(c.criterion_id, c.criterion_text)
    return {cid: build_fewshot(cid, txt, k=k) for cid, txt in text_by_crit.items()}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_CAL_ROOT / "src"))
    from judex_calibration.aireg import load_cells

    cells = load_cells()
    fs = build_fewshot_by_criterion(cells)
    print(f"built few-shot for {len(fs)} Article criteria at k={default_k()}")
    cid = "article_10"
    block = fs[cid]
    letters = [ln.split("Answer:")[1].strip() for ln in block.splitlines() if ln.startswith("Answer:")]
    print(f"  {cid}: {len(letters)} examples, letters (ordinal ramp): {letters}, {len(block)} chars")
