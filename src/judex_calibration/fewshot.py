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
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# --- sibling-repo layout (this repo is a sibling checkout of judex-corpus) ---
# JUDEX_UMBRELLA overrides the parent-dir default (worktrees live under <umbrella>/worktrees/).
_CAL_ROOT = Path(__file__).resolve().parents[2]          # .../judex-calibration
_UMBRELLA = Path(os.environ.get("JUDEX_UMBRELLA") or _CAL_ROOT.parent)  # .../judex
CORPUS = _UMBRELLA / "judex-corpus"
# The Article-level (dimension) store is keyed by criterion_id (article_9..article_15),
# matching aireg.Cell.criterion_id. The sibling exemplar_store.json is leaf-keyed only.
# Corpus v2 (2026-07-13) is the adopted vintage: the same 44 Article-level excerpts
# re-authored in the ambiguity_structured style and re-annotated by the same 7-seat
# panel; the v1 tree (judex_leaf_exemplar_construction/) is the archived baseline.
DIMENSION_STORE = (
    CORPUS / "leaf_exemplars" / "judex_leaf_exemplar_construction_v2"
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
    """fewshot_k from configs/models.yaml (elicitation.base), default 5.

    The fallback is 5, not 4: k=5 is the adopted protocol (models.yaml raised it
    2026-07-19) because one exemplar per compliance level is what makes the scaffold
    coverage-complete — k=4's stratified round-robin terminates one step short of
    ``very_high``. A yaml read failure must not silently regress a live leg to the
    deprecated four-level scaffold, so the fallback matches the pin.
    """
    try:
        import yaml
        cfg = yaml.safe_load(MODELS_YAML.read_text())
        return int(cfg["elicitation"]["base"]["fewshot_k"])
    except Exception:
        return 5


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


# ---------------------------------------------------------------------------
# Scaffold-sensitivity variants (Phase 1a, plan_2026_08_08_verbalized_arm_gap_closure §1g)
# ---------------------------------------------------------------------------
# The instrument in use is k=5, one exemplar per compliance level. Bounding the
# scaffold confound means perturbing THAT instrument while keeping it
# coverage-complete — never re-running the deprecated k=4 (E-hole) scaffold.
#
#   baseline  — the shipped selection + very_low -> very_high rendering (unchanged).
#   alt_set   — same policy, same k, but the baseline draw's source excerpts are
#               excluded, giving a fully disjoint alternate one-per-level set.
#               Exclusion is by ``source_item_label`` (the existing ``exclude``
#               semantics), which is STRICTER than excluding row ids: it swaps the
#               excerpt texts, not merely the rater annotations of the same excerpt.
#               Feasibility probed 2026-08-08 over all five criteria the 120 AIReg
#               cells use — no (criterion, level) bucket starves and every alternate
#               draw is row-disjoint from its baseline (the tightest case,
#               article_14 very_low with 2 stored rows, retains 1).
#   rev_order — the SAME exemplar rows as baseline, rendered very_high -> very_low.
#               A pure presentation permutation: selection is untouched.
SCAFFOLD_VARIANTS = ("baseline", "alt_set", "rev_order")


def scaffold_rows(criterion_id: str, k: int, variant: str = "baseline",
                  exclude: Optional[Sequence[str]] = None) -> List[dict]:
    """The SELECTION stage of a scaffold variant (rendering order applied separately).

    ``baseline``/``rev_order`` return the shipped draw verbatim; ``alt_set`` returns a
    fully disjoint alternate draw. Raises if the alternate draw would lose coverage or
    overlap the baseline — a scaffold-sensitivity variant that is not coverage-complete
    would confound the perturbation with the k=4 E-hole and must never run silently.
    """
    if variant not in SCAFFOLD_VARIANTS:
        raise ValueError(f"unknown scaffold variant {variant!r}; expected one of {SCAFFOLD_VARIANTS}")
    rows = select_rows(criterion_id, k, exclude=exclude)
    if variant != "alt_set":
        return rows
    base_ids = {r.get("id") for r in rows}
    excluded = sorted(set(exclude or ()) | {r.get("source_item_label") for r in rows})
    alt = select_rows(criterion_id, k, exclude=excluded)
    if len(alt) != len(rows):
        raise RuntimeError(
            f"alt_set for {criterion_id!r} drew {len(alt)} of {len(rows)} exemplars — a "
            f"(criterion, level) bucket starved under source-label exclusion; the variant "
            f"would not be coverage-preserving")
    overlap = base_ids & {r.get("id") for r in alt}
    if overlap:
        raise RuntimeError(f"alt_set for {criterion_id!r} overlaps the baseline draw: {sorted(overlap)}")
    return alt


def order_rows(rows: Sequence[dict], variant: str = "baseline") -> List[dict]:
    """Render order: the ordinal ramp very_low -> very_high, reversed for ``rev_order``."""
    ordered = sorted(rows, key=lambda r: int(r["compliance_1to5"]))
    if variant == "rev_order":
        ordered.reverse()
    return ordered


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
                  exclude: Optional[Sequence[str]] = None,
                  variant: str = "baseline") -> str:
    """Assemble the k-example few-shot prefix for one Article criterion.

    Examples are ordered very_low -> very_high (A -> E) for a clean ordinal ramp
    (reversed under the ``rev_order`` scaffold variant).
    """
    if k is None:
        k = default_k()
    rows = order_rows(scaffold_rows(criterion_id, k, variant, exclude), variant)
    return "".join(render_block(r, criterion_text) for r in rows)


def build_fewshot_by_criterion(cells, k: Optional[int] = None,
                               variant: str = "baseline") -> Dict[str, str]:
    """{criterion_id: few-shot string} for every Article criterion present in ``cells``."""
    text_by_crit: Dict[str, str] = {}
    for c in cells:
        text_by_crit.setdefault(c.criterion_id, c.criterion_text)
    return {cid: build_fewshot(cid, txt, k=k, variant=variant) for cid, txt in text_by_crit.items()}


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
