"""Few-shot selection for the defined-answer control scaffold (k=4, one per letter).

The control's exemplar store is the corpus-side artifact
``judex-corpus/leaf_exemplars/mmlu_control_exemplars_v1/`` (design D4, revised): MMLU
dev/validation items for the six control subjects, annotated by the SAME 7-seat rater
panel that produced the AIReg scaffold's exemplars, under an MCQ-adapted contract. The
exemplar distributions are therefore the panel's organic credences — the same generating
process as the AIReg scaffold, nothing invented or sharpness-tuned.

Selection mirrors ``fewshot.select_rows`` semantics exactly, with ONE substitution: the
stratification axis is the answer LETTER (A-D) instead of the compliance LEVEL. That is
the same coverage rule (k >= #categories, one exemplar per category) that made k=5
mandatory on the 5-level ordinal task; here #categories = 4, so k=4 is coverage-complete
— this is NOT a return to the deprecated AIReg k=4, which was defective only because
4 < 5. Round-robin over letters in A->D order, least-used ``rater_model`` at each step,
ties keeping stored order. Deterministic.

FIREWALL: the store draws exclusively from MMLU **dev/validation**; the scored slice is
**test**. Split-disjointness is structural, and the test suite additionally asserts zero
overlap of ``source_item_label`` and of exact question text against the 120 scored items.

``fewshot.py`` is NOT edited by this module — its ``LEVELS``/``LETTERS``/``SCAFFOLD_VARIANTS``
constants stay the AIReg instrument's. The variant names are re-declared here so the
control scaffold carries its own (identically-behaving) machinery.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .mmlu import OPTION_LABELS

# --- sibling-repo layout (this repo is a sibling checkout of judex-corpus) ---
# JUDEX_UMBRELLA overrides the parent-dir default (worktrees live under <umbrella>/worktrees/).
_CAL_ROOT = Path(__file__).resolve().parents[2]          # .../judex-calibration
_UMBRELLA = Path(os.environ.get("JUDEX_UMBRELLA") or _CAL_ROOT.parent)  # .../judex
CORPUS = _UMBRELLA / "judex-corpus"
CONTROL_STORE = (
    CORPUS / "leaf_exemplars" / "mmlu_control_exemplars_v1"
    / "exemplar_store" / "mmlu_control_exemplar_store.json"
)
# Pin of the control exemplar store (237 rows, keyed by subject). Verified on load and
# recorded in every leg's meta sidecar, exactly like the slice pin: a scaffold that moved
# between the pre and post leg of a pair would confound the paired ECE delta (E2).
STORE_SHA256 = "c4174d4587618c2aa54a71620fbcb4d2514db611d586fa093ad0ee039c3fa3f2"

# k=4 is the CONTROL protocol constant (one exemplar per option letter). It is NOT read
# from configs/models.yaml — that yaml pins the AIReg instrument's k=5, and a control leg
# must never inherit it (5 exemplars over 4 letters breaks one-per-category balance).
CONTROL_FEWSHOT_K = 4

SCAFFOLD_VARIANTS = ("baseline", "alt_set", "rev_order")

_STORE_CACHE: Optional[Dict[str, list]] = None


def store_sha256(path: Optional[Path] = None) -> str:
    return hashlib.sha256((path or CONTROL_STORE).read_bytes()).hexdigest()


def verify_store_sha256(path: Optional[Path] = None, expected: str = STORE_SHA256) -> str:
    p = path or CONTROL_STORE
    got = store_sha256(p)
    if got != expected:
        raise RuntimeError(
            f"{p} sha256 {got} != pinned {expected} — the control exemplar store moved; "
            f"a scaffold that changes between the pre and post leg of a pair confounds "
            f"the paired ECE delta. Re-pin deliberately, never silently")
    return got


def load_store(verify: bool = True) -> Dict[str, list]:
    """{subject: [row, ...]} from the pinned control store (cached)."""
    global _STORE_CACHE
    if _STORE_CACHE is None:
        if verify:
            verify_store_sha256()
        data = json.loads(CONTROL_STORE.read_text())
        _STORE_CACHE = data.get("store", data)
    return _STORE_CACHE


def default_k() -> int:
    """The control protocol's k — a constant, deliberately not a yaml read."""
    return CONTROL_FEWSHOT_K


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_rows(subject: str, k: int, exclude: Optional[Sequence[str]] = None) -> List[dict]:
    """Stratified fixed_set over letters x raters — ``fewshot.select_rows`` semantics."""
    excluded = set(exclude or ())
    rows = [r for r in load_store().get(subject, [])
            if r.get("source_item_label") not in excluded]
    by_letter: Dict[str, list] = {L: [] for L in OPTION_LABELS}
    for r in rows:
        by_letter.setdefault(r.get("answer_letter"), []).append(r)
    rater_use: Dict[str, int] = {}
    picked: List[dict] = []
    used_ids = set()
    while len(picked) < k:
        progressed = False
        for L in OPTION_LABELS:
            if len(picked) >= k:
                break
            cands = [r for r in by_letter.get(L, []) if r.get("id") not in used_ids]
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


def scaffold_rows(subject: str, k: int, variant: str = "baseline",
                  exclude: Optional[Sequence[str]] = None) -> List[dict]:
    """The SELECTION stage of a control scaffold variant (render order applied separately).

    ``baseline``/``rev_order`` return the shipped draw verbatim; ``alt_set`` re-walks the
    selector with the baseline draw's SOURCE ITEMS excluded, giving a fully disjoint
    alternate one-per-letter set. The coverage + disjointness guards are FAIL-LOUD, for
    the same reason they are on the AIReg scaffold: a variant that silently loses a
    letter is not a scaffold perturbation, it is a broken scaffold, and would confound
    the sensitivity read with a coverage hole.

    NOTE (measured 2026-08-09, ``scripts/control_scaffold_probe.py``): the control store
    holds only ONE source item for at least one letter in every subject (it was built
    from 6 candidate items per subject), so ``alt_set`` currently raises for all six
    subjects. That is the guard working, not a bug: the optional robustness check needs a
    deeper store (more annotated dev/validation candidates) before it can run.
    """
    if variant not in SCAFFOLD_VARIANTS:
        raise ValueError(f"unknown scaffold variant {variant!r}; expected one of {SCAFFOLD_VARIANTS}")
    rows = select_rows(subject, k, exclude=exclude)
    if variant != "alt_set":
        return rows
    base_ids = {r.get("id") for r in rows}
    excluded = sorted(set(exclude or ()) | {r.get("source_item_label") for r in rows})
    alt = select_rows(subject, k, exclude=excluded)
    if len(alt) != len(rows):
        raise RuntimeError(
            f"alt_set for {subject!r} drew {len(alt)} of {len(rows)} exemplars — a "
            f"(subject, letter) bucket starved under source-label exclusion; the variant "
            f"would not be coverage-preserving")
    if {r.get("answer_letter") for r in alt} != {r.get("answer_letter") for r in rows}:
        raise RuntimeError(
            f"alt_set for {subject!r} does not cover the same option letters as the "
            f"baseline draw — not coverage-preserving")
    overlap = base_ids & {r.get("id") for r in alt}
    if overlap:
        raise RuntimeError(f"alt_set for {subject!r} overlaps the baseline draw: {sorted(overlap)}")
    return alt


def order_rows(rows: Sequence[dict], variant: str = "baseline") -> List[dict]:
    """Render order: the letter ramp A -> D, reversed for ``rev_order``."""
    ordered = sorted(rows, key=lambda r: int(r["answer_1to4"]))
    if variant == "rev_order":
        ordered.reverse()
    return ordered


def alt_set_feasibility(subject: str, k: int = CONTROL_FEWSHOT_K) -> dict:
    """Non-raising report of whether ``alt_set`` can be drawn for one subject."""
    base = select_rows(subject, k)
    base_src = {r["source_item_label"] for r in base}
    left = [r for r in load_store().get(subject, []) if r["source_item_label"] not in base_src]
    left_letters = {r["answer_letter"] for r in left}
    starved = [L for L in OPTION_LABELS if L not in left_letters]
    try:
        alt = scaffold_rows(subject, k, "alt_set")
        ok, err = True, None
    except RuntimeError as e:
        alt, ok, err = [], False, str(e)
    return {"subject": subject, "baseline_ids": [r["id"] for r in base],
            "baseline_source_items": sorted(base_src),
            "rows_after_source_exclusion": len(left), "starved_letters": starved,
            "alt_set_feasible": ok, "alt_set_ids": [r["id"] for r in alt], "error": err}
