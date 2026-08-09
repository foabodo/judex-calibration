"""Defined-answer control: the vendored MMLU slice as elicitation-ready items.

Phase 2 of the defined-answer control (design:
``docs/phase2_defined_answer_control_design.md``, D1/D2/D7). The control runs the
adoption panel's base/post pairs through the SAME verbalized harness on a benchmark
whose answers are defined, so the AIReg base-miscalibration finding can be read as
task/reference/scaffold-specific or as generic-under-this-harness.

The slice is **vendored** (``data/mmlu_control_slice_v1.json``, 120 items = 6 subjects
x 20) and pinned by sha256 — stdlib-only doctrine: no ``datasets`` dependency anywhere
in the elicitation path, and no silent drift of the scored set between legs. The sha is
verified on every load and re-pinned into each leg's meta sidecar.

Items are duck-typed for the elicitation machinery (``aireg.Cell``'s protocol):

    item_label      "mmlu:<subject>:<split>:<NNNN>"   (== item_id)
    evidence_text   the question stem
    criterion_text  the rendered A-D option list
    criterion_id    the subject (so the few-shot cache builds ONE block per subject,
                    D2: 6 blocks x k=4 = 24 exemplars, not 228 across all 57 subjects)

plus the scoring side:

    gt_labels       ("A", "B", "C", "D")
    gt_argmax       0..3, the answer key index
    gt_probs        the one-hot key distribution
    document_id     D7's clustering unit — the ITEM by default (MMLU items are
                    independent draws, so the item-level bootstrap is primary), the
                    SUBJECT under ``document_id="subject"`` for the clustered
                    sensitivity read (6 clusters is too few for a primary CI).

Nothing here is ordinal: A-D are nominal options. No RPS, no W1, no Murphy is ever
computed on these items (see ``score_control``, which enforces that structurally).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_CAL_ROOT = Path(__file__).resolve().parents[2]          # .../judex-calibration
SLICE_PATH = _CAL_ROOT / "data" / "mmlu_control_slice_v1.json"

# Pin of the vendored scored slice (120 items). Verified on every load; a mismatch is a
# hard error, never a warning — a moved slice silently invalidates every leg elicited
# against the old one, and the sidecar guard can only catch it if the value is real.
SLICE_SHA256 = "31822879ab5e960557ecf2a1f4fb7ccc56a6110458d577a8d3cf788fedf88c7e"

OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D")
SUBJECTS: Tuple[str, ...] = ("clinical_knowledge", "econometrics", "formal_logic",
                             "high_school_mathematics", "philosophy", "professional_law")
DOCUMENT_ID_MODES = ("item", "subject")


# ---------------------------------------------------------------------------
# Rendering (shared with the control exemplar store's own ``text`` field)
# ---------------------------------------------------------------------------

def render_options(choices: Sequence[str]) -> str:
    """The A-D option list, one per line — the control scaffold's ``Criterion:`` body."""
    if len(choices) != len(OPTION_LABELS):
        raise ValueError(f"expected {len(OPTION_LABELS)} choices, got {len(choices)}")
    return "\n".join(f"{L}. {c}" for L, c in zip(OPTION_LABELS, choices))


def render_item_text(question: str, choices: Sequence[str]) -> str:
    """question + blank line + options — byte-identical to the store rows' ``text``.

    Asserted against all 237 control-store rows in the test suite, which is what makes
    the store's ``text`` and a live item's (evidence, criterion) pair the same object
    seen through two seams.
    """
    return f"{question.strip()}\n\n{render_options(choices)}"


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MMLUItem:
    """One scored control item — duck-typed for the verbalized elicitation harness."""
    item_id: str
    item_label: str
    document_id: str
    subject: str
    split: str
    index: int
    question: str
    choices: Tuple[str, ...]
    answer_index: int
    answer_letter: str
    criterion_id: str
    criterion_text: str
    evidence_text: str
    gt_labels: Tuple[str, ...]
    gt_argmax: int
    gt_probs: Tuple[float, ...]

    @property
    def text(self) -> str:
        """The store-shaped single-blob rendering (question + options)."""
        return render_item_text(self.question, self.choices)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slice_sha256(path: Optional[Path] = None) -> str:
    return _sha256(path or SLICE_PATH)


def verify_slice_sha256(path: Optional[Path] = None, expected: str = SLICE_SHA256) -> str:
    """Hard-error unless the vendored slice hashes to the pin. Returns the digest."""
    p = path or SLICE_PATH
    got = _sha256(p)
    if got != expected:
        raise RuntimeError(
            f"{p} sha256 {got} != pinned {expected} — the vendored control slice moved; "
            f"legs elicited against the old slice are not comparable to legs against the "
            f"new one. Re-pin deliberately (and re-run every leg), never silently")
    return got


def load_slice(path: Optional[Path] = None, verify: bool = True) -> dict:
    p = path or SLICE_PATH
    if verify:
        verify_slice_sha256(p)
    return json.loads(p.read_text())


def load_control_items(path: Optional[Path] = None, *, document_id: str = "item",
                       verify: bool = True) -> List[MMLUItem]:
    """The 120 vendored control items, in slice order.

    ``document_id`` picks D7's clustering unit: ``"item"`` (default, the primary
    i.i.d. bootstrap) or ``"subject"`` (the 6-cluster sensitivity read).
    """
    if document_id not in DOCUMENT_ID_MODES:
        raise ValueError(f"unknown document_id mode {document_id!r}; "
                         f"expected one of {DOCUMENT_ID_MODES}")
    data = load_slice(path, verify=verify)
    items: List[MMLUItem] = []
    for raw in data["items"]:
        choices = tuple(raw["choices"])
        ai = int(raw["answer_index"])
        letter = raw["answer_letter"]
        if OPTION_LABELS[ai] != letter:
            raise RuntimeError(f"{raw['item_id']}: answer_index {ai} != letter {letter!r}")
        subject = raw["subject"]
        items.append(MMLUItem(
            item_id=raw["item_id"], item_label=raw["item_id"],
            document_id=(raw["item_id"] if document_id == "item" else subject),
            subject=subject, split=raw["split"], index=int(raw["index"]),
            question=raw["question"].strip(), choices=choices,
            answer_index=ai, answer_letter=letter,
            criterion_id=subject, criterion_text=render_options(choices),
            evidence_text=raw["question"].strip(),
            gt_labels=OPTION_LABELS, gt_argmax=ai,
            gt_probs=tuple(1.0 if i == ai else 0.0 for i in range(len(OPTION_LABELS))),
        ))
    return items


def criteria_texts(items: Sequence[MMLUItem]) -> Dict[str, str]:
    """{subject: criterion_text of its first item} — the per-subject few-shot cache key.

    NOTE the control's criterion_text is item-specific (it IS the option list), so the
    few-shot block is built per SUBJECT from the store rows' own options, not from this
    map; the map exists only where the AIReg-shaped API expects one.
    """
    out: Dict[str, str] = {}
    for it in items:
        out.setdefault(it.criterion_id, it.criterion_text)
    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke
    its = load_control_items()
    print(f"{len(its)} control items, sha256 {slice_sha256()}")
    from collections import Counter
    print("per subject:", dict(Counter(i.subject for i in its)))
    print("key letters:", dict(Counter(i.answer_letter for i in its)))
