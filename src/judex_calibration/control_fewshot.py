"""Few-shot selection for the defined-answer control scaffold (k=4, one per letter).

The control's exemplar store is the corpus-side artifact
``judex-corpus/leaf_exemplars/mmlu_control_exemplars_v2/`` (design D4, revised): MMLU
dev/validation items for the six control subjects, annotated by the SAME 7-seat rater
panel that produced the AIReg scaffold's exemplars, under an MCQ-adapted contract. The
exemplar distributions are therefore the panel's organic credences — the same generating
process as the AIReg scaffold, nothing invented or sharpness-tuned.

Selection mirrors ``fewshot.select_rows`` semantics exactly, with TWO substitutions:

  1. the stratification axis is the answer LETTER (A-D) instead of the compliance LEVEL.
     That is the same coverage rule (k >= #categories, one exemplar per category) that
     made k=5 mandatory on the 5-level ordinal task; here #categories = 4, so k=4 is
     coverage-complete — this is NOT a return to the deprecated AIReg k=4, which was
     defective only because 4 < 5.
  2. within each (subject, letter) bucket the candidate pool is first restricted to rows
     whose ``answer_justification`` length falls in ``EXEMPLAR_SPAN_BAND`` (see
     "Band-targeted selection" below).

Round-robin over letters in A->D order, least-used ``rater_model`` at each step, ties
keeping stored order. Deterministic.

BAND-TARGETED SELECTION (``exemplar_selection = "band_400_800"``, E1 remediation
2026-08-09). Store v2 fixed the *floor* problem (no exemplar is short enough to teach a
base checkpoint that the reasoning span is optional) but introduced a *ceiling* problem:
its spans run to 1509 characters, and a length-blind draw hands base checkpoints an
elaborate model to imitate. The offline+Mac smoke ladder measured exactly that on
qwen3-4b-base, 13 items, greedy, v2 store throughout:

    default (length-blind) draw   spans 648-1509   parse 0.538   <- WORST
    band-targeted draw            spans  507-781   parse 0.846   <- BEST

The failure the length-blind draw adds is not empty reasoning (0 empty cells in every v2
iteration) — it is JSON-breaking output: models imitate the elaborateness, write longer
and more ornate justifications, and emit illegal escapes and raw control characters
inside the strings. Restricting to 400-800 characters keeps the span comfortably above
the AIReg dimension exemplars' floor (min 246, median 610) while removing the long tail
that elicits the imitation.

The band is a *preference*, never a coverage cost: if a (subject, letter) bucket holds no
in-band row, the selector falls back to the row(s) nearest the band and the letter is
still covered. Only a genuinely EMPTY bucket is an error, and that is the pre-existing
coverage guard's business, not the band filter's.

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
    CORPUS / "leaf_exemplars" / "mmlu_control_exemplars_v2"
    / "exemplar_store" / "mmlu_control_exemplar_store.json"
)
# Pin of the control exemplar store (333 rows, keyed by subject). Verified on load and
# recorded in every leg's meta sidecar, exactly like the slice pin: a scaffold that moved
# between the pre and post leg of a pair would confound the paired ECE delta (E2).
STORE_SHA256 = "d34d855eac8c0e0a7a9430f2db711b6692d008404bd2a96b229fe64b04f5f9ba"

# PROVENANCE OF THE PREVIOUS PIN (do not delete — the Phase-2c legs are scored against it):
#   v1  mmlu_control_exemplars_v1/exemplar_store/mmlu_control_exemplar_store.json
#       sha256 c4174d4587618c2aa54a71620fbcb4d2514db611d586fa093ad0ee039c3fa3f2, 237 rows.
# Re-pinned to v2 on 2026-08-09 (E1 remediation). Phase 2c ran all eight legs against v1
# and NONE of the four base (pre) legs cleared the parse_ok >= 0.90 gate, through two
# content-driven modes that were traced to the exemplars rather than to the harness:
#   Mode A  LaTeX backslashes inside JSON string fields -> `Invalid \escape` on stage 2;
#   Mode B  exemplar reasoning spans ~3x shorter than the AIReg scaffold's (median 188 vs
#           610 chars, 56% under 200 chars vs 0%), which taught base checkpoints that the
#           reasoning span is optional — they emitted 0-1 chars and fell into stage-2 EOS.
# v2 re-annotates a depth-2 candidate pool under prompt version mmlu-control-annotations-v2,
# whose two instruction amendments target exactly those modes (a full distractor-elimination
# paragraph of roughly 400-800 characters; plain text with no backslash anywhere). Measured
# on the v2 store: justification spans min 370 / median 810, 0% under 200 chars, and zero
# backslashes in any of the 364 annotated rows.
#
# ONLY THE SCAFFOLD CONTENT MOVED. The two-stage harness, the stop lists, the K=4 contract,
# the 0.05 grid, the nominal (non-unimodal) answer support and the correct-answer filter are
# unchanged, and the distributions are still the panel's ORGANIC credences — the v2 prompt
# says nothing about sharpness or confidence. A v1-scored leg and a v2-scored leg are
# therefore comparable in every respect except the exemplars, which is the intended contrast.
#
# Re-pinning is deliberate and load-bearing: legs elicited under the v1 pin hard-error on
# resume against this one (see elicit_control._guard_resume), which is the desired behaviour.
STORE_SHA256_V1 = "c4174d4587618c2aa54a71620fbcb4d2514db611d586fa093ad0ee039c3fa3f2"

# k=4 is the CONTROL protocol constant (one exemplar per option letter). It is NOT read
# from configs/models.yaml — that yaml pins the AIReg instrument's k=5, and a control leg
# must never inherit it (5 exemplars over 4 letters breaks one-per-category balance).
CONTROL_FEWSHOT_K = 4

SCAFFOLD_VARIANTS = ("baseline", "alt_set", "rev_order")

# Band-targeted exemplar selection (E1 remediation) — see the module docstring for the
# smoke evidence. Inclusive bounds on len(answer_justification), in characters.
EXEMPLAR_SPAN_BAND = (400, 800)
# Recorded verbatim in every leg's meta sidecar and guarded on resume: a leg elicited
# under a length-blind draw and one elicited under the band draw are DIFFERENT legs, and
# the smoke ladder measured a 31-point parse-rate gap between them.
EXEMPLAR_SELECTION = f"band_{EXEMPLAR_SPAN_BAND[0]}_{EXEMPLAR_SPAN_BAND[1]}"

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

def span_len(row: dict) -> int:
    """Length in characters of the exemplar's ``answer_justification`` — the band axis."""
    return len(row.get("answer_justification") or "")


def _band_distance(row: dict, band=EXEMPLAR_SPAN_BAND) -> int:
    """0 inside the band, else the characters by which the row misses it."""
    n = span_len(row)
    lo, hi = band
    return 0 if lo <= n <= hi else (lo - n if n < lo else n - hi)


def band_restrict(cands: Sequence[dict], band=EXEMPLAR_SPAN_BAND) -> List[dict]:
    """The band PREFERENCE applied to one (subject, letter) candidate pool.

    In-band rows if there are any; otherwise the rows tied at the smallest distance to
    the band (documented fallback — the letter is still covered, and the least-used-rater
    walk still gets to balance among ties rather than being handed a single row).

    Order is preserved, so the caller's stable sort still resolves rater ties by stored
    order. FAIL-LOUD on an empty pool: the band filter must never be the thing that
    silently drops a letter — an empty bucket is the caller's coverage problem and is
    surfaced as such.
    """
    if not cands:
        raise ValueError("band_restrict got an empty candidate pool — an empty "
                         "(subject, letter) bucket is a coverage failure, not a band miss")
    in_band = [r for r in cands if _band_distance(r, band) == 0]
    if in_band:
        return in_band
    best = min(_band_distance(r, band) for r in cands)
    return [r for r in cands if _band_distance(r, band) == best]


def select_rows(subject: str, k: int, exclude: Optional[Sequence[str]] = None) -> List[dict]:
    """Stratified fixed_set over letters x raters, band-targeted within each bucket.

    ``fewshot.select_rows`` semantics with the letter axis and ONE added step: the
    per-bucket candidate pool passes through :func:`band_restrict` before the least-used
    rater walk. The band is re-applied on every round-robin pass, so a k > 4 draw that
    exhausts a bucket's in-band rows falls back per pass rather than once.
    """
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
            cands = band_restrict(cands)
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

    STATUS (2026-08-09, v2 re-pin): ``alt_set`` is now FEASIBLE for all six subjects. On
    the v1 store it raised everywhere, because that store was built from a candidate pool
    with one item per (subject, letter) and a source-exclusion re-walk could not be
    coverage-preserving. The v2 pool is depth 2 by construction and survives the
    correct-answer filter at depth 2 in every bucket, so the disjoint draw goes through.
    The guard is unchanged and still raises on a shallower store.

    BAND INTERPLAY (E1 remediation): both draws are band-targeted. The primary is drawn
    band-first, its SOURCE ITEMS are excluded, and the alternate is then band-drawn among
    what is left. Four (subject, letter) buckets hold a source item with no in-band row
    at all, so the alternate legitimately falls back to nearest-the-band there — which is
    precisely why the fallback exists rather than a hard in-band requirement: an in-band
    *requirement* would make ``alt_set`` infeasible on a store that is deep enough to
    support it, trading a real coverage property for a soft length preference.
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
            "alt_set_feasible": ok, "alt_set_ids": [r["id"] for r in alt], "error": err,
            "exemplar_selection": EXEMPLAR_SELECTION,
            "baseline_span_lens": [span_len(r) for r in base],
            "alt_set_span_lens": [span_len(r) for r in alt],
            "baseline_out_of_band": [r["id"] for r in base if _band_distance(r)],
            "alt_set_out_of_band": [r["id"] for r in alt if _band_distance(r)]}
