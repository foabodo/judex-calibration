"""Scoring for the defined-answer control — E1 (level) and E2 (direction).

E1  base-leg top-1 ECE on MMLU under the identical harness, against the AIReg values as
    printed in Paper B v14 (six bases 0.175-0.443; the five floor-clearing legs
    0.175-0.398, median 0.260).
E2  paired post-minus-pre ECE delta per family, against the AIReg direction (post
    improves 6/6, CI-significant in 4) and DACA's opposite MMLU direction.

Conventions are the frozen battery's, imported from ``ece``: confidence = max_j p_j after
the epsilon floor; correct = 1{argmax pred == key}; ECE10 equal-width / AECE5 equal-mass
/ MCE / over-confidence gap / AUROC; T_ece and T_aece by grid argmin with the bounds
passed explicitly. Uncertainty per D7: the ITEM-level bootstrap is primary (MMLU items
are independent draws in a way AIReg cells are not) with a SUBJECT-clustered read
reported as sensitivity only — six clusters is too few for a primary CI.

STRUCTURAL GUARANTEE (D7). Nothing ordinal is emitted, and this module cannot emit it:
its import graph contains no ``study_a``, no ``judex.experiments`` (Murphy) and no
``judex.core.metrics`` (RPS / W1). A-D are nominal options; an ordinal metric over them
would be an artifact of the alphabet. ``tests/test_ece_lift.py`` asserts the import graph
in a fresh interpreter, so the guarantee cannot rot into a comment.

The D6 tau companion is DESCRIPTIVE ONLY. It is fitted under a TOTAL-VARIATION objective
(permutation-invariant, the only kind that means anything on nominal options) — never
W1, never RPS, never Murphy. An MMLU tau is a different estimand on a different task with
a different reference regime, and MUST NOT enter the frozen r3/rA1 cluster/band/LOFO
constants.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from . import mmlu
from .ece import (BOOT_N, BOOT_SEED, EPSILON, GRID, T_BOUNDS, ComplianceDistribution,
                  apply_temperature, boot_ece, boot_paired_ece_delta, conf_correct,
                  correctness_block, ece_equal_width, fit_T_ece, floor_renorm, saturated)

CONTRACT = "mmlu_control_v1"

# The AIReg comparands, as PRINTED in Paper B v14 (not the pre-v13 figures). Carried here
# so the control report is self-contained and the E1 read never gets pointed at a stale
# range by accident.
AIREG_V14_BASE_ECE = {
    "all_six_bases": [0.175, 0.443],
    "five_floor_clearing": [0.175, 0.398],
    "five_floor_clearing_median": 0.260,
}
# DACA Fig. 1 (MMLU, max-prob over A-D, scaffold-free logit slice) — the qualitative
# comparand at MATCHED K=4. Verbatim from arXiv:2505.16690v2 Figure 1.
DACA_FIG1_ECE = {
    "Llama-3-8B": {"pre": 0.0352, "post": 0.1900},
    "Qwen-2.5-7B": {"pre": 0.0541, "post": 0.2084},
    "DeepSeek-V2-Lite": {"pre": 0.0339, "post": 0.2081},
    "Yi-1.5-6B": {"pre": 0.0685, "post": 0.2415},
}
CHANCE_FLOOR_K4 = 0.25      # D8: stated, never corrected for


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def answer_view(recs: Dict[str, dict]) -> Dict[str, List[float]]:
    """{item_label: eps-floored A-D vector} over parse_ok records.

    The floor is applied HERE, at analysis time — stored vectors keep the contract's
    exact grid zeros, exactly as on the AIReg side.
    """
    return {label: floor_renorm(r["compliance"])
            for label, r in recs.items()
            if r.get("parse_ok") and r.get("compliance")}


# ---------------------------------------------------------------------------
# D6 descriptive tau under a TVD objective
# ---------------------------------------------------------------------------

def _tvd(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def fit_tau_tvd(post: Dict[str, List[float]], pre: Dict[str, List[float]], items,
                grid: Optional[Sequence[float]] = None, bounds=T_BOUNDS) -> dict:
    """DESCRIPTIVE tau: the grid temperature aligning POST -> PRE under mean TVD.

    ``fit_tau_oc``'s W1 objective is order-dependent on the option alphabet and therefore
    meaningless on nominal options; TVD is permutation-invariant and drops into the same
    grid-argmin pattern. Report-only, appendix-grade, fenced: this number never touches
    the frozen r3/rA1 constants.
    """
    g = list(grid) if grid is not None else GRID
    by_label = {it.item_label: it for it in items}
    pairs = []
    for label in post.keys() & pre.keys():
        it = by_label.get(label)
        if it is None:
            continue
        pairs.append((ComplianceDistribution.from_values(list(post[label]), it.gt_labels),
                      list(pre[label])))
    if not pairs:
        return {"n": 0, "tau_tvd": None}

    def obj(T):
        return sum(_tvd(apply_temperature(p, T).probabilities, q) for p, q in pairs) / len(pairs)

    tau = min(g, key=obj)
    return {"n": len(pairs), "tau_tvd": tau, "tau_tvd_saturated": saturated(tau, bounds=bounds),
            "mean_tvd_at_tau": obj(tau), "mean_tvd_at_T1": obj(1.0),
            "objective": "mean total-variation distance (permutation-invariant)",
            "status": "DESCRIPTIVE ONLY (design D6) — different task, different reference "
                      "regime; never enters the r3/rA1 cluster/band/LOFO constants"}


# ---------------------------------------------------------------------------
# Per-leg and paired scoring
# ---------------------------------------------------------------------------

def score_leg(preds: Dict[str, List[float]], items_item, items_subject,
              *, bootstrap: bool = True, boot_n: int = BOOT_N,
              boot_seed: int = BOOT_SEED, bounds=T_BOUNDS) -> dict:
    """E1 for one leg: accuracy + the correctness-sense battery + both bootstraps."""
    if not preds:
        return {"n": 0}
    block = correctness_block(preds, items_item)
    out = {
        "n": block["n"],
        "accuracy": block["accuracy"],
        "mean_confidence": block["mean_confidence"],
        "overconfidence_gap": block["overconfidence_gap"],
        "ece10": block["ece10"], "aece5": block["aece5"], "mce10": block["mce10"],
        "auroc_conf_vs_correct": block["auroc_conf_vs_correct"],
        "n_argmax_ties": block["n_argmax_ties"],
        "ece10_bins": block["ece10_bins"], "aece5_bins": block["aece5_bins"],
        "chance_floor": CHANCE_FLOOR_K4,
        "accuracy_above_chance": block["accuracy"] - CHANCE_FLOOR_K4,
        "temperature": fit_T_ece(preds, items_item, bounds=bounds),
    }
    if bootstrap:
        out["ece10_boot_item"] = boot_ece(preds, items_item, n=boot_n, seed=boot_seed)
        out["ece10_boot_subject"] = boot_ece(preds, items_subject, n=boot_n, seed=boot_seed)
    return out


def score_pair(pre: Dict[str, List[float]], post: Dict[str, List[float]],
               items_item, items_subject, *, boot_n: int = BOOT_N,
               boot_seed: int = BOOT_SEED) -> dict:
    """E2: the paired post-minus-pre ECE delta, item-primary + subject sensitivity."""
    ece_pre = ece_equal_width(conf_correct(pre, items_item))["ece"]
    ece_post = ece_equal_width(conf_correct(post, items_item))["ece"]
    return {
        "ece10_pre": ece_pre, "ece10_post": ece_post,
        "delta_post_minus_pre": ece_post - ece_pre,
        "direction": ("post_improves" if ece_post < ece_pre else
                      "post_degrades" if ece_post > ece_pre else "tie"),
        "boot_item": boot_paired_ece_delta(pre, post, items_item, n=boot_n, seed=boot_seed),
        "boot_subject": boot_paired_ece_delta(pre, post, items_subject, n=boot_n, seed=boot_seed),
        "primary": "boot_item (D7: MMLU items are independent draws; the 6-subject "
                   "cluster bootstrap is a sensitivity read, not a CI of record)",
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def convention_block() -> dict:
    return {
        "contract": CONTRACT,
        "epsilon": EPSILON,
        "T_bounds": list(T_BOUNDS),
        "grid_points": len(GRID),
        "ece_bins_equal_width": 10,
        "aece_bins_equal_mass": 5,
        "bootstrap": {"reps": BOOT_N, "seed": BOOT_SEED,
                      "primary_cluster": "item (i.i.d.)",
                      "sensitivity_cluster": "subject (6 clusters)"},
        "correctness_definition":
            "confidence = max_j p_j after the epsilon floor ; correct = 1{argmax pred == key}",
        "ordinal_metrics": "NONE — A-D are nominal options (design D7); this module's "
                           "import graph contains no RPS/W1/Murphy",
        "cross_K_note": "ECE at K=4 sits over a 0.25 chance floor vs 0.20 at K=5 on the "
                        "AIReg task; the floors are stated, no numeric equality is claimed "
                        "(design D8)",
    }


def analyze(recs_by_leg: Dict[str, Dict[str, dict]], *, bootstrap: bool = True,
            slice_path=None, boot_n: int = BOOT_N, boot_seed: int = BOOT_SEED) -> dict:
    """Full control report from {leg: {item_label: record}}.

    Loads the pinned slice twice — once with ``document_id`` = item (primary) and once
    = subject (sensitivity) — so both bootstraps run through the identical machinery.
    """
    items_item = mmlu.load_control_items(slice_path, document_id="item")
    items_subject = mmlu.load_control_items(slice_path, document_id="subject")
    report = {
        "contract": CONTRACT,
        "convention": convention_block(),
        "slice_sha256": mmlu.slice_sha256(slice_path),
        "n_items": len(items_item),
        "aireg_v14_base_ece_comparand": AIREG_V14_BASE_ECE,
        "daca_fig1_reference_ece": DACA_FIG1_ECE,
        "legs": {},
    }
    views: Dict[str, Dict[str, List[float]]] = {}
    for leg, recs in recs_by_leg.items():
        preds = answer_view(recs)
        views[leg] = preds
        report["legs"][leg] = score_leg(preds, items_item, items_subject,
                                        bootstrap=bootstrap, boot_n=boot_n,
                                        boot_seed=boot_seed)
    if "pre" in views and "post" in views and views["pre"] and views["post"]:
        report["e2_paired_ece_delta"] = score_pair(views["pre"], views["post"], items_item,
                                                   items_subject, boot_n=boot_n,
                                                   boot_seed=boot_seed)
        report["tau_tvd_descriptive"] = fit_tau_tvd(views["post"], views["pre"], items_item)
    if "pre" in report["legs"] and report["legs"]["pre"].get("n"):
        e1 = report["legs"]["pre"]["ece10"]
        lo, hi = AIREG_V14_BASE_ECE["five_floor_clearing"]
        report["e1_read"] = {
            "base_ece10": e1,
            "band": ("daca_like_below_0.10" if e1 < 0.10 else
                     "intermediate_0.10_to_0.20" if e1 < 0.20 else "aireg_like_above_0.20"),
            "inside_aireg_v14_floor_clearing_range": lo <= e1 <= hi,
            "note": "read against the pre-specified E1 interpretation table (design §3)",
        }
    return report
