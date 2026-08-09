"""Correctness-sense calibration primitives (ECE / AECE / MCE / AUROC / bootstrap).

Lifted VERBATIM out of ``scripts/base_calibration_premise_probe.py`` (2026-08-09) so the
defined-answer control and the AIReg probe compute the literature's calibration
quantities with ONE implementation. The probe now imports from here; a regression test
holds the lifted functions against frozen copies of their pre-lift bodies, and the probe
artifact re-runs identical modulo ``study_a``'s unrelated ``T_rel`` grid argmin.

Conventions (frozen — these ARE the reported quantities):
    confidence  := max_j p_j after the epsilon floor
    correct     := 1{argmax pred == argmax GT}
    ECE10 equal-width / AECE5 equal-mass / MCE / over-confidence gap / AUROC
    bootstrap   B = 2000, seed 20260721, resampling ``document_id`` clusters

DELIBERATELY NOTHING ORDINAL LIVES HERE. This module never imports ``study_a``,
``judex.experiments`` (Murphy) or ``judex.core.metrics`` (RPS/W1) — that is what lets
``score_control`` depend on it while remaining structurally incapable of emitting an
ordinal metric on the nominal K=4 control (design D7; guarded by a test on the import
graph). The temperature grid is re-declared here from the same formula rather than
imported from ``study_a`` for exactly that reason; a test asserts the two are identical
element for element.
"""
from __future__ import annotations

import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Sibling-repo layout: .../judex/{judex-calibration,judex-evaluator}; JUDEX_UMBRELLA wins.
_UMBRELLA = Path(os.environ.get("JUDEX_UMBRELLA") or Path(__file__).resolve().parents[3])
_EVAL_SRC = _UMBRELLA / "judex-evaluator" / "src"
if str(_EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(_EVAL_SRC))

from judex.core.distributions import ComplianceDistribution     # noqa: E402
from judex.calibration import apply_temperature                 # noqa: E402

EPSILON = 0.005          # the registered verbalized-channel floor
BOOT_SEED = 20260721
BOOT_N = 2000
ECE_BINS = 10            # equal-width, the standard reporting choice
AECE_BINS = 5            # equal-mass (adaptive); more stable at n ~ 120

# The temperature search range and its 60-point log grid — the same formula and the same
# floats as ``study_a.T_BOUNDS`` / ``study_a.GRID`` (asserted by test). Re-declared, not
# imported, so this module's dependency graph stays free of the ordinal machinery.
T_BOUNDS = (0.25, 20.0)
GRID = [math.exp(math.log(T_BOUNDS[0]) + (math.log(T_BOUNDS[1]) - math.log(T_BOUNDS[0])) * i / 59)
        for i in range(60)]


def saturated(T: float, tol: float = 1e-6, bounds=T_BOUNDS) -> bool:
    """True when a fitted temperature sits on the search boundary — a peg, not a fit."""
    return (not math.isfinite(T)) or T <= bounds[0] * (1 + tol) or T >= bounds[1] * (1 - tol)


def floor_renorm(v: Sequence[float], eps: float = EPSILON) -> List[float]:
    w = [max(float(x), eps) for x in v]
    z = sum(w)
    return [x / z for x in w]


# ------------------------------------------------------- correctness-sense rows

def conf_correct(preds, cells, temperature: float = 1.0):
    """[(document_id, confidence, correct, tied)] in the literature's sense.

    confidence = max_j p_j after optional temperature scaling through the same
    invert-softmax seam the distributional fits use, so both senses land on ONE
    temperature axis.
    correct    = argmax(pred) == argmax(GT).

    K-agnostic: ``gt_labels`` supplies the support, so the AIReg 5-level ordinal scale
    and the control's nominal A-D options both flow through unchanged.
    """
    by_label = {c.item_label: c for c in cells}
    rows = []
    for label, probs in preds.items():
        c = by_label.get(label)
        if c is None:
            continue
        d = ComplianceDistribution.from_values(list(probs), c.gt_labels)
        if temperature != 1.0:
            d = apply_temperature(d, temperature)
        p = list(d.probabilities)
        m = max(p)
        tied = sum(1 for x in p if abs(x - m) <= 1e-12) > 1
        rows.append((c.document_id, m, int(p.index(m) == c.gt_argmax), int(tied)))
    return rows


_conf_correct = conf_correct        # the probe's historical private name


# ------------------------------------------------------------------- estimators

def ece_equal_width(rows, bins=ECE_BINS):
    """Standard ECE: equal-width confidence bins over [0, 1]. Also returns MCE
    and the per-bin table (the reliability diagram in numbers)."""
    n = len(rows)
    if n == 0:
        return {"ece": float("nan"), "mce": float("nan"), "bins": []}
    buckets = [[] for _ in range(bins)]
    for _, conf, corr, _ in rows:
        idx = min(bins - 1, int(conf * bins))
        buckets[idx].append((conf, corr))
    ece = 0.0
    mce = 0.0
    table = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        acc = sum(c for _, c in b) / len(b)
        conf = sum(cf for cf, _ in b) / len(b)
        gap = abs(conf - acc)
        ece += (len(b) / n) * gap
        mce = max(mce, gap)
        table.append({"bin": f"[{i / bins:.1f},{(i + 1) / bins:.1f})", "n": len(b),
                      "mean_conf": round(conf, 4), "acc": round(acc, 4),
                      "signed_gap": round(conf - acc, 4)})
    return {"ece": ece, "mce": mce, "bins": table}


def mce(rows, bins=ECE_BINS) -> float:
    """Maximum calibration error — the worst equal-width bin gap."""
    return ece_equal_width(rows, bins)["mce"]


def aece_equal_mass(rows, bins=AECE_BINS):
    """Adaptive ECE: equal-MASS bins (each holds ~n/bins items). Stable at n~120
    where equal-width bins are mostly empty. DACA reports AECE alongside ECE."""
    n = len(rows)
    if n == 0:
        return {"aece": float("nan"), "bins": []}
    order = sorted(rows, key=lambda r: r[1])
    edges = [round(i * n / bins) for i in range(bins + 1)]
    aece = 0.0
    table = []
    for i in range(bins):
        b = order[edges[i]:edges[i + 1]]
        if not b:
            continue
        acc = sum(r[2] for r in b) / len(b)
        conf = sum(r[1] for r in b) / len(b)
        aece += (len(b) / n) * abs(conf - acc)
        table.append({"n": len(b), "conf_lo": round(b[0][1], 4), "conf_hi": round(b[-1][1], 4),
                      "mean_conf": round(conf, 4), "acc": round(acc, 4),
                      "signed_gap": round(conf - acc, 4)})
    return {"aece": aece, "bins": table}


def auroc(rows):
    """AUROC of confidence as a ranker of correctness — DISCRIMINATION, which is
    invariant to any monotone recalibration (temperature included). Separates
    "the confidence signal is informative but mis-levelled" (0.5 << AUROC,
    fixable by one temperature) from "the confidence carries no information"
    (AUROC ~ 0.5, not fixable by any recalibration)."""
    pos = [r[1] for r in rows if r[2] == 1]
    neg = [r[1] for r in rows if r[2] == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


def correctness_block(preds, cells, temperature: float = 1.0):
    """The per-leg correctness-sense summary (accuracy, gap, ECE10/AECE5/MCE, AUROC)."""
    rows = conf_correct(preds, cells, temperature)
    n = len(rows)
    if n == 0:
        return {"n": 0}
    acc = sum(r[2] for r in rows) / n
    conf = sum(r[1] for r in rows) / n
    ew = ece_equal_width(rows)
    em = aece_equal_mass(rows)
    return {"n": n, "accuracy": acc, "mean_confidence": conf,
            "overconfidence_gap": conf - acc,
            "ece10": ew["ece"], "mce10": ew["mce"], "ece10_bins": ew["bins"],
            "aece5": em["aece"], "aece5_bins": em["bins"],
            "auroc_conf_vs_correct": auroc(rows),
            "n_argmax_ties": sum(r[3] for r in rows)}


def fit_T_ece(preds, cells, bounds=T_BOUNDS, grid: Optional[Sequence[float]] = None):
    """Grid argmin of the correctness-sense ECE — the correctness analogue of T_abs.

    ECE is a non-smooth binned summary, so this is a grid argmin, not an optimiser; both
    ECE10 and AECE5 targets are reported because the binning choice can move a non-smooth
    argmin. ``bounds`` is passed explicitly (never inherited) so a saturated fit is
    reported as the peg it is.
    """
    g = list(grid) if grid is not None else GRID

    def obj_ece(T):
        return ece_equal_width(conf_correct(preds, cells, T))["ece"]

    def obj_aece(T):
        return aece_equal_mass(conf_correct(preds, cells, T))["aece"]

    T_ece = min(g, key=obj_ece)
    T_aece = min(g, key=obj_aece)
    return {"T_ece10": T_ece, "T_ece10_saturated": saturated(T_ece, bounds=bounds),
            "T_ece10_value": obj_ece(T_ece), "ece10_at_T1": obj_ece(1.0),
            "T_aece5": T_aece, "T_aece5_saturated": saturated(T_aece, bounds=bounds),
            "T_aece5_value": obj_aece(T_aece), "aece5_at_T1": obj_aece(1.0)}


# ------------------------------------------------------------------- inference

def boot_ece(preds, cells, n=BOOT_N, seed=BOOT_SEED):
    """Cluster bootstrap CI on correctness ECE10 — resamples ``document_id`` groups.

    The clustering unit is whatever ``document_id`` carries: the 24 AIReg documents on
    the judging task, the ITEM on the control (i.i.d. primary read, D7) or the SUBJECT
    for the control's clustered sensitivity read. Fewer than 3 clusters returns None
    rather than a meaningless interval.
    """
    rows = conf_correct(preds, cells)
    by_doc = {}
    for doc, conf, corr, tie in rows:
        by_doc.setdefault(doc, []).append((doc, conf, corr, tie))
    docs = sorted(by_doc)
    if len(docs) < 3:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        s = []
        for _ in docs:
            s.extend(by_doc[docs[rng.randrange(len(docs))]])
        vals.append(ece_equal_width(s)["ece"])
    vals.sort()
    return {"lo": vals[int(0.025 * len(vals))], "hi": vals[int(0.975 * len(vals)) - 1],
            "median": statistics.median(vals), "n_docs": len(docs), "n_reps": len(vals)}


def boot_paired_ece_delta(pre, post, cells, n=BOOT_N, seed=BOOT_SEED):
    """Paired cluster bootstrap on ECE(post) - ECE(pre): the same cluster resample
    drives both legs, so the pairing is preserved."""
    rpre = conf_correct(pre, cells)
    rpost = conf_correct(post, cells)
    dpre, dpost = {}, {}
    for r in rpre:
        dpre.setdefault(r[0], []).append(r)
    for r in rpost:
        dpost.setdefault(r[0], []).append(r)
    docs = sorted(set(dpre) & set(dpost))
    if len(docs) < 3:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        a, b = [], []
        for _ in docs:
            d = docs[rng.randrange(len(docs))]
            a.extend(dpre[d])
            b.extend(dpost[d])
        vals.append(ece_equal_width(b)["ece"] - ece_equal_width(a)["ece"])
    vals.sort()
    lo, hi = vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]
    return {"lo": lo, "hi": hi, "median": statistics.median(vals),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_docs": len(docs)}
