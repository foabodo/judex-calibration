#!/usr/bin/env python3
"""Does the literature's "pre-trained LMs are well-calibrated" premise transfer here?

Motivation (2026-07-21). Study A/B found that open PRE-trained bases need large
ABSOLUTE temperature corrections against JUDEX's distributional ground truth
(verbalized T_abs(pre) 2.55-5.53; logit T*_pre 2.43-3.83). That looks like it
contradicts DACA (Luo et al., NeurIPS 2025, arXiv:2505.16690), whose motivating
premise is "the inherently well-calibrated property of PLMs".

But the two statements are measured in DIFFERENT SENSES, and this script keeps
them apart:

  CORRECTNESS sense (the literature's, incl. DACA Fig. 1):
      confidence  := max_j p_j
      correct     := 1{argmax p == argmax GT}
      metric      := ECE / MCE / AECE over confidence bins, plus the
                     over-confidence gap  mean(conf) - accuracy.
      DACA's own Figure 1 (MMLU, max-prob over A-D): base ECE
      {Llama-3-8B 3.52%, Qwen-2.5-7B 5.41%, DeepSeek-V2-Lite 3.39%,
       Yi-1.5-6B 6.85%} vs post {19.00, 20.84, 20.81, 24.15}%.

  DISTRIBUTIONAL sense (JUDEX's):
      metric      := T_abs (RPS-optimal temperature vs the 7-rater GT
                     distribution), Murphy reliability, mean W1 / RPS.

HYPOTHESIS UNDER TEST
  H1  Bases are reasonably calibrated in the CORRECTNESS sense while being
      badly miscalibrated in the DISTRIBUTIONAL sense. If H1 holds, our result
      does not falsify the literature's claim; it shows the claim does not
      TRANSFER to distributional judging against a dispersed human reference.
  H0' Bases are ALSO poorly calibrated in the correctness sense -> the
      challenge to the literature is broader.

TESTS
  P1  Per-leg correctness-sense calibration (ECE10 / AECE5 / MCE / gap) beside
      the distributional quantities (T_abs, Murphy reliability, W1, RPS), for
      every Study B verbalized leg and every Study A logit leg.
  P2  Both senses expressed in TEMPERATURE units on one axis: T_ECE (the
      grid temperature minimising correctness-ECE) vs T_abs (RPS vs GT).
  P3  Structural controls that show the two senses can diverge by construction:
        - ORACLE-DIST: predict the GT distribution itself (accuracy 1.0,
          perfect distributionally) -> its correctness-ECE.
        - ORACLE-ARGMAX: one-hot on the GT argmax (correctness-ECE ~ 0,
          maximally wrong distributionally).
  P4  Sharpness vs location: entropy gap, |E_pred - E_gt| (a lower bound on
      W1), the shape residual W1 - |dE|, and how much of W1/RPS the optimal
      temperature can actually remove.
  P5  Association across legs: Spearman(correctness ECE, T_abs) and
      Spearman(correctness ECE, Murphy reliability). If ECE ranked legs the
      same way T_abs does, the two senses would be redundant.
  P6  Doc-clustered bootstrap (24 AIReg documents) on ECE per leg and on the
      PAIRED post-minus-base ECE difference.

$0 -- analysis-only on existing artifacts. REPORT-ONLY: r3 F1-F8 are frozen,
the evaluator seam stays `mode: noop`, nothing here is adopted and no config
is touched.

Run from judex-calibration:
  JUDEX_UMBRELLA=/path/to/judex PYTHONPATH=src \
      /path/to/judex-arm/bin/python scripts/base_calibration_premise_probe.py \
      [--out runs/base_calibration_premise]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from judex_calibration import aireg, study_a  # noqa: E402
from judex.calibration import apply_temperature, fit_temperature  # noqa: E402
from judex.core.distributions import ComplianceDistribution  # noqa: E402
from judex.core.metrics import ranked_probability_score, wasserstein_1  # noqa: E402

EPSILON = 0.005          # Study B analysis convention (also applied to logit legs here)
BOOT_SEED = 20260721
BOOT_N = 2000
ECE_BINS = 10            # equal-width, the standard reporting choice
AECE_BINS = 5            # equal-mass (adaptive); more stable at n ~ 120

# DACA Fig. 1 (MMLU, max-prob confidence over A-D choices), for the writeup's
# side-by-side. Verbatim from arXiv:2505.16690v2 Figure 1.
DACA_FIG1_ECE = {
    "Llama-3-8B":      {"pre": 0.0352, "post": 0.1900},
    "Qwen-2.5-7B":     {"pre": 0.0541, "post": 0.2084},
    "DeepSeek-V2-Lite": {"pre": 0.0339, "post": 0.2081},
    "Yi-1.5-6B":       {"pre": 0.0685, "post": 0.2415},
}

# (key, verbalized run dir, logit run dir or None, label, in adoption panel)
FAMILIES = [
    ("qwen",     "study_b_qwen",        "qwen_k5",     "Qwen3.5-35B-A3B",   True),
    ("gemma31",  "study_b_gemma31_api", "gemma31_k5",  "Gemma-4-31B",       True),
    ("glm",      "study_b_glm",         "glm",         "GLM-4.5",           True),
    ("maverick", "study_b_maverick",    None,          "Llama-4-Maverick",  True),
    ("llama31",  "study_b_llama31",     "llama31",     "Llama-3.1-405B",    False),
    ("gemma26",  "study_b_gemma26_api", None,          "Gemma-4-26B-A4B",   False),
]


# --------------------------------------------------------------------------- io

def floor_renorm(v, eps=EPSILON):
    w = [max(float(x), eps) for x in v]
    z = sum(w)
    return [x / z for x in w]


def load_verbalized(run_dir: Path, leg: str):
    """{item_label: floored 5-vector} for parse_ok cells (Study B)."""
    p = run_dir / f"{leg}_verbalized.json"
    if not p.exists():
        return None
    recs = json.loads(p.read_text())
    out = {label: floor_renorm(r["compliance"])
           for label, r in recs.items()
           if r.get("parse_ok") and r.get("compliance")}
    return out or None


def load_logit(run_dir: Path, leg: str):
    """{item_label: floored 5-vector} (Study A token-slice legs)."""
    p = run_dir / f"{leg}.json"
    if not p.exists():
        return None
    recs = json.loads(p.read_text())
    out = {label: floor_renorm(v) for label, v in recs.items() if v}
    return out or None


# ------------------------------------------------------- correctness-sense ECE

def _conf_correct(preds, cells, temperature: float = 1.0):
    """[(document_id, confidence, correct, tied)] in the literature's sense.

    confidence = max_j p_j after optional temperature scaling through the same
    invert-softmax seam the distributional fits use, so the two senses are
    expressed on ONE temperature axis (P2).
    correct    = argmax(pred) == argmax(GT)  (GT argmax is the hard label the
                 correctness sense needs; the GT's own dispersion is discarded,
                 which is exactly the information the literature's metric drops)
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


def correctness_block(preds, cells, temperature: float = 1.0):
    rows = _conf_correct(preds, cells, temperature)
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


def _lodo_folds(preds, cells):
    by_label = {c.item_label: c for c in cells}
    docs = {}
    for label in preds:
        c = by_label.get(label)
        if c is not None:
            docs.setdefault(c.document_id, []).append(label)
    return docs


def lodo_crossfit(preds, cells):
    """Leave-one-DOCUMENT-out cross-fitted repair, for BOTH senses on equal terms.

    In-sample argmins are optimistically biased -- ECE especially, because it is a
    non-smooth binned statistic and we pick T on a 60-point grid at n~120. Here T
    is fitted on 23 documents and scored on the 24th, so the reported residuals
    are honest and the two senses are compared under the same protocol.
    """
    folds = _lodo_folds(preds, cells)
    if len(folds) < 3:
        return None
    by_label = {c.item_label: c for c in cells}
    held_rows, held_pairs_1, held_pairs_T = [], [], []
    for held in sorted(folds):
        tr = {l: preds[l] for d, ls in folds.items() if d != held for l in ls}
        te = {l: preds[l] for l in folds[held]}
        if not tr or not te:
            continue
        # correctness sense: T minimising in-fold ECE10
        T_e = min(study_a.GRID, key=lambda T: ece_equal_width(_conf_correct(tr, cells, T))["ece"])
        held_rows.extend(_conf_correct(te, cells, T_e))
        # distributional sense: T minimising in-fold RPS vs GT
        tr_pairs = [(ComplianceDistribution.from_values(list(v), by_label[l].gt_labels),
                     ComplianceDistribution.from_values(list(by_label[l].gt_probs), by_label[l].gt_labels))
                    for l, v in tr.items() if l in by_label]
        T_d = fit_temperature(tr_pairs, bounds=study_a.T_BOUNDS).temperature
        for l, v in te.items():
            c = by_label.get(l)
            if c is None:
                continue
            p = ComplianceDistribution.from_values(list(v), c.gt_labels)
            g = ComplianceDistribution.from_values(list(c.gt_probs), c.gt_labels)
            held_pairs_1.append((p, g))
            held_pairs_T.append((apply_temperature(p, T_d), g))
    if not held_rows or not held_pairs_1:
        return None
    base_rows = _conf_correct(preds, cells)
    w1_1 = statistics.mean(wasserstein_1(p, g) for p, g in held_pairs_1)
    w1_T = statistics.mean(wasserstein_1(p, g) for p, g in held_pairs_T)
    rps_1 = statistics.mean(ranked_probability_score(p, g) for p, g in held_pairs_1)
    rps_T = statistics.mean(ranked_probability_score(p, g) for p, g in held_pairs_T)
    e_1 = ece_equal_width(base_rows)["ece"]
    e_T = ece_equal_width(held_rows)["ece"]
    return {
        "n_folds": len(folds),
        "ece10_uncorrected": e_1, "ece10_crossfit": e_T,
        "ece10_frac_removed": (e_1 - e_T) / e_1 if e_1 else float("nan"),
        "w1_uncorrected": w1_1, "w1_crossfit": w1_T,
        "w1_frac_removed": (w1_1 - w1_T) / w1_1 if w1_1 else float("nan"),
        "rps_uncorrected": rps_1, "rps_crossfit": rps_T,
        "rps_frac_removed": (rps_1 - rps_T) / rps_1 if rps_1 else float("nan"),
    }


def fit_T_ece(preds, cells):
    """Temperature on Study A's own 60-point log grid that minimises the
    correctness-sense ECE. The correctness-sense analogue of T_abs, so both
    senses land on the SAME axis (P2). ECE is a non-smooth binned summary, so
    this is a grid argmin, not an optimiser; both ECE10 and AECE5 targets are
    reported because the binning choice can move a non-smooth argmin."""
    def obj_ece(T):
        return ece_equal_width(_conf_correct(preds, cells, T))["ece"]

    def obj_aece(T):
        return aece_equal_mass(_conf_correct(preds, cells, T))["aece"]

    T_ece = min(study_a.GRID, key=obj_ece)
    T_aece = min(study_a.GRID, key=obj_aece)
    return {"T_ece10": T_ece, "T_ece10_saturated": study_a.saturated(T_ece),
            "T_ece10_value": obj_ece(T_ece), "ece10_at_T1": obj_ece(1.0),
            "T_aece5": T_aece, "T_aece5_saturated": study_a.saturated(T_aece),
            "T_aece5_value": obj_aece(T_aece), "aece5_at_T1": obj_aece(1.0)}


# ------------------------------------------------------ sharpness vs location

def _mean_level(p):
    """Expected ordinal level on the 1..5 support."""
    return sum((i + 1) * x for i, x in enumerate(p))


def shape_block(preds, cells):
    """P4. |E_pred - E_gt| lower-bounds W1 on the line, so
          mean W1 = mean|dE| (LOCATION)  +  (W1 - |dE|) (SHAPE/SPREAD residual)
    is a valid non-negative split. Recomputed at T_abs to show what a pure
    dispersion correction can and cannot remove."""
    by_label = {c.item_label: c for c in cells}
    pairs, dE, w1, hp, hg, dmax = [], [], [], [], [], []
    for label, probs in preds.items():
        c = by_label.get(label)
        if c is None:
            continue
        pred = ComplianceDistribution.from_values(list(probs), c.gt_labels)
        gt = ComplianceDistribution.from_values(list(c.gt_probs), c.gt_labels)
        pairs.append((pred, gt))
        dmax.append(max(pred.probabilities) - max(gt.probabilities))
        dE.append(_mean_level(pred.probabilities) - _mean_level(gt.probabilities))
        w1.append(wasserstein_1(pred, gt))
        hp.append(pred.normalized_entropy())
        hg.append(gt.normalized_entropy())
    if not pairs:
        return {"n": 0}
    T = fit_temperature(pairs, bounds=study_a.T_BOUNDS).temperature
    scaled = [(apply_temperature(p, T), g) for p, g in pairs]
    dE_T = [_mean_level(p.probabilities) - _mean_level(g.probabilities) for p, g in scaled]
    w1_T = [wasserstein_1(p, g) for p, g in scaled]
    rps_1 = statistics.mean(ranked_probability_score(p, g) for p, g in pairs)
    rps_T = statistics.mean(ranked_probability_score(p, g) for p, g in scaled)
    m_w1, m_dE = statistics.mean(w1), statistics.mean(abs(x) for x in dE)
    return {
        "n": len(pairs), "T_abs": T, "T_abs_saturated": study_a.saturated(T),
        "mean_pred_norm_entropy": statistics.mean(hp),
        "mean_gt_norm_entropy": statistics.mean(hg),
        "entropy_gap_pred_minus_gt": statistics.mean(hp) - statistics.mean(hg),
        # CAPABILITY-INVARIANT sharpness: how much peakier the prediction is than
        # the human panel on the SAME item, paired. Involves no notion of being
        # right, so it survives the "your bases are just weak judges" objection.
        "sharpness_gap_max_prob": statistics.mean(dmax),
        "frac_items_sharper_than_gt": sum(1 for x in dmax if x > 0) / len(dmax),
        "mean_signed_dE": statistics.mean(dE),
        "mean_abs_dE_location": m_dE,
        "mean_w1": m_w1,
        "shape_residual_w1_minus_absdE": m_w1 - m_dE,
        "location_share_of_w1": (m_dE / m_w1) if m_w1 else float("nan"),
        "at_T_abs": {"mean_signed_dE": statistics.mean(dE_T),
                     "mean_abs_dE_location": statistics.mean(abs(x) for x in dE_T),
                     "mean_w1": statistics.mean(w1_T),
                     "mean_rps": rps_T},
        "mean_rps": rps_1,
        "w1_removed_by_temperature": (m_w1 - statistics.mean(w1_T)) / m_w1 if m_w1 else float("nan"),
        "rps_removed_by_temperature": (rps_1 - rps_T) / rps_1 if rps_1 else float("nan"),
    }


# ------------------------------------------------------------------ inference

def boot_ece(preds, cells, n=BOOT_N, seed=BOOT_SEED):
    """Doc-clustered bootstrap CI on correctness ECE10 (resample the 24 AIReg
    documents with replacement; the item is not the independent unit)."""
    rows = _conf_correct(preds, cells)
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
    """Paired doc-clustered bootstrap on ECE(post) - ECE(pre): the same document
    resample drives both legs, so the pairing is preserved."""
    rpre = _conf_correct(pre, cells)
    rpost = _conf_correct(post, cells)
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


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


# ----------------------------------------------------------------------- main

def leg_report(preds, cells, bootstrap=True):
    out = {}
    out["correctness"] = correctness_block(preds, cells)
    out["correctness_temperature"] = fit_T_ece(preds, cells)
    out["distributional"] = study_a.score_variant(preds, cells)
    out["shape"] = shape_block(preds, cells)
    out["lodo_crossfit"] = lodo_crossfit(preds, cells)
    if bootstrap:
        out["ece10_boot"] = boot_ece(preds, cells)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/base_calibration_premise")
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()
    boot = not args.no_bootstrap

    cells = aireg.load_cells()
    runs = REPO / "runs"
    report = {
        "convention": {"epsilon": EPSILON, "T_bounds": list(study_a.T_BOUNDS),
                       "ece_bins_equal_width": ECE_BINS, "aece_bins_equal_mass": AECE_BINS,
                       "bootstrap": {"reps": BOOT_N, "seed": BOOT_SEED,
                                     "cluster": "AIReg document"},
                       "correctness_definition":
                           "confidence = max_j p_j ; correct = 1{argmax pred == argmax GT}"},
        "daca_fig1_reference_ece": DACA_FIG1_ECE,
        "legs": {}, "families": {},
    }

    # ---- GT reference properties (what the correctness sense throws away)
    gt_pairs = [(ComplianceDistribution.from_values(list(c.gt_probs), c.gt_labels), c)
                for c in cells]
    gt_max = [max(d.probabilities) for d, _ in gt_pairs]
    marg = [0] * 5
    for c in cells:
        marg[c.gt_argmax] += 1
    report["ground_truth"] = {
        "n_cells": len(cells),
        "mean_gt_max_prob": statistics.mean(gt_max),
        "median_gt_max_prob": statistics.median(gt_max),
        "frac_gt_max_prob_above_0.9": sum(1 for x in gt_max if x > 0.9) / len(gt_max),
        "mean_gt_norm_entropy": statistics.mean(d.normalized_entropy() for d, _ in gt_pairs),
        "gt_argmax_marginal": [x / len(cells) for x in marg],
        "majority_class_rate": max(marg) / len(cells),
    }

    # ---- P3 structural controls
    oracle_dist = {c.item_label: list(c.gt_probs) for c in cells}
    oracle_argmax = {c.item_label: floor_renorm([1.0 if i == c.gt_argmax else 0.0 for i in range(5)])
                     for c in cells}
    report["controls"] = {
        "oracle_distributional": {
            "note": "predict the GT distribution itself: distributionally PERFECT "
                    "(W1 = RPS = 0, T_abs = 1) and argmax-accuracy 1.0 by construction",
            "correctness": correctness_block(oracle_dist, cells),
            "shape": shape_block(oracle_dist, cells)},
        "oracle_argmax_onehot": {
            "note": "eps-floored one-hot on the GT argmax: correctness-ECE near 0 "
                    "(confident and always right) while distributionally maximally sharp",
            "correctness": correctness_block(oracle_argmax, cells),
            "shape": shape_block(oracle_argmax, cells)},
    }

    # ---- P1/P2/P4/P6 per leg
    for key, vdir, ldir, label, panel in FAMILIES:
        fam = {"label": label, "in_adoption_panel": panel, "channels": {}}
        for channel, loader, d in (("verbalized", load_verbalized, vdir),
                                   ("logit", load_logit, ldir)):
            if d is None:
                continue
            rd = runs / d
            if not rd.exists():
                continue
            pre, post = loader(rd, "pre"), loader(rd, "post")
            if not pre or not post:
                continue
            ch = {"run_dir": d,
                  "pre": leg_report(pre, cells, boot),
                  "post": leg_report(post, cells, boot)}
            if boot:
                ch["paired_ece_delta_post_minus_pre"] = boot_paired_ece_delta(pre, post, cells)
            ch["tau_v"] = study_a.fit_tau_oc(post, pre, cells)
            fam["channels"][channel] = ch
            for leg in ("pre", "post"):
                report["legs"][f"{key}|{channel}|{leg}"] = {
                    "family": label, "channel": channel, "leg": leg,
                    "in_adoption_panel": panel,
                    "n": ch[leg]["correctness"]["n"],
                    "accuracy": ch[leg]["correctness"]["accuracy"],
                    "mean_confidence": ch[leg]["correctness"]["mean_confidence"],
                    "overconfidence_gap": ch[leg]["correctness"]["overconfidence_gap"],
                    "ece10": ch[leg]["correctness"]["ece10"],
                    "aece5": ch[leg]["correctness"]["aece5"],
                    "mce10": ch[leg]["correctness"]["mce10"],
                    "T_ece10": ch[leg]["correctness_temperature"]["T_ece10"],
                    "T_ece10_saturated": ch[leg]["correctness_temperature"]["T_ece10_saturated"],
                    "T_aece5": ch[leg]["correctness_temperature"]["T_aece5"],
                    "T_abs": ch[leg]["distributional"]["T_rps"],
                    "T_abs_saturated": ch[leg]["distributional"]["T_rps_saturated"],
                    "murphy_reliability": ch[leg]["distributional"]["murphy"]["reliability"],
                    "murphy_resolution": ch[leg]["distributional"]["murphy"]["resolution"],
                    "mean_rps": ch[leg]["distributional"]["mean_rps"],
                    "mean_w1": ch[leg]["distributional"]["mean_w1"],
                    "pred_norm_entropy": ch[leg]["distributional"]["mean_pred_norm_entropy"],
                    "location_share_of_w1": ch[leg]["shape"]["location_share_of_w1"],
                    "mean_signed_dE": ch[leg]["shape"]["mean_signed_dE"],
                    "w1_removed_by_temperature": ch[leg]["shape"]["w1_removed_by_temperature"],
                    "auroc": ch[leg]["correctness"]["auroc_conf_vs_correct"],
                    "sharpness_gap_max_prob": ch[leg]["shape"]["sharpness_gap_max_prob"],
                    "frac_items_sharper_than_gt": ch[leg]["shape"]["frac_items_sharper_than_gt"],
                    "entropy_gap_pred_minus_gt": ch[leg]["shape"]["entropy_gap_pred_minus_gt"],
                    "mean_abs_dE_location": ch[leg]["shape"]["mean_abs_dE_location"],
                    "ece10_at_T_ece": ch[leg]["correctness_temperature"]["T_ece10_value"],
                    "lodo_ece10_frac_removed":
                        (ch[leg]["lodo_crossfit"] or {}).get("ece10_frac_removed"),
                    "lodo_ece10_crossfit": (ch[leg]["lodo_crossfit"] or {}).get("ece10_crossfit"),
                    "lodo_w1_frac_removed": (ch[leg]["lodo_crossfit"] or {}).get("w1_frac_removed"),
                    "lodo_rps_frac_removed": (ch[leg]["lodo_crossfit"] or {}).get("rps_frac_removed"),
                }
        if fam["channels"]:
            report["families"][key] = fam

    # ---- P5 association between the two senses, across legs
    L = report["legs"]
    def assoc(sel, xk, yk):
        ks = [k for k in L if sel(L[k])]
        if len(ks) < 3:
            return None
        return {"n": len(ks), "keys": ks,
                "spearman_rho": spearman([L[k][xk] for k in ks], [L[k][yk] for k in ks])}
    report["association"] = {
        "all_legs_ece10_vs_T_abs": assoc(lambda r: True, "ece10", "T_abs"),
        "base_legs_ece10_vs_T_abs": assoc(lambda r: r["leg"] == "pre", "ece10", "T_abs"),
        "verbalized_legs_ece10_vs_T_abs":
            assoc(lambda r: r["channel"] == "verbalized", "ece10", "T_abs"),
        "all_legs_ece10_vs_reliability": assoc(lambda r: True, "ece10", "murphy_reliability"),
        "all_legs_ece10_vs_accuracy": assoc(lambda r: True, "ece10", "accuracy"),
    }

    # ---- headline summary
    def grp(ch, leg):
        return [v for v in L.values() if v["channel"] == ch and v["leg"] == leg]
    summary = {}
    for ch in ("verbalized", "logit"):
        for leg in ("pre", "post"):
            g = grp(ch, leg)
            if not g:
                continue
            summary[f"{ch}_{leg}"] = {
                "n_legs": len(g),
                "ece10": {"min": min(v["ece10"] for v in g), "max": max(v["ece10"] for v in g),
                          "median": statistics.median(v["ece10"] for v in g)},
                "aece5": {"median": statistics.median(v["aece5"] for v in g)},
                "overconfidence_gap": {
                    "min": min(v["overconfidence_gap"] for v in g),
                    "max": max(v["overconfidence_gap"] for v in g),
                    "median": statistics.median(v["overconfidence_gap"] for v in g)},
                "accuracy": {"median": statistics.median(v["accuracy"] for v in g)},
                "T_ece10": {"min": min(v["T_ece10"] for v in g), "max": max(v["T_ece10"] for v in g),
                            "median": statistics.median(v["T_ece10"] for v in g)},
                "T_abs": {"min": min(v["T_abs"] for v in g), "max": max(v["T_abs"] for v in g),
                          "median": statistics.median(v["T_abs"] for v in g)},
            }
    summary["daca_fig1_pre_ece_range"] = [min(v["pre"] for v in DACA_FIG1_ECE.values()),
                                          max(v["pre"] for v in DACA_FIG1_ECE.values())]
    summary["daca_fig1_post_ece_range"] = [min(v["post"] for v in DACA_FIG1_ECE.values()),
                                           max(v["post"] for v in DACA_FIG1_ECE.values())]
    report["summary"] = summary

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "base_calibration_premise.json").write_text(json.dumps(report, indent=2))

    # ------------------------------------------------------------------ console
    g = report["ground_truth"]
    print(f"\nGT reference: n={g['n_cells']}  mean max-prob={g['mean_gt_max_prob']:.3f}  "
          f"mean norm-entropy={g['mean_gt_norm_entropy']:.3f}  "
          f"majority class={g['majority_class_rate']:.3f}")
    for name, c in report["controls"].items():
        cc = c["correctness"]
        print(f"  CONTROL {name:26} acc={cc['accuracy']:.3f} conf={cc['mean_confidence']:.3f} "
              f"gap={cc['overconfidence_gap']:+.3f} ECE10={cc['ece10']:.3f} "
              f"W1={c['shape']['mean_w1']:.3f} T_abs={c['shape']['T_abs']:.3f}")

    hdr = (f"\n{'leg':30} {'n':>4} {'acc':>6} {'conf':>6} {'gap':>7} {'ECE10':>7} "
           f"{'AECE5':>7} {'T_ECE':>7} {'T_abs':>7} {'relia':>7} {'W1':>6} {'h_pred':>7}")
    print(hdr)
    for k, v in L.items():
        print(f"{k:30} {v['n']:4d} {v['accuracy']:6.3f} {v['mean_confidence']:6.3f} "
              f"{v['overconfidence_gap']:+7.3f} {v['ece10']:7.3f} {v['aece5']:7.3f} "
              f"{v['T_ece10']:7.3f} {v['T_abs']:7.3f} {v['murphy_reliability']:7.4f} "
              f"{v['mean_w1']:6.3f} {v['pred_norm_entropy']:7.3f}")

    print(f"\n{'REPAIRABILITY (leave-one-document-out cross-fitted)':30}"
          f" {'AUROC':>6} {'ECE@1':>7} {'ECE_cf':>7} {'ECE-rem':>8} {'W1-rem':>7} {'RPS-rem':>8} {'locshr':>7}")
    for k, v in L.items():
        lo = v.get("lodo_ece10_crossfit")
        print(f"{k:30} {v['auroc']:6.3f} {v['ece10']:7.3f} "
              f"{(lo if lo is not None else float('nan')):7.3f} "
              f"{(v['lodo_ece10_frac_removed'] or float('nan')):8.1%} "
              f"{(v['lodo_w1_frac_removed'] or float('nan')):7.1%} "
              f"{(v['lodo_rps_frac_removed'] or float('nan')):8.1%} "
              f"{v['location_share_of_w1']:7.3f}")

    print(f"\n{'CAPABILITY-INVARIANT SHARPNESS (paired vs the human panel)':30}"
          f" {'d_maxp':>8} {'%sharper':>9} {'d_entropy':>10} {'|dE|':>7} {'signed dE':>10}")
    for k, v in L.items():
        print(f"{k:30} {v['sharpness_gap_max_prob']:+8.3f} "
              f"{v['frac_items_sharper_than_gt']:9.1%} {v['entropy_gap_pred_minus_gt']:+10.3f} "
              f"{v['mean_abs_dE_location']:7.3f} {v['mean_signed_dE']:+10.3f}")

    print("\nsummary by channel x leg:")
    for k, v in summary.items():
        if not isinstance(v, dict) or "ece10" not in v:
            continue
        print(f"  {k:18} n={v['n_legs']}  ECE10 median {v['ece10']['median']:.3f} "
              f"[{v['ece10']['min']:.3f},{v['ece10']['max']:.3f}]  "
              f"gap median {v['overconfidence_gap']['median']:+.3f}  "
              f"T_ECE median {v['T_ece10']['median']:.3f}  "
              f"T_abs median {v['T_abs']['median']:.3f}")
    print(f"  DACA Fig.1 pre  ECE range {summary['daca_fig1_pre_ece_range']}")
    print(f"  DACA Fig.1 post ECE range {summary['daca_fig1_post_ece_range']}")

    print("\nassociation (Spearman rho across legs):")
    for k, v in report["association"].items():
        if v:
            print(f"  {k:36} n={v['n']:2d}  rho={v['spearman_rho']:+.3f}")

    print("\npaired post-minus-pre ECE10 delta (doc-clustered bootstrap):")
    for key, fam in report["families"].items():
        for ch, cv in fam["channels"].items():
            d = cv.get("paired_ece_delta_post_minus_pre")
            if d:
                print(f"  {fam['label']:20} {ch:11} median {d['median']:+.3f} "
                      f"[{d['lo']:+.3f}, {d['hi']:+.3f}] excludes 0: {d['excludes_zero']}")

    print(f"\nwrote {out / 'base_calibration_premise.json'}")


if __name__ == "__main__":
    main()
