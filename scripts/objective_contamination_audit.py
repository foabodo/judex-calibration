#!/usr/bin/env python3
"""Objective-contamination audit: which recorded JUDEX conclusions depend on WHICH
objective a temperature was fit under, and which are metric-free?

Motivation (2026-07-21 reliability probe, spec/analysis_2026_07_21_reliability_objective_probe.md):
three defects were established in the instrument stack.

  D1  MIXED OBJECTIVES ACROSS ONE DECISION CHAIN.
        study_a.fit_tau_oc              minimizes  W1                  -> tau_oc, tau_v, T_J
        judex.calibration.fit_temperature minimizes RPS                -> T_rps, T_abs, arm2 T*
        study_a.closed_side_check        accepts on Murphy RELIABILITY -> F4
        judex.calibration.fit_dispersion_temperature minimizes ENTROPY GAP -> arm3b T_raw
        study_b.fit_Tc                   minimizes BRIER               -> arm4 T_c
      Numbers produced by one are compared to numbers produced by another as if
      commensurable.
  D2  W1 IS NEARLY NON-IDENTIFYING for temperature in the verbalized channel: its
      objective moves <=1% between T=1 and its optimum and is NEGATIVE for three
      families (exact T=1 beats the grid optimum).
  D3  BIN-DEPENDENCE: Murphy reliability/resolution are computed on binned data;
      study_a._reliability fits at bins=3, closed_side_check/score_variant read at
      bins=10, and closed-pair T_rel moves 2.10x over bins in {3..20}.

This script recomputes, on the artifacts of record, the four highest-value
conclusions that could be contaminated, plus the bin-sensitivity of the
highest-stakes CLASS-C decision (Study A's resolution-primary capability gate,
which drove two family swaps).

Checks
  A0  Reproduction gate. Re-derive the published numbers before questioning them.
  A1  CHECK 1 -- Study A's Q3-NEGATIVE verdict under a matched objective. The
      tau_oc set {qwen 1.601, llama31 1.857, gemma31 4.877, glm 8.835} is W1-fitted;
      ratio 3.05 > 2 killed the giant tier. Refit under RPS and under reliability
      (bin sweep) on the same logit-channel legs. Does the clustering verdict flip?
      Plus the direction-symmetry test on the Study A legs and the objective-movement
      percentages -- was W1 better identified in the logit channel than in the
      verbalized channel?
  A2  CHECK 2 -- tau_DACA's F7 corroboration. The 12 cross-family fits are RPS-fitted;
      the F7 window [T_J/2, 2 T_J] is anchored on a W1-derived T_J. Recompute the
      fits under W1 and re-anchor the window under each objective (a 2x2 verdict
      table), so the mixed comparison is separated from the finding.
  A3  CHECK 3 -- arm 3b's dispersion pool. Identify the objective, then build the
      matched-objective band (entropy-consistency / entropy-matching post->base per
      panel family) and ask whether T_raw is in band under a matched comparison.
  A4  CHECK 4 -- the "post-training supplies ~1/6 of judge overconfidence" claim now
      in judex_paper_v5_absolute_calibration.tex: ln(1.153)/ln(2.353), a W1-fitted
      numerator over an RPS-fitted denominator. Test (i) whether the multiplicative
      identity the decomposition presupposes -- T_abs(post) = tau * T_abs(pre) --
      holds at all, and (ii) what the all-RPS, all-W1 and objective-free
      (entropy-matched) versions of the share are.
  A5  CLASS C -- Study A's resolution-primary capability gate across bins in
      {3,5,10,15,20}. The pass/fail calls produced the "bimodal cliff" (passing bases
      0.039-0.046 vs failing 0.018-0.024) and drove the Gemma-26B->31B and
      Llama-4-Maverick->Llama-3.1-405B family swaps. Does any call flip?
  A6  W1 identification in the logit channel (the D2 question, Study A side).

$0 -- analysis-only on existing artifacts. REPORT-ONLY: r3 F1-F8 are frozen; nothing
here is adopted; no config is touched; the evaluator seam stays ``mode: noop``.

Run from judex-calibration:
  JUDEX_UMBRELLA=/path/to/judex PYTHONPATH=src \
      /Users/fabodo/anaconda3/envs/judex-arm/bin/python \
      scripts/objective_contamination_audit.py [--out runs/objective_contamination_audit]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from judex_calibration import aireg, study_a  # noqa: E402
from judex.calibration import (apply_temperature, fit_temperature,  # noqa: E402
                               fit_dispersion_temperature, realized_dispersion)
from judex.core.distributions import ComplianceDistribution  # noqa: E402
from judex.core.metrics import ranked_probability_score, wasserstein_1  # noqa: E402
from judex.metrics_report import murphy_decomposition, calibration_groups  # noqa: E402

# ------------------------------------------------------------------ constants
EPSILON_B = 0.005                  # Study B analysis convention (verbalized legs only)
T_J = 1.153                        # r3 F2 (FROZEN -- tested against, never edited)
BAND = (1.0251785151221313, 1.379820350674421)   # r3 F3 (FROZEN)
LN2 = math.log(2.0)
DACA_RANGE = (T_J / 2.0, 2.0 * T_J)              # r3 F7 (FROZEN)
CLUSTER_MAX_RATIO = study_a.CLUSTER_RULE_MAX_RATIO   # 2.0, the Q3 stopping rule
BIN_SWEEP = (3, 5, 10, 15, 20)
GRID = study_a.GRID                # 60-pt log grid over T_BOUNDS -- study_a's fitter grid
CLOSED_RUN = "stage9-onpair-e6-20260721"

# Study A (token-slice / logit channel) legs.  key -> (run dir, label, protocol, role)
# role: "panel_gate_pass" = in the Q3 gate-passing trio; "gate_fail" = failed the
# resolution-primary capability gate; "k4" = protocol ablation, not a panel datum.
STUDY_A = [
    ("qwen_k5",    "qwen_k5",    "Qwen3.5-35B-A3B",   "k5", "panel_gate_pass"),
    ("gemma31_k5", "gemma31_k5", "Gemma-4-31B",       "k5", "panel_gate_pass"),
    ("llama31",    "llama31",    "Llama-3.1-405B",    "k5", "panel_gate_pass"),
    ("glm",        "glm",        "GLM-4.5",           "k5", "gate_fail"),
    ("gemma26_k5", "gemma26_k5", "Gemma-4-26B-A4B",   "k5", "gate_fail"),
    ("maverick",   "llama",      "Llama-4-Maverick",  "k5", "gate_fail"),
    ("qwen_k4",    "qwen",       "Qwen3.5-35B-A3B",   "k4", "k4"),
    ("gemma31_k4", "gemma31",    "Gemma-4-31B",       "k4", "k4"),
    ("gemma26_k4", "gemma",      "Gemma-4-26B-A4B",   "k4", "k4"),
]
# The Q3 clustering verdict of record was taken over the GATE-PASSING trio.
Q3_TRIO = ("qwen_k5", "llama31", "gemma31_k5")
Q3_CLEAN4 = ("qwen_k5", "llama31", "gemma31_k5", "glm")

# Study B (verbalized channel) legs.  key -> (run dir, label, in adoption panel)
STUDY_B = [
    ("qwen",     "study_b_qwen",        "Qwen3.5-35B-A3B",  True),
    ("gemma31",  "study_b_gemma31_api", "Gemma-4-31B",      True),
    ("glm",      "study_b_glm",         "GLM-4.5",          True),
    ("maverick", "study_b_maverick",    "Llama-4-Maverick", True),
    ("llama31",  "study_b_llama31",     "Llama-3.1-405B",   False),
    ("gemma26",  "study_b_gemma26_api", "Gemma-4-26B-A4B",  False),
]
PANEL = ("qwen", "gemma31", "glm", "maverick")

# Published values to reproduce (A0).  Sources: spec/handoff_2026_07_19_gemma26_to_31b_
# replacement.md (Study A), spec/analysis_2026_07_21_absolute_vs_ratio_estimand.md and
# docs/verbalized_r0_analyses_2026_07_21.md (Study B), judex_paper_v5 (the 1/6 claim).
REF_TAU_OC = {"qwen_k5": 1.601, "gemma31_k5": 4.877, "llama31": 1.857, "glm": 8.835}
REF_BASE_RES = {"qwen_k5": 0.0456, "gemma31_k5": 0.0457, "llama31": 0.0393,
                "glm": 0.0221, "gemma26_k5": 0.0239, "maverick": 0.0197,
                "gemma26_k4": 0.0177, "gemma31_k4": 0.0407, "qwen_k4": 0.0332}
REF_TAU_V = {"qwen": 1.281, "gemma31": 1.025, "glm": 1.025, "maverick": 1.380}
REF_T_ABS_POST = {"qwen": 2.637, "gemma31": 2.242, "glm": 2.464, "maverick": 2.035}
REF_T_ABS_PRE = {"qwen": 3.190, "gemma31": 2.629, "glm": 2.552, "maverick": 4.941}
REF_PAPER = {"total_ln": 0.856, "increment_ln": 0.142, "share": 0.166,
             "T_abs_post_median": 2.353}
# tau_DACA of record (RPS-fitted, 12 cross-family, verbalized full-scale study)
REF_DACA_MEDIAN = 0.922


# --------------------------------------------------------------------- loading
def floor_renorm(v, eps=EPSILON_B):
    w = [max(float(x), eps) for x in v]
    z = sum(w)
    return [x / z for x in w]


def load_study_a_leg(run_dir: Path, leg: str):
    """Study A token-slice vectors -- raw, NO epsilon floor (they are strictly
    positive softmax slices; the record was computed on them raw)."""
    p = run_dir / f"{leg}.json"
    if not p.exists():
        return None
    return {k: [float(x) for x in v] for k, v in json.loads(p.read_text()).items()}


def load_study_b_leg(run_dir: Path, leg: str):
    """Study B verbalized vectors over parse_ok cells, epsilon-floored (convention)."""
    p = run_dir / f"{leg}_verbalized.json"
    if not p.exists():
        return None
    recs = json.loads(p.read_text())
    out = {lb: floor_renorm(r["compliance"]) for lb, r in recs.items()
           if r.get("parse_ok") and r.get("compliance")}
    return out or None


def load_closed_run(eval_runs: Path):
    try:
        from e6_r3_arms import load_closed
        return load_closed(eval_runs / CLOSED_RUN)
    except Exception:
        return None


def items_vs_gt(preds, cells):
    by = {c.item_label: c for c in cells}
    out, docs = [], []
    for lb, p in preds.items():
        c = by.get(lb)
        if c is None:
            continue
        out.append(study_a._metric_item(
            lb, study_a._pred_dist(p, c.gt_labels),
            ComplianceDistribution.from_values(c.gt_probs, c.gt_labels)))
        docs.append(c.document_id)
    return out, docs


def align_pairs(src, tgt, cells):
    """[(src_dist, tgt_dist)] on the shared labels -- the RATIO estimand's pair list."""
    by = {c.item_label: c for c in cells}
    keys = sorted(src.keys() & tgt.keys() & by.keys())
    return [(study_a._pred_dist(src[k], by[k].gt_labels),
             study_a._pred_dist(tgt[k], by[k].gt_labels)) for k in keys]


# ------------------------------------------------------------------- fitters
def grid_fit(pairs, metric, grid=None):
    """Objective-parametrized grid fit -- the SAME 60-pt grid study_a.fit_tau_oc uses,
    so W1 and RPS results differ only in the objective."""
    grid = grid or GRID
    if not pairs:
        return {"T": float("nan"), "movement_pct": float("nan")}

    def obj(T):
        return sum(metric(apply_temperature(p, T), q) for p, q in pairs) / len(pairs)

    best = min(grid, key=obj)
    o_best, o_one = obj(best), obj(1.0)
    return {"T": best, "obj_at_best": o_best, "obj_at_exact_T1": o_one,
            # <=0 means the exact-T=1 objective BEATS the grid optimum: the fit is
            # the null correction and the objective is non-identifying for T.
            "movement_pct": 100.0 * (o_one - o_best) / o_one if o_one > 0 else float("nan"),
            "saturated": study_a.saturated(best)}


def rel_fit(items, bins, grid=None):
    """argmin over the grid of Murphy reliability at a fixed bin count."""
    grid = grid or GRID
    return min(grid, key=lambda T: study_a._reliability(items, T, bins=bins))


def entropy_match_T(src_pairs, grid=None):
    """Objective-FREE temperature: the T making mean normalized predictive entropy of
    the source equal the target's.  No scoring rule, no binning, monotone in T, so the
    root is unique -- a well-conditioned reference point for every objective-contaminated
    quantity below.  Bisection on log T over study_a.T_BOUNDS."""
    if not src_pairs:
        return float("nan")
    target = statistics.mean(q.normalized_entropy() for _, q in src_pairs)

    def mean_h(T):
        return statistics.mean(apply_temperature(p, T).normalized_entropy() for p, _ in src_pairs)

    lo, hi = study_a.T_BOUNDS
    if mean_h(lo) > target:
        return lo
    if mean_h(hi) < target:
        return hi
    for _ in range(80):
        mid = math.exp((math.log(lo) + math.log(hi)) / 2)
        if mean_h(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.exp((math.log(lo) + math.log(hi)) / 2)


def entropy_match_vs_gt(preds, cells):
    """Absolute analog: T making the leg's mean normalized entropy match the GT's."""
    items, _ = items_vs_gt(preds, cells)
    return entropy_match_T([(it.prediction, it.ground_truth) for it in items])


def cluster_block(vals: dict, name: str) -> dict:
    """The registered Q3 clustering read (max/min <= 2 over the gate-passing set)."""
    finite = {k: v for k, v in vals.items() if isinstance(v, float) and math.isfinite(v) and v > 0}
    if len(finite) < 2:
        return {"estimand": name, "values": vals, "ratio": None, "rule_ok": None}
    s = sorted(finite.values())
    return {"estimand": name, "values": finite, "n": len(finite),
            "min": s[0], "max": s[-1], "ratio": s[-1] / s[0],
            "rule_ok": bool(s[-1] / s[0] <= CLUSTER_MAX_RATIO),
            "median": statistics.median(s),
            "any_saturated": any(study_a.saturated(v) for v in finite.values())}


def direction_symmetry(pre, post, cells) -> dict:
    """A temperature that measures a SHARPNESS RATIO inverts when the legs swap
    (fwd*rev ~ 1).  A hedging response to item-level disagreement does not."""
    out = {}
    for tag, (a, b) in (("post_to_pre", (post, pre)), ("pre_to_post", (pre, post))):
        pairs = align_pairs(a, b, cells)
        out[tag] = {"n": len(pairs),
                    "w1": grid_fit(pairs, wasserstein_1),
                    "rps": grid_fit(pairs, ranked_probability_score),
                    "mean_norm_H_source": statistics.mean(p.normalized_entropy() for p, _ in pairs),
                    "mean_norm_H_target": statistics.mean(q.normalized_entropy() for _, q in pairs)}
    for obj in ("w1", "rps"):
        f, r = out["post_to_pre"][obj]["T"], out["pre_to_post"][obj]["T"]
        out[f"{obj}_fwd_times_rev"] = f * r
    return out


# --------------------------------------------------- tau_DACA under any objective
def daca_fit(closed, reference, cells, metric=None) -> dict:
    """study_a.fit_tau_daca's agreement filter with a swappable alignment objective.

    metric=None  -> the fitter of record (judex.calibration.fit_temperature, RPS,
                    49-pt grid + golden refine).
    metric=W1/RPS -> the same filter, aligned on the 60-pt GRID under that metric,
                    so the W1 and RPS variants are directly comparable to each other
                    AND to fit_tau_oc (identical grid + objective machinery).
    """
    by = {c.item_label: c for c in cells}
    pairs, overlap, ref_correct = [], 0, 0
    for lb in closed.keys() & reference.keys():
        c = by.get(lb)
        if c is None:
            continue
        overlap += 1
        pc = study_a._pred_dist(closed[lb], c.gt_labels)
        pr = study_a._pred_dist(reference[lb], c.gt_labels)
        ref_correct += pr.argmax_index() == c.gt_argmax
        if pc.argmax_index() == pr.argmax_index():
            pairs.append((pc, pr))
    if not pairs:
        return {"n_overlap": overlap, "n_agreement": 0, "tau": float("nan"),
                "saturated": True, "reference_below_chance": ref_correct / max(overlap, 1) < 0.2}
    if metric is None:
        tau = fit_temperature(pairs, bounds=study_a.T_BOUNDS).temperature
        mv = float("nan")
    else:
        f = grid_fit(pairs, metric)
        tau, mv = f["T"], f["movement_pct"]
    return {"n_overlap": overlap, "n_agreement": len(pairs),
            "agreement_rate": len(pairs) / overlap, "tau": tau,
            "movement_pct": mv, "saturated": study_a.saturated(tau),
            "reference_below_chance": bool(ref_correct / overlap < 0.2)}


# ================================================================== the audit
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/objective_contamination_audit")
    ap.add_argument("--skip-a5-post", action="store_true")
    args = ap.parse_args()

    cells = aireg.load_cells()
    runs = REPO / "runs"
    eval_runs = Path(__file__).resolve().parents[2] / "judex-evaluator" / "runs"
    R = {"provenance": {
        "frozen_r3": {"T_J": T_J, "band": list(BAND), "F5_ln2": LN2,
                      "F7_range": list(DACA_RANGE), "Q3_cluster_max_ratio": CLUSTER_MAX_RATIO},
        "T_bounds": list(study_a.T_BOUNDS), "grid_points": len(GRID),
        "epsilon_study_b": EPSILON_B, "epsilon_study_a": None,
        "bin_sweep": list(BIN_SWEEP),
        "note": "REPORT-ONLY. r3 F1-F8 frozen; nothing adopted; seam stays mode: noop.",
    }}

    # ------------------------------------------------------ load every leg once
    A = {}
    for key, d, label, proto, role in STUDY_A:
        rd = runs / d
        pre, post = load_study_a_leg(rd, "pre"), load_study_a_leg(rd, "post")
        if pre and post:
            A[key] = {"label": label, "protocol": proto, "role": role,
                      "pre": pre, "post": post}
    B = {}
    for key, d, label, in_panel in STUDY_B:
        rd = runs / d
        pre, post = load_study_b_leg(rd, "pre"), load_study_b_leg(rd, "post")
        if pre and post:
            B[key] = {"label": label, "in_panel": in_panel, "pre": pre, "post": post}

    # =========================================================== A0 reproduction
    a0 = {"study_a_tau_oc_w1": {}, "study_a_base_resolution_bins10": {},
          "study_b_tau_v_w1": {}, "study_b_T_abs_post_rps": {},
          "study_b_T_abs_pre_rps": {}}
    for k, v in A.items():
        t = study_a.fit_tau_oc(v["post"], v["pre"], cells)
        v["tau_oc_w1"] = t
        if k in REF_TAU_OC:
            a0["study_a_tau_oc_w1"][k] = {"published": REF_TAU_OC[k], "recomputed": t,
                                          "match": abs(t - REF_TAU_OC[k]) < 5e-3}
        it_pre, _ = items_vs_gt(v["pre"], cells)
        it_post, _ = items_vs_gt(v["post"], cells)
        v["items_pre"], v["items_post"] = it_pre, it_post
        res = murphy_decomposition(it_pre, bins=10)["resolution"]
        v["base_resolution_b10"] = res
        if k in REF_BASE_RES:
            a0["study_a_base_resolution_bins10"][k] = {
                "published": REF_BASE_RES[k], "recomputed": round(res, 5),
                "match": abs(res - REF_BASE_RES[k]) < 5e-4}
    for k, v in B.items():
        v["tau_v_w1"] = study_a.fit_tau_oc(v["post"], v["pre"], cells)
        v["s_post"] = study_a.score_variant(v["post"], cells)
        v["s_pre"] = study_a.score_variant(v["pre"], cells)
        if k in REF_TAU_V:
            a0["study_b_tau_v_w1"][k] = {"published": REF_TAU_V[k], "recomputed": v["tau_v_w1"],
                                         "match": abs(v["tau_v_w1"] - REF_TAU_V[k]) < 5e-3}
            a0["study_b_T_abs_post_rps"][k] = {
                "published": REF_T_ABS_POST[k], "recomputed": v["s_post"]["T_rps"],
                "match": abs(v["s_post"]["T_rps"] - REF_T_ABS_POST[k]) < 5e-3}
            a0["study_b_T_abs_pre_rps"][k] = {
                "published": REF_T_ABS_PRE[k], "recomputed": v["s_pre"]["T_rps"],
                "match": abs(v["s_pre"]["T_rps"] - REF_T_ABS_PRE[k]) < 5e-3}
    a0["all_match"] = all(r["match"] for blk in a0.values() if isinstance(blk, dict)
                          for r in blk.values() if isinstance(r, dict) and "match" in r)
    R["A0_reproduction"] = a0

    # ============================== A1 -- CHECK 1: Study A Q3 under matched objectives
    a1 = {"per_family": {}}
    for k, v in A.items():
        pairs = align_pairs(v["post"], v["pre"], cells)
        w1 = grid_fit(pairs, wasserstein_1)
        rps = grid_fit(pairs, ranked_probability_score)
        # ratio estimand under Murphy reliability: post -> pre with pre in the GT seat
        by = {c.item_label: c for c in cells}
        it_ratio = [study_a._metric_item(lb, study_a._pred_dist(v["post"][lb], by[lb].gt_labels),
                                         study_a._pred_dist(v["pre"][lb], by[lb].gt_labels))
                    for lb in sorted(v["post"].keys() & v["pre"].keys() & by.keys())]
        rel = {str(b): rel_fit(it_ratio, b) for b in BIN_SWEEP}
        tH = entropy_match_T(pairs)
        a1["per_family"][k] = {
            "label": v["label"], "protocol": v["protocol"], "role": v["role"], "n_pairs": len(pairs),
            "tau_w1_of_record": w1["T"], "w1_movement_pct": w1["movement_pct"],
            "tau_rps": rps["T"], "rps_movement_pct": rps["movement_pct"],
            "tau_rel_by_bins": rel, "tau_entropy_match": tH,
            "w1_saturated": w1["saturated"], "rps_saturated": rps["saturated"],
            "direction_symmetry": direction_symmetry(v["pre"], v["post"], cells),
        }
    a1["clustering"] = {}
    for setname, keys in (("gate_passing_trio", Q3_TRIO), ("clean_four", Q3_CLEAN4)):
        sel = [k for k in keys if k in a1["per_family"]]
        blk = {}
        blk["W1_of_record"] = cluster_block({k: a1["per_family"][k]["tau_w1_of_record"] for k in sel},
                                            "tau_oc (ratio, W1) -- the fit of record")
        blk["RPS"] = cluster_block({k: a1["per_family"][k]["tau_rps"] for k in sel},
                                   "tau_oc (ratio, RPS)")
        for b in BIN_SWEEP:
            blk[f"reliability_bins{b}"] = cluster_block(
                {k: a1["per_family"][k]["tau_rel_by_bins"][str(b)] for k in sel},
                f"tau_oc (ratio, Murphy reliability, bins={b})")
        blk["entropy_match"] = cluster_block(
            {k: a1["per_family"][k]["tau_entropy_match"] for k in sel},
            "tau (ratio, entropy-matching -- objective-free)")
        a1["clustering"][setname] = blk
    a1["verdict"] = {
        "Q3_of_record": "NEGATIVE (W1 ratio 3.05 > 2 over the gate-passing trio)",
        "flips_under": [name for name, blk in a1["clustering"]["gate_passing_trio"].items()
                        if blk.get("rule_ok") is True],
        "holds_under": [name for name, blk in a1["clustering"]["gate_passing_trio"].items()
                        if blk.get("rule_ok") is False],
    }
    R["A1_check1_study_a_Q3"] = a1

    # ================================= A2 -- CHECK 2: tau_DACA under matched objectives
    a2 = {"cross_family": {}, "own_family": {}}
    for tgt in PANEL:
        if tgt not in B:
            continue
        for ref in PANEL:
            if ref not in B:
                continue
            row = {"rps_of_record": daca_fit(B[tgt]["post"], B[ref]["pre"], cells, None),
                   "rps_grid": daca_fit(B[tgt]["post"], B[ref]["pre"], cells, ranked_probability_score),
                   "w1_grid": daca_fit(B[tgt]["post"], B[ref]["pre"], cells, wasserstein_1)}
            (a2["own_family"] if tgt == ref else a2["cross_family"])[f"{tgt}<-{ref}"] = row
    cross_rps = [r["rps_of_record"]["tau"] for r in a2["cross_family"].values()
                 if not r["rps_of_record"]["saturated"]]
    cross_w1 = [r["w1_grid"]["tau"] for r in a2["cross_family"].values()
                if not r["w1_grid"]["saturated"]]
    a2["summary"] = {
        "n_cross": len(a2["cross_family"]),
        "rps": {"n_valid": len(cross_rps), "min": min(cross_rps), "max": max(cross_rps),
                "median": statistics.median(cross_rps)},
        "w1": {"n_valid": len(cross_w1), "min": min(cross_w1), "max": max(cross_w1),
               "median": statistics.median(cross_w1)},
    }
    # Window anchors: the W1 anchor is T_J (record); the RPS anchor is the panel median
    # RPS-fitted ratio; the entropy anchor is the panel median entropy-matched ratio.
    anchors = {"W1_of_record_T_J": T_J,
               "RPS_matched": statistics.median(
                   [grid_fit(align_pairs(B[k]["post"], B[k]["pre"], cells),
                             ranked_probability_score)["T"] for k in PANEL if k in B]),
               "entropy_matched": statistics.median(
                   [entropy_match_T(align_pairs(B[k]["post"], B[k]["pre"], cells))
                    for k in PANEL if k in B])}
    a2["window_anchors"] = anchors
    a2["verdict_matrix"] = {}
    for fit_name, vals in (("fit_RPS", cross_rps), ("fit_W1", cross_w1)):
        for anc_name, anc in anchors.items():
            lo, hi = anc / 2.0, anc * 2.0
            inside = [t for t in vals if lo <= t <= hi]
            a2["verdict_matrix"][f"{fit_name}__window_{anc_name}"] = {
                "anchor": anc, "window": [lo, hi], "n_valid": len(vals),
                "n_inside": len(inside), "all_inside": len(inside) == len(vals),
                "F7_corroborates": bool(len(vals) >= 3 and len(inside) == len(vals)),
                "fit_range": [min(vals), max(vals)]}
    # Why the objectives agree HERE but not elsewhere: the agreement filter removes the
    # item-level disagreement that drives RPS's hedging response, so on the filtered
    # subpopulation W1 and RPS see nearly the same thing -- and BOTH objectives move very
    # little (near-non-identifying), which is what makes a factor-2 window easy to satisfy.
    a2["objective_movement_on_filtered_subpopulation"] = {
        "w1_movement_pct": {k: r["w1_grid"]["movement_pct"] for k, r in a2["cross_family"].items()},
        "rps_movement_pct": {k: r["rps_grid"]["movement_pct"] for k, r in a2["cross_family"].items()},
        "w1_movement_pct_max": max(r["w1_grid"]["movement_pct"] for r in a2["cross_family"].values()),
        "rps_movement_pct_max": max(r["rps_grid"]["movement_pct"] for r in a2["cross_family"].values()),
        "max_abs_ln_ratio_w1_vs_rps": max(
            abs(math.log(r["w1_grid"]["tau"] / r["rps_grid"]["tau"])) for r in a2["cross_family"].values()),
    }
    a2["matched_pairs"] = {
        "fit_W1__window_W1_of_record_T_J": a2["verdict_matrix"]["fit_W1__window_W1_of_record_T_J"]["F7_corroborates"],
        "fit_RPS__window_RPS_matched": a2["verdict_matrix"]["fit_RPS__window_RPS_matched"]["F7_corroborates"],
        "mixed_of_record__fit_RPS__window_W1": a2["verdict_matrix"]["fit_RPS__window_W1_of_record_T_J"]["F7_corroborates"],
    }
    R["A2_check2_tau_daca"] = a2

    # ============================ A3 -- CHECK 3: arm 3b dispersion pool commensurability
    a3 = {}
    probe = fit_dispersion_temperature(
        [(study_a._pred_dist(B["qwen"]["post"][lb], c.gt_labels),
          [study_a._pred_dist(B["qwen"]["pre"][lb], c.gt_labels)])
         for lb, c in [(l, {cc.item_label: cc for cc in cells}[l])
                       for l in sorted(B["qwen"]["post"].keys() & B["qwen"]["pre"].keys()
                                       & {cc.item_label for cc in cells})]],
        bounds=study_a.T_BOUNDS)
    a3["fitter_objective_declared"] = {"objective": probe.objective,
                                       "target_metric": probe.target_metric}
    by = {c.item_label: c for c in cells}
    # (i) reproduce the pool fit on the panel BASE legs, closed pair in the reported seat
    closed = load_closed_run(eval_runs)
    if closed:
        labels = [lb for lb in closed if lb in by and all(lb in B[f]["pre"] for f in PANEL)]
        pool_pairs = [(study_a._pred_dist(closed[lb], by[lb].gt_labels),
                       [study_a._pred_dist(B[f]["pre"][lb], by[lb].gt_labels) for f in PANEL])
                      for lb in labels]
        fit = fit_dispersion_temperature(pool_pairs, bounds=study_a.T_BOUNDS)
        a3["executed_sweep_pool"] = {
            "n": len(labels), "T_raw": fit.temperature, "objective": fit.objective,
            "claimed_entropy_at_unit": fit.claimed_entropy_at_unit,
            "realized_entropy_mean": fit.realized_entropy_mean,
            "in_W1_band_of_record": bool(BAND[0] <= fit.temperature <= BAND[1])}
    # (ii) matched-objective band: the SAME entropy-consistency estimator applied to the
    #      panel's own post->base ratio, i.e. the objective the band edges would have had
    #      if F3 had been derived under arm 3b's objective instead of W1.
    ent_band, entmatch_band = {}, {}
    for k in PANEL:
        if k not in B:
            continue
        labs = sorted(B[k]["post"].keys() & B[k]["pre"].keys() & set(by))
        f = fit_dispersion_temperature(
            [(study_a._pred_dist(B[k]["post"][lb], by[lb].gt_labels),
              [study_a._pred_dist(B[k]["pre"][lb], by[lb].gt_labels)]) for lb in labs],
            bounds=study_a.T_BOUNDS)
        ent_band[k] = f.temperature
        entmatch_band[k] = entropy_match_T(align_pairs(B[k]["post"], B[k]["pre"], cells))
    a3["matched_objective_band_entropy_consistency"] = cluster_block(
        ent_band, "tau (ratio, entropy-consistency == arm3b's own objective)")
    a3["matched_objective_band_entropy_match"] = cluster_block(
        entmatch_band, "tau (ratio, entropy-matching -- objective-free)")
    # (iii) FULLY matched band: same objective AND same reference structure as arm 3b --
    #       each panel POST leg fit against the identical 4-base pool the closed pair is
    #       fit against.  This is the band F3 would have carried had it been derived under
    #       arm 3b's own instrument; the (ii) bands match only the objective.
    pool_band = {}
    for k in PANEL:
        if k not in B:
            continue
        labs = [lb for lb in B[k]["post"] if lb in by and all(lb in B[f]["pre"] for f in PANEL)]
        if not labs:
            continue
        f = fit_dispersion_temperature(
            [(study_a._pred_dist(B[k]["post"][lb], by[lb].gt_labels),
              [study_a._pred_dist(B[g]["pre"][lb], by[lb].gt_labels) for g in PANEL])
             for lb in labs], bounds=study_a.T_BOUNDS)
        pool_band[k] = f.temperature
    a3["matched_band_same_pool_reference"] = cluster_block(
        pool_band, "T (panel POST vs the SAME 4-base pool, entropy-consistency) "
                   "-- objective AND reference matched to arm 3b")
    for name, blk in (("entropy_consistency", a3["matched_objective_band_entropy_consistency"]),
                      ("entropy_match", a3["matched_objective_band_entropy_match"]),
                      ("same_pool_reference", a3["matched_band_same_pool_reference"])):
        lo, hi = blk["min"], blk["max"]
        a3[f"in_band_under_{name}"] = {
            "band": [lo, hi],
            "T_raw_prototype_1.278": bool(lo <= 1.2780628080769258 <= hi),
            "T_raw_executed": (bool(lo <= a3["executed_sweep_pool"]["T_raw"] <= hi)
                               if "executed_sweep_pool" in a3 else None)}
    R["A3_check3_dispersion_pool"] = a3

    # ================ A4 -- CHECK 4: the 1/6 decomposition in judex_paper_v5
    a4 = {"per_family": {}}
    for k in PANEL:
        if k not in B:
            continue
        pairs = align_pairs(B[k]["post"], B[k]["pre"], cells)
        tau_w1 = B[k]["tau_v_w1"]
        tau_rps = grid_fit(pairs, ranked_probability_score)["T"]
        tau_H = entropy_match_T(pairs)
        Tpost_rps, Tpre_rps = B[k]["s_post"]["T_rps"], B[k]["s_pre"]["T_rps"]
        it_post, _ = items_vs_gt(B[k]["post"], cells)
        it_pre, _ = items_vs_gt(B[k]["pre"], cells)
        Tpost_w1 = grid_fit([(i.prediction, i.ground_truth) for i in it_post], wasserstein_1)
        Tpre_w1 = grid_fit([(i.prediction, i.ground_truth) for i in it_pre], wasserstein_1)
        a4["per_family"][k] = {
            "label": B[k]["label"],
            "tau_w1": tau_w1, "tau_rps": tau_rps, "tau_entropy": tau_H,
            "T_abs_post_rps": Tpost_rps, "T_abs_pre_rps": Tpre_rps,
            "T_abs_post_w1": Tpost_w1["T"], "T_abs_post_w1_movement_pct": Tpost_w1["movement_pct"],
            "T_abs_pre_w1": Tpre_w1["T"], "T_abs_pre_w1_movement_pct": Tpre_w1["movement_pct"],
            "T_abs_post_entropy": entropy_match_vs_gt(B[k]["post"], cells),
            "T_abs_pre_entropy": entropy_match_vs_gt(B[k]["pre"], cells),
            # The multiplicative identity the decomposition PRESUPPOSES:
            #   T_abs(post) == tau * T_abs(pre)   (temperatures compose)
            "composition_residual_ln_mixed": math.log(Tpost_rps) - math.log(tau_w1 * Tpre_rps),
            "composition_residual_ln_all_rps": math.log(Tpost_rps) - math.log(tau_rps * Tpre_rps),
            "composition_residual_ln_entropy": (
                math.log(entropy_match_vs_gt(B[k]["post"], cells))
                - math.log(tau_H * entropy_match_vs_gt(B[k]["pre"], cells))),
        }
    P = a4["per_family"]

    def med(f):
        return statistics.median([f(v) for v in P.values()])

    variants = {}
    # (a) the claim as published: W1 numerator over an RPS denominator
    variants["published_mixed_W1_over_RPS"] = {
        "increment_T": med(lambda v: v["tau_w1"]), "total_T": med(lambda v: v["T_abs_post_rps"]),
        "increment_objective": "W1", "total_objective": "RPS"}
    # (b) matched all-RPS
    variants["matched_all_RPS"] = {
        "increment_T": med(lambda v: v["tau_rps"]), "total_T": med(lambda v: v["T_abs_post_rps"]),
        "increment_objective": "RPS", "total_objective": "RPS"}
    # (c) matched all-W1
    variants["matched_all_W1"] = {
        "increment_T": med(lambda v: v["tau_w1"]), "total_T": med(lambda v: v["T_abs_post_w1"]),
        "increment_objective": "W1", "total_objective": "W1"}
    # (d) objective-free entropy matching
    variants["objective_free_entropy_match"] = {
        "increment_T": med(lambda v: v["tau_entropy"]),
        "total_T": med(lambda v: v["T_abs_post_entropy"]),
        "increment_objective": "entropy-match", "total_objective": "entropy-match"}
    for name, d in variants.items():
        inc, tot = d["increment_T"], d["total_T"]
        d["increment_ln"] = math.log(inc)
        d["total_ln"] = math.log(tot)
        d["share"] = math.log(inc) / math.log(tot) if tot > 0 and abs(math.log(tot)) > 1e-12 else float("nan")
        d["one_over_share"] = 1.0 / d["share"] if d["share"] and math.isfinite(d["share"]) and d["share"] != 0 else float("nan")
    a4["share_variants"] = variants
    a4["published_claim"] = REF_PAPER
    # Is the post-training increment distinguishable from ZERO at a matched objective?
    # Doc-clustered bootstrap (24 AIReg documents, the real cluster) on the panel-median
    # entropy-matched ratio -- if the CI on ln(increment) covers 0 there is no increment
    # to take a share OF, and the whole decomposition is undefined, not merely mis-scaled.
    import random as _random
    by_lab = {c.item_label: c for c in cells}
    fam_pairs = {}
    for k in PANEL:
        if k not in B:
            continue
        tagged = []
        for lb in sorted(B[k]["post"].keys() & B[k]["pre"].keys() & set(by_lab)):
            c = by_lab[lb]
            tagged.append((c.document_id,
                           study_a._pred_dist(B[k]["post"][lb], c.gt_labels),
                           study_a._pred_dist(B[k]["pre"][lb], c.gt_labels),
                           ComplianceDistribution.from_values(c.gt_probs, c.gt_labels)))
        fam_pairs[k] = tagged
    docs = sorted({t[0] for tg in fam_pairs.values() for t in tg})
    rng = _random.Random(20260720)
    inc_boot, tot_boot, share_boot = [], [], []
    for _ in range(600):
        pick = [docs[rng.randrange(len(docs))] for _ in docs]
        incs, tots = [], []
        for k, tg in fam_pairs.items():
            sel = [t for d in pick for t in tg if t[0] == d]
            if len(sel) < 10:
                continue
            incs.append(entropy_match_T([(t[1], t[2]) for t in sel]))
            tots.append(entropy_match_T([(t[1], t[3]) for t in sel]))
        if len(incs) < 3:
            continue
        mi, mt = statistics.median(incs), statistics.median(tots)
        inc_boot.append(math.log(mi))
        tot_boot.append(math.log(mt))
        if abs(math.log(mt)) > 1e-9:
            share_boot.append(math.log(mi) / math.log(mt))

    def ci(v):
        v = sorted(v)
        return {"lo": v[int(0.025 * len(v))], "hi": v[int(0.975 * len(v)) - 1],
                "median": statistics.median(v), "n_reps": len(v)}

    a4["entropy_matched_bootstrap"] = {
        "n_docs": len(docs), "ln_increment": ci(inc_boot), "ln_total": ci(tot_boot),
        "share": ci(share_boot),
        "increment_ci_covers_zero": ci(inc_boot)["lo"] <= 0.0 <= ci(inc_boot)["hi"]}
    a4["composition_identity"] = {
        "statement": "the decomposition presupposes T_abs(post) = tau * T_abs(pre)",
        "max_abs_residual_ln_mixed": max(abs(v["composition_residual_ln_mixed"]) for v in P.values()),
        "max_abs_residual_ln_all_rps": max(abs(v["composition_residual_ln_all_rps"]) for v in P.values()),
        "max_abs_residual_ln_entropy": max(abs(v["composition_residual_ln_entropy"]) for v in P.values()),
        "implied_base_requirement_from_paper_numbers": REF_PAPER["T_abs_post_median"] / T_J,
        "measured_panel_median_T_abs_pre_rps": med(lambda v: v["T_abs_pre_rps"]),
    }
    R["A4_check4_one_sixth_decomposition"] = a4

    # ============= A5 -- CLASS C: capability-gate resolutions across the bin sweep
    a5 = {"base_legs": {}, "post_legs": {}}
    for k, v in A.items():
        for leg, itemkey, out in (("pre", "items_pre", a5["base_legs"]),
                                  ("post", "items_post", a5["post_legs"])):
            if args.skip_a5_post and leg == "post":
                continue
            items = v[itemkey]
            row = {"label": v["label"], "protocol": v["protocol"], "role": v["role"],
                   "n": len(items),
                   "argmax_acc": sum(i.argmax_agreement for i in items) / len(items),
                   "resolution_by_bins": {}, "reliability_by_bins": {},
                   "resolution_over_uncertainty_by_bins": {}, "n_groups_by_bins": {}}
            for b in BIN_SWEEP:
                m = murphy_decomposition(items, bins=b)
                row["resolution_by_bins"][str(b)] = m["resolution"]
                row["reliability_by_bins"][str(b)] = m["reliability"]
                row["resolution_over_uncertainty_by_bins"][str(b)] = m["resolution"] / m["uncertainty"]
                row["n_groups_by_bins"][str(b)] = len(calibration_groups(items, b))
            vals = list(row["resolution_by_bins"].values())
            row["resolution_bin_ratio_max_over_min"] = max(vals) / min(vals)
            out[k] = row
    # the decision read: does the pass/fail SEPARATION survive at every bin count?
    passing = [k for k, v in a5["base_legs"].items() if v["role"] == "panel_gate_pass"]
    failing = [k for k, v in a5["base_legs"].items() if v["role"] == "gate_fail"]
    sep = {}
    for b in BIN_SWEEP:
        p = {k: a5["base_legs"][k]["resolution_by_bins"][str(b)] for k in passing}
        f = {k: a5["base_legs"][k]["resolution_by_bins"][str(b)] for k in failing}
        sep[str(b)] = {
            "min_passing": min(p.values()), "min_passing_family": min(p, key=p.get),
            "max_failing": max(f.values()), "max_failing_family": max(f, key=f.get),
            "gap_absolute": min(p.values()) - max(f.values()),
            "gap_ratio": min(p.values()) / max(f.values()),
            "separated": bool(min(p.values()) > max(f.values())),
            "passing": p, "failing": f}
    a5["separation_by_bins"] = sep
    a5["verdict"] = {
        "separated_at_every_bin_count": all(s["separated"] for s in sep.values()),
        "min_gap_ratio_over_bins": min(s["gap_ratio"] for s in sep.values()),
        "any_call_flips": not all(s["separated"] for s in sep.values()),
        "argmax_leg_independent_check": {
            k: {"argmax_acc": a5["base_legs"][k]["argmax_acc"],
                "vs_majority_floor_0.333": a5["base_legs"][k]["argmax_acc"] - 1 / 3,
                "role": a5["base_legs"][k]["role"]}
            for k in list(passing) + list(failing)},
    }
    R["A5_classC_capability_gate_bins"] = a5

    # ============== A6 -- W1 identification, logit channel vs verbalized channel
    a6 = {"logit_channel_study_a": {}, "verbalized_channel_study_b": {}}
    for k, v in A.items():
        f = a1["per_family"][k]
        a6["logit_channel_study_a"][k] = {
            "label": v["label"], "role": v["role"],
            "w1_movement_pct": f["w1_movement_pct"], "rps_movement_pct": f["rps_movement_pct"],
            "w1_fwd_times_rev": f["direction_symmetry"]["w1_fwd_times_rev"],
            "rps_fwd_times_rev": f["direction_symmetry"]["rps_fwd_times_rev"],
            "tau_w1": f["tau_w1_of_record"], "tau_rps": f["tau_rps"]}
    for k, v in B.items():
        pairs = align_pairs(v["post"], v["pre"], cells)
        w1, rps = grid_fit(pairs, wasserstein_1), grid_fit(pairs, ranked_probability_score)
        ds = direction_symmetry(v["pre"], v["post"], cells)
        a6["verbalized_channel_study_b"][k] = {
            "label": v["label"], "in_panel": v["in_panel"],
            "w1_movement_pct": w1["movement_pct"], "rps_movement_pct": rps["movement_pct"],
            "w1_fwd_times_rev": ds["w1_fwd_times_rev"], "rps_fwd_times_rev": ds["rps_fwd_times_rev"],
            "tau_w1": w1["T"], "tau_rps": rps["T"]}
    a6["summary"] = {
        "study_a_w1_movement_pct_range": [
            min(v["w1_movement_pct"] for v in a6["logit_channel_study_a"].values()),
            max(v["w1_movement_pct"] for v in a6["logit_channel_study_a"].values())],
        "study_b_w1_movement_pct_range": [
            min(v["w1_movement_pct"] for v in a6["verbalized_channel_study_b"].values()),
            max(v["w1_movement_pct"] for v in a6["verbalized_channel_study_b"].values())],
        "study_a_n_negative_w1_movement": sum(
            v["w1_movement_pct"] <= 0 for v in a6["logit_channel_study_a"].values()),
        "study_b_n_negative_w1_movement": sum(
            v["w1_movement_pct"] <= 0 for v in a6["verbalized_channel_study_b"].values()),
    }
    R["A6_w1_identification_by_channel"] = a6

    # ------------------------------------------------------------------- write
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "objective_contamination_audit.json").write_text(json.dumps(R, indent=2, default=str))

    # ---------------------------------------------------------------- console
    print(f"\n=== A0 reproduction: all_match={a0['all_match']}")
    for blk_name, blk in a0.items():
        if not isinstance(blk, dict):
            continue
        bad = {k: v for k, v in blk.items() if isinstance(v, dict) and not v.get("match", True)}
        if bad:
            print(f"  MISMATCH in {blk_name}: {bad}")

    print("\n=== A1 CHECK 1 — Study A tau_oc under matched objectives (logit channel)")
    print(f"{'family':22} {'role':17} {'W1 (rec)':>9} {'W1 mv%':>8} {'RPS':>8} {'RPSmv%':>8} "
          f"{'rel b3':>8} {'rel b10':>8} {'entH':>7}")
    for k, f in a1["per_family"].items():
        print(f"{f['label']+' '+f['protocol']:22} {f['role']:17} {f['tau_w1_of_record']:9.3f} "
              f"{f['w1_movement_pct']:8.2f} {f['tau_rps']:8.3f} {f['rps_movement_pct']:8.2f} "
              f"{f['tau_rel_by_bins']['3']:8.3f} {f['tau_rel_by_bins']['10']:8.3f} "
              f"{f['tau_entropy_match']:7.3f}")
    for setname, blk in a1["clustering"].items():
        print(f"  -- clustering over {setname} (rule: max/min <= {CLUSTER_MAX_RATIO})")
        for name, c in blk.items():
            if c.get("ratio") is None:
                continue
            print(f"     {name:24} ratio {c['ratio']:8.3f}  rule_ok={str(c['rule_ok']):5}  "
                  f"median {c['median']:7.3f}  [{c['min']:.3f}, {c['max']:.3f}]")
    print(f"  VERDICT: Q3 flips (clusters) under {a1['verdict']['flips_under']}; "
          f"holds (does not cluster) under {a1['verdict']['holds_under']}")

    print("\n=== A2 CHECK 2 — tau_DACA fit objective x window anchor")
    s = a2["summary"]
    print(f"  cross-family fits (n={s['n_cross']}): RPS median {s['rps']['median']:.3f} "
          f"[{s['rps']['min']:.3f}, {s['rps']['max']:.3f}]  |  "
          f"W1 median {s['w1']['median']:.3f} [{s['w1']['min']:.3f}, {s['w1']['max']:.3f}]")
    print(f"  window anchors: " + "  ".join(f"{k}={v:.3f}" for k, v in a2["window_anchors"].items()))
    print(f"{'combination':46} {'window':>22} {'inside':>8} {'F7':>6}")
    for name, d in a2["verdict_matrix"].items():
        print(f"{name:46} [{d['window'][0]:7.3f},{d['window'][1]:7.3f}] "
              f"{d['n_inside']:3d}/{d['n_valid']:<4d} {'CORROB' if d['F7_corroborates'] else 'NO':>6}")

    print("\n=== A3 CHECK 3 — dispersion pool")
    print(f"  fit_dispersion_temperature objective = {a3['fitter_objective_declared']}")
    if "executed_sweep_pool" in a3:
        e = a3["executed_sweep_pool"]
        print(f"  executed sweep T_raw = {e['T_raw']:.4f}  (in W1 band of record: {e['in_W1_band_of_record']})")
    for name, key in (("entropy_consistency", "matched_objective_band_entropy_consistency"),
                      ("entropy_match", "matched_objective_band_entropy_match"),
                      ("same_pool_reference", "matched_band_same_pool_reference")):
        blk = a3[key]
        ib = a3[f"in_band_under_{name}"]
        print(f"  matched band ({name}): [{blk['min']:.3f}, {blk['max']:.3f}] ratio {blk['ratio']:.3f}"
              f"  -> prototype in band: {ib['T_raw_prototype_1.278']}, executed in band: {ib['T_raw_executed']}")

    print("\n=== A4 CHECK 4 — the one-sixth decomposition")
    print(f"{'family':22} {'tau_W1':>7} {'tau_RPS':>8} {'tau_H':>7} {'Tpost_RPS':>10} "
          f"{'Tpost_W1':>9} {'Tpost_H':>8} {'compResid':>10}")
    for k, v in P.items():
        print(f"{v['label']:22} {v['tau_w1']:7.3f} {v['tau_rps']:8.3f} {v['tau_entropy']:7.3f} "
              f"{v['T_abs_post_rps']:10.3f} {v['T_abs_post_w1']:9.3f} {v['T_abs_post_entropy']:8.3f} "
              f"{v['composition_residual_ln_mixed']:+10.3f}")
    print(f"{'variant':34} {'increment':>10} {'total':>8} {'ln inc':>8} {'ln tot':>8} "
          f"{'share':>8} {'1/share':>9}")
    for name, d in a4["share_variants"].items():
        print(f"{name:34} {d['increment_T']:10.3f} {d['total_T']:8.3f} {d['increment_ln']:8.3f} "
              f"{d['total_ln']:8.3f} {d['share']:8.3f} {d['one_over_share']:9.2f}")
    ci = a4["composition_identity"]
    print(f"  composition identity max |residual| (ln): mixed {ci['max_abs_residual_ln_mixed']:.3f}  "
          f"all-RPS {ci['max_abs_residual_ln_all_rps']:.3f}  entropy {ci['max_abs_residual_ln_entropy']:.3f}")
    print(f"  paper's numbers imply base needs {ci['implied_base_requirement_from_paper_numbers']:.3f}; "
          f"measured panel median T_abs(pre) = {ci['measured_panel_median_T_abs_pre_rps']:.3f}")
    eb = a4["entropy_matched_bootstrap"]
    print(f"  entropy-matched increment ln: {eb['ln_increment']['median']:+.3f} "
          f"CI [{eb['ln_increment']['lo']:+.3f}, {eb['ln_increment']['hi']:+.3f}] "
          f"covers 0: {eb['increment_ci_covers_zero']}  (share CI "
          f"[{eb['share']['lo']:.2f}, {eb['share']['hi']:.2f}])")

    print("\n=== A5 CLASS C — capability-gate base resolutions across bins")
    print(f"{'family':24} {'role':17} {'argmax':>7} " + " ".join(f"{'b'+str(b):>8}" for b in BIN_SWEEP)
          + f" {'binratio':>9}")
    for k, v in sorted(a5["base_legs"].items(), key=lambda kv: -kv[1]["resolution_by_bins"]["10"]):
        print(f"{v['label']+' '+v['protocol']:24} {v['role']:17} {v['argmax_acc']:7.3f} "
              + " ".join(f"{v['resolution_by_bins'][str(b)]:8.4f}" for b in BIN_SWEEP)
              + f" {v['resolution_bin_ratio_max_over_min']:9.3f}")
    print(f"{'bins':>6} {'min passing':>12} {'max failing':>12} {'gap':>9} {'ratio':>7} {'separated':>10}")
    for b in BIN_SWEEP:
        d = sep[str(b)]
        print(f"{b:6d} {d['min_passing']:12.4f} {d['max_failing']:12.4f} {d['gap_absolute']:9.4f} "
              f"{d['gap_ratio']:7.3f} {str(d['separated']):>10}")
    print(f"  VERDICT: separated at every bin count = {a5['verdict']['separated_at_every_bin_count']}; "
          f"min gap ratio = {a5['verdict']['min_gap_ratio_over_bins']:.3f}; "
          f"any call flips = {a5['verdict']['any_call_flips']}")

    print("\n=== A6 W1 identification by channel (objective movement vs exact T=1)")
    su = a6["summary"]
    print(f"  Study A (logit)      W1 movement % range {su['study_a_w1_movement_pct_range'][0]:+.2f} .. "
          f"{su['study_a_w1_movement_pct_range'][1]:+.2f}   "
          f"({su['study_a_n_negative_w1_movement']}/{len(a6['logit_channel_study_a'])} non-positive)")
    print(f"  Study B (verbalized) W1 movement % range {su['study_b_w1_movement_pct_range'][0]:+.2f} .. "
          f"{su['study_b_w1_movement_pct_range'][1]:+.2f}   "
          f"({su['study_b_n_negative_w1_movement']}/{len(a6['verbalized_channel_study_b'])} non-positive)")

    print(f"\nwrote {out / 'objective_contamination_audit.json'}")


if __name__ == "__main__":
    main()
