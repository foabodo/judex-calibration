#!/usr/bin/env python3
"""Reliability-objective probe: is r3 arm 2's F5 non-concurrence objective-driven,
estimand-driven, or both?

Motivating observation (2026-07-21).  On the executed 24-doc closed-pair sweep
(``stage9-onpair-e6-20260721``) the r3 validation arm fits, by MINIMIZING RPS,

    T*_rps = 2.5446   =>  |ln(T*_rps / T_J)| = 0.7916 > ln 2  =>  F5 FAILS

but the same run's RELIABILITY-optimized fit is

    T*_rel = 1.8574   =>  |ln(T*_rel / T_J)| = 0.4767 < ln 2  =>  F5 would CONCUR.

Since the system's own acceptance criterion F4 is reliability-led ("reliability
strictly better AND RPS no worse AND resolution within +-10%"), an RPS-optimized
validation arm may be the mismatched instrument.  This script tests whether the
reliability-optimized fit is a real, transferable estimand or a fragile artifact.

Probes
  P0  regression check: reproduce the published closed-run reference values.
  P1  panel legs (pre and post) -- T_rps and T_rel for every Study B family, at
      bins in {3, 5, 10, 15, 20} and on two search grids (study_a's 60-point GRID
      and a dense 601-point grid), so bin-count and grid-resolution effects are
      separated.
  P2  closed pair -- same sweep of T_rel, plus the whole reliability(T) curve.
  P3  closed pair -- doc-clustered bootstrap (24 document clusters, 2000 reps,
      seed 20260720) on BOTH objectives, computed by one mechanism so the two CIs
      are comparable.
  P4  does the reliability estimand transfer?  Panel max/min clustering ratio,
      leave-one-family-out transfer error |ln(transferred/own)| (the exact
      convention of ``verbalized_reframe_r0.lofo_transfer``: transferred =
      median of the other three), and saturation -- head to head against the two
      known estimands (tau_v, T_abs_rps).
  P5  apply and compare: ``study_a.closed_side_check`` at six temperatures with
      the F4 verdict at each, plus the same read recomputed at bins=3 to expose
      F4's own bin sensitivity.
  P6  additive log-scale decomposition of the F5 gap into an OBJECTIVE term and
      an ESTIMAND term, routed through the panel (the closed pair has no base
      leg, so no ratio estimand exists for it directly).
  P7  what does each objective actually identify?  A direction-symmetry test
      (fit post->pre and pre->post under both W1 and RPS: a genuine sharpness
      ratio must invert, a disagreement-hedging response must not), a
      self-alignment sanity fit, how far the frozen tau_v objective actually
      moves, and the objective-free entropy-matching temperature T_H for the
      closed pair.

Structural note that P2/P4 quantify empirically.  ``murphy_decomposition``
groups items by (predicted argmax level, argmax-confidence bin), so it has at
most 5*bins groups.  As bins grows the groups become singletons, and for
singleton groups group_pred == pred and group_gt == gt, hence

    reliability -> mean RPS   and   resolution -> uncertainty   as bins -> inf.

So T_rel is not a different objective in the limit -- it CONVERGES to T_rps.
The size of the "objective effect" is therefore a function of the bin count,
which is a free parameter no frozen constant pins.  Note also the live
inconsistency this probe was written to check: ``study_a._reliability`` (the
grid target that produces T_rel) uses bins=3, while ``study_a.closed_side_check``
(the F4 acceptance read) and ``score_variant``'s reported Murphy block both use
bins=10.

$0 -- analysis-only on existing artifacts.  REPORT-ONLY: r3 F1-F8 are frozen,
nothing here is adopted, no temperature is written to any config, the evaluator
seam stays ``mode: noop``.

Run from judex-calibration:
  JUDEX_UMBRELLA=/path/to/judex PYTHONPATH=src \
      /Users/fabodo/anaconda3/envs/judex-arm/bin/python \
      scripts/reliability_objective_probe.py [--out runs/reliability_objective_probe]
  ... --no-bootstrap   skips P3 (the only slow probe, ~2-4 min)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from judex_calibration import aireg, study_a  # noqa: E402
from judex.calibration import fit_temperature, apply_temperature  # noqa: E402
from judex.core.distributions import ComplianceDistribution  # noqa: E402
from judex.core.metrics import wasserstein_1, ranked_probability_score  # noqa: E402
from judex.metrics_report import murphy_decomposition  # noqa: E402

from e6_r3_arms import load_closed  # canonical closed-run loader (long-form label map)  # noqa: E402

# ------------------------------------------------------------------ constants
EPSILON = 0.005              # Study B analysis convention (floor then renormalize)
T_J = 1.153                  # r3 F2 (FROZEN -- tested against, never edited)
BAND = (1.0251785151221313, 1.379820350674421)   # r3 F3 (FROZEN)
LN2 = math.log(2.0)          # r3 F5 concurrence threshold
RESOLUTION_TOL = 0.10        # r3 F4
CLOSED_RUN = "stage9-onpair-e6-20260721"
BOOT_SEED = 20260720
BOOT_N = 2000
BIN_SWEEP = (3, 5, 10, 15, 20)
LIMIT_BINS = (30, 50, 100, 200, 500, 1000)   # outside the sweep: the bins -> inf limit check
DENSE_N = 601                # dense log grid over T_BOUNDS for the refinement check
BOOT_GRID_N = 241            # coarser dense grid for the bootstrap (1.8% log spacing)

# Published reference values to reproduce (P0).
REF = {
    "uncalibrated_reliability": 0.02092,
    "uncalibrated_resolution": 0.04916,
    "uncalibrated_mean_rps": 0.03350,
    "T_J_reliability_improvement": 0.00151,
    "T_J_rps_change": -0.00218,
    "T_2353_reliability_improvement": 0.00772,
    "T_2353_rps_change": -0.00650,
    "T_2353_resolution_change": -0.00422,
    "T_star_rps": 2.544640471713656,
}

# Known estimands, from the record (spec/analysis_2026_07_21_absolute_vs_ratio_estimand.md
# and docs/verbalized_r0_analyses_2026_07_21.md).  Recomputed here, not trusted.
KNOWN_TAU_V = {"qwen": 1.281, "gemma31": 1.025, "glm": 1.025, "maverick": 1.380}
KNOWN_T_ABS_RPS = {"qwen": 2.637, "gemma31": 2.242, "glm": 2.464, "maverick": 2.035}

# (key, run dir, label, in adoption panel, note)
FAMILIES = [
    ("qwen",     "study_b_qwen",        "Qwen3.5-35B-A3B",   True,  ""),
    ("gemma31",  "study_b_gemma31_api", "Gemma-4-31B",       True,  "chat-API collection"),
    ("glm",      "study_b_glm",         "GLM-4.5",           True,  ""),
    ("maverick", "study_b_maverick",    "Llama-4-Maverick",  True,  ""),
    ("llama31",  "study_b_llama31",     "Llama-3.1-405B",    False, "non-panel (excluded pre-hoc)"),
    ("gemma26",  "study_b_gemma26_api", "Gemma-4-26B-A4B",   False, "non-panel; capability-gate FAIL"),
]


def log_grid(n: int) -> list:
    lo, hi = study_a.T_BOUNDS
    return [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (n - 1)) for i in range(n)]


DENSE = log_grid(DENSE_N)
BOOT_GRID = log_grid(BOOT_GRID_N)


# ------------------------------------------------------------------- loading
def floor_renorm(v, eps=EPSILON):
    w = [max(float(x), eps) for x in v]
    z = sum(w)
    return [x / z for x in w]


def load_leg(run_dir: Path, leg: str):
    """{item_label: floored compliance vector} over parse_ok cells (Study B convention)."""
    p = run_dir / f"{leg}_verbalized.json"
    if not p.exists():
        return None
    recs = json.loads(p.read_text())
    out = {}
    for label, r in recs.items():
        if r.get("parse_ok") and r.get("compliance"):
            out[label] = floor_renorm(r["compliance"])
    return out or None


def build_items(preds, cells, ref_preds=None):
    """MetricItemResult list.  ``ref_preds`` puts another leg in the ground-truth
    seat (the RATIO estimand); default is the canonical AIReg GT (ABSOLUTE)."""
    by_label = {c.item_label: c for c in cells}
    items, docs = [], []
    for label, probs in preds.items():
        c = by_label.get(label)
        if c is None:
            continue
        if ref_preds is not None:
            if label not in ref_preds:
                continue
            target = study_a._pred_dist(ref_preds[label], c.gt_labels)
        else:
            target = ComplianceDistribution.from_values(c.gt_probs, c.gt_labels)
        items.append(study_a._metric_item(label, study_a._pred_dist(probs, c.gt_labels), target))
        docs.append(c.document_id)
    return items, docs


# --------------------------------------------------- exact-but-fast precompute
class Scaled:
    """Per-(grid T, item) tables built through the REFERENCE code path.

    ``apply_temperature`` + ``ComplianceDistribution`` are used verbatim, so the
    CDFs, argmaxes and argmax-confidences below are bit-identical to what
    ``study_a._reliability`` / ``murphy_decomposition`` would see; only the
    grouping arithmetic is vectorized.  Verified in ``verify_fast_path``.
    """

    def __init__(self, items, grid):
        n, G = len(items), len(grid)
        self.grid = list(grid)
        self.n = n
        self.gt_cdf = np.array([np.cumsum(it.ground_truth.probabilities)[:-1] for it in items])
        pred_cdf = np.empty((G, n, self.gt_cdf.shape[1]))
        argmax = np.empty((G, n), dtype=np.int64)
        conf = np.empty((G, n))
        for g, T in enumerate(grid):
            for i, it in enumerate(items):
                d = apply_temperature(it.prediction, T)
                p = np.asarray(d.probabilities)
                pred_cdf[g, i] = np.cumsum(p)[:-1]
                a = d.argmax_index()
                argmax[g, i] = a
                conf[g, i] = d.probabilities[a]
        self.pred_cdf, self.argmax, self.conf = pred_cdf, argmax, conf
        d = pred_cdf - self.gt_cdf[None, :, :]
        self.rps = (d * d).sum(axis=2) / self.gt_cdf.shape[1]      # (G, n)

    def reliability_curve(self, idx, bins):
        """Murphy reliability at every grid T for the item multiset ``idx``."""
        G, m, C = self.pred_cdf.shape[0], len(idx), self.gt_cdf.shape[1]
        K = 5 * bins
        bucket = np.minimum((self.conf[:, idx] * bins).astype(np.int64), bins - 1)
        key = (self.argmax[:, idx] * bins + bucket)
        off = (np.arange(G)[:, None] * K + key).ravel()
        counts = np.bincount(off, minlength=G * K).astype(float)
        P = self.pred_cdf[:, idx, :]
        Q = np.broadcast_to(self.gt_cdf[idx], (G, m, C))
        sp = np.stack([np.bincount(off, weights=P[:, :, j].ravel(), minlength=G * K) for j in range(C)], 1)
        sq = np.stack([np.bincount(off, weights=np.ascontiguousarray(Q[:, :, j]).ravel(),
                                   minlength=G * K) for j in range(C)], 1)
        nz = counts > 0
        mp = np.zeros_like(sp)
        mq = np.zeros_like(sq)
        mp[nz] = sp[nz] / counts[nz, None]
        mq[nz] = sq[nz] / counts[nz, None]
        dd = mp - mq
        contrib = (counts / m) * (dd * dd).sum(axis=1) / C
        return contrib.reshape(G, K).sum(axis=1)

    def rps_curve(self, idx):
        return self.rps[:, idx].mean(axis=1)

    def argmin_T(self, curve):
        return float(self.grid[int(np.argmin(curve))])


def group_structure(items, T, bins):
    """Why reliability is bin-fragile: how coarse is the (argmax, conf-bin) partition?"""
    scaled = [study_a._metric_item(it.item_label, apply_temperature(it.prediction, T),
                                   it.ground_truth) for it in items]
    from judex.metrics_report import calibration_groups
    g = calibration_groups(scaled, bins)
    sizes = sorted((len(v) for v in g.values()), reverse=True)
    return {"T": T, "bins": bins, "n_items": len(items), "n_groups": len(g),
            "max_group_size": sizes[0], "singleton_groups": sum(s == 1 for s in sizes),
            "frac_items_in_singletons": sum(s for s in sizes if s == 1) / len(items),
            "frac_items_in_largest_group": sizes[0] / len(items)}


def verify_fast_path(items, probes=(0.5, 1.0, 1.153, 1.8574, 2.5446, 7.0)):
    """Assert the vectorized reliability equals study_a._reliability exactly."""
    sc = Scaled(items, list(probes))
    idx = np.arange(len(items))
    out = []
    for bins in BIN_SWEEP:
        fast = sc.reliability_curve(idx, bins)
        for g, T in enumerate(probes):
            ref = study_a._reliability(items, T, bins=bins)
            out.append(abs(fast[g] - ref))
    ref_rps = [study_a._metric_item(it.item_label, apply_temperature(it.prediction, 1.153),
                                    it.ground_truth).rps for it in items]
    out.append(abs(sc.rps_curve(idx)[2] - statistics.mean(ref_rps)))
    return max(out)


# ------------------------------------------------------------------ P1 and P2
def fit_block(items, label):
    """T_rps (the fitter of record) + T_rel over the bin sweep on both grids."""
    pairs = [(it.prediction, it.ground_truth) for it in items]
    T_rps = fit_temperature(pairs, bounds=study_a.T_BOUNDS).temperature
    sc_grid = Scaled(items, study_a.GRID)
    sc_dense = Scaled(items, DENSE)
    idx = np.arange(len(items))
    blk = {"label": label, "n": len(items), "T_rps": T_rps,
           "T_rps_saturated": study_a.saturated(T_rps),
           "T_rps_dense_grid": sc_dense.argmin_T(sc_dense.rps_curve(idx)),
           "T_rel_by_bins": {}, "T_rel_dense_by_bins": {}, "T_rel_saturated_by_bins": {}}
    for bins in BIN_SWEEP:
        t_g = sc_grid.argmin_T(sc_grid.reliability_curve(idx, bins))
        t_d = sc_dense.argmin_T(sc_dense.reliability_curve(idx, bins))
        blk["T_rel_by_bins"][str(bins)] = t_g
        blk["T_rel_dense_by_bins"][str(bins)] = t_d
        blk["T_rel_saturated_by_bins"][str(bins)] = study_a.saturated(t_g)
    vals = list(blk["T_rel_by_bins"].values())
    blk["T_rel_bin_ratio_max_over_min"] = max(vals) / min(vals)
    blk["T_rel_grid_vs_dense_max_abs_log_ratio"] = max(
        abs(math.log(blk["T_rel_dense_by_bins"][str(b)] / blk["T_rel_by_bins"][str(b)]))
        for b in BIN_SWEEP)
    return blk, sc_grid, sc_dense


# ------------------------------------------------------------------ P3 bootstrap
def doc_bootstrap(items, docs, n=BOOT_N, seed=BOOT_SEED):
    """Doc-clustered bootstrap CIs on BOTH objectives, one mechanism.

    Resamples the ``document_id`` clusters with replacement (24 docs x 5 Articles)
    and re-minimizes each objective on the resampled item multiset over the same
    dense log grid, so the RPS and reliability CIs differ only in the objective.
    """
    sc = Scaled(items, BOOT_GRID)
    by_doc = {}
    for i, d in enumerate(docs):
        by_doc.setdefault(d, []).append(i)
    keys = sorted(by_doc)
    rng = random.Random(seed)
    fits = {"rps": []}
    for b in BIN_SWEEP:
        fits[f"rel_bins{b}"] = []
    for _ in range(n):
        idx = []
        for _ in keys:
            idx.extend(by_doc[keys[rng.randrange(len(keys))]])
        idx = np.asarray(idx)
        fits["rps"].append(sc.argmin_T(sc.rps_curve(idx)))
        for b in BIN_SWEEP:
            fits[f"rel_bins{b}"].append(sc.argmin_T(sc.reliability_curve(idx, b)))
    out = {"n_docs": len(keys), "n_reps": n, "seed": seed,
           "grid_points": BOOT_GRID_N, "grid_log_spacing_pct":
           100 * (math.exp((math.log(study_a.T_BOUNDS[1]) - math.log(study_a.T_BOUNDS[0]))
                           / (BOOT_GRID_N - 1)) - 1)}
    for k, v in fits.items():
        v = sorted(v)
        out[k] = {"lo": v[int(0.025 * len(v))], "hi": v[int(0.975 * len(v)) - 1],
                  "median": statistics.median(v),
                  "covers_T_J": v[int(0.025 * len(v))] <= T_J <= v[int(0.975 * len(v)) - 1],
                  "frac_concurrent_with_T_J": sum(abs(math.log(t / T_J)) <= LN2 for t in v) / len(v)}
    return out


# ------------------------------------------------------------------ P4 transfer
def transfer_criteria(vals: dict, name: str, saturated: dict | None = None) -> dict:
    """Clustering ratio + LOFO transfer error, exactly the registered conventions."""
    fams = sorted(vals)
    s = sorted(vals.values())
    lofo = {}
    for f in fams:
        med = statistics.median([vals[g] for g in fams if g != f])
        lofo[f] = {"own": vals[f], "transferred": med, "log_ratio": math.log(med / vals[f])}
    return {
        "estimand": name,
        "values": vals,
        "panel_min": s[0], "panel_max": s[-1],
        "cluster_ratio": s[-1] / s[0],
        "cluster_rule_ok": bool(s[-1] / s[0] <= study_a.CLUSTER_RULE_MAX_RATIO),
        "median_interpolated": statistics.median(s),
        "lofo": lofo,
        "lofo_max_abs_log_ratio": max(abs(r["log_ratio"]) for r in lofo.values()),
        "any_saturated": bool(saturated and any(saturated.values())),
        "saturated": saturated or {},
    }


# ------------------------------------------- P7 what does each objective identify?
def _align_pairs(src, tgt, cells):
    by = {c.item_label: c for c in cells}
    ov = sorted(src.keys() & tgt.keys() & by.keys())
    return [(study_a._pred_dist(src[k], by[k].gt_labels),
             study_a._pred_dist(tgt[k], by[k].gt_labels)) for k in ov]


def _grid_fit(pairs, metric, grid=None):
    grid = grid or study_a.GRID
    def obj(T):
        return sum(metric(apply_temperature(p, T), q) for p, q in pairs) / len(pairs)
    best = min(grid, key=obj)
    o1, ob = obj(1.0), obj(best)
    return {"T": best, "obj_at_T1_grid_nearest": obj(min(grid, key=lambda t: abs(math.log(t)))),
            "obj_at_best": ob, "obj_at_exact_T1": o1,
            # how much the objective actually MOVES: a near-zero (or negative) gain means the
            # objective is nearly non-identifying for T and the "fit" is the null correction.
            "relative_gain_pct_vs_exact_T1": 100.0 * (o1 - ob) / o1 if o1 > 0 else float("nan")}


def direction_symmetry(pre, post, cells) -> dict:
    """A temperature that measures a SHARPNESS RATIO must invert when the two legs
    swap roles (tau_reverse ~ 1/tau_forward).  A temperature that is a hedging
    response to item-level disagreement does not invert -- it stays > 1 both ways.
    This test tells the two apart without any ground truth."""
    out = {}
    for tag, (a, b) in (("post_to_pre", (post, pre)), ("pre_to_post", (pre, post))):
        pairs = _align_pairs(a, b, cells)
        out[tag] = {
            "n": len(pairs),
            "w1": _grid_fit(pairs, wasserstein_1),
            "rps": _grid_fit(pairs, ranked_probability_score),
            "mean_norm_entropy_source": statistics.mean(p.normalized_entropy() for p, _ in pairs),
            "mean_norm_entropy_target": statistics.mean(q.normalized_entropy() for _, q in pairs),
        }
    for obj in ("w1", "rps"):
        f, r = out["post_to_pre"][obj]["T"], out["pre_to_post"][obj]["T"]
        # product == 1 <=> perfectly inverting (a true ratio); product >> 1 <=> hedging
        out[f"{obj}_forward_times_reverse"] = f * r
        out[f"{obj}_ln_product"] = math.log(f * r)
    return out


def entropy_track(items, temps, gt_mean_norm_H) -> dict:
    """Mean normalized predictive entropy at each candidate T, plus the
    objective-free entropy-matching temperature T_H (pred dispersion == GT
    dispersion).  A fit far above T_H is over-softening the predictor relative
    to the target's own dispersion."""
    def mean_H(T):
        return statistics.mean(apply_temperature(it.prediction, T).normalized_entropy() for it in items)
    track = {f"{T:.4f}": mean_H(T) for T in temps}
    lo, hi = study_a.T_BOUNDS
    for _ in range(60):
        mid = math.exp((math.log(lo) + math.log(hi)) / 2)
        if mean_H(mid) < gt_mean_norm_H:
            lo = mid
        else:
            hi = mid
    return {"gt_mean_norm_entropy": gt_mean_norm_H, "pred_mean_norm_entropy_at_T": track,
            "T_entropy_match": math.exp((math.log(lo) + math.log(hi)) / 2)}


# ------------------------------------------------------------------ P5 apply
def f4_row(preds, cells, T, tag, bins=10):
    chk = study_a.closed_side_check(preds, cells, T, bins=bins)
    res_un = chk["uncalibrated"]["resolution"]
    ok = (abs(chk["resolution_change"]) <= RESOLUTION_TOL * res_un) if res_un > 0 else False
    chk["resolution_within_tol"] = bool(ok)
    chk["f4_pass"] = bool(chk["verdict_ok"] and ok)
    chk["tag"] = tag
    chk["murphy_bins"] = bins
    chk["abs_log_ratio_vs_T_J"] = abs(math.log(T / T_J))
    chk["f5_concurrent_vs_T_J"] = bool(chk["abs_log_ratio_vs_T_J"] <= LN2)
    return chk


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/reliability_objective_probe")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--boot-n", type=int, default=BOOT_N)
    args = ap.parse_args()

    cells = aireg.load_cells()
    runs = REPO / "runs"
    eval_runs = Path(__file__).resolve().parents[2] / "judex-evaluator" / "runs"
    report = {
        "provenance": {
            "frozen_r3": {"T_J": T_J, "band": list(BAND), "F5_threshold_ln2": LN2,
                          "F4_resolution_tol": RESOLUTION_TOL},
            "epsilon": EPSILON, "T_bounds": list(study_a.T_BOUNDS),
            "closed_run": CLOSED_RUN,
            "study_a_GRID_points": len(study_a.GRID),
            "study_a_GRID_log_spacing_pct":
                100 * (study_a.GRID[1] / study_a.GRID[0] - 1),
            "dense_grid_points": DENSE_N,
            "dense_grid_log_spacing_pct": 100 * (DENSE[1] / DENSE[0] - 1),
            "bin_sweep": list(BIN_SWEEP),
            "note": "REPORT-ONLY. r3 F1-F8 frozen; nothing adopted; seam stays mode: noop.",
        }
    }

    # ---------------- closed pair
    closed = load_closed(eval_runs / CLOSED_RUN)
    c_items, c_docs = build_items(closed, cells)
    report["provenance"]["fast_path_max_abs_error_vs_reference"] = verify_fast_path(c_items)

    # ---- P0 regression check
    p0 = {}
    for tag, T in (("uncalibrated_probe", 1.0), ("T_J", T_J), ("panel_abs_rps_median_2.353", 2.353)):
        chk = f4_row(closed, cells, T, tag)
        p0[tag] = {k: chk[k] for k in ("uncalibrated", "calibrated", "reliability_improvement",
                                       "resolution_change", "rps_change", "f4_pass")}
    got = {
        "uncalibrated_reliability": p0["T_J"]["uncalibrated"]["reliability"],
        "uncalibrated_resolution": p0["T_J"]["uncalibrated"]["resolution"],
        "uncalibrated_mean_rps": p0["T_J"]["uncalibrated"]["mean_rps"],
        "T_J_reliability_improvement": p0["T_J"]["reliability_improvement"],
        "T_J_rps_change": p0["T_J"]["rps_change"],
        "T_2353_reliability_improvement": p0["panel_abs_rps_median_2.353"]["reliability_improvement"],
        "T_2353_rps_change": p0["panel_abs_rps_median_2.353"]["rps_change"],
        "T_2353_resolution_change": p0["panel_abs_rps_median_2.353"]["resolution_change"],
    }
    report["P0_regression"] = {
        "expected": REF, "got": got,
        "all_match": all(abs(got[k] - REF[k]) < 1e-5 for k in got),
        "mismatches": {k: [REF[k], got[k]] for k in got if abs(got[k] - REF[k]) >= 1e-5},
    }

    # ---- P2 closed-pair fits
    c_blk, _, _ = fit_block(c_items, "closed pair (Sonnet 4.6 + GPT 5.4)")
    c_blk["T_rps_matches_record"] = abs(c_blk["T_rps"] - REF["T_star_rps"]) < 1e-6
    sc_dense = Scaled(c_items, DENSE)
    idx = np.arange(len(c_items))
    c_blk["reliability_curve_dense"] = {
        str(b): {"T": DENSE, "reliability": list(map(float, sc_dense.reliability_curve(idx, b)))}
        for b in BIN_SWEEP}
    c_blk["rps_curve_dense"] = list(map(float, sc_dense.rps_curve(idx)))
    # how flat is the reliability objective near its minimum? (fragility read)
    for b in BIN_SWEEP:
        cur = np.asarray(c_blk["reliability_curve_dense"][str(b)]["reliability"])
        lo = cur.min()
        within = [DENSE[i] for i in range(len(DENSE)) if cur[i] <= lo * 1.05 + 1e-12]
        c_blk.setdefault("flatness_T_within_5pct_of_min", {})[str(b)] = [min(within), max(within)]
    # bins -> inf limit check: reliability collapses onto mean RPS, so T_rel -> T_rps
    sc_lim = Scaled(c_items, study_a.GRID)
    c_blk["T_rel_limit_bins"] = {
        str(b): sc_lim.argmin_T(sc_lim.reliability_curve(idx, b)) for b in LIMIT_BINS}
    c_blk["T_rel_limit_bins_dense"] = {
        str(b): sc_dense.argmin_T(sc_dense.reliability_curve(idx, b)) for b in LIMIT_BINS}
    c_blk["T_rps_on_same_grid"] = sc_lim.argmin_T(sc_lim.rps_curve(idx))
    c_blk["group_structure"] = [group_structure(c_items, T, b)
                                for T in (1.0, T_J, REF["T_star_rps"])
                                for b in tuple(BIN_SWEEP) + LIMIT_BINS]
    report["P2_closed_pair"] = c_blk

    # ---- P1 panel legs
    panel = {}
    for key, d, label, in_panel, note in FAMILIES:
        rd = runs / d
        if not rd.exists():
            continue
        pre, post = load_leg(rd, "pre"), load_leg(rd, "post")
        if not pre or not post:
            continue
        entry = {"label": label, "in_adoption_panel": in_panel, "note": note}
        for leg, preds in (("pre", pre), ("post", post)):
            it, _ = build_items(preds, cells)
            entry[leg], _, _ = fit_block(it, f"{label} {leg}")
        # RATIO estimand (post -> own base) under all three objectives
        it_ratio, _ = build_items(post, cells, ref_preds=pre)
        sc_r = Scaled(it_ratio, study_a.GRID)
        ridx = np.arange(len(it_ratio))
        entry["ratio_tau_v_w1_of_record"] = study_a.fit_tau_oc(post, pre, cells)
        entry["ratio_tau_rps"] = fit_temperature(
            [(i.prediction, i.ground_truth) for i in it_ratio],
            bounds=study_a.T_BOUNDS).temperature
        entry["ratio_tau_rel_by_bins"] = {
            str(b): sc_r.argmin_T(sc_r.reliability_curve(ridx, b)) for b in BIN_SWEEP}
        panel[key] = entry
    report["P1_panel"] = panel

    # ---- P4 transfer criteria, head to head
    pk = [k for k, v in panel.items() if v["in_adoption_panel"]]
    est = {}
    est["tau_v_ratio_w1_FROZEN"] = transfer_criteria(
        {k: panel[k]["ratio_tau_v_w1_of_record"] for k in pk}, "tau_v (ratio, W1)")
    est["T_abs_rps"] = transfer_criteria(
        {k: panel[k]["post"]["T_rps"] for k in pk}, "T_abs(post) (absolute, RPS)",
        {k: panel[k]["post"]["T_rps_saturated"] for k in pk})
    for b in BIN_SWEEP:
        est[f"T_abs_rel_bins{b}"] = transfer_criteria(
            {k: panel[k]["post"]["T_rel_by_bins"][str(b)] for k in pk},
            f"T_abs(post) (absolute, reliability, bins={b})",
            {k: panel[k]["post"]["T_rel_saturated_by_bins"][str(b)] for k in pk})
    est["tau_ratio_rps"] = transfer_criteria(
        {k: panel[k]["ratio_tau_rps"] for k in pk}, "tau (ratio, RPS)")
    for b in BIN_SWEEP:
        est[f"tau_ratio_rel_bins{b}"] = transfer_criteria(
            {k: panel[k]["ratio_tau_rel_by_bins"][str(b)] for k in pk}, f"tau (ratio, rel, bins={b})")
    report["P4_transfer"] = est
    report["P4_known_value_check"] = {
        "tau_v": {k: [KNOWN_TAU_V[k], panel[k]["ratio_tau_v_w1_of_record"]] for k in pk},
        "T_abs_rps": {k: [KNOWN_T_ABS_RPS[k], panel[k]["post"]["T_rps"]] for k in pk},
    }

    # ---- P5 apply and compare
    med_rel = {str(b): est[f"T_abs_rel_bins{b}"]["median_interpolated"] for b in BIN_SWEEP}
    points = [
        ("identity", 1.0),
        ("T_J (r3 F2)", T_J),
        ("panel median T_abs_rps", est["T_abs_rps"]["median_interpolated"]),
        ("panel median T_abs_rel bins=3", med_rel["3"]),
        ("panel median T_abs_rel bins=10", med_rel["10"]),
        ("closed T*_rps (arm 2 of record)", c_blk["T_rps"]),
        ("closed T*_rel bins=3", c_blk["T_rel_by_bins"]["3"]),
        ("closed T*_rel bins=10", c_blk["T_rel_by_bins"]["10"]),
    ]
    seen, rows = set(), []
    for tag, T in points:
        if round(T, 9) in seen and tag != "identity":
            rows.append({"tag": tag, "T": T, "duplicate_of_prior_row": True})
            continue
        seen.add(round(T, 9))
        r10 = f4_row(closed, cells, T, tag, bins=10)
        r3 = f4_row(closed, cells, T, tag, bins=3)
        rows.append({
            "tag": tag, "T": T,
            "bins10": {k: r10[k] for k in ("uncalibrated", "calibrated", "reliability_improvement",
                                           "resolution_change", "rps_change", "verdict_ok",
                                           "resolution_within_tol", "f4_pass")},
            "bins3_reliability_improvement": r3["reliability_improvement"],
            "bins3_f4_pass": r3["f4_pass"],
            "abs_log_ratio_vs_T_J": r10["abs_log_ratio_vs_T_J"],
            "f5_concurrent_vs_T_J": r10["f5_concurrent_vs_T_J"],
        })
    report["P5_apply"] = rows

    # ---- P6 decomposition
    def L(x):
        return math.log(x)
    med_abs_rps = est["T_abs_rps"]["median_interpolated"]
    med_abs_rel3 = est["T_abs_rel_bins3"]["median_interpolated"]
    med_ratio_rel3 = est["tau_ratio_rel_bins3"]["median_interpolated"]
    med_ratio_rps = est["tau_ratio_rps"]["median_interpolated"]
    total = L(c_blk["T_rps"]) - L(T_J)
    all_bins = tuple(BIN_SWEEP) + LIMIT_BINS
    ladder = {str(b): (c_blk["T_rel_by_bins"].get(str(b)) or c_blk["T_rel_limit_bins"][str(b)])
              for b in all_bins}
    ladder_dense = {str(b): (c_blk["T_rel_dense_by_bins"].get(str(b))
                             or c_blk["T_rel_limit_bins_dense"][str(b)]) for b in all_bins}
    report["P6_decomposition"] = {
        "total_gap_ln_T_star_rps_over_T_J": total,
        "threshold_ln2": LN2,
        "closed_side_objective_effect_ln_Trps_over_Trel": {
            str(b): L(c_blk["T_rps"]) - L(c_blk["T_rel_by_bins"][str(b)]) for b in BIN_SWEEP},
        "residual_after_objective_ln_Trel_over_T_J": {
            str(b): L(c_blk["T_rel_by_bins"][str(b)]) - L(T_J) for b in BIN_SWEEP},
        "panel_routed_chain_bins3": {
            "1_closed_idiosyncrasy_ln_Tstar_rps_over_panel_med_abs_rps":
                L(c_blk["T_rps"]) - L(med_abs_rps),
            "2_objective_within_absolute_ln_panel_med_abs_rps_over_abs_rel":
                L(med_abs_rps) - L(med_abs_rel3),
            "3_estimand_at_matched_reliability_ln_abs_rel_over_ratio_rel":
                L(med_abs_rel3) - L(med_ratio_rel3),
            "4_objective_within_ratio_ln_ratio_rel_over_T_J":
                L(med_ratio_rel3) - L(T_J),
            "sum": (L(c_blk["T_rps"]) - L(med_abs_rps)) + (L(med_abs_rps) - L(med_abs_rel3))
                   + (L(med_abs_rel3) - L(med_ratio_rel3)) + (L(med_ratio_rel3) - L(T_J)),
            "check_equals_total": abs(((L(c_blk["T_rps"]) - L(T_J))) - total) < 1e-12,
            "objective_terms_total": (L(med_abs_rps) - L(med_abs_rel3)) + (L(med_ratio_rel3) - L(T_J)),
            "estimand_term": L(med_abs_rel3) - L(med_ratio_rel3),
        },
        # Same total, routed with the ESTIMAND step taken at a matched RPS objective.
        # A path decomposition is not unique; both chains are reported so the reader can
        # see that the conclusion does not depend on which intermediate is chosen.
        "panel_routed_chain_matched_rps": {
            "1_closed_idiosyncrasy": L(c_blk["T_rps"]) - L(med_abs_rps),
            "2_estimand_at_matched_rps_ln_abs_rps_over_ratio_rps": L(med_abs_rps) - L(med_ratio_rps),
            "3_objective_within_ratio_W1_to_RPS_ln_ratio_rps_over_T_J": L(med_ratio_rps) - L(T_J),
            "sum": (L(c_blk["T_rps"]) - L(med_abs_rps)) + (L(med_abs_rps) - L(med_ratio_rps))
                   + (L(med_ratio_rps) - L(T_J)),
        },
        "f5_verdict_over_full_bin_ladder": {
            "bins": list(all_bins),
            "T_rel_60pt_GRID": ladder,
            "T_rel_601pt": ladder_dense,
            "concur_60pt": [b for b in all_bins if abs(L(ladder[str(b)] / T_J)) <= LN2],
            "concur_601pt": [b for b in all_bins if abs(L(ladder_dense[str(b)] / T_J)) <= LN2],
            "margin_60pt": {str(b): LN2 - abs(L(ladder[str(b)] / T_J)) for b in all_bins},
        },
        "limit_note": ("murphy_decomposition groups by (argmax, confidence bin); as bins -> inf "
                       "groups become singletons and reliability -> mean RPS, so T_rel -> T_rps. "
                       "The objective effect is therefore a function of the bin count. On this "
                       "run the collapse is only partial even at bins=1000 because the closed "
                       "pair's verbalized vectors repeat exactly (a 0.05 elicitation grid), so "
                       "tied argmax confidences keep ~28 non-singleton groups at every bin count."),
    }

    # ---- P7 objective diagnostics
    p7 = {"direction_symmetry": {}}
    for key, d, label, in_panel, note in FAMILIES:
        rd = runs / d
        if key not in panel or not rd.exists():
            continue
        pre, post = load_leg(rd, "pre"), load_leg(rd, "post")
        p7["direction_symmetry"][key] = {"label": label, "in_adoption_panel": in_panel,
                                         **direction_symmetry(pre, post, cells)}
    # sanity: a leg aligned to ITSELF must fit the grid point nearest T=1
    qpost = load_leg(runs / "study_b_qwen", "post")
    self_pairs = _align_pairs(qpost, qpost, cells)
    p7["self_alignment_sanity"] = {
        "family": "qwen post -> itself",
        "T_rps": _grid_fit(self_pairs, ranked_probability_score)["T"],
        "T_w1": _grid_fit(self_pairs, wasserstein_1)["T"],
        "grid_point_nearest_1": min(study_a.GRID, key=lambda t: abs(math.log(t))),
    }
    gt_H = statistics.mean(it.ground_truth.normalized_entropy() for it in c_items)
    p7["closed_entropy_track"] = entropy_track(
        c_items, [1.0, T_J, BAND[1], est["T_abs_rel_bins3"]["median_interpolated"],
                  est["T_abs_rps"]["median_interpolated"], c_blk["T_rel_by_bins"]["3"],
                  c_blk["T_rps"], c_blk["T_rel_by_bins"]["10"]], gt_H)
    report["P7_objective_diagnostics"] = p7

    # ---- P3 bootstrap (last: slowest)
    if not args.no_bootstrap:
        report["P3_bootstrap_closed"] = doc_bootstrap(c_items, c_docs, n=args.boot_n)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "reliability_objective_probe.json").write_text(json.dumps(report, indent=2))

    # -------------------------------------------------------------- console
    print(f"\nP0 regression: all_match={report['P0_regression']['all_match']} "
          f"mismatches={report['P0_regression']['mismatches']}")
    print(f"fast-path max |err| vs study_a._reliability = "
          f"{report['provenance']['fast_path_max_abs_error_vs_reference']:.3e}")

    print(f"\nP2 closed pair  T_rps={c_blk['T_rps']:.4f} (record match "
          f"{c_blk['T_rps_matches_record']})")
    print(f"{'bins':>6} {'T_rel (60-pt GRID)':>20} {'T_rel (601-pt)':>16} {'|ln(T_rel/T_J)|':>16} {'F5':>5}")
    for b in BIN_SWEEP:
        t = c_blk["T_rel_by_bins"][str(b)]
        td = c_blk["T_rel_dense_by_bins"][str(b)]
        lr = abs(math.log(t / T_J))
        print(f"{b:6d} {t:20.4f} {td:16.4f} {lr:16.4f} {'YES' if lr <= LN2 else 'no':>5}")
    print(f"  T_rel bin ratio max/min = {c_blk['T_rel_bin_ratio_max_over_min']:.3f}"
          f"   grid-vs-dense max |ln ratio| = {c_blk['T_rel_grid_vs_dense_max_abs_log_ratio']:.3f}")
    print("  bins -> inf limit (601-pt grid): " + " ".join(
        f"b{b}={c_blk['T_rel_limit_bins_dense'][str(b)]:.3f}" for b in LIMIT_BINS)
        + f"   (T_rps dense = {c_blk['T_rps_dense_grid']:.3f})")
    print("  group structure at T=1: " + " ".join(
        f"b{g['bins']}:{g['n_groups']}grp/{g['frac_items_in_largest_group']:.2f}top"
        for g in c_blk["group_structure"] if g["T"] == 1.0))

    for leg in ("post", "pre"):
        print(f"\nP1 panel {leg.upper()} legs — T_rel by bins (60-pt GRID)")
        print(f"{'family':22} {'T_rps':>7} " + " ".join(f"{'b'+str(b):>7}" for b in BIN_SWEEP)
              + f" {'binratio':>9}")
        for k, v in panel.items():
            row = " ".join(f"{v[leg]['T_rel_by_bins'][str(b)]:7.3f}" for b in BIN_SWEEP)
            flag = "" if v["in_adoption_panel"] else "  (non-panel)"
            print(f"{v['label']:22} {v[leg]['T_rps']:7.3f} {row} "
                  f"{v[leg]['T_rel_bin_ratio_max_over_min']:9.3f}{flag}")

    print("\nP4 transfer criteria (adoption panel n=4)")
    print(f"{'estimand':38} {'ratio':>7} {'ok':>4} {'LOFOmax':>8} {'median':>8}")
    for k, v in est.items():
        print(f"{v['estimand']:38} {v['cluster_ratio']:7.3f} {str(v['cluster_rule_ok']):>4} "
              f"{v['lofo_max_abs_log_ratio']:8.3f} {v['median_interpolated']:8.3f}")

    print("\nP5 apply on the executed sweep (F4 at bins=10)")
    print(f"{'point':34} {'T':>7} {'d_rel':>8} {'d_rps':>9} {'d_res':>9} {'F4':>4} {'F5':>4}")
    for r in rows:
        if r.get("duplicate_of_prior_row"):
            continue
        b = r["bins10"]
        print(f"{r['tag']:34} {r['T']:7.3f} {b['reliability_improvement']:8.5f} "
              f"{b['rps_change']:9.5f} {b['resolution_change']:9.5f} "
              f"{'PASS' if b['f4_pass'] else 'FAIL':>4} "
              f"{'YES' if r['f5_concurrent_vs_T_J'] else 'no':>4}")

    d = report["P6_decomposition"]
    print(f"\nP6 total ln(T*_rps/T_J) = {d['total_gap_ln_T_star_rps_over_T_J']:.4f} (ln2={LN2:.4f})")
    for b in BIN_SWEEP:
        print(f"   bins={b:2d}: objective {d['closed_side_objective_effect_ln_Trps_over_Trel'][str(b)]:+.4f}"
              f"  residual {d['residual_after_objective_ln_Trel_over_T_J'][str(b)]:+.4f}")
    ch = d["panel_routed_chain_bins3"]
    print(f"   panel-routed (bins=3): idiosyncrasy {ch['1_closed_idiosyncrasy_ln_Tstar_rps_over_panel_med_abs_rps']:+.4f}"
          f" | obj_abs {ch['2_objective_within_absolute_ln_panel_med_abs_rps_over_abs_rel']:+.4f}"
          f" | estimand {ch['3_estimand_at_matched_reliability_ln_abs_rel_over_ratio_rel']:+.4f}"
          f" | obj_ratio {ch['4_objective_within_ratio_ln_ratio_rel_over_T_J']:+.4f}")
    cm = d["panel_routed_chain_matched_rps"]
    print(f"   panel-routed (matched RPS): idiosyncrasy {cm['1_closed_idiosyncrasy']:+.4f}"
          f" | estimand {cm['2_estimand_at_matched_rps_ln_abs_rps_over_ratio_rps']:+.4f}"
          f" | objective W1->RPS {cm['3_objective_within_ratio_W1_to_RPS_ln_ratio_rps_over_T_J']:+.4f}")
    fl = d["f5_verdict_over_full_bin_ladder"]
    print(f"   F5 over the {len(fl['bins'])}-point bin ladder {fl['bins']}: "
          f"concurring bins (60-pt grid) = {fl['concur_60pt']}, (601-pt) = {fl['concur_601pt']}")

    print("\nP7 direction symmetry — a true sharpness RATIO inverts (product ~ 1); "
          "a hedging response does not")
    print(f"{'family':22} {'W1 fwd':>7} {'W1 rev':>7} {'prod':>6} {'W1gain%':>8} | "
          f"{'RPS fwd':>8} {'RPS rev':>8} {'prod':>6} {'RPSgain%':>9}")
    for k, v in p7["direction_symmetry"].items():
        print(f"{v['label']:22} {v['post_to_pre']['w1']['T']:7.3f} {v['pre_to_post']['w1']['T']:7.3f} "
              f"{v['w1_forward_times_reverse']:6.2f} "
              f"{v['post_to_pre']['w1']['relative_gain_pct_vs_exact_T1']:8.2f} | "
              f"{v['post_to_pre']['rps']['T']:8.3f} "
              f"{v['pre_to_post']['rps']['T']:8.3f} {v['rps_forward_times_reverse']:6.2f} "
              f"{v['post_to_pre']['rps']['relative_gain_pct_vs_exact_T1']:9.2f}"
              f"{'' if v['in_adoption_panel'] else '  (non-panel)'}")
    et = p7["closed_entropy_track"]
    print(f"  self-alignment sanity: T_rps={p7['self_alignment_sanity']['T_rps']:.4f} "
          f"(grid point nearest 1 = {p7['self_alignment_sanity']['grid_point_nearest_1']:.4f})")
    print(f"  closed entropy match: GT mean norm H = {et['gt_mean_norm_entropy']:.4f}, "
          f"T_H = {et['T_entropy_match']:.4f}")
    for t, h in et["pred_mean_norm_entropy_at_T"].items():
        print(f"     T={t}: pred mean norm H = {h:.4f}")

    if not args.no_bootstrap:
        bs = report["P3_bootstrap_closed"]
        print(f"\nP3 doc-clustered bootstrap ({bs['n_docs']} docs, {bs['n_reps']} reps, seed {bs['seed']})")
        for k in ["rps"] + [f"rel_bins{b}" for b in BIN_SWEEP]:
            v = bs[k]
            print(f"  {k:12} median {v['median']:7.3f}  95% CI [{v['lo']:.3f}, {v['hi']:.3f}]  "
                  f"P(F5 concurs) = {v['frac_concurrent_with_T_J']:.3f}")

    print(f"\nwrote {out / 'reliability_objective_probe.json'}")


if __name__ == "__main__":
    main()
