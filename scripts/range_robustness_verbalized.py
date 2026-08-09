#!/usr/bin/env python3
"""Verbalized-channel tau range-robustness sweep on a closed-pair run.

The verbalized-native successor of ``scripts/q4_range_robustness.py`` (the
logit-era instrument, which sweeps [0.75, 40] against the retired logit-channel
reference taus and is kept unedited as that channel's record).

What this measures: how sensitive the r3 **F4 acceptance verdict** is to WHERE
the applied temperature falls, across and around the r3 sensitivity band, on a
closed-pair evaluator run. It is the full-range generalisation of the 9-point
across-band profile that ``scripts/e6_r3_arms.py`` emits as ladder item 6.

F4 (``docs/e6_onpair_decision_protocol_r3.md``, frozen): a temperature is
accepted iff Murphy **Reliability** is strictly better AND mean **RPS** is no
worse (tolerance 1e-9) AND **Resolution** stays within +/-10% of its
uncalibrated value, all on the 10-bin Murphy decomposition. Nothing here adopts
anything: the frozen constants (T_J = 1.153, band [1.0252, 1.3798]) are inputs,
not outputs.

Grid (spec: ``spec/analysis_2026_07_21_q4_range_robustness_verbalized.md`` §2):
95 log-spaced points over [0.800, 2.000], 30 more across the band
[1.025, 1.380], plus markers at T_J, both exact band edges, the panel tau_v
values, identity, and the nine across-band profile points so the committed
record can be compared point-for-point.

Bootstrap: 2000 doc-clustered replicates (resample the 24 documents with
replacement; every one carries its 5 Articles), seed 20260720 — the same
convention the 2026-07-21 analysis used on its proxy datasets.

Joins are on **item identity** (``item_label``), never on position: the stored
prediction order and the ground-truth cell order are not the same ordering.
Level ordering is taken from the prediction's own label names, never from the
order they happen to appear in.

Ground truth comes from ``judex_calibration.aireg.load_cells()`` only — the
canonical manifest-verified bundle. Never from a ``runs/`` metrics_report.

Run from judex-calibration:
  JUDEX_UMBRELLA=/path/to/judex \\
    python scripts/range_robustness_verbalized.py \\
      --closed-run stage9-onpair-e6-20260721 --out runs/q4_range_onpair_<date>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------- frozen (r3)
T_J = 1.153
BAND = (1.0251785151221313, 1.379820350674421)
RESOLUTION_TOL = 0.10
PANEL_TAU_V = {  # F1 adoption panel, verbalized channel
    "qwen": 1.281052072726771,
    "gemma31": 1.0251785151221313,
    "glm": 1.0251785151221313,
    "maverick": 1.379820350674421,
}
BAND_PROFILE_N = 9  # e6_r3_arms.py ladder item 6 — the cross-check points

# Sweep spec (analysis 2026-07-21 §2)
SWEEP_RANGE = (0.800, 2.000)
SWEEP_N = 95
INBAND_N = 30
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260720

# Level order of the 5-point ordinal scale, keyed by the contract's own label text.
FULL_TO_SHORT = {
    "Very low probability of compliance": "very_low",
    "Low probability of compliance": "low",
    "Moderate probability of compliance": "moderate",
    "High probability of compliance": "high",
    "Very high probability of compliance": "very_high",
}


def default_umbrella() -> Path:
    env = os.environ.get("JUDEX_UMBRELLA")
    if env:
        return Path(env)
    if REPO.parent.name == "worktrees":
        return REPO.parent.parent
    return REPO.parent


def load_closed(umbrella: Path, run_id: str) -> dict[str, list[float]]:
    """{item_label: [p(very_low) .. p(very_high)]} — identity join, name-ordered levels.

    Identical in effect to ``e6_r3_arms.load_closed``; restated here so this
    instrument does not import a frozen driver.
    """
    path = umbrella / "judex-evaluator" / "runs" / run_id / "metrics_report.json"
    if not path.exists():
        raise SystemExit(f"no metrics_report.json at {path} "
                         f"(judex-evaluator/runs/ is gitignored — host-local only)")
    report = json.loads(path.read_text())
    preds = {}
    for it in report["items"]:
        p = it["prediction"]
        by = dict(zip(p["labels"], p["probabilities"]))
        missing = set(FULL_TO_SHORT) - set(by)
        if missing:
            raise SystemExit(f"item {it['item_label']!r} is missing levels {sorted(missing)}")
        preds[it["item_label"]] = [float(by[full]) for full in FULL_TO_SHORT]
    return preds


def floor_renorm(vec, eps: float):
    """The channel's analysis-time epsilon floor, applied to a probability vector."""
    if eps <= 0:
        return list(vec)
    v = [max(float(x), eps) for x in vec]
    s = sum(v)
    return [x / s for x in v]


# --------------------------------------------------------------------- F4 read

def f4_check(preds, cells, T, study_a) -> dict:
    """One F4 acceptance read at temperature ``T``.

    Delegates every number to ``study_a.closed_side_check`` (the production
    helper the protocol names) and adds the Resolution guard exactly as
    ``e6_r3_arms._f4_check`` does — including its rounding, so verdicts are
    bit-comparable with the committed across-band profile.
    """
    chk = study_a.closed_side_check(preds, cells, T)
    res_un = chk["uncalibrated"]["resolution"]
    res_ok = (abs(chk["resolution_change"]) <= RESOLUTION_TOL * res_un) if res_un > 0 else False
    chk["resolution_within_tol"] = bool(res_ok)
    chk["f4_pass"] = bool(chk["verdict_ok"] and res_ok)
    return chk


# ------------------------------------------------- vectorized bootstrap replica

def cdf_matrix(dists) -> np.ndarray:
    return np.array([np.cumsum(d.probabilities)[:-1] for d in dists], dtype=float)


def group_ids(items) -> np.ndarray:
    """(argmax level, confidence decile) as a dense int id — metrics_report
    .calibration_groups at bins=10."""
    keys = []
    for it in items:
        conf = it.prediction.probabilities[it.predicted_argmax_index]
        keys.append((it.predicted_argmax_index, min(int(conf * 10), 9)))
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([uniq[k] for k in keys], dtype=int)


def weighted_murphy(w, gt_cdfs, pred_cdfs, gids, rps):
    """Multiplicity-weighted Murphy decomposition + mean RPS."""
    n = w.sum()
    clim = (w @ gt_cdfs) / n
    g = int(gids.max()) + 1
    wsum = np.bincount(gids, weights=w, minlength=g)
    live = wsum > 0
    k = gt_cdfs.shape[1]
    gt_sum = np.zeros((g, k)); pred_sum = np.zeros((g, k))
    for j in range(k):
        gt_sum[:, j] = np.bincount(gids, weights=w * gt_cdfs[:, j], minlength=g)
        pred_sum[:, j] = np.bincount(gids, weights=w * pred_cdfs[:, j], minlength=g)
    gt_mean = np.zeros((g, k)); pred_mean = np.zeros((g, k))
    gt_mean[live] = gt_sum[live] / wsum[live, None]
    pred_mean[live] = pred_sum[live] / wsum[live, None]
    res = float(np.sum(wsum[live] / n * np.mean((gt_mean[live] - clim) ** 2, axis=1)))
    rel = float(np.sum(wsum[live] / n * np.mean((pred_mean[live] - gt_mean[live]) ** 2, axis=1)))
    return rel, res, float(w @ rps / n)


def f4_indicator(rel, res, rps, rel0, res0, rps0):
    """F4 on rounded deltas — the same arithmetic ``closed_side_check`` reports."""
    rel_imp = np.round(rel0 - rel, 5)
    rps_chg = np.round(rps - rps0, 5)
    res_chg = np.round(res - res0, 5)
    res_un = np.round(res0, 5)
    return ((rel < rel0) & (rps <= rps0 + 1e-9)
            & (np.abs(res_chg) <= RESOLUTION_TOL * res_un) & (res_un > 0)), rel_imp, rps_chg, res_chg


def region_edges(grid, ind):
    idx = np.flatnonzero(ind)
    if idx.size == 0:
        return None, None, True
    return float(grid[idx[0]]), float(grid[idx[-1]]), bool(ind[idx[0]:idx[-1] + 1].all())


def refine_edge(lo, hi, keep_lo, inb, steps=40):
    """Bisect (geometrically) between adjacent grid points; ``keep_lo`` says
    whether the in-region side is at ``lo``."""
    for _ in range(steps):
        mid = math.sqrt(lo * hi)
        if inb(mid) == keep_lo:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def build_grid() -> np.ndarray:
    band_profile_pts = [BAND[0] * (BAND[1] / BAND[0]) ** (i / (BAND_PROFILE_N - 1))
                        for i in range(BAND_PROFILE_N)]
    return np.unique(np.concatenate([
        np.geomspace(SWEEP_RANGE[0], SWEEP_RANGE[1], SWEEP_N),
        np.geomspace(BAND[0], BAND[1], INBAND_N),
        np.array(band_profile_pts),
        np.array(sorted(set(list(PANEL_TAU_V.values()) + [T_J, 1.0, BAND[0], BAND[1]]))),
    ]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed-run", default="stage9-onpair-e6-20260721",
                    help="judex-evaluator run id (the on-pair confirmation run of record)")
    ap.add_argument("--umbrella", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, default=0.0,
                    help="analysis-time zero floor on the closed predictions. 0.0 (default) "
                         "reproduces the committed across-band profile, which reads the stored "
                         "vectors as emitted; 0.005 is the channel's floor convention and is "
                         "reported as a sensitivity")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--cross-check", default=None,
                    help="path to e6_r3_arms.json whose arm1 band_profile this sweep must "
                         "reproduce point-for-point")
    args = ap.parse_args()

    umbrella = Path(args.umbrella) if args.umbrella else default_umbrella()
    os.environ.setdefault("JUDEX_UMBRELLA", str(umbrella))

    from judex_calibration import study_a
    from judex_calibration.aireg import load_cells
    from judex.calibration import apply_temperature, fit_temperature
    from judex.metrics_report import murphy_decomposition
    from judex.core.distributions import ComplianceDistribution

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    by_label = {c.item_label: c for c in cells}
    closed_raw = load_closed(umbrella, args.closed_run)

    # --- identity join, explicit and audited ------------------------------
    matched = sorted(lb for lb in closed_raw if lb in by_label)
    unmatched_pred = sorted(lb for lb in closed_raw if lb not in by_label)
    unmatched_gt = sorted(lb for lb in by_label if lb not in closed_raw)
    if unmatched_pred or unmatched_gt:
        print(f"[join] WARNING unmatched predictions={unmatched_pred} cells={unmatched_gt}")
    n_zero = sum(1 for lb in matched for p in closed_raw[lb] if p == 0.0)
    preds = {lb: floor_renorm(closed_raw[lb], args.eps) for lb in matched}
    docs = np.array([by_label[lb].document_id for lb in matched])
    print(f"[load] {len(matched)}/{len(closed_raw)} items joined by item_label on "
          f"{len(set(docs.tolist()))} documents; {n_zero} exact zeros in the stored vectors; "
          f"eps={args.eps}")

    rows = [(lb,
             ComplianceDistribution.from_values(list(preds[lb]), by_label[lb].gt_labels),
             ComplianceDistribution.from_values(by_label[lb].gt_probs, by_label[lb].gt_labels))
            for lb in matched]

    def items_at(T):
        return [study_a._metric_item(lb, p if T is None else apply_temperature(p, T), gt)
                for lb, p, gt in rows]

    base_items = items_at(None)
    mb = murphy_decomposition(base_items, bins=10)
    base = {"reliability": mb["reliability"], "resolution": mb["resolution"],
            "uncertainty": mb["uncertainty"],
            "mean_rps": sum(i.rps for i in base_items) / len(base_items),
            "mean_w1": sum(i.w1 for i in base_items) / len(base_items),
            "argmax_acc": sum(i.argmax_agreement for i in base_items) / len(base_items)}
    print(f"[base] uncalibrated rel {base['reliability']:.5f} res {base['resolution']:.5f} "
          f"rps {base['mean_rps']:.5f} argmax {base['argmax_acc']:.5f}")

    # The identity baseline is the STORED vectors, not apply_temperature(.,1.0):
    # the inverse-softmax seam clips exact zeros at 1e-12, so a T=1 pass is a
    # near-identity, not an identity. Quantify the gap rather than assert it away.
    one_items = items_at(1.0)
    m1 = murphy_decomposition(one_items, bins=10)
    t1_gap = {
        "max_abs_prob_delta": max(abs(a - b) for i1, i0 in zip(one_items, base_items)
                                  for a, b in zip(i1.prediction.probabilities,
                                                  i0.prediction.probabilities)),
        "mean_rps_delta": (sum(i.rps for i in one_items) / len(one_items)) - base["mean_rps"],
        "reliability_delta": m1["reliability"] - base["reliability"],
        "resolution_delta": m1["resolution"] - base["resolution"],
        "note": "apply_temperature(T=1) is a near-identity, not an identity: invert_softmax "
                "clips exact zeros at 1e-12 before renormalising. The uncalibrated baseline "
                "is always the stored vectors.",
    }
    print(f"[T=1] near-identity gap: max|dp| {t1_gap['max_abs_prob_delta']:.3e}, "
          f"d(mean_rps) {t1_gap['mean_rps_delta']:.3e}")

    # --- sweep -------------------------------------------------------------
    grid = build_grid()
    curves = {k: [] for k in ("reliability", "resolution", "mean_rps", "mean_w1", "argmax_acc",
                              "reliability_improvement", "rps_change", "resolution_change")}
    f4 = []
    per_T = []
    gt_cdfs = cdf_matrix([gt for _, _, gt in rows])
    for T in grid:
        chk = f4_check(preds, cells, float(T), study_a)
        for k in ("reliability", "resolution", "mean_rps", "mean_w1", "argmax_acc"):
            curves[k].append(chk["calibrated"][k])
        for k in ("reliability_improvement", "rps_change", "resolution_change"):
            curves[k].append(chk[k])
        f4.append(chk["f4_pass"])
        its = items_at(float(T))
        per_T.append({"pred_cdfs": cdf_matrix([i.prediction for i in its]),
                      "gids": group_ids(its),
                      "rps": np.array([i.rps for i in its])})
    curves = {k: np.array(v) for k, v in curves.items()}
    f4 = np.array(f4, dtype=bool)

    # vectorized replica must equal the production path at unit weights
    ones = np.ones(len(rows))
    rel0, res0, rps0 = base["reliability"], base["resolution"], base["mean_rps"]
    rel_v = np.empty(len(grid)); res_v = np.empty(len(grid)); rps_v = np.empty(len(grid))
    for i in range(len(grid)):
        rel_v[i], res_v[i], rps_v[i] = weighted_murphy(
            ones, gt_cdfs, per_T[i]["pred_cdfs"], per_T[i]["gids"], per_T[i]["rps"])
    ind_v, rel_imp_v, rps_chg_v, res_chg_v = f4_indicator(rel_v, res_v, rps_v, rel0, res0, rps0)
    assert np.array_equal(ind_v, f4), "vectorized F4 replica disagrees with closed_side_check"
    assert np.allclose(rel_imp_v, curves["reliability_improvement"], atol=1e-9)
    assert np.allclose(rps_chg_v, curves["rps_change"], atol=1e-9)
    assert np.allclose(res_chg_v, curves["resolution_change"], atol=1e-9)
    print("[replica] vectorized F4 == closed_side_check at every grid point")

    lo, hi, contig = region_edges(grid, f4)

    def inb(T):
        return f4_check(preds, cells, float(T), study_a)["f4_pass"]

    refined = {"lower": None, "upper": None}
    if lo is not None:
        gi = int(np.flatnonzero(grid >= lo)[0])
        refined["lower"] = (refine_edge(float(grid[gi - 1]), lo, False, inb)
                            if gi > 0 else float(grid[0]))
        gj = int(np.flatnonzero(grid <= hi)[-1])
        refined["upper"] = (refine_edge(hi, float(grid[gj + 1]), True, inb)
                            if gj < len(grid) - 1 else float(grid[-1]))

    # which criterion binds just outside each edge
    def binding(T):
        chk = f4_check(preds, cells, float(T), study_a)
        why = []
        if not (chk["calibrated"]["reliability"] < chk["uncalibrated"]["reliability"]):
            why.append("reliability not strictly better")
        if chk["rps_change"] > 1e-9:
            why.append("RPS worse")
        if not chk["resolution_within_tol"]:
            why.append("resolution beyond +/-10%")
        return why

    verdict = {
        "acceptance_region_on_grid": [lo, hi], "contiguous_on_grid": contig,
        "acceptance_region_refined": refined,
        "n_grid_points_passing": int(f4.sum()), "n_grid_points": int(len(grid)),
        "T_J_in_region": bool(inb(T_J)),
        "band_low_in_region": bool(inb(BAND[0])),
        "band_high_in_region": bool(inb(BAND[1])),
        "band_fully_covered": bool(lo is not None and refined["lower"] is not None
                                   and refined["lower"] <= BAND[0]
                                   and refined["upper"] >= BAND[1]
                                   and f4[(grid >= BAND[0] - 1e-12) & (grid <= BAND[1] + 1e-12)].all()),
        "identity_in_region": bool(inb(1.0)),
        "binding_below_lower_edge": binding(refined["lower"] * 0.995) if refined["lower"] else None,
        "binding_above_upper_edge": binding(refined["upper"] * 1.005) if refined["upper"] else None,
    }
    print(f"[region] F4 acceptance region = [{refined['lower']}, {refined['upper']}] "
          f"(contiguous={contig}); T_J in: {verdict['T_J_in_region']}; "
          f"band covered: {verdict['band_fully_covered']}")
    print(f"[region] binds below: {verdict['binding_below_lower_edge']}; "
          f"above: {verdict['binding_above_upper_edge']}")

    # --- point reads -------------------------------------------------------
    marks = {"identity": 1.0, "band_low": BAND[0], "T_J": T_J,
             "tau_v_qwen": PANEL_TAU_V["qwen"], "band_high": BAND[1]}
    points = {name: {k: f4_check(preds, cells, T, study_a)[k]
                     for k in ("T", "reliability_improvement", "rps_change",
                               "resolution_change", "resolution_within_tol",
                               "verdict_ok", "f4_pass")}
              for name, T in marks.items()}

    fit = fit_temperature([(p, gt) for _, p, gt in rows], bounds=study_a.T_BOUNDS)
    optima = {
        "T_supervised_rps_fit": fit.temperature,
        "T_supervised_saturated": study_a.saturated(fit.temperature),
        "T_min_reliability_on_grid": float(grid[int(np.argmin(curves["reliability"]))]),
        "T_min_rps_on_grid": float(grid[int(np.argmin(curves["mean_rps"]))]),
        "T_min_w1_on_grid": float(grid[int(np.argmin(curves["mean_w1"]))]),
    }
    print(f"[optima] supervised T* = {optima['T_supervised_rps_fit']:.4f}; "
          f"grid argmin reliability {optima['T_min_reliability_on_grid']:.4f}, "
          f"RPS {optima['T_min_rps_on_grid']:.4f}")

    # --- cross-check against the committed across-band profile -------------
    cross = None
    if args.cross_check:
        ref = json.loads(Path(args.cross_check).read_text())["arm1_transferred_T_J"]["band_profile"]
        rowsx, agree = [], True
        for r in ref:
            i = int(np.argmin(np.abs(grid - r["T"])))
            assert abs(grid[i] - r["T"]) < 1e-9, f"profile T={r['T']} not on this grid"
            mine = {"reliability_improvement": float(curves["reliability_improvement"][i]),
                    "rps_change": float(curves["rps_change"][i]),
                    "resolution_change": float(curves["resolution_change"][i]),
                    "f4_pass": bool(f4[i])}
            same = all(mine[k] == r[k] for k in mine)
            agree &= same
            rowsx.append({"T": r["T"], "committed": {k: r[k] for k in mine}, "this_sweep": mine,
                          "identical": same})
        cross = {"source": str(args.cross_check), "n_points": len(ref),
                 "all_identical": bool(agree), "points": rowsx}
        print(f"[cross-check] committed 9-point band profile reproduced exactly: {agree}")

    # --- doc-clustered bootstrap on the region edges ------------------------
    rng = np.random.default_rng(args.seed)
    uniq_docs = sorted(set(docs.tolist()))
    doc_items = {d: np.flatnonzero(docs == d) for d in uniq_docs}
    base_gids = group_ids(base_items)
    base_pred_cdfs = cdf_matrix([i.prediction for i in base_items])
    base_rps = np.array([i.rps for i in base_items])
    B = args.bootstrap
    boot_lo, boot_hi = [], []
    empty = noncontig = 0
    covers_band = 0
    pass_counts = {name: 0 for name in marks}
    mark_idx = {name: int(np.argmin(np.abs(grid - T))) for name, T in marks.items()}
    band_mask = (grid >= BAND[0] - 1e-12) & (grid <= BAND[1] + 1e-12)
    for _ in range(B):
        w = np.zeros(len(rows))
        for d in rng.choice(uniq_docs, size=len(uniq_docs), replace=True):
            w[doc_items[d]] += 1.0
        b_rel0, b_res0, b_rps0 = weighted_murphy(w, gt_cdfs, base_pred_cdfs, base_gids, base_rps)
        rel_b = np.empty(len(grid)); res_b = np.empty(len(grid)); rps_b = np.empty(len(grid))
        for i in range(len(grid)):
            rel_b[i], res_b[i], rps_b[i] = weighted_murphy(
                w, gt_cdfs, per_T[i]["pred_cdfs"], per_T[i]["gids"], per_T[i]["rps"])
        bi, _, _, _ = f4_indicator(rel_b, res_b, rps_b, b_rel0, b_res0, b_rps0)
        for name, i in mark_idx.items():
            pass_counts[name] += int(bi[i])
        l, h, c = region_edges(grid, bi)
        if l is None:
            empty += 1
            continue
        if not c:
            noncontig += 1
        boot_lo.append(l); boot_hi.append(h)
        if l <= BAND[0] + 1e-12 and h >= BAND[1] - 1e-12 and bi[band_mask].all():
            covers_band += 1
    lo_arr, hi_arr = np.array(boot_lo), np.array(boot_hi)
    bootstrap = {
        "B": B, "seed": args.seed, "cluster": "document",
        "n_clusters": len(uniq_docs), "empty_regions": empty,
        "noncontiguous_regions": noncontig,
        "lower_edge": {"median": float(np.median(lo_arr)),
                       "p2.5": float(np.percentile(lo_arr, 2.5)),
                       "p97.5": float(np.percentile(lo_arr, 97.5))} if lo_arr.size else None,
        "upper_edge": {"median": float(np.median(hi_arr)),
                       "p2.5": float(np.percentile(hi_arr, 2.5)),
                       "p97.5": float(np.percentile(hi_arr, 97.5))} if hi_arr.size else None,
        "f4_pass_rate": {name: pass_counts[name] / B for name in marks},
        "p_band_fully_covered": covers_band / B,
        "note": "region edges are quantized to the sweep grid inside the bootstrap",
    }
    if lo_arr.size:
        print(f"[bootstrap] B={B}: lower edge {bootstrap['lower_edge']['median']:.4f} "
              f"[{bootstrap['lower_edge']['p2.5']:.4f}, {bootstrap['lower_edge']['p97.5']:.4f}], "
              f"upper edge {bootstrap['upper_edge']['median']:.4f} "
              f"[{bootstrap['upper_edge']['p2.5']:.4f}, {bootstrap['upper_edge']['p97.5']:.4f}]")
    print("[bootstrap] F4 pass rates: "
          + ", ".join(f"{k}={v:.3f}" for k, v in bootstrap["f4_pass_rate"].items())
          + f"; P(band covered)={bootstrap['p_band_fully_covered']:.3f}")

    dump = {
        "meta": {
            "instrument": "range_robustness_verbalized.py",
            "channel": "verbalized",
            "closed_run": args.closed_run,
            "n_items": len(rows), "n_documents": len(uniq_docs),
            "join": "item_label identity join; level order from the contract's label names",
            "unmatched_predictions": unmatched_pred, "unmatched_cells": unmatched_gt,
            "epsilon_floor_on_closed_predictions": args.eps,
            "exact_zeros_in_stored_vectors": n_zero,
            "gt": "aireg.load_cells() canonical bundle (re-materialized 2026-07-09, "
                  "freethresh, readout tau=0.675, continuous)",
            "murphy_bins": 10,
            "f4": "reliability strictly better AND mean RPS <= uncalibrated + 1e-9 AND "
                  "|resolution change| <= 10% of uncalibrated resolution",
            "frozen": {"T_J": T_J, "band": list(BAND), "panel_tau_v": PANEL_TAU_V},
            "grid": {"range": list(SWEEP_RANGE), "n_logspaced": SWEEP_N,
                     "n_inband_extra": INBAND_N, "n_total": int(len(grid)),
                     "markers": "T_J, both band edges, panel tau_v, identity, "
                                "the 9 across-band profile points"},
        },
        "uncalibrated": base,
        "identity_roundtrip_diagnostic": t1_gap,
        "grid_T": grid.tolist(),
        "curves": {k: v.tolist() for k, v in curves.items()},
        "f4_membership": f4.tolist(),
        "verdict": verdict,
        "point_reads": points,
        "optima": optima,
        "cross_check_band_profile": cross,
        "bootstrap": bootstrap,
    }
    p = out / "range_robustness_verbalized.json"
    p.write_text(json.dumps(dump, indent=2))
    print(f"[out] wrote {p}")


if __name__ == "__main__":
    main()
