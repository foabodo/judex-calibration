#!/usr/bin/env python3
"""Q4 range-robustness sweep — is the closed-side correction flat in tau?

Study A returned Q3-NEGATIVE (gate-passing tau_oc = {qwen 1.60, llama31 1.86,
gemma31 4.88}; max/min 3.05 > 2), so no transferable constant exists. This sweep
tests the salvage hypothesis: the closed-side benefit may not need the *right*
tau. It reproduces the Q4 computation (study_a.closed_side_check) standalone,
sweeps tau over a fine log grid, and reports the **benefit band** — the
tau-interval over which Murphy reliability improves while resolution and RPS do
not degrade beyond a stated tolerance — plus a doc-clustered bootstrap on the
band edges and per-Article / per-GT-level heterogeneity.

INDICATIVE / OFF-PAIR: the default closed run (stage9-gemini-gpt-medium) is the
LEGACY Gemini+GPT pair — the only 120-cell run on disk — NOT the current
Claude+GPT pair (Sonnet 4.6 medium + GPT 5.4 medium). Nothing here is an E6
result and nothing here may be pasted into any pipeline.yaml; the real Q4/E6
runs after the held on-pair sweep.

Orientation pin (E3 cost engine is orientation-sensitive): the 5-level scale is
positional very_low..very_high = A..E; grade 1 = very_low.

Run from the calibration worktree:
  JUDEX_UMBRELLA=/path/to/judex PYTHONPATH=src python scripts/q4_range_robustness.py \
      --out <dir> [--regression-runs-dir /path/to/main-checkout/judex-calibration/runs]

GT: judex_calibration.aireg.load_cells() ONLY (canonical manifest-verified
bundle, re-materialized 2026-07-09, readout tau=0.675, continuous). Never GT
from a runs/ metrics_report.
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

# The measured open-panel range this analysis interrogates (gate-passing tau_oc
# min .. GLM soft max). GLM's base leg passed the resolution gate only marginally,
# so 8.84 is the soft outer edge; the hard gate-passing spread is 1.60-4.88.
REFERENCE_TAUS = {
    "tau_oc_qwen": 1.6007889248849985,
    "tau_oc_llama31": 1.8571440700827275,
    "tau_oc_gemma31": 4.877199362107526,
    "tau_oc_glm_soft": 8.835204542762193,
    "identity": 1.0,
}
OPEN_PANEL_RANGE = (REFERENCE_TAUS["tau_oc_qwen"], REFERENCE_TAUS["tau_oc_glm_soft"])
HARD_GATE_RANGE = (REFERENCE_TAUS["tau_oc_qwen"], REFERENCE_TAUS["tau_oc_gemma31"])


def default_umbrella() -> Path:
    env = os.environ.get("JUDEX_UMBRELLA")
    if env:
        return Path(env)
    # worktrees live under <umbrella>/worktrees/<name>; the main checkout is a
    # direct child of the umbrella.
    if REPO.parent.name == "worktrees":
        return REPO.parent.parent
    return REPO.parent


def load_closed_predictions(umbrella: Path, run_id: str) -> dict[str, list[float]]:
    path = umbrella / "judex-evaluator" / "runs" / run_id / "metrics_report.json"
    if not path.exists():
        raise SystemExit(f"no metrics_report.json at {path} (judex-evaluator/runs/ is gitignored; "
                         f"this analysis only runs on a host that has the closed run)")
    report = json.loads(path.read_text())
    return {it["item_label"]: it["prediction"]["probabilities"] for it in report["items"]}


# ---------------------------------------------------------------------------
# Production-path evaluation (study_a / judex-evaluator verbatim)
# ---------------------------------------------------------------------------

def build_items(closed, cells, study_a, ComplianceDistribution):
    """(label, pred_dist, gt_dist, document_id, article, gt_argmax) per overlapping cell."""
    by_label = {c.item_label: c for c in cells}
    rows = []
    for label, probs in closed.items():
        c = by_label.get(label)
        if c is None:
            continue
        rows.append((label,
                     study_a._pred_dist(probs, c.gt_labels),
                     ComplianceDistribution.from_values(c.gt_probs, c.gt_labels),
                     c.document_id, c.article, c.gt_argmax))
    return rows


def eval_at(tau, rows, study_a, apply_temperature, murphy_decomposition):
    """closed_side_check's arithmetic at one tau, via the production helpers."""
    items = [study_a._metric_item(label, pred if tau is None else apply_temperature(pred, tau), gt)
             for label, pred, gt, *_ in rows]
    m = murphy_decomposition(items)  # bins=10, matching closed_side_check
    return {
        "reliability": m["reliability"], "resolution": m["resolution"],
        "uncertainty": m["uncertainty"],
        "mean_rps": sum(i.rps for i in items) / len(items),
        "mean_w1": sum(i.w1 for i in items) / len(items),
        "argmax_acc": sum(i.argmax_agreement for i in items) / len(items),
    }, items


def regression_check(closed, cells, runs_dir: Path, study_a):
    """Recompute the shipped Q4 blocks at each report's own tau and diff every field."""
    results = {}
    for name in ("qwen_k5", "llama31", "gemma31_k5", "glm", "study_a_4family"):
        path = runs_dir / name / "study_a_report.json"
        if not path.exists():
            results[name] = {"status": "SKIPPED", "reason": f"missing {path}"}
            continue
        report = json.loads(path.read_text())
        expected = report.get("closed_side_check_Q4")
        if not expected or expected.get("skipped"):
            results[name] = {"status": "SKIPPED", "reason": "no Q4 block in report"}
            continue
        got = study_a.closed_side_check(closed, cells, expected["T"])
        diffs = {k: {"expected": expected[k], "got": got[k]}
                 for k in expected if got.get(k) != expected[k]}
        results[name] = {"status": "PASS" if not diffs else "FAIL",
                         "T": expected["T"], "diffs": diffs}
    return results


# ---------------------------------------------------------------------------
# Vectorized replica for the bootstrap (asserted equal to the production path
# at unit weights before use)
# ---------------------------------------------------------------------------

def cdf_matrix(dists) -> np.ndarray:
    """Interior CDF vectors (n x K-1), matching metrics_report._murphy_cdf."""
    return np.array([np.cumsum(d.probabilities)[:-1] for d in dists], dtype=float)


def group_ids(items) -> np.ndarray:
    """(argmax level, confidence decile) group per item as a dense int id,
    matching metrics_report.calibration_groups at bins=10."""
    keys = []
    for it in items:
        conf = it.prediction.probabilities[it.predicted_argmax_index]
        keys.append((it.predicted_argmax_index, min(int(conf * 10), 9)))
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([uniq[k] for k in keys], dtype=int)


def weighted_murphy(w, gt_cdfs, pred_cdfs, gids, rps):
    """Multiplicity-weighted Murphy decomposition + mean RPS (w = item weights)."""
    n = w.sum()
    clim = (w @ gt_cdfs) / n
    unc = float(w @ np.mean((gt_cdfs - clim) ** 2, axis=1) / n)
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
    return rel, res, unc, float(w @ rps / n)


def band_indicator(rel, res, rps, rel0, res0, rps0, res_tol_frac):
    """Benefit-band membership per grid point: reliability strictly improves AND
    RPS does not get worse (closed_side_check's own verdict criterion) AND
    resolution does not degrade by more than res_tol_frac of its uncalibrated
    value."""
    return (rel < rel0 - 1e-12) & (rps <= rps0 + 1e-9) & (res >= res0 - res_tol_frac * res0)


def band_edges(grid, ind):
    """(lower, upper, contiguous) of the indicator's support on the grid."""
    idx = np.flatnonzero(ind)
    if idx.size == 0:
        return None, None, True
    return float(grid[idx[0]]), float(grid[idx[-1]]), bool(ind[idx[0]:idx[-1] + 1].all())


def refine_edge(lo, hi, keep_lo, evalf, steps=30):
    """Bisect a band edge between adjacent grid points; keep_lo = whether the
    in-band side is at lo."""
    for _ in range(steps):
        mid = math.sqrt(lo * hi)
        if evalf(mid) == keep_lo:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed-run", default="stage9-gemini-gpt-medium",
                    help="OFF-PAIR legacy Gemini+GPT run — the only 120-cell run on disk")
    ap.add_argument("--umbrella", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regression-runs-dir", default=None,
                    help="judex-calibration/runs dir holding the shipped Q4 blocks "
                         "(gitignored, main-checkout-only)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--res-tol", type=float, default=0.10,
                    help="tolerated resolution degradation as a fraction of the "
                         "uncalibrated resolution")
    args = ap.parse_args()

    umbrella = Path(args.umbrella) if args.umbrella else default_umbrella()
    os.environ.setdefault("JUDEX_UMBRELLA", str(umbrella))

    # Imports AFTER the env pin so aireg/study_a resolve siblings from a worktree.
    from judex_calibration import study_a
    from judex_calibration.aireg import load_cells
    from judex.calibration import apply_temperature, fit_temperature
    from judex.metrics_report import murphy_decomposition
    from judex.core.distributions import ComplianceDistribution

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    closed = load_closed_predictions(umbrella, args.closed_run)
    rows = build_items(closed, cells, study_a, ComplianceDistribution)
    print(f"[load] {len(rows)} overlapping cells from {args.closed_run} (OFF-PAIR legacy run)")

    # -- 1. Regression: reproduce the shipped Q4 blocks ---------------------
    runs_dir = Path(args.regression_runs_dir) if args.regression_runs_dir else None
    regression = {}
    if runs_dir:
        regression = regression_check(closed, cells, runs_dir, study_a)
        for name, r in regression.items():
            print(f"[regression] {name}: {r['status']}"
                  + (f" (T={r['T']:.4f})" if "T" in r else f" — {r.get('reason','')}"))
        if any(r["status"] == "FAIL" for r in regression.values()):
            raise SystemExit("regression against shipped Q4 blocks FAILED — do not trust the sweep")

    # -- 2. Sweep -----------------------------------------------------------
    # Ceiling 40: probing showed the band closes between tau=20 and tau=30
    # (reliability and RPS both re-cross their uncalibrated values there), so 40
    # brackets the upper edge with margin; the tau->inf (uniform) limit is
    # evaluated separately as a diagnostic.
    grid = np.unique(np.concatenate([
        np.geomspace(0.75, 40.0, 73),
        np.geomspace(1.0, 10.0, 121),
        np.array(sorted(REFERENCE_TAUS.values())),
    ]))
    base_metrics, base_items = eval_at(None, rows, study_a, apply_temperature, murphy_decomposition)
    print(f"[base] uncalibrated: rel {base_metrics['reliability']:.5f} "
          f"res {base_metrics['resolution']:.5f} rps {base_metrics['mean_rps']:.5f}")

    curves = {k: [] for k in ("reliability", "resolution", "uncertainty",
                              "mean_rps", "mean_w1", "argmax_acc")}
    per_tau = []  # bootstrap ingredients per grid point
    per_item_rps = {}  # tau_name -> per-item rps deltas for heterogeneity
    gt_cdfs = cdf_matrix([gt for _, _, gt, *_ in rows])
    for tau in grid:
        m, items = eval_at(float(tau), rows, study_a, apply_temperature, murphy_decomposition)
        for k in curves:
            curves[k].append(m[k])
        per_tau.append({
            "pred_cdfs": cdf_matrix([it.prediction for it in items]),
            "gids": group_ids(items),
            "rps": np.array([it.rps for it in items]),
            "w1": np.array([it.w1 for it in items]),
        })
    curves = {k: np.array(v) for k, v in curves.items()}

    # identity-tau sanity: apply_temperature at T=1 must round-trip
    i1 = int(np.flatnonzero(np.isclose(grid, 1.0))[0])
    assert abs(curves["mean_rps"][i1] - base_metrics["mean_rps"]) < 1e-12, \
        "apply_temperature(T=1) does not round-trip"

    # vectorized-replica sanity at unit weights, every grid point
    ones = np.ones(len(rows))
    for i, tau in enumerate(grid):
        rel, res, unc, mrps = weighted_murphy(ones, gt_cdfs, per_tau[i]["pred_cdfs"],
                                              per_tau[i]["gids"], per_tau[i]["rps"])
        assert max(abs(rel - curves["reliability"][i]), abs(res - curves["resolution"][i]),
                   abs(mrps - curves["mean_rps"][i])) < 1e-10, f"replica mismatch at tau={tau}"

    # argmax invariance audit: temperature scaling preserves ordering, so any
    # argmax shift is a floating-point tie-break, not a real re-ranking
    base_argmax = np.array([it.predicted_argmax_index for it in base_items])
    shift_report = {}
    for name, tau in REFERENCE_TAUS.items():
        _, items = eval_at(tau, rows, study_a, apply_temperature, murphy_decomposition)
        moved = [(rows[j][0], int(base_argmax[j]), int(items[j].predicted_argmax_index),
                  [round(p, 4) for p in rows[j][1].probabilities])
                 for j in range(len(rows))
                 if items[j].predicted_argmax_index != base_argmax[j]]
        if moved:
            shift_report[name] = moved

    # -- 3. Band ------------------------------------------------------------
    rel0, res0, rps0 = (base_metrics["reliability"], base_metrics["resolution"],
                        base_metrics["mean_rps"])
    ind = band_indicator(curves["reliability"], curves["resolution"], curves["mean_rps"],
                         rel0, res0, rps0, args.res_tol)
    ind_strict = band_indicator(curves["reliability"], curves["resolution"], curves["mean_rps"],
                                rel0, res0, rps0, 0.0)
    lo, hi, contig = band_edges(grid, ind)
    lo_s, hi_s, contig_s = band_edges(grid, ind_strict)

    def in_band(tau, tol):
        m, _ = eval_at(tau, rows, study_a, apply_temperature, murphy_decomposition)
        return bool(band_indicator(np.array([m["reliability"]]), np.array([m["resolution"]]),
                                   np.array([m["mean_rps"]]), rel0, res0, rps0, tol)[0])

    refined = {}
    for tag, (l, h, tol) in {"tolerant": (lo, hi, args.res_tol),
                             "strict": (lo_s, hi_s, 0.0)}.items():
        if l is None:
            refined[tag] = {"lower": None, "upper": None}
            continue
        gi = int(np.flatnonzero(grid >= l)[0])
        lo_ref = (refine_edge(grid[gi - 1], l, False, lambda t: in_band(t, tol))
                  if gi > 0 and not in_band(grid[gi - 1], tol) else float(grid[0]))
        gj = int(np.flatnonzero(grid <= h)[-1])
        hi_ref = (refine_edge(h, grid[gj + 1], True, lambda t: in_band(t, tol))
                  if gj < len(grid) - 1 and not in_band(grid[gj + 1], tol) else float(grid[-1]))
        refined[tag] = {"lower": lo_ref, "upper": hi_ref}

    optima = {
        "tau_star_reliability": float(grid[int(np.argmin(curves["reliability"]))]),
        "tau_star_rps": float(grid[int(np.argmin(curves["mean_rps"]))]),
        "tau_star_w1": float(grid[int(np.argmin(curves["mean_w1"]))]),
        "tau_star_resolution": float(grid[int(np.argmax(curves["resolution"]))]),
    }
    fit = fit_temperature([(pred, gt) for _, pred, gt, *_ in rows], bounds=study_a.T_BOUNDS)
    optima["T_supervised_rps_fit"] = fit.temperature
    optima["T_supervised_saturated"] = study_a.saturated(fit.temperature)
    print("[optima] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in optima.items()))

    # Diagnostics: (a) the tau=1 bin-edge artifact — apply_temperature(T=1)
    # round-trips probabilities to ~1e-16 but items whose argmax confidence sits
    # exactly on a decile edge can change calibration group, moving reliability/
    # resolution without moving any distribution. Reliability deltas below this
    # magnitude are grouping noise. (b) the tau->inf (uniform) limit.
    uni_items = [study_a._metric_item(label, ComplianceDistribution.from_values(
        [1.0 / len(pred.probabilities)] * len(pred.probabilities), pred.labels), gt)
        for label, pred, gt, *_ in rows]
    mu = murphy_decomposition(uni_items)
    diagnostics = {
        "tau1_bin_edge_artifact": {
            "reliability_at_tau1_minus_uncalibrated": float(curves["reliability"][i1] - rel0),
            "resolution_at_tau1_minus_uncalibrated": float(curves["resolution"][i1]
                                                           - base_metrics["resolution"]),
            "note": "pure calibration-group reassignment at bin edges; distributions are "
                    "numerically identical at tau=1 (RPS delta < 1e-12)",
        },
        "uniform_limit": {
            "reliability": mu["reliability"], "resolution": mu["resolution"],
            "mean_rps": sum(i.rps for i in uni_items) / len(uni_items),
            "note": "tau->inf limit; decisively out of band (resolution -> 0, RPS ~2x worse)",
        },
    }

    def covers(lo_, hi_, rng):
        if lo_ is None:
            return False
        inside = (grid >= rng[0] - 1e-9) & (grid <= rng[1] + 1e-9)
        return bool(lo_ <= rng[0] + 1e-9 and hi_ >= rng[1] - 1e-9 and ind[inside].all())

    verdict = {
        "band_tolerant": {**refined["tolerant"], "contiguous_on_grid": contig,
                          "res_tol_frac": args.res_tol},
        "band_strict": {**refined["strict"], "contiguous_on_grid": contig_s},
        "covers_open_panel_range_tolerant": covers(refined["tolerant"]["lower"],
                                                   refined["tolerant"]["upper"], OPEN_PANEL_RANGE),
        "covers_hard_gate_range_tolerant": covers(refined["tolerant"]["lower"],
                                                  refined["tolerant"]["upper"], HARD_GATE_RANGE),
        "open_panel_range": list(OPEN_PANEL_RANGE),
        "hard_gate_range": list(HARD_GATE_RANGE),
    }
    print(f"[band] tolerant (res -{args.res_tol:.0%} allowed): "
          f"[{refined['tolerant']['lower']}, {refined['tolerant']['upper']}] "
          f"contiguous={contig}")
    print(f"[band] strict   (res may not drop): "
          f"[{refined['strict']['lower']}, {refined['strict']['upper']}] "
          f"contiguous={contig_s}")
    print(f"[band] covers open-panel range {OPEN_PANEL_RANGE}: "
          f"{verdict['covers_open_panel_range_tolerant']}")

    # -- 4. Heterogeneity ---------------------------------------------------
    articles = np.array([r[4] for r in rows])
    gt_levels = np.array([r[5] for r in rows])
    docs = np.array([r[3] for r in rows])
    level_names = ["very_low", "low", "moderate", "high", "very_high"]
    base_rps = np.array([it.rps for it in base_items])
    base_w1 = np.array([it.w1 for it in base_items])
    het = {}
    for name, tau in {**{k: v for k, v in REFERENCE_TAUS.items() if k != "identity"},
                      "tau_star_reliability": optima["tau_star_reliability"]}.items():
        i = int(np.argmin(np.abs(grid - tau)))
        d_rps = per_tau[i]["rps"] - base_rps
        d_w1 = per_tau[i]["w1"] - base_w1
        het[name] = {
            "tau": float(grid[i]),
            "frac_cells_rps_improved": float(np.mean(d_rps < 0)),
            "by_article": {str(a): {"n": int((articles == a).sum()),
                                    "mean_d_rps": float(d_rps[articles == a].mean()),
                                    "mean_d_w1": float(d_w1[articles == a].mean())}
                           for a in sorted(set(articles))},
            "by_gt_level": {level_names[l]: {"n": int((gt_levels == l).sum()),
                                             "mean_d_rps": float(d_rps[gt_levels == l].mean()),
                                             "mean_d_w1": float(d_w1[gt_levels == l].mean())}
                            for l in sorted(set(gt_levels.tolist()))},
        }

    # -- 5. Doc-clustered bootstrap on the band edges -----------------------
    rng = np.random.default_rng(args.seed)
    uniq_docs = sorted(set(docs.tolist()))
    doc_items = {d: np.flatnonzero(docs == d) for d in uniq_docs}
    B = args.bootstrap
    base_gids = group_ids(base_items)
    base_pred_cdfs = cdf_matrix([it.prediction for it in base_items])
    boot = {"lower": [], "upper": [], "empty": 0, "noncontiguous": 0,
            "covers_open_panel": 0, "covers_hard_gate": 0,
            "open_panel_fail_lower": 0, "open_panel_fail_upper": 0,
            "open_panel_fail_interior_gap": 0}
    n_items = len(rows)
    for b in range(B):
        w = np.zeros(n_items)
        for d in rng.choice(uniq_docs, size=len(uniq_docs), replace=True):
            w[doc_items[d]] += 1.0
        b_rel0, b_res0, _, b_rps0 = weighted_murphy(w, gt_cdfs, base_pred_cdfs,
                                                    base_gids, base_rps)
        rels = np.empty(len(grid)); ress = np.empty(len(grid)); rpss = np.empty(len(grid))
        for i in range(len(grid)):
            rels[i], ress[i], _, rpss[i] = weighted_murphy(
                w, gt_cdfs, per_tau[i]["pred_cdfs"], per_tau[i]["gids"], per_tau[i]["rps"])
        bi = band_indicator(rels, ress, rpss, b_rel0, b_res0, b_rps0, args.res_tol)
        l, h, c = band_edges(grid, bi)
        if l is None:
            boot["empty"] += 1
            continue
        if not c:
            boot["noncontiguous"] += 1
        boot["lower"].append(l); boot["upper"].append(h)
        for rng_name, rng_ in (("covers_open_panel", OPEN_PANEL_RANGE),
                               ("covers_hard_gate", HARD_GATE_RANGE)):
            inside = (grid >= rng_[0] - 1e-9) & (grid <= rng_[1] + 1e-9)
            if l <= rng_[0] + 1e-9 and h >= rng_[1] - 1e-9 and bi[inside].all():
                boot[rng_name] += 1
            elif rng_name == "covers_open_panel":
                if l > rng_[0] + 1e-9:
                    boot["open_panel_fail_lower"] += 1
                elif h < rng_[1] - 1e-9:
                    boot["open_panel_fail_upper"] += 1
                else:
                    boot["open_panel_fail_interior_gap"] += 1
    lo_arr, hi_arr = np.array(boot["lower"]), np.array(boot["upper"])
    bootstrap = {
        "B": B, "seed": args.seed, "empty_bands": boot["empty"],
        "noncontiguous_bands": boot["noncontiguous"],
        "lower_edge": {"median": float(np.median(lo_arr)),
                       "p2.5": float(np.percentile(lo_arr, 2.5)),
                       "p97.5": float(np.percentile(lo_arr, 97.5))},
        "upper_edge": {"median": float(np.median(hi_arr)),
                       "p2.5": float(np.percentile(hi_arr, 2.5)),
                       "p97.5": float(np.percentile(hi_arr, 97.5))},
        "p_covers_open_panel_range": boot["covers_open_panel"] / B,
        "p_covers_hard_gate_range": boot["covers_hard_gate"] / B,
        "open_panel_noncoverage_breakdown": {
            "lower_edge_above_1.60": boot["open_panel_fail_lower"],
            "upper_edge_below_8.84": boot["open_panel_fail_upper"],
            "interior_grid_point_dropout": boot["open_panel_fail_interior_gap"],
        },
        "note": "band edges quantized to the sweep grid inside the bootstrap; "
                "cluster = document (24 docs x 5 Articles)",
    }
    print(f"[bootstrap] B={B}: lower edge {bootstrap['lower_edge']['median']:.3f} "
          f"[{bootstrap['lower_edge']['p2.5']:.3f}, {bootstrap['lower_edge']['p97.5']:.3f}], "
          f"upper edge {bootstrap['upper_edge']['median']:.3f} "
          f"[{bootstrap['upper_edge']['p2.5']:.3f}, {bootstrap['upper_edge']['p97.5']:.3f}]")
    print(f"[bootstrap] P(band covers open-panel range) = "
          f"{bootstrap['p_covers_open_panel_range']:.3f}; "
          f"P(covers hard-gate range) = {bootstrap['p_covers_hard_gate_range']:.3f}; "
          f"empty={boot['empty']} noncontiguous={boot['noncontiguous']}")

    # -- 6. Dump ------------------------------------------------------------
    dump = {
        "meta": {
            "date": "2026-07-19",
            "closed_run": args.closed_run,
            "off_pair_warning": "stage9-gemini-gpt-medium is the LEGACY Gemini+GPT pair, NOT "
                                "the current Claude+GPT pair. Indicative only; never an E6 "
                                "result; never paste any temperature into pipeline.yaml.",
            "gt": "canonical aireg.load_cells() bundle (re-materialized 2026-07-09; freethresh, "
                  "readout tau=0.675, continuous)",
            "orientation_pin": "very_low..very_high = A..E; grade 1 = very_low",
            "murphy_bins": 10,
            "n_cells": len(rows),
            "grid": {"min": float(grid[0]), "max": float(grid[-1]), "n": int(len(grid)),
                     "spec": "geomspace(0.75,14,49) U geomspace(1,10,121) U reference taus"},
            "reference_taus": REFERENCE_TAUS,
            "band_criteria": "reliability strictly improves AND mean RPS <= uncalibrated + 1e-9 "
                             "AND resolution >= uncalibrated * (1 - res_tol)",
            "res_tol_frac": args.res_tol,
            "study_a_noise_note": "single-measurement tau_oc noise band ~ +/-20% "
                                  "(per-leg T ~ +/-10-15%), 2026-07-19 convention",
        },
        "regression_vs_shipped_q4": regression,
        "uncalibrated": base_metrics,
        "grid_taus": grid.tolist(),
        "curves": {k: v.tolist() for k, v in curves.items()},
        "band_membership_tolerant": ind.tolist(),
        "band_membership_strict": ind_strict.tolist(),
        "verdict": verdict,
        "optima": optima,
        "diagnostics": diagnostics,
        "argmax_shifts_at_reference_taus": shift_report,
        "heterogeneity": het,
        "bootstrap": bootstrap,
    }
    (out / "q4_range_robustness.json").write_text(json.dumps(dump, indent=2))
    print(f"[out] wrote {out / 'q4_range_robustness.json'}")


if __name__ == "__main__":
    main()
