#!/usr/bin/env python3
"""2(d) band-mechanism formalization: zero-fit fallback + composite decision protocol.

$0, analysis-only. Characterizes a fixed T = 3.761 (the pre-registered band's
geometric midpoint, [1.60, 8.84]) as a zero-supervision fallback, quantifies
direction-safety and bounded-regret guarantees for any T inside the band, and
computes the concurrence table feeding the composite decision protocol drafted
in spec/analysis_2026_07_XX_band_mechanism_composite.md.

Every target is scored with judex_calibration.study_a.closed_side_check
(Murphy decomposition at bins=10, verbatim production path) applied to a
FIXED T — never fit against the target. GT: aireg.load_cells() only. GT is
used to SCORE every mechanism's output, never to select 3.761 (which is fixed
by the pre-existing band, chosen before this script ran).

Targets (7): 4 open post legs (gate-passing; the promotion doc's honest
validation set minus the augmented closed legs) + 2 legacy closed runs
(stage9-gemini-gpt-medium, phase23-deference-fix-native; OFF-PAIR, indicative)
+ glm_pre (known blind-spot control; failed the Study A gate).

Run from this worktree:
  JUDEX_UMBRELLA=<umbrella> PYTHONPATH=src \
    <conda judex-arm>/bin/python scripts/band_mechanism_composite.py --out <dir>
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

# ---------------------------------------------------------------------------
# Frozen constants (fixed by prior, already-merged instruments; not fit here)
# ---------------------------------------------------------------------------
BAND = (1.60, 8.84)  # pre-registered E6 sensitivity band (open-panel range)
ANCHOR_MIDPOINT = math.sqrt(BAND[0] * BAND[1])  # 3.7607... — the zero-fit point
TRANSFERRED_CONSTANT = 1.86  # median(tau_oc) baseline, Q3-dead as a point mechanism
TOLERANT_BAND_STAGE9 = (1.00, 22.8)  # q4_range_robustness verdict, stage9-only

# Frozen pool-shrunk T-hat per target, verbatim from
# spec/analysis_2026_07_19_dispersion_pool_promotion.md Table (Task 1b / Task 2).
# NOT recomputed here — reused as a fixed T to score, exactly like 3.761.
POOL_THAT = {
    "stage9": 6.013,
    "phase23": 6.280,
    "qwen_post": 1.387,
    "llama31_post": 3.007,
    "glm_post": 3.871,
    "gemma31_post": 5.373,
    "glm_pre": 2.130,
}

RES_TOL_FRAC = 0.10  # q4_range_robustness's resolution tolerance, reused verbatim


def default_umbrella() -> Path:
    env = os.environ.get("JUDEX_UMBRELLA")
    if env:
        return Path(env)
    if REPO.parent.name == "worktrees":
        return REPO.parent.parent
    return REPO.parent


def load_closed_predictions(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        raise SystemExit(f"missing {path} (gitignored judex-evaluator/runs/ — needs a host with the legacy run)")
    report = json.loads(path.read_text())
    return {it["item_label"]: it["prediction"]["probabilities"] for it in report["items"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbrella", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260719)
    args = ap.parse_args()

    umbrella = Path(args.umbrella) if args.umbrella else default_umbrella()
    os.environ.setdefault("JUDEX_UMBRELLA", str(umbrella))

    from judex_calibration import study_a
    from judex_calibration.aireg import load_cells
    from judex.calibration import apply_temperature, fit_temperature
    from judex.metrics_report import murphy_decomposition
    from judex.core.distributions import ComplianceDistribution

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    by_label = {c.item_label: c for c in cells}

    # -- 1. Assemble targets: (name, preds dict, kind, off_pair) -------------
    calib_runs = umbrella / "judex-calibration" / "runs"
    eval_runs = umbrella / "judex-evaluator" / "runs"
    targets_spec = [
        ("qwen_post", calib_runs / "qwen_k5" / "post.json", "open_post", False),
        ("llama31_post", calib_runs / "llama31" / "post.json", "open_post", False),
        ("glm_post", calib_runs / "glm" / "post.json", "open_post", False),
        ("gemma31_post", calib_runs / "gemma31_k5" / "post.json", "open_post", False),
        ("glm_pre", calib_runs / "glm" / "pre.json", "blind_spot_control", False),
        ("stage9", eval_runs / "stage9-gemini-gpt-medium" / "metrics_report.json", "legacy_closed", True),
        ("phase23", eval_runs / "phase23-deference-fix-native" / "metrics_report.json", "legacy_closed", True),
    ]

    def build_rows(preds):
        rows = []
        for label, probs in preds.items():
            c = by_label.get(label)
            if c is None:
                continue
            pred = study_a._pred_dist(probs, c.gt_labels)
            gt = ComplianceDistribution.from_values(c.gt_probs, c.gt_labels)
            rows.append((label, pred, gt, c.document_id, c.article, c.gt_argmax))
        return rows

    def eval_at(T, rows):
        items = [study_a._metric_item(label, pred if T is None else apply_temperature(pred, T), gt)
                 for label, pred, gt, *_ in rows]
        m = murphy_decomposition(items, bins=10)
        return {
            "reliability": m["reliability"], "resolution": m["resolution"],
            "uncertainty": m["uncertainty"],
            "mean_rps": sum(i.rps for i in items) / len(items),
            "mean_w1": sum(i.w1 for i in items) / len(items),
            "argmax_acc": sum(i.argmax_agreement for i in items) / len(items),
        }, items

    def cdf_matrix(dists):
        return np.array([np.cumsum(d.probabilities)[:-1] for d in dists], dtype=float)

    def group_ids(items):
        keys = []
        for it in items:
            conf = it.prediction.probabilities[it.predicted_argmax_index]
            keys.append((it.predicted_argmax_index, min(int(conf * 10), 9)))
        uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
        return np.array([uniq[k] for k in keys], dtype=int)

    def weighted_murphy(w, gt_cdfs, pred_cdfs, gids, rps, w1):
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
        return rel, res, float(w @ rps / n), float(w @ w1 / n)

    results = {}
    rng_master = np.random.default_rng(args.seed)
    band_grid = np.unique(np.concatenate([
        np.geomspace(BAND[0], BAND[1], 61),
        np.array([ANCHOR_MIDPOINT, TRANSFERRED_CONSTANT]),
    ]))

    for name, path, kind, off_pair in targets_spec:
        preds = (json.loads(path.read_text()) if kind != "legacy_closed"
                 else load_closed_predictions(path))
        rows = build_rows(preds)
        if not rows:
            results[name] = {"status": "SKIPPED", "reason": "no overlapping cells"}
            continue

        pairs = [(pred, gt) for _, pred, gt, *_ in rows]
        T_star = fit_temperature(pairs, bounds=study_a.T_BOUNDS).temperature
        T_star_saturated = study_a.saturated(T_star)

        m0, items0 = eval_at(None, rows)
        fixed_points = {
            "identity": 1.0,
            "transferred_constant_1.86": TRANSFERRED_CONSTANT,
            "zero_fit_3.761": ANCHOR_MIDPOINT,
            "pool_shrunk_that": POOL_THAT[name],
            "supervised_Tstar": T_star,
        }
        scored = {}
        per_T_items = {}
        for tag, T in fixed_points.items():
            m, items = eval_at(T, rows)
            per_T_items[tag] = items
            scored[tag] = {**m, "T": T}

        rps0 = m0["mean_rps"]
        rps_star = scored["supervised_Tstar"]["mean_rps"]
        denom = rps0 - rps_star
        small_gap = denom < 0.005

        def gap_closed(rps_T):
            if abs(denom) < 1e-12:
                return None
            return (rps0 - rps_T) / denom

        gap_closure = {tag: gap_closed(scored[tag]["mean_rps"]) for tag in fixed_points
                       if tag != "supervised_Tstar"}

        # -- band-criteria membership at the fixed points (direction-safety) --
        def band_ok(m, tol=RES_TOL_FRAC):
            return {
                "reliability_improves": bool(m["reliability"] < m0["reliability"] - 1e-12),
                "rps_not_worse": bool(m["mean_rps"] <= rps0 + 1e-9),
                "resolution_within_tol": bool(m["resolution"] >= m0["resolution"] - tol * m0["resolution"]),
                "d_reliability": m["reliability"] - m0["reliability"],
                "d_resolution": m["resolution"] - m0["resolution"],
                "d_rps": m["mean_rps"] - rps0,
            }

        band_membership = {tag: band_ok(scored[tag]) for tag in ("zero_fit_3.761",)}
        band_membership["band_lower_1.60"] = band_ok(eval_at(BAND[0], rows)[0])
        band_membership["band_upper_8.84"] = band_ok(eval_at(BAND[1], rows)[0])

        # -- doc-clustered bootstrap on gap-closure at 3.761 and pool T-hat ---
        docs = np.array([r[3] for r in rows])
        uniq_docs = sorted(set(docs.tolist()))
        doc_items = {d: np.flatnonzero(docs == d) for d in uniq_docs}
        gt_cdfs = cdf_matrix([gt for _, _, gt, *_ in rows])
        pred_cdfs_by_tag = {tag: cdf_matrix([it.prediction for it in per_T_items[tag]])
                            for tag in fixed_points}
        gids_by_tag = {tag: group_ids(per_T_items[tag]) for tag in fixed_points}
        rps_by_tag = {tag: np.array([it.rps for it in per_T_items[tag]]) for tag in fixed_points}
        w1_by_tag = {tag: np.array([it.w1 for it in per_T_items[tag]]) for tag in fixed_points}
        n_items = len(rows)
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        B = args.bootstrap
        boot_gap = {tag: [] for tag in ("transferred_constant_1.86", "zero_fit_3.761", "pool_shrunk_that")}
        boot_delta_rel = {"zero_fit_3.761": [], "band_lower_1.60": [], "band_upper_8.84": []}
        boot_delta_res = {"zero_fit_3.761": [], "band_lower_1.60": [], "band_upper_8.84": []}
        boot_delta_rps = {"zero_fit_3.761": [], "band_lower_1.60": [], "band_upper_8.84": []}
        edge_items = {"band_lower_1.60": eval_at(BAND[0], rows)[1],
                      "band_upper_8.84": eval_at(BAND[1], rows)[1]}
        edge_pred_cdfs = {k: cdf_matrix([it.prediction for it in v]) for k, v in edge_items.items()}
        edge_gids = {k: group_ids(v) for k, v in edge_items.items()}
        edge_rps = {k: np.array([it.rps for it in v]) for k, v in edge_items.items()}
        edge_w1 = {k: np.array([it.w1 for it in v]) for k, v in edge_items.items()}
        for b in range(B):
            w = np.zeros(n_items)
            for d in rng.choice(uniq_docs, size=len(uniq_docs), replace=True):
                w[doc_items[d]] += 1.0
            rel0, res0, rps0_b, _ = weighted_murphy(w, gt_cdfs, pred_cdfs_by_tag["identity"],
                                                      gids_by_tag["identity"], rps_by_tag["identity"],
                                                      w1_by_tag["identity"])
            _, _, rps_star_b, _ = weighted_murphy(w, gt_cdfs, pred_cdfs_by_tag["supervised_Tstar"],
                                                    gids_by_tag["supervised_Tstar"],
                                                    rps_by_tag["supervised_Tstar"], w1_by_tag["supervised_Tstar"])
            denom_b = rps0_b - rps_star_b
            for tag in boot_gap:
                _, _, rps_t, _ = weighted_murphy(w, gt_cdfs, pred_cdfs_by_tag[tag], gids_by_tag[tag],
                                                   rps_by_tag[tag], w1_by_tag[tag])
                boot_gap[tag].append((rps0_b - rps_t) / denom_b if abs(denom_b) > 1e-12 else np.nan)
            for tag in boot_delta_rel:
                if tag == "zero_fit_3.761":
                    rel_t, res_t, rps_t, _ = weighted_murphy(w, gt_cdfs, pred_cdfs_by_tag[tag],
                                                               gids_by_tag[tag], rps_by_tag[tag], w1_by_tag[tag])
                else:
                    rel_t, res_t, rps_t, _ = weighted_murphy(w, gt_cdfs, edge_pred_cdfs[tag], edge_gids[tag],
                                                               edge_rps[tag], edge_w1[tag])
                boot_delta_rel[tag].append(rel_t - rel0)
                boot_delta_res[tag].append(res_t - res0)
                boot_delta_rps[tag].append(rps_t - rps0_b)

        def summarize(arr):
            a = np.array(arr, dtype=float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return None
            return {"median": float(np.median(a)), "p5": float(np.percentile(a, 5)),
                    "p95": float(np.percentile(a, 95))}

        results[name] = {
            "kind": kind, "off_pair": off_pair, "n": len(rows),
            "T_star_supervised": T_star, "T_star_saturated": T_star_saturated,
            "small_gap_branch": small_gap, "uncalibrated_mean_rps": rps0,
            "uncalibrated": m0,
            "scored": scored,
            "gap_closure_point": gap_closure,
            "gap_closure_bootstrap": {tag: summarize(v) for tag, v in boot_gap.items()},
            "band_membership": band_membership,
            "band_edge_bootstrap_deltas": {
                tag: {"d_reliability": summarize(boot_delta_rel[tag]),
                      "d_resolution": summarize(boot_delta_res[tag]),
                      "d_rps": summarize(boot_delta_rps[tag])}
                for tag in boot_delta_rel
            },
        }
        print(f"[{name}] n={len(rows)} T*={T_star:.3f}{' SAT' if T_star_saturated else ''} "
              f"gap@3.761={gap_closure['zero_fit_3.761']} gap@pool={gap_closure['pool_shrunk_that']}")

    # -- 2. Exploratory minimax fixed point over the band --------------------
    # Uses the point (non-bootstrap) gap-closure across the 5 in-band-T* honest
    # validation targets (T* strictly inside [1.60, 8.84]); labeled exploratory
    # because it is selected against the validation targets. 3.761 stays primary.
    inband_targets = [n for n in results if results[n].get("T_star_supervised") is not None
                      and BAND[0] < results[n]["T_star_supervised"] < BAND[1]]
    minimax = {}
    minimax_curve = []
    if inband_targets:
        per_target_eval = {}
        for name, path, kind, off_pair in targets_spec:
            if name not in inband_targets:
                continue
            preds = (json.loads(path.read_text()) if kind != "legacy_closed"
                     else load_closed_predictions(path))
            rows = build_rows(preds)
            rps0 = results[name]["uncalibrated_mean_rps"]
            rps_star = results[name]["scored"]["supervised_Tstar"]["mean_rps"]
            denom = rps0 - rps_star
            per_target_eval[name] = (rows, rps0, denom)
        for T in band_grid:
            gaps = []
            for name, (rows, rps0, denom) in per_target_eval.items():
                m, _ = eval_at(float(T), rows)
                gaps.append((rps0 - m["mean_rps"]) / denom if abs(denom) > 1e-12 else np.nan)
            gaps = np.array(gaps, dtype=float)
            minimax_curve.append({"T": float(T), "min_gap": float(np.nanmin(gaps)),
                                   "mean_gap": float(np.nanmean(gaps))})
        best_min = max(minimax_curve, key=lambda r: r["min_gap"])
        best_mean = max(minimax_curve, key=lambda r: r["mean_gap"])
        minimax = {
            "in_band_Tstar_targets": inband_targets,
            "curve": minimax_curve,
            "best_by_min_gap": best_min,
            "best_by_mean_gap": best_mean,
            "distance_from_3.761_log": {
                "best_by_min_gap": math.log(best_min["T"] / ANCHOR_MIDPOINT),
                "best_by_mean_gap": math.log(best_mean["T"] / ANCHOR_MIDPOINT),
            },
            "label": "EXPLORATORY — selected against the validation targets; never promoted "
                     "over the pre-registered 3.761 midpoint",
        }

    # -- 3. Guarantees summary -------------------------------------------------
    worst_reliability_harm = None
    worst_res_harm = None
    worst_rps_harm = None
    for tag in ("band_lower_1.60", "zero_fit_3.761", "band_upper_8.84"):
        for name, r in results.items():
            if r.get("status") == "SKIPPED":
                continue
            bm = r["band_membership"].get(tag)
            if bm is None:
                continue
            if worst_reliability_harm is None or bm["d_reliability"] > worst_reliability_harm[0]:
                worst_reliability_harm = (bm["d_reliability"], name, tag)
            if worst_res_harm is None or bm["d_resolution"] < worst_res_harm[0]:
                worst_res_harm = (bm["d_resolution"], name, tag)
            if worst_rps_harm is None or bm["d_rps"] > worst_rps_harm[0]:
                worst_rps_harm = (bm["d_rps"], name, tag)

    max_regret = None
    for name, r in results.items():
        if r.get("status") == "SKIPPED":
            continue
        t_star = r["T_star_supervised"]
        if not (BAND[0] < t_star < BAND[1]):
            continue
        regret = r["scored"]["zero_fit_3.761"]["mean_rps"] - r["scored"]["supervised_Tstar"]["mean_rps"]
        if max_regret is None or regret > max_regret[0]:
            max_regret = (regret, name)

    guarantees = {
        "direction_safety": {
            "worst_reliability_delta": worst_reliability_harm,
            "worst_resolution_delta": worst_res_harm,
            "worst_rps_delta": worst_rps_harm,
            "reconciliation_note": "q4_range_robustness's tolerant band (1.00, 22.8] is stage9-only; "
                                    "these deltas are computed across all 7 targets here and are the "
                                    "authoritative multi-target direction-safety statement",
        },
        "bounded_regret": {
            "max_rps_shortfall_zero_fit_vs_supervised": max_regret,
            "definition": "max over targets with T* strictly inside (1.60, 8.84) of "
                          "mean_rps(T=3.761) - mean_rps(T=T*)",
        },
    }

    dump = {
        "meta": {
            "date": "2026-07-19", "band": list(BAND), "anchor_midpoint": ANCHOR_MIDPOINT,
            "transferred_constant": TRANSFERRED_CONSTANT, "res_tol_frac": RES_TOL_FRAC,
            "bootstrap_B": args.bootstrap, "seed": args.seed,
            "gt": "canonical aireg.load_cells() bundle (2026-07-09 vintage, tau=0.675, continuous)",
            "off_pair_warning": "stage9/phase23 are LEGACY closed runs; indicative only, never an "
                                "E6 result; pool T-hat values are copied verbatim from "
                                "analysis_2026_07_19_dispersion_pool_promotion.md, not refit here",
        },
        "targets": results,
        "minimax_exploratory": minimax,
        "guarantees": guarantees,
    }
    (out / "band_mechanism_composite.json").write_text(json.dumps(dump, indent=2, default=float))
    print(f"[out] wrote {out / 'band_mechanism_composite.json'}")


if __name__ == "__main__":
    main()
