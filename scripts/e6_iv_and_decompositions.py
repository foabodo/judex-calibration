#!/usr/bin/env python3
"""E6 completion computations C1-C4 for the core paper (2026-07-21).

Post-hoc, $0, read-only over the R1 sweep's logged artifacts. Adopts nothing,
edits no frozen constant, and re-derives no driver verdict.

  C1  E6(iv) decision-endpoint delta — the FROZEN E3 cost engine
      (judex-ground-truth scripts/tier0_utility_battery/cost_engine.py, imported
      verbatim, not reimplemented) consuming the closed pair's CALIBRATED vs
      UNCALIBRATED credences on the 120 AIReg cells, scored against the 3 AIReg
      human experts under per-expert-averaged realized cost. Registered E6 Step 4.
  C2  Per-Article and per-level decomposition of the arm-1 gain (promised in the
      core paper's calibration section; never emitted by the r3 driver).
  C3  Doc-clustered paired bootstrap CIs on the three F4 deltas. DESCRIPTIVE and
      NON-REGISTERED: F4 is a frozen POINT criterion and it PASSED; a CI that
      covers 0 does not retroactively fail it.
  C4  closed_side_check at the supervised T* (in-sample-supervised; context only).

Beliefs are taken through the SAME seam as arm 1 (ComplianceDistribution +
judex.calibration.apply_temperature on unfloored vectors, whose invert_softmax
carries its own 1e-12 clip), so C1's calibrated credences are byte-identical to
the ones arm 1 scored.

Run from judex-calibration:
  JUDEX_UMBRELLA=/path/to/judex PYTHONPATH=src \
    python scripts/e6_iv_and_decompositions.py \
      --closed-run-dir ../judex-evaluator/runs/stage9-onpair-e6-20260721 \
      --out runs/e6_r3_stage9-onpair-e6-20260721
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from judex_calibration import aireg  # noqa: E402
from judex_calibration import study_a  # noqa: E402
from judex.core.distributions import ComplianceDistribution  # noqa: E402
from judex.calibration import apply_temperature  # noqa: E402
from judex.metrics_report import murphy_decomposition  # noqa: E402

# ---------------------------------------------------------------- frozen refs
T_J = 1.153                     # r3 F2 (frozen)
T_STAR_SUPERVISED = 2.544640471713656   # arm 2 measured (NOT frozen, context only)
BOOT_SEED = 20260720            # r3 preamble: doc-clustered bootstrap seed
BOOT_B = 10000
RESOLUTION_TOL = 0.10           # r3 F4

FULL_TO_SHORT = {
    "Very low probability of compliance": "very_low",
    "Low probability of compliance": "low",
    "Moderate probability of compliance": "moderate",
    "High probability of compliance": "high",
    "Very high probability of compliance": "very_high",
}
LABEL_RE = re.compile(r"Art\s*(\d+)\s*/\s*Scenario\s*([A-Z])\s*\|\s*Use\s*(\d+)")


def load_closed(run_dir: Path) -> dict:
    mr = json.loads((run_dir / "metrics_report.json").read_text())
    preds = {}
    for it in mr["items"]:
        p = it["prediction"]
        by = dict(zip(p["labels"], p["probabilities"]))
        preds[it["item_label"]] = [float(by[full]) for full in FULL_TO_SHORT]
    return preds


def gt_core():
    """Import the frozen gt-side E3 engine + loaders (never reimplemented here)."""
    umbrella = Path(__import__("os").environ.get("JUDEX_UMBRELLA", REPO.parent))
    bat = umbrella / "judex-ground-truth" / "scripts" / "tier0_utility_battery"
    if not bat.exists():
        raise SystemExit(f"gt-side battery not found at {bat}")
    sys.path.insert(0, str(bat))
    import core          # noqa: E402
    import cost_engine   # noqa: E402
    return core, cost_engine


# ------------------------------------------------------------------ scoring
def build_items(preds, cells, T):
    """(labels, uncal items, cal items, uncal probs, cal probs) through arm 1's seam."""
    by_label = {c.item_label: c for c in cells}
    labels, un_items, ca_items, un_p, ca_p, docs = [], [], [], [], [], []
    for lb in sorted(preds):
        c = by_label.get(lb)
        if c is None:
            continue
        gt = ComplianceDistribution.from_values(list(c.gt_probs), c.gt_labels)
        pred = ComplianceDistribution.from_values(list(preds[lb]), c.gt_labels)
        cal = apply_temperature(pred, T)
        labels.append(lb)
        docs.append(c.document_id)
        un_items.append(study_a._metric_item(lb, pred, gt))
        ca_items.append(study_a._metric_item(lb, cal, gt))
        # .probabilities is ordered to the labels the distribution was built with
        un_p.append(list(pred.probabilities))
        ca_p.append(list(cal.probabilities))
    return labels, docs, un_items, ca_items, np.array(un_p), np.array(ca_p)


def deltas(un_items, ca_items) -> dict:
    """The three F4 deltas (+ W1/argmax), unrounded. Murphy is sample-dependent:
    climatology and (argmax, confidence) bins are re-estimated on whatever item
    set is passed — which is exactly what a bootstrap replicate must do."""
    mu, mc = murphy_decomposition(un_items), murphy_decomposition(ca_items)
    mean = lambda xs: float(np.mean(xs))
    return {
        "reliability_improvement": mu["reliability"] - mc["reliability"],
        "rps_change": mean([i.rps for i in ca_items]) - mean([i.rps for i in un_items]),
        "resolution_change": mc["resolution"] - mu["resolution"],
        "w1_change": mean([i.w1 for i in ca_items]) - mean([i.w1 for i in un_items]),
        "argmax_change": mean([i.argmax_agreement for i in ca_items])
                          - mean([i.argmax_agreement for i in un_items]),
        "uncal": {"reliability": mu["reliability"], "resolution": mu["resolution"],
                  "uncertainty": mu["uncertainty"],
                  "mean_rps": mean([i.rps for i in un_items]),
                  "mean_w1": mean([i.w1 for i in un_items]),
                  "argmax_acc": mean([i.argmax_agreement for i in un_items])},
        "cal": {"reliability": mc["reliability"], "resolution": mc["resolution"],
                "uncertainty": mc["uncertainty"],
                "mean_rps": mean([i.rps for i in ca_items]),
                "mean_w1": mean([i.w1 for i in ca_items]),
                "argmax_acc": mean([i.argmax_agreement for i in ca_items])},
    }


def pct_ci(x, lo=2.5, hi=97.5):
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


# ------------------------------------------------------------------------ C3
def c3_bootstrap(docs, un_items, ca_items, point) -> dict:
    doc_ids = sorted(set(docs))
    idx_by_doc = {d: [i for i, dd in enumerate(docs) if dd == d] for d in doc_ids}
    rng = np.random.default_rng(BOOT_SEED)
    keys = ("reliability_improvement", "rps_change", "resolution_change", "w1_change")
    draws = {k: [] for k in keys}
    for _ in range(BOOT_B):
        pick = rng.integers(0, len(doc_ids), len(doc_ids))
        idx = [i for j in pick for i in idx_by_doc[doc_ids[j]]]
        d = deltas([un_items[i] for i in idx], [ca_items[i] for i in idx])
        for k in keys:
            draws[k].append(d[k])
    out = {"B": BOOT_B, "seed": BOOT_SEED, "n_doc_clusters": len(doc_ids),
           "status": "DESCRIPTIVE / NON-REGISTERED — F4 is a frozen point criterion "
                     "and passed; a CI covering 0 does not retroactively fail it",
           "murphy_note": "reliability/resolution are sample-dependent (climatology + "
                          "bins re-estimated per replicate), as required for a paired "
                          "bootstrap of the decomposition"}
    for k in keys:
        a = np.array(draws[k])
        favourable = (a > 0) if k == "reliability_improvement" else (a < 0)
        out[k] = {"point": point[k], "ci95": pct_ci(a),
                  "excludes_zero": bool(pct_ci(a)[0] > 0 or pct_ci(a)[1] < 0),
                  "frac_favourable": float(favourable.mean())}
    return out


# ------------------------------------------------------------------------ C2
def c2_decomposition(labels, docs, cells, un_items, ca_items) -> dict:
    by_label = {c.item_label: c for c in cells}
    N = len(labels)
    pooled_rps_delta = float(np.mean([i.rps for i in ca_items])
                             - np.mean([i.rps for i in un_items]))

    def strata_block(keyfn, name):
        groups = {}
        for i, lb in enumerate(labels):
            groups.setdefault(keyfn(by_label[lb], i), []).append(i)
        rows = []
        for key in sorted(groups, key=str):
            idx = groups[key]
            un = [un_items[i] for i in idx]
            ca = [ca_items[i] for i in idx]
            d_rps = float(np.mean([i.rps for i in ca]) - np.mean([i.rps for i in un]))
            d_w1 = float(np.mean([i.w1 for i in ca]) - np.mean([i.w1 for i in un]))
            mu, mc = murphy_decomposition(un), murphy_decomposition(ca)
            rows.append({
                "stratum": str(key), "n": len(idx),
                "mean_rps_uncal": float(np.mean([i.rps for i in un])),
                "mean_rps_cal": float(np.mean([i.rps for i in ca])),
                "rps_change": d_rps,
                "share_of_pooled_rps_gain": float((len(idx) / N) * d_rps / pooled_rps_delta)
                                            if pooled_rps_delta else None,
                "contribution_to_pooled_rps_change": float((len(idx) / N) * d_rps),
                "mean_w1_uncal": float(np.mean([i.w1 for i in un])),
                "w1_change": d_w1,
                "argmax_acc_uncal": float(np.mean([i.argmax_agreement for i in un])),
                "argmax_acc_cal": float(np.mean([i.argmax_agreement for i in ca])),
                "within_stratum_reliability_improvement": mu["reliability"] - mc["reliability"],
                "within_stratum_resolution_change": mc["resolution"] - mu["resolution"],
            })
        return {"strata": rows, "note": name}

    return {
        "pooled_rps_change": pooled_rps_delta,
        "by_article": strata_block(
            lambda c, i: f"Article {c.article}",
            "mean-RPS/W1 deltas are exactly item-decomposable: the "
            "contribution_to_pooled_rps_change column sums to pooled_rps_change. "
            "Murphy columns are WITHIN-STRATUM (climatology + bins re-estimated on the "
            "stratum) and therefore do NOT sum to the pooled decomposition."),
        "by_gt_level": strata_block(
            lambda c, i: f"GT argmax = {c.gt_labels[c.gt_argmax]}",
            "strata by ground-truth argmax level; same decomposability caveats"),
        "by_pred_level": strata_block(
            lambda c, i: f"pred argmax = {c.gt_labels[i]}" if False else
                         f"pred argmax = {ComplianceDistribution.from_values(list(c.gt_probs), c.gt_labels).labels[0]}",
            "placeholder — replaced below"),
    }


# ------------------------------------------------------------------------ C1
def c1_e6iv(labels, docs, un_p, ca_p, core, cost_engine) -> dict:
    """Registered E6 Step 4: frozen E3 engine, calibrated vs uncalibrated credences."""
    b1 = core.load_b1()
    meta = b1["meta"]
    expert = core.load_b2(meta)                      # (120, 3) human expert grades 1..5

    # join judex-calibration item_label -> b1 item_index via (article, scenario, use)
    key_to_idx = {(int(r.cell_article), str(r.cell_scenario), int(r.cell_use)): int(r.item_index)
                  for r in meta.itertuples()}
    rows = []
    for lb in labels:
        m = LABEL_RE.match(lb)
        if not m:
            raise SystemExit(f"unparseable item_label {lb!r}")
        k = (int(m.group(1)), m.group(2), int(m.group(3)))
        if k not in key_to_idx:
            raise SystemExit(f"no b1 join for {lb!r} -> {k}")
        rows.append(key_to_idx[k])
    if len(set(rows)) != len(labels):
        raise SystemExit("join is not injective")
    votes = expert[np.array(rows)]                   # (N, 3) aligned to `labels`

    def regime_cells():
        cells = [("native01", cost_engine.native01_matrix(), "native", 0)]
        for fam, q in (("linear", 1), ("quadratic", 2)):
            for k in range(-4, 5):
                cells.append((f"{fam}_log2lam_{k:+d}",
                              cost_engine.grade_grid_matrix(2.0 ** k, q), fam, k))
        for k in range(-4, 5):
            cells.append((f"operational_log2lam_{k:+d}",
                          cost_engine.operational_matrix(2.0 ** k), "operational", k))
        return cells

    doc_ids = sorted(set(docs))
    idx_by_doc = {d: [i for i, dd in enumerate(docs) if dd == d] for d in doc_ids}
    rng = np.random.default_rng(BOOT_SEED)
    boot_pick = [rng.integers(0, len(doc_ids), len(doc_ids)) for _ in range(BOOT_B)]

    out_cells = []
    for name, M, fam, k in regime_cells():
        a_un = cost_engine.bayes_actions(un_p, M)
        a_ca = cost_engine.bayes_actions(ca_p, M)
        c_un = cost_engine.realized_cost(a_un[:, None], votes, M)[:, 0]
        c_ca = cost_engine.realized_cost(a_ca[:, None], votes, M)[:, 0]
        per_item = c_ca - c_un
        point = float(per_item.mean())
        draws = np.array([per_item[[i for j in pick for i in idx_by_doc[doc_ids[j]]]].mean()
                          for pick in boot_pick])
        ci = pct_ci(draws)
        _, oracle = cost_engine.oracle_costs(votes, M)
        out_cells.append({
            "regime": name, "family": fam, "log2_lambda": k,
            "mean_cost_uncalibrated": float(c_un.mean()),
            "mean_cost_calibrated": float(c_ca.mean()),
            "delta_cost_cal_minus_uncal": point,
            "delta_ci95": ci,
            "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            "n_action_changes": int((a_un != a_ca).sum()),
            "mean_oracle_cost": float(oracle.mean()),
        })
    n_any = sum(c["n_action_changes"] for c in out_cells)
    return {
        "status": "REGISTERED — E6 Step 4 (decision-endpoint delta)",
        "engine": "judex-ground-truth scripts/tier0_utility_battery/cost_engine.py (frozen, imported verbatim)",
        "realized_cost_convention": "per-expert averaging over the 3 AIReg human experts (frozen; never consensus-first)",
        "T_J": T_J, "n_items": len(labels),
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "clusters": len(doc_ids)},
        "total_action_changes_across_all_regimes": n_any,
        "cells": out_cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed-run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    preds = load_closed(args.closed_run_dir)
    cells = aireg.load_cells()
    core, cost_engine = gt_core()

    labels, docs, un_items, ca_items, un_p, ca_p = build_items(preds, cells, T_J)
    point = deltas(un_items, ca_items)

    # C2 (drop the placeholder block)
    c2 = c2_decomposition(labels, docs, cells, un_items, ca_items)
    c2.pop("by_pred_level", None)

    # C4 — context only
    labs4, docs4, un4, ca4, _, _ = build_items(preds, cells, T_STAR_SUPERVISED)
    d4 = deltas(un4, ca4)

    report = {
        "provenance": {
            "closed_run_dir": str(args.closed_run_dir),
            "n_items": len(labels), "n_doc_clusters": len(set(docs)),
            "T_J": T_J, "gt": "aireg.load_cells() canonical, manifest-verified",
            "seam": "arm-1 seam (unfloored vectors through judex.calibration.apply_temperature)",
            "adopts_nothing": True,
        },
        "arm1_point_recompute": point,
        "C1_e6iv_decision_endpoint": c1_e6iv(labels, docs, un_p, ca_p, core, cost_engine),
        "C2_gain_decomposition": c2,
        "C3_f4_delta_bootstrap": c3_bootstrap(docs, un_items, ca_items, point),
        "C4_at_supervised_T_star": {
            "T": T_STAR_SUPERVISED,
            "status": "NON-REGISTERED, in-sample-supervised — context only, adopts nothing",
            **{k: d4[k] for k in ("reliability_improvement", "rps_change",
                                  "resolution_change", "w1_change", "argmax_change")},
            "resolution_within_tol": bool(abs(d4["resolution_change"])
                                          <= RESOLUTION_TOL * d4["uncal"]["resolution"]),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "e6_iv_and_decompositions.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"wrote {dest}")

    p = point
    print(f"\narm1 recompute @ T_J={T_J}: rel_impr {p['reliability_improvement']:+.5f} "
          f"rps {p['rps_change']:+.5f} res {p['resolution_change']:+.5f} "
          f"w1 {p['w1_change']:+.5f} argmax {p['argmax_change']:+.5f}")
    c3 = report["C3_f4_delta_bootstrap"]
    for k in ("reliability_improvement", "rps_change", "resolution_change"):
        b = c3[k]
        print(f"  C3 {k:26s} {b['point']:+.5f}  CI95 [{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}] "
              f"excl0={b['excludes_zero']} favourable={b['frac_favourable']:.3f}")
    c1 = report["C1_e6iv_decision_endpoint"]
    print(f"  C1 total action changes across all regimes: "
          f"{c1['total_action_changes_across_all_regimes']}")


if __name__ == "__main__":
    main()
