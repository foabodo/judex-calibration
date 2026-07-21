#!/usr/bin/env python3
"""E6 post-run driver — protocol r3 arms (verbalized-first).

Computes arms 1, 2, 3a, 3b of ``docs/e6_onpair_decision_protocol_r3.md``
(ADOPTED 2026-07-21) against a closed-pair evaluator run:

  Arm 1  closed_side_check at the frozen T_J = 1.153 and at both band edges
         (the F4 acceptance read + the always-on band profile).
  Arm 2  supervised held-out T* (validation arm): full-sample fit + doc-clustered
         leave-one-doc-out read; F5 concurrence |ln(T*/T_J)| <= ln 2.
  Arm 3b verbalized dispersion pool (F6): the four panel families' verbalized
         BASE legs, raw fit, NO shrinkage; vetting = pool mixture entropy >= 1.00
         nats on the run's cells, LOBO min >= 1.00, fit unsaturated; concurrence
         = T_raw in the band.
  Arm 3a tau_DACA-verbalized (F7): report-only; per-reference validity =
         unsaturated + reference not below chance; corroboration iff >= 3 valid
         fits AND their full range within [T_J/2, 2*T_J]; per-reference
         entropy-gap diagnostic (reference vs closed mean normalized entropy).

Arm 4 (the confidence instrument, F8) is NOT computed here: it consumes the
closed pair's PER-SEAT confidence distributions, which live in the run's
cross-family payloads rather than metrics_report.json — extract them and feed
``judex_calibration.study_b.analyze_confidence`` per seat + pooled.

Nothing here adopts anything: the script measures and reports; adoption is the
user acting on the r3 failure ladder. Runs with < 120 items are flagged smoke.

Run from judex-calibration:
  JUDEX_UMBRELLA=/path/to/judex PYTHONPATH=src python scripts/e6_r3_arms.py \
      --closed-run-dir ../judex-evaluator/runs/<run-id> [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from judex_calibration import aireg  # noqa: E402
from judex_calibration.elicit_verbalized import compliance_view, floor_and_renormalize, EPSILON  # noqa: E402
from judex_calibration import study_a  # noqa: E402
from judex_calibration.study_a import T_BOUNDS, saturated  # noqa: E402

from judex.core.distributions import ComplianceDistribution  # noqa: E402
from judex.core.metrics import ranked_probability_score  # noqa: E402
from judex.calibration import fit_temperature, fit_dispersion_temperature, realized_dispersion  # noqa: E402

# ------------------------------------------------------------ FROZEN (r3 §0)
T_J = 1.153                                   # F2 — interpolated panel median
BAND = (1.0251785151221313, 1.379820350674421)  # F3 — exact panel-range grid values
RESOLUTION_TOL = 0.10                         # F4 — Resolution within 10%
CONCURRENCE_LOG_RATIO = math.log(2.0)         # F5 — factor-2 rule
POOL_BASES = {                                # F6 — panel BASE legs, verbalized channel
    "qwen": "study_b_qwen/pre_verbalized.json",
    "gemma31": "study_b_gemma31_api/pre_verbalized.json",
    "glm": "study_b_glm/pre_verbalized.json",
    "maverick": "study_b_maverick/pre_verbalized.json",
}
MIN_POOL_ENTROPY = 1.00                       # F6 vetting floor (nats), also the LOBO floor
DACA_MIN_VALID = 3                            # F7
DACA_RANGE = (T_J / 2.0, 2.0 * T_J)           # F7

LABELS = ("very_low", "low", "moderate", "high", "very_high")
FULL_TO_SHORT = {
    "Very low probability of compliance": "very_low",
    "Low probability of compliance": "low",
    "Moderate probability of compliance": "moderate",
    "High probability of compliance": "high",
    "Very high probability of compliance": "very_high",
}


def load_closed(run_dir: Path) -> dict:
    mr = json.loads((run_dir / "metrics_report.json").read_text())
    preds = {}
    for it in mr["items"]:
        p = it["prediction"]
        by = dict(zip(p["labels"], p["probabilities"]))
        preds[it["item_label"]] = [float(by[full]) for full in FULL_TO_SHORT]
    return preds


def dist(probs, cell) -> ComplianceDistribution:
    return ComplianceDistribution.from_values(list(probs), cell.gt_labels)


def arm1(preds, cells) -> dict:
    """F4 acceptance at T_J + the band profile at both edges."""
    points = {"T_J": T_J, "band_low": BAND[0], "band_high": BAND[1]}
    out = {"frozen": {"T_J": T_J, "band": list(BAND), "resolution_tol": RESOLUTION_TOL}}
    for name, T in points.items():
        chk = study_a.closed_side_check(preds, cells, T)
        res_un = chk["uncalibrated"]["resolution"]
        res_ok = (abs(chk["resolution_change"]) <= RESOLUTION_TOL * res_un) if res_un > 0 else False
        chk["resolution_within_tol"] = bool(res_ok)
        chk["f4_pass"] = bool(chk["verdict_ok"] and res_ok)
        out[name] = chk
    out["recommendation"] = ("RECOMMEND adoption of T_J (user pastes the block)"
                             if out["T_J"]["f4_pass"] else
                             "F4 FAILED at T_J -> descend the r3 failure ladder (arm 2, else band-only terminal)")
    return out


def arm2(preds, cells) -> dict:
    """Supervised held-out T* (validation arm) + F5 concurrence."""
    by_label = {c.item_label: c for c in cells}
    rows = [(lb, by_label[lb]) for lb in preds if lb in by_label]
    pairs = [(dist(preds[lb], c), ComplianceDistribution.from_values(c.gt_probs, c.gt_labels))
             for lb, c in rows]
    T_full = fit_temperature(pairs, bounds=T_BOUNDS).temperature
    score = study_a.score_variant(preds, cells)

    docs = sorted({c.document_id for _, c in rows})
    per_doc = {}
    for d in docs:
        train = [(dist(preds[lb], c), ComplianceDistribution.from_values(c.gt_probs, c.gt_labels))
                 for lb, c in rows if c.document_id != d]
        test = [(lb, c) for lb, c in rows if c.document_id == d]
        if not train or not test:
            continue
        T = fit_temperature(train, bounds=T_BOUNDS).temperature

        def mean_rps(temp):
            from judex.calibration import apply_temperature
            return statistics.mean(
                ranked_probability_score(apply_temperature(dist(preds[lb], c), temp),
                                         ComplianceDistribution.from_values(c.gt_probs, c.gt_labels))
                for lb, c in test)

        per_doc[d] = {"T": T, "T_saturated": saturated(T), "n": len(test),
                      "oos_rps_delta_vs_T1": mean_rps(T) - mean_rps(1.0)}
    log_ratio = math.log(T_full / T_J) if math.isfinite(T_full) and T_full > 0 else float("nan")
    return {
        "T_star_full_sample": T_full, "T_star_saturated": saturated(T_full),
        "in_band": BAND[0] <= T_full <= BAND[1],
        "accuracy_gate_inputs": {"argmax_acc": score.get("argmax_acc"),
                                 "resolution": score.get("murphy", {}).get("resolution")},
        "lodo_per_doc": per_doc,
        "lodo_T_range": ([min(v["T"] for v in per_doc.values()),
                          max(v["T"] for v in per_doc.values())] if per_doc else None),
        "f5_log_ratio_vs_T_J": log_ratio,
        "f5_concurrent": bool(abs(log_ratio) <= CONCURRENCE_LOG_RATIO) if math.isfinite(log_ratio) else False,
        "estimand_note": "T* absorbs base-vs-panel miscalibration on top of the post-training increment; T* >= T_J is expected",
    }


def load_pool(bases_dir: Path) -> dict:
    pool = {}
    for fam, rel in POOL_BASES.items():
        recs = json.loads((bases_dir / rel).read_text())
        pool[fam] = {lb: floor_and_renormalize(v, EPSILON) for lb, v in compliance_view(recs).items()}
    return pool


def arm3b(preds, cells, bases_dir: Path) -> dict:
    """F6 — verbalized dispersion pool, raw fit, no shrinkage."""
    by_label = {c.item_label: c for c in cells}
    pool = load_pool(bases_dir)
    labels = [lb for lb in preds if lb in by_label and all(lb in pool[f] for f in POOL_BASES)]
    pool_dists = {lb: [dist(pool[f][lb], by_label[lb]) for f in POOL_BASES] for lb in labels}
    fit = fit_dispersion_temperature([(dist(preds[lb], by_label[lb]), pool_dists[lb]) for lb in labels],
                                     bounds=T_BOUNDS)

    def pool_entropy(fams):
        return statistics.mean(
            realized_dispersion([dist(pool[f][lb], by_label[lb]) for f in fams]).mixture_entropy
            for lb in labels)

    h = pool_entropy(list(POOL_BASES))
    lobo = {f: pool_entropy([g for g in POOL_BASES if g != f]) for f in POOL_BASES}
    t = fit.temperature
    vetting_ok = bool(h >= MIN_POOL_ENTROPY and min(lobo.values()) >= MIN_POOL_ENTROPY
                      and not saturated(t))
    return {
        "frozen": {"pool_bases": POOL_BASES, "estimator": "raw fit_dispersion_temperature, no shrinkage",
                   "min_pool_entropy": MIN_POOL_ENTROPY, "band": list(BAND)},
        "n_items": len(labels),
        "T_raw": t, "T_raw_saturated": saturated(t),
        "pool_mixture_entropy": h, "lobo_pool_entropy": lobo,
        "vetting_pass": vetting_ok,
        "in_band_concurrence": bool(vetting_ok and BAND[0] <= t <= BAND[1]),
    }


def arm3a(preds, cells, bases_dir: Path, T_supervised: float) -> dict:
    """F7 — tau_DACA-verbalized, report-only, + entropy-gap diagnostic."""
    pool = load_pool(bases_dir)
    tri = study_a.daca_triangulation(preds, pool, cells,
                                     tau_transfer=T_J, T_supervised=T_supervised)
    closed_h = statistics.mean(
        ComplianceDistribution.from_values(floor_and_renormalize(p, EPSILON), LABELS).normalized_entropy()
        for p in preds.values())
    for fam, refs in pool.items():
        overlap = preds.keys() & refs.keys()
        ref_h = statistics.mean(
            ComplianceDistribution.from_values(refs[lb], LABELS).normalized_entropy() for lb in overlap)
        tri["references"][fam]["entropy_gap_ref_minus_closed"] = ref_h - closed_h
    valid = [r["tau_daca"] for r in tri["references"].values()
             if not r["tau_daca_saturated"] and not r.get("reference_below_chance", False)]
    tri["f7"] = {
        "n_valid": len(valid), "min_valid": DACA_MIN_VALID, "range_criterion": list(DACA_RANGE),
        "corroborates": bool(len(valid) >= DACA_MIN_VALID
                             and all(DACA_RANGE[0] <= v <= DACA_RANGE[1] for v in valid)),
        "note": "report-only; never gates (r3 F7)",
    }
    return tri


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--closed-run-dir", type=Path, required=True)
    ap.add_argument("--bases-runs-dir", type=Path, default=REPO / "runs")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cells = aireg.load_cells()
    preds = load_closed(args.closed_run_dir)
    smoke = len(preds) < 120

    a2 = arm2(preds, cells)
    report = {
        "protocol": "e6_onpair_decision_protocol_r3 (ADOPTED 2026-07-21)",
        "closed_run_dir": str(args.closed_run_dir),
        "n_items": len(preds), "smoke": smoke,
        "epsilon": EPSILON, "T_bounds": list(T_BOUNDS),
        "arm1_transferred_T_J": arm1(preds, cells),
        "arm2_supervised_validation": a2,
        "arm3b_dispersion_pool": arm3b(preds, cells, args.bases_runs_dir),
        "arm3a_tau_daca": arm3a(preds, cells, args.bases_runs_dir, a2["T_star_full_sample"]),
        "arm4_note": ("NOT computed here: extract per-seat confidence distributions from the "
                      "run's cross-family payloads and feed study_b.analyze_confidence (F8 bar: "
                      "LODO calibratable AND T_c unsaturated AND kendall_tau_b <= -0.10)"),
    }
    if smoke:
        report["smoke_note"] = "[SMOKE] < 120 items — prototype-labeled, nothing adoptable"

    print(json.dumps({k: report[k] for k in ("n_items", "smoke")}, indent=1))
    a1 = report["arm1_transferred_T_J"]
    print(f"arm1 F4 at T_J={T_J}: pass={a1['T_J']['f4_pass']} "
          f"(rel_impr={a1['T_J']['reliability_improvement']}, rps_change={a1['T_J']['rps_change']}, "
          f"res_within_tol={a1['T_J']['resolution_within_tol']})")
    print(f"  -> {a1['recommendation']}")
    print(f"arm2 T*={a2['T_star_full_sample']:.3f} (sat={a2['T_star_saturated']}) "
          f"in_band={a2['in_band']} F5 concurrent={a2['f5_concurrent']}")
    p3b = report["arm3b_dispersion_pool"]
    print(f"arm3b T_raw={p3b['T_raw']:.3f} vetting={p3b['vetting_pass']} in_band={p3b['in_band_concurrence']}")
    print(f"arm3a: {json.dumps(report['arm3a_tau_daca']['f7'])}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "e6_r3_arms.json").write_text(json.dumps(report, indent=1, default=float))
        print(f"wrote {args.out / 'e6_r3_arms.json'}")


if __name__ == "__main__":
    main()
