#!/usr/bin/env python3
"""E6 post-run driver — protocol r3 arms (verbalized-first).

Computes ALL FOUR arms of ``docs/e6_onpair_decision_protocol_r3.md``
(ADOPTED 2026-07-21) against a closed-pair evaluator run, plus the always-on
band-only sensitivity report and the paper's E6 exhibit blocks:

  Arm 1  closed_side_check at the frozen T_J = 1.153 and at both band edges
         (the F4 acceptance read), plus a 9-point log-spaced profile ACROSS
         the band (r3 ladder item 6, "deltas across [1.025, 1.380]") and the
         E6(i)/(iii) diagnostics: per-item entropy deficit vs GT before/after
         correction, and the tie-aware argmax-invariance audit (registered up
         to exact top-two ties: scored on tie-free items, tied items reported
         separately — the r3-era successor of q4's retired ``shift_report``).
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
  Arm 4  confidence instrument (F8, report-only): PER-SEAT + pooled T_c on the
         closed pair's phase-1 dimension-level ``confidence_distribution``
         (extracted from documents/*/cross_family_evaluation.json), scored with
         the Study B machinery (fixed link, LODO dBrier with the seeded
         doc-clustered bootstrap, Kendall association, AURC); F8 bar = LODO
         calibratable AND T_c unsaturated AND kendall_tau_b <= -0.10.

The emitted ``concurrence_table`` block mirrors r3 §4 row-for-row so the
published table fills mechanically. Nothing here adopts anything: the script
measures and reports; adoption is the user acting on the r3 failure ladder.
Runs with < 120 items are flagged smoke.

``--selftest`` recomputes the whole report against the pinned reference run
(``stage9-claude-gpt-medium``, 3 docs / 15 items, host-local gitignored) and
compares against the values pinned from the 2026-07-21 pilot
(``runs/e6_r3_arms_pilot``) — the regression gate for this instrument.

Run from judex-calibration:
  JUDEX_UMBRELLA=/path/to/judex PYTHONPATH=src python scripts/e6_r3_arms.py \
      --closed-run-dir ../judex-evaluator/runs/<run-id> [--out <dir>]
  JUDEX_UMBRELLA=/path/to/judex PYTHONPATH=src python scripts/e6_r3_arms.py --selftest
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
from judex_calibration import study_b  # noqa: E402
from judex_calibration.study_a import T_BOUNDS, saturated  # noqa: E402

from judex.core.distributions import ComplianceDistribution  # noqa: E402
from judex.core.metrics import ranked_probability_score  # noqa: E402
from judex.calibration import apply_temperature, fit_temperature, fit_dispersion_temperature, realized_dispersion  # noqa: E402

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
KENDALL_BAR = -0.10                           # F8 — association bar for "carries signal"
BAND_PROFILE_N = 9                            # log-spaced points across the band (ladder item 6)
COVERAGE_QS = (0.5, 0.9)                      # credible-set masses for the (non-registered) coverage read

LABELS = ("very_low", "low", "moderate", "high", "very_high")
CONF_LABELS = ("low", "medium", "high")
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


def _f4_check(preds, cells, T) -> dict:
    chk = study_a.closed_side_check(preds, cells, T)
    res_un = chk["uncalibrated"]["resolution"]
    res_ok = (abs(chk["resolution_change"]) <= RESOLUTION_TOL * res_un) if res_un > 0 else False
    chk["resolution_within_tol"] = bool(res_ok)
    chk["f4_pass"] = bool(chk["verdict_ok"] and res_ok)
    return chk


def _joined(preds, cells):
    by_label = {c.item_label: c for c in cells}
    return [(lb, dist(preds[lb], by_label[lb]),
             ComplianceDistribution.from_values(by_label[lb].gt_probs, by_label[lb].gt_labels))
            for lb in preds if lb in by_label]


def band_profile(preds, cells) -> list:
    """closed_side_check deltas ACROSS the band — r3 ladder item 6 (always-on)."""
    grid = [BAND[0] * (BAND[1] / BAND[0]) ** (i / (BAND_PROFILE_N - 1)) for i in range(BAND_PROFILE_N)]
    prof = []
    for t in grid:
        chk = _f4_check(preds, cells, t)
        prof.append({k: chk[k] for k in ("T", "reliability_improvement", "rps_change",
                                         "resolution_change", "resolution_within_tol",
                                         "verdict_ok", "f4_pass")})
    return prof


def _credible_set_coverage(pairs, q: float) -> float:
    """Fraction of items whose GT argmax lies in the smallest prediction-ordered
    level set with cumulative predicted mass >= q. NON-REGISTERED operationalization."""
    hits = 0
    for _, pred, gt in pairs:
        order = sorted(range(len(pred.probabilities)), key=lambda i: -pred.probabilities[i])
        cum, chosen = 0.0, set()
        for i in order:
            chosen.add(i)
            cum += pred.probabilities[i]
            if cum >= q - 1e-12:
                break
        hits += int(gt.argmax_index() in chosen)
    return hits / len(pairs)


def e6i_diagnostics(preds, cells) -> dict:
    """E6(i)/(iii) exhibit inputs: per-item entropy deficit vs GT, before and after
    correction, plus a credible-set coverage read (definition NOT frozen in r3 —
    flagged; the registered phrase 'coverage against distributional ground truth'
    has no frozen operationalization, so this block is report-only diagnostics)."""
    pairs = _joined(preds, cells)
    gt_H = statistics.mean(gt.entropy() for _, _, gt in pairs)
    gt_Hn = statistics.mean(gt.normalized_entropy() for _, _, gt in pairs)
    out = {"n": len(pairs), "mean_gt_entropy": gt_H, "mean_gt_normalized_entropy": gt_Hn,
           "coverage_definition_note": (
               "NON-REGISTERED: E6(i) registers 'coverage against distributional ground truth' "
               "but neither r3 nor any frozen instrument pins a definition; emitted here as "
               "smallest prediction-ordered credible-set coverage of the GT argmax at "
               f"q in {list(COVERAGE_QS)} — report-only, enters no acceptance criterion")}
    for name, T in (("uncalibrated", None), ("T_J", T_J), ("band_low", BAND[0]), ("band_high", BAND[1])):
        scaled = [(lb, apply_temperature(p, T) if T else p, gt) for lb, p, gt in pairs]
        out[name] = {
            "mean_pred_entropy": statistics.mean(p.entropy() for _, p, _ in scaled),
            "mean_pred_normalized_entropy": statistics.mean(p.normalized_entropy() for _, p, _ in scaled),
            "mean_entropy_deficit_gt_minus_pred": statistics.mean(
                gt.entropy() - p.entropy() for _, p, gt in scaled),
            "credible_set_coverage": {str(q): _credible_set_coverage(scaled, q) for q in COVERAGE_QS},
        }
    return out


def argmax_invariance(preds, cells) -> dict:
    """E6(iii), tie-aware (registered up to exact top-two ties): temperature preserves
    the credence ordering, so on tie-free items any argmax shift is a floating-point
    artifact; exact top-two ties may flip on fp tie-break through the logit round-trip
    and are reported separately. Successor of q4's retired ``shift_report``, evaluated
    on the r3 band [1.025, 1.380] + T_J instead of the retired logit-era taus."""
    pairs = _joined(preds, cells)
    tied, tie_free = [], []
    for lb, pred, gt in pairs:
        top2 = sorted(pred.probabilities, reverse=True)[:2]
        (tied if top2[0] == top2[1] else tie_free).append((lb, pred, gt))
    out = {"n": len(pairs), "n_tied_exact_top2": len(tied),
           "tied_items": [{"label": lb, "probabilities": list(p.probabilities)} for lb, p, _ in tied]}
    all_clean = True
    for name, T in (("band_low", BAND[0]), ("T_J", T_J), ("band_high", BAND[1])):
        shifts = [lb for lb, p, _ in tie_free
                  if apply_temperature(p, T).argmax_index() != p.argmax_index()]
        tied_flips = [{"label": lb, "argmax_before": p.argmax_index(),
                       "argmax_after": apply_temperature(p, T).argmax_index()}
                      for lb, p, _ in tied
                      if apply_temperature(p, T).argmax_index() != p.argmax_index()]
        acc_before = (statistics.mean(int(p.argmax_index() == gt.argmax_index()) for _, p, gt in tie_free)
                      if tie_free else float("nan"))
        acc_after = (statistics.mean(int(apply_temperature(p, T).argmax_index() == gt.argmax_index())
                                     for _, p, gt in tie_free) if tie_free else float("nan"))
        all_clean = all_clean and not shifts
        out[name] = {"tie_free_argmax_shifts": shifts,
                     "tie_free_argmax_acc_delta": acc_after - acc_before,
                     "tied_item_flips": tied_flips}
    out["invariance_pass_tie_free"] = bool(all_clean)
    return out


def arm1(preds, cells) -> dict:
    """F4 acceptance at T_J + band edges, the across-band profile, and the E6
    exhibit diagnostics (entropy deficit, coverage read, tie-aware invariance)."""
    points = {"T_J": T_J, "band_low": BAND[0], "band_high": BAND[1]}
    out = {"frozen": {"T_J": T_J, "band": list(BAND), "resolution_tol": RESOLUTION_TOL}}
    for name, T in points.items():
        out[name] = _f4_check(preds, cells, T)
    out["recommendation"] = ("RECOMMEND adoption of T_J (user pastes the block)"
                             if out["T_J"]["f4_pass"] else
                             "F4 FAILED at T_J -> descend the r3 failure ladder (arm 2, else band-only terminal)")
    out["band_profile"] = band_profile(preds, cells)
    out["e6i_diagnostics"] = e6i_diagnostics(preds, cells)
    out["argmax_invariance_tie_aware"] = argmax_invariance(preds, cells)
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


def _conf_block(rows, eps: float = EPSILON) -> dict:
    """study_b.analyze_confidence's summary over pre-joined rows (per-seat or pooled),
    plus the F8 verdict fields. Field-compatible with analyze_confidence."""
    if not rows:
        return {"n": 0}
    T_full = study_b.fit_Tc(rows, eps)
    s_raw = [study_b.link(floor_and_renormalize(r["conf"], eps)) for r in rows]
    kend = study_b.kendall_tau_b(s_raw, [r["w1"] for r in rows])
    lod = study_b.lodo(rows, eps)
    unsat = not saturated(T_full)
    kend_ok = bool(math.isfinite(kend) and kend <= KENDALL_BAR)
    return {
        "n": len(rows), "epsilon": eps,
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "mean_link_phat": sum(s_raw) / len(rows),
        "brier_T1": study_b.brier(rows, 1.0, eps),
        "T_c_full_sample": T_full, "T_c_saturated": saturated(T_full),
        "brier_at_Tc": study_b.brier(rows, T_full, eps),
        "kendall_tau_b_s_vs_w1": kend,
        "aurc_raw": study_b.aurc(rows, None, eps),
        "aurc_at_Tc": study_b.aurc(rows, T_full, eps),
        "lodo": lod,
        "link_weights": list(study_b.CONF_WEIGHTS), "T_bounds": list(T_BOUNDS),
        "f8": {"lodo_calibratable": bool(lod.get("calibratable", False)),
               "T_c_unsaturated": unsat,
               "kendall_bar": KENDALL_BAR, "kendall_pass": kend_ok,
               "carries_signal": bool(lod.get("calibratable", False) and unsat and kend_ok),
               "note": "report-only within E6 (r3 F8); production use is a separate user decision"},
    }


def arm4(run_dir: Path, cells) -> dict:
    """F8 — the confidence instrument on the closed pair's PER-SEAT phase-1
    dimension-level confidence distributions (+ pooled across both seats).

    Stage choice: phase-1 cell (dimension) level — the seat's independent judgment,
    the closest closed-pair analog of Study B's single-pass FULL-contract read.
    Correctness event C per cell = argmax agreement of the SEAT'S OWN phase-1
    final distribution with canonical GT (mirrors study_b._rows)."""
    by_dc = {(c.document_id, c.article): c.item_label for c in cells}
    seats: dict = {}
    for f in sorted((run_dir / "documents").glob("*/cross_family_evaluation.json")):
        payload = json.loads(f.read_text())
        doc_id = payload["document_id"]
        for fam in (payload["family_a_id"], payload["family_b_id"]):
            for lv in payload["phase1"][fam]["levels"]:
                if lv.get("hierarchy_level") != "dimension":
                    continue
                art = lv["level_id"].split(":")[0].split("_")[-1]  # 'article_9:dimension' -> '9'
                label = by_dc.get((doc_id, art))
                if label is None:
                    continue
                comp_by = lv["final_distribution"]["by_label"]
                conf_by = lv["confidence_distribution"]["by_label"]
                seats.setdefault(fam, {})[label] = {
                    "parse_ok": True,
                    "compliance": [float(comp_by[full]) for full in FULL_TO_SHORT],
                    "confidence": [float(conf_by[k]) for k in CONF_LABELS],
                }
    out = {"frozen": {"f8_bar": "LODO calibratable AND T_c unsaturated AND kendall_tau_b <= -0.10",
                      "kendall_bar": KENDALL_BAR,
                      "stage": "phase1 dimension level (per-seat independent judgment)",
                      "bootstrap_seed": 20260720},
           "per_seat": {}}
    all_rows = []
    for fam in sorted(seats):
        rows = study_b._rows(seats[fam], cells)
        all_rows.extend(rows)
        out["per_seat"][fam] = _conf_block(rows)
    out["pooled_both_seats"] = _conf_block(all_rows)
    return out


def concurrence_table(report: dict) -> list:
    """The r3 §4 concurrence table, one row per instrument, filled mechanically."""
    a1 = report["arm1_transferred_T_J"]
    a2 = report["arm2_supervised_validation"]
    a3b = report["arm3b_dispersion_pool"]
    a3a = report["arm3a_tau_daca"]
    a4 = report["arm4_confidence_instrument"]
    rows = [
        {"instrument": "arm1_T_J", "T": T_J, "vs_T_J": None,
         "gate_verdict": {"f4_pass": a1["T_J"]["f4_pass"]}},
        {"instrument": "arm2_supervised_T_star", "T": a2["T_star_full_sample"],
         "vs_T_J": {"abs_log_ratio": abs(a2["f5_log_ratio_vs_T_J"]),
                    "threshold": CONCURRENCE_LOG_RATIO, "f5_concurrent": a2["f5_concurrent"]},
         "gate_verdict": {"saturated": a2["T_star_saturated"], "in_band": a2["in_band"],
                          "accuracy_gate_inputs": a2["accuracy_gate_inputs"]}},
        {"instrument": "arm3b_pool_T_raw", "T": a3b["T_raw"],
         "vs_T_J": {"in_band_f6": a3b["in_band_concurrence"]},
         "gate_verdict": {"vetting_pass": a3b["vetting_pass"]}},
        {"instrument": "arm3a_tau_daca", "T": {fam: r["tau_daca"] for fam, r in a3a["references"].items()},
         "vs_T_J": {"f7_range": list(DACA_RANGE), "f7_corroborates": a3a["f7"]["corroborates"]},
         "gate_verdict": {"n_valid": a3a["f7"]["n_valid"], "min_valid": DACA_MIN_VALID}},
        {"instrument": "arm4_T_c",
         "T": {fam: blk.get("T_c_full_sample") for fam, blk in a4["per_seat"].items()},
         "vs_T_J": "n/a (different object)",
         "gate_verdict": {fam: blk.get("f8", {}).get("carries_signal")
                          for fam, blk in a4["per_seat"].items()}},
        {"instrument": "r2_logit_era_instruments", "T": None, "vs_T_J": None,
         "gate_verdict": "retired with their channel (r2 retained as record)"},
    ]
    return rows


# Pinned from the 2026-07-21 pilot (runs/e6_r3_arms_pilot, reference run
# stage9-claude-gpt-medium — 3 docs / 15 items, smoke-grade by design; the pin is
# a REGRESSION gate for this instrument, never a scientific number).
SELFTEST_RUN = "stage9-claude-gpt-medium"
SELFTEST_PINS = {
    "arm1_T_J.reliability_improvement": 0.00211,
    "arm1_T_J.rps_change": -0.00097,
    "arm1_T_J.f4_pass": True,
    "arm2.T_star_full_sample": 1.5034115698817216,
    "arm2.f5_concurrent": True,
    "arm3b.T_raw": 1.2780628080769258,
    "arm3b.pool_mixture_entropy": 1.155957303767346,
    "arm3b.vetting_pass": True,
    "arm3b.in_band_concurrence": True,
    "arm3a.f7.n_valid": 2,
    "arm3a.f7.corroborates": False,
    # new-block pins (computed 2026-07-21 on the same reference run, first green run)
    "arm1_inv.n_tied_exact_top2": 1,
    "arm1_inv.invariance_pass_tie_free": True,
    "arm4.per_seat.anthropic_claude_medium.T_c_full_sample": 19.99999999999999,
    "arm4.per_seat.anthropic_claude_medium.kendall_tau_b_s_vs_w1": -0.2514253554661761,
    "arm4.per_seat.openai_gpt_standard.T_c_full_sample": 2.692305933904886,
    "arm4.per_seat.openai_gpt_standard.kendall_tau_b_s_vs_w1": -0.015641237740115502,
}


def _dig(report: dict, dotted: str):
    key_map = {"arm1_T_J": ("arm1_transferred_T_J", "T_J"), "arm2": ("arm2_supervised_validation",),
               "arm3b": ("arm3b_dispersion_pool",), "arm3a": ("arm3a_tau_daca",),
               "arm1_inv": ("arm1_transferred_T_J", "argmax_invariance_tie_aware"),
               "arm4": ("arm4_confidence_instrument",)}
    parts = dotted.split(".")
    node = report
    for k in key_map[parts[0]]:
        node = node[k]
    for k in parts[1:]:
        node = node[k]
    return node


def selftest(report: dict) -> bool:
    ok = True
    for dotted, want in SELFTEST_PINS.items():
        got = _dig(report, dotted)
        good = (abs(got - want) <= 1e-9) if isinstance(want, float) else (got == want)
        if not good:
            print(f"[selftest] MISMATCH {dotted}: got {got!r} want {want!r}")
            ok = False
    # structural: every top-level block the concurrence table consumes must exist
    for key in ("arm1_transferred_T_J", "arm2_supervised_validation", "arm3b_dispersion_pool",
                "arm3a_tau_daca", "arm4_confidence_instrument", "concurrence_table"):
        if key not in report:
            print(f"[selftest] MISSING block {key}")
            ok = False
    print(f"[selftest] {'PASS' if ok else 'FAIL'} ({len(SELFTEST_PINS)} pins + structure)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--closed-run-dir", type=Path, default=None)
    ap.add_argument("--bases-runs-dir", type=Path, default=REPO / "runs")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="recompute against the pinned reference run and compare to the pilot pins")
    args = ap.parse_args()

    if args.closed_run_dir is None:
        if not args.selftest:
            ap.error("--closed-run-dir is required (unless --selftest)")
        umbrella = Path(os.environ.get("JUDEX_UMBRELLA") or REPO.parent)
        args.closed_run_dir = umbrella / "judex-evaluator" / "runs" / SELFTEST_RUN

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
        "arm4_confidence_instrument": arm4(args.closed_run_dir, cells),
    }
    report["concurrence_table"] = concurrence_table(report)
    if smoke:
        report["smoke_note"] = "[SMOKE] < 120 items — prototype-labeled, nothing adoptable"

    print(json.dumps({k: report[k] for k in ("n_items", "smoke")}, indent=1))
    a1 = report["arm1_transferred_T_J"]
    print(f"arm1 F4 at T_J={T_J}: pass={a1['T_J']['f4_pass']} "
          f"(rel_impr={a1['T_J']['reliability_improvement']}, rps_change={a1['T_J']['rps_change']}, "
          f"res_within_tol={a1['T_J']['resolution_within_tol']})")
    print(f"  -> {a1['recommendation']}")
    inv = a1["argmax_invariance_tie_aware"]
    print(f"arm1 tie-aware invariance: tie_free_pass={inv['invariance_pass_tie_free']} "
          f"n_tied={inv['n_tied_exact_top2']}")
    print(f"arm2 T*={a2['T_star_full_sample']:.3f} (sat={a2['T_star_saturated']}) "
          f"in_band={a2['in_band']} F5 concurrent={a2['f5_concurrent']}")
    p3b = report["arm3b_dispersion_pool"]
    print(f"arm3b T_raw={p3b['T_raw']:.3f} vetting={p3b['vetting_pass']} in_band={p3b['in_band_concurrence']}")
    print(f"arm3a: {json.dumps(report['arm3a_tau_daca']['f7'])}")
    a4 = report["arm4_confidence_instrument"]
    for fam, blk in a4["per_seat"].items():
        f8 = blk.get("f8", {})
        print(f"arm4 {fam}: T_c={blk.get('T_c_full_sample')} kendall={blk.get('kendall_tau_b_s_vs_w1')} "
              f"carries_signal={f8.get('carries_signal')}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "e6_r3_arms.json").write_text(json.dumps(report, indent=1, default=float))
        print(f"wrote {args.out / 'e6_r3_arms.json'}")
    if args.selftest and not selftest(report):
        sys.exit(1)


if __name__ == "__main__":
    main()
