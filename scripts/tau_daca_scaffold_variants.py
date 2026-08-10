#!/usr/bin/env python3
"""tau_DACA under SCAFFOLD-VARIANT references (2026-08-09) -- $0, report-only.

LABELED VARIANT READ. The frozen drivers are never edited:
  * scripts/tau_daca_verbalized_fullscale.py  -- the study of record (baseline scaffold)
  * scripts/objective_contamination_audit.py  -- the matched-objective (W1) audit
  * scripts/absolute_vs_ratio_estimand.py     -- the rA1-adjacent frozen estimand driver
This file re-runs their exact tau_DACA machinery with ONE thing changed: which pre legs
supply the references.

Question (transfer-condition (iv) of docs/daca_standing_assessment_20260809.md):
the verbalized tau_DACA corroboration was computed against BASELINE-scaffold base legs,
and Phase 1a/1b showed those legs move materially under coverage-preserving scaffold
perturbation (alt_set inflates T_abs(pre) +0.46--0.72 ln on three families). Does the F7
corroboration survive when the references are the V1 (alt_set) / V2 (rev_order) pre legs?

Method, inherited verbatim from the study of record:
  targets    = the four panel POST legs of record (closed-evaluator analogs)
  references = the four panel PRE legs, swapped per arm
  filter     = strict argmax agreement, epsilon = 0.005 floor at analysis time
  fit        = judex.calibration.fit_temperature (RPS, T_BOUNDS = (0.25, 20.0))
  W1 variant = the same filter on study_a.GRID under W1 (objective_contamination_audit A2)
  validity   = not saturated AND reference not below chance (argmax acc >= 0.2, gate-only)
  F7         = per target over the 3 CROSS-family references: n_valid >= 3 and every valid
               fit inside [T_J/2, 2 T_J] = [0.5765, 2.3060]

Nothing here adopts anything. F7 was registered for the baseline scaffold only, so every
variant verdict is EXPLORATORY: it says whether the corroboration is scaffold-robust.

Run from judex-calibration:
  python scripts/tau_daca_scaffold_variants.py [--out runs/tau_daca_scaffold_variants]
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

from judex_calibration import aireg, study_a                      # noqa: E402
from judex_calibration.study_a import T_BOUNDS, GRID, saturated   # noqa: E402

from judex.core.distributions import ComplianceDistribution       # noqa: E402
from judex.core.metrics import wasserstein_1, ranked_probability_score  # noqa: E402
from judex.calibration import apply_temperature, fit_temperature  # noqa: E402

EPSILON = 0.005                     # Study B analysis convention
T_J = 1.153                         # r3 F2
DACA_MIN_VALID = 3                  # r3 F7
DACA_RANGE = (T_J / 2.0, 2.0 * T_J)  # [0.5765, 2.3060]

PANEL = ("qwen", "gemma31", "glm", "maverick")

# ---------------------------------------------------------------- leg inventory
#
# TARGETS are held fixed at the study of record's POST legs. gemma31's post of record is
# the OpenRouter chat leg (study_b_gemma31_api) -- the pairing every published constant was
# fitted on; the matched-transport post (study_b_gemma31_vllmchat_base) is carried as a
# labeled sensitivity, per the Phase-1 convention in scripts/absolute_estimand_k5variants.py.
TARGET_DIRS = {
    "qwen":     "study_b_qwen",
    "gemma31":  "study_b_gemma31_api",
    "glm":      "study_b_glm",
    "maverick": "study_b_maverick",
}
TARGET_DIRS_MATCHED_TRANSPORT = dict(TARGET_DIRS, gemma31="study_b_gemma31_vllmchat_base")

# REFERENCES. Baseline = the study-of-record pre legs. Variant pre legs live in the k5v*
# dirs for every family INCLUDING gemma31 (its variant *posts* are the chat legs in the
# same dirs; the OpenRouter study_b_gemma31_api_k5v1 dir is a legacy annex and holds no pre
# leg at all).  Every gemma31 pre leg -- baseline and both variants -- is the same
# transport: bf16 vast raw /v1/completions.  So the reference swap is transport-matched.
REFERENCE_DIRS = {
    "baseline": {
        "qwen":     "study_b_qwen",
        "gemma31":  "study_b_gemma31_api",       # pre = provenance copy of study_b_gemma31 pre
        "glm":      "study_b_glm",
        "maverick": "study_b_maverick",
    },
    "V1_alt_set": {
        "qwen":     "study_b_qwen_k5v1",
        "gemma31":  "study_b_gemma31_k5v1",
        "glm":      "study_b_glm_k5v1",
        "maverick": "study_b_maverick_k5v1",
    },
    "V2_rev_order": {
        "qwen":     "study_b_qwen_k5v2",
        "gemma31":  "study_b_gemma31_k5v2",
        "glm":      "study_b_glm_k5v2",
        "maverick": "study_b_maverick_k5v2",
    },
}

# B-Q1 contract-gate blemishes recorded in Phase 1a/1b (run READMEs). Both are POST legs,
# so neither enters the PRIMARY (references-only) swap.
GATE_BLEMISHES = {
    ("V2_rev_order", "qwen", "post"): 0.80,
    ("V1_alt_set", "glm", "post"): 0.8833,
}

# Baseline values of record to reproduce (study of record + objective-contamination A2).
RECORD = {
    "rps": {"n_valid": 12, "median": 0.9224116756116703,
            "range": [0.767484205941036, 1.072714731330948]},
    "w1":  {"n_valid": 12, "median": 0.9884871021863636,
            "range": [0.761686829270434, 1.0251785151221313]},
}
TOL = 5e-9


# ------------------------------------------------------------------------ loading
def floor_renorm(v, eps=EPSILON):
    w = [max(float(x), eps) for x in v]
    z = sum(w)
    return [x / z for x in w]


def load_leg(run_dir: Path, leg: str):
    """{item_label: floored compliance vector} over parse_ok cells -- the Study B
    convention shared by elicit_verbalized.compliance_view + floor_and_renormalize."""
    p = run_dir / f"{leg}_verbalized.json"
    if not p.exists():
        return None
    recs = json.loads(p.read_text())
    out = {lb: floor_renorm(r["compliance"]) for lb, r in recs.items()
           if r.get("parse_ok") and r.get("compliance")}
    return out or None


def contract_rate(run_dir: Path, leg: str):
    p = run_dir / f"{leg}_verbalized.json"
    if not p.exists():
        return None
    recs = json.loads(p.read_text())
    if not recs:
        return None
    parsed = [r for r in recs.values() if r.get("parse_ok")]
    return {"n": len(recs), "parse_rate": len(parsed) / len(recs),
            "contract_complete": (sum(1 for r in parsed if r.get("contract_complete"))
                                  / len(recs))}


# ------------------------------------------------------------------------ fitters
def grid_fit(pairs, metric):
    """The 60-pt study_a.GRID fit under a swappable objective -- identical machinery to
    study_a.fit_tau_oc, so W1 and RPS differ ONLY in the objective."""
    def obj(T):
        return sum(metric(apply_temperature(p, T), q) for p, q in pairs) / len(pairs)
    best = min(GRID, key=obj)
    o_best, o_one = obj(best), obj(1.0)
    return {"T": best, "movement_pct": (100.0 * (o_one - o_best) / o_one) if o_one > 0 else float("nan"),
            "saturated": saturated(best)}


def daca_fit(closed: dict, reference: dict, cells, metric=None) -> dict:
    """study_a.fit_tau_daca's agreement filter with a swappable alignment objective.

    metric=None -> the fitter OF RECORD (judex.calibration.fit_temperature, RPS, 49-pt
                   grid + golden refine).  metric=W1/RPS -> the 60-pt grid variant.
    Same function as objective_contamination_audit.daca_fit, with ONE hardening: the
    overlap is iterated in SORTED label order (as scripts/tau_daca_verbalized_fullscale.py
    does). Set iteration over str keys is PYTHONHASHSEED-dependent, so the unsorted form
    varies the floating-point summation order run to run -- harmless at 1e-13 in the
    objective, but it perturbed one golden-section refine in the 7th digit. Sorting makes
    every number here reproduce bit-for-bit and matches the frozen study-of-record driver.
    """
    by = {c.item_label: c for c in cells}
    pairs, overlap, ref_correct, closed_correct = [], 0, 0, 0
    h_ref_all, h_closed_all, h_ref_agree, h_closed_agree = [], [], [], []
    for lb in sorted(closed.keys() & reference.keys()):
        c = by.get(lb)
        if c is None:
            continue
        overlap += 1
        pc = study_a._pred_dist(closed[lb], c.gt_labels)
        pr = study_a._pred_dist(reference[lb], c.gt_labels)
        ref_correct += pr.argmax_index() == c.gt_argmax
        closed_correct += pc.argmax_index() == c.gt_argmax
        h_ref_all.append(pr.normalized_entropy())
        h_closed_all.append(pc.normalized_entropy())
        if pc.argmax_index() == pr.argmax_index():
            pairs.append((pc, pr))
            h_ref_agree.append(pr.normalized_entropy())
            h_closed_agree.append(pc.normalized_entropy())
    if not pairs:
        return {"n_overlap": overlap, "n_agreement": 0, "agreement_rate": 0.0,
                "tau": float("nan"), "saturated": True,
                "reference_below_chance": bool(ref_correct / max(overlap, 1) < 0.2),
                "valid": False}
    if metric is None:
        tau = fit_temperature(pairs, bounds=T_BOUNDS).temperature
        mv = float("nan")
    else:
        f = grid_fit(pairs, metric)
        tau, mv = f["T"], f["movement_pct"]
    sat = saturated(tau)
    below = bool(ref_correct / overlap < 0.2)
    return {
        "n_overlap": overlap, "n_agreement": len(pairs),
        "agreement_rate": len(pairs) / overlap,
        "tau": tau, "movement_pct": mv, "saturated": sat,
        "reference_argmax_acc_gate_only": ref_correct / overlap,
        "closed_argmax_acc_gate_only": closed_correct / overlap,
        "reference_below_chance": below,
        "valid": bool(not sat and not below),
        "in_daca_range": bool(not sat and DACA_RANGE[0] <= tau <= DACA_RANGE[1]),
        "mean_h_ref_all": statistics.mean(h_ref_all),
        "mean_h_closed_all": statistics.mean(h_closed_all),
        "mean_h_ref_agree": statistics.mean(h_ref_agree),
        "mean_h_closed_agree": statistics.mean(h_closed_agree),
    }


def f7_verdict(cross_fits: dict) -> dict:
    """r3 F7 over the CROSS-family references of one target."""
    valid = sorted(r["tau"] for r in cross_fits.values() if r["valid"])
    excluded = sorted(f for f, r in cross_fits.items() if not r["valid"])
    inside = [t for t in valid if DACA_RANGE[0] <= t <= DACA_RANGE[1]]
    return {"n_valid": len(valid), "valid_taus": valid, "n_in_range": len(inside),
            "excluded": excluded, "min_valid": DACA_MIN_VALID,
            "range_criterion": list(DACA_RANGE),
            "corroborates": bool(len(valid) >= DACA_MIN_VALID and len(inside) == len(valid))}


def summarize(taus) -> dict:
    s = sorted(taus)
    if not s:
        return {"n_valid": 0}
    return {"n_valid": len(s), "median": statistics.median(s),
            "min": s[0], "max": s[-1], "range": [s[0], s[-1]],
            "all_inside_window": bool(all(DACA_RANGE[0] <= t <= DACA_RANGE[1] for t in s))}


# --------------------------------------------------------------------------- arms
def run_arm(targets: dict, references: dict, cells, *, arm: str) -> dict:
    """One reference-set: 4x4 fits under both objectives, F7 per target, summaries."""
    out = {"arm": arm, "per_target": {}, "reference_entropy": {}}
    for tgt in PANEL:
        if tgt not in targets:
            continue
        rows = {}
        for ref in PANEL:
            if ref not in references:
                continue
            rows[ref] = {
                "rps_of_record": daca_fit(targets[tgt], references[ref], cells, None),
                "w1_grid": daca_fit(targets[tgt], references[ref], cells, wasserstein_1),
                "rps_grid": daca_fit(targets[tgt], references[ref], cells, ranked_probability_score),
                "own_family": ref == tgt,
            }
        cross_rps = {r: v["rps_of_record"] for r, v in rows.items() if not v["own_family"]}
        cross_w1 = {r: v["w1_grid"] for r, v in rows.items() if not v["own_family"]}
        out["per_target"][tgt] = {
            "fits": rows,
            "f7_cross_family_rps": f7_verdict(cross_rps),
            "f7_cross_family_w1": f7_verdict(cross_w1),
        }
    # pooled cross-family summaries
    for obj_key, sub in (("rps", "rps_of_record"), ("w1", "w1_grid")):
        taus = [v[sub]["tau"] for t in out["per_target"].values()
                for v in t["fits"].values() if not v["own_family"] and v[sub]["valid"]]
        out[f"summary_{obj_key}"] = summarize(taus)
    out["f7_all_targets_corroborate_rps"] = bool(
        out["per_target"] and all(t["f7_cross_family_rps"]["corroborates"]
                                  for t in out["per_target"].values()))
    out["f7_all_targets_corroborate_w1"] = bool(
        out["per_target"] and all(t["f7_cross_family_w1"]["corroborates"]
                                  for t in out["per_target"].values()))
    # objective-free reference sharpness on each reference leg's own parse-ok support
    for ref in PANEL:
        if ref in references:
            out["reference_entropy"][ref] = statistics.mean(
                ComplianceDistribution.from_values(v, aireg_labels).normalized_entropy()
                for v in references[ref].values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/tau_daca_scaffold_variants")
    args = ap.parse_args()

    cells = aireg.load_cells()
    global aireg_labels
    aireg_labels = cells[0].gt_labels

    runs = REPO / "runs"

    # ---- load every leg once
    targets = {f: load_leg(runs / d, "post") for f, d in TARGET_DIRS.items()}
    targets_mt = {f: load_leg(runs / d, "post") for f, d in TARGET_DIRS_MATCHED_TRANSPORT.items()}
    refs = {arm: {f: load_leg(runs / d, "pre") for f, d in m.items()}
            for arm, m in REFERENCE_DIRS.items()}
    variant_posts = {arm: {f: load_leg(runs / d, "post") for f, d in m.items()}
                     for arm, m in REFERENCE_DIRS.items() if arm != "baseline"}

    missing = [f"{arm}/{f}" for arm, m in refs.items() for f, v in m.items() if v is None]
    if missing:
        raise SystemExit(f"missing reference legs: {missing}")

    # ---- leg inventory / gate provenance
    inventory = {}
    for arm, m in REFERENCE_DIRS.items():
        inventory[arm] = {f: {"dir": d,
                              "pre": contract_rate(runs / d, "pre"),
                              "post": contract_rate(runs / d, "post")}
                          for f, d in m.items()}
    inventory["targets_of_record"] = {f: {"dir": d, "post": contract_rate(runs / d, "post")}
                                      for f, d in TARGET_DIRS.items()}

    report = {
        "what": "tau_DACA recomputed with scaffold-variant PRE legs as references",
        "status": "EXPLORATORY / DIAGNOSTIC -- F7 was registered for the baseline scaffold "
                  "only; nothing adopted, r3 F1-F8 and rA1 A1-A8 untouched",
        "cost_usd": 0.0,
        "epsilon": EPSILON,
        "T_bounds": list(T_BOUNDS),
        "frozen": {"T_J": T_J, "DACA_RANGE": list(DACA_RANGE),
                   "DACA_MIN_VALID": DACA_MIN_VALID},
        "leg_inventory": inventory,
        "gate_blemishes": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in GATE_BLEMISHES.items()},
        "arms": {},
    }

    # ---- ARM 0..2: references swapped, targets held at the study of record
    for arm in ("baseline", "V1_alt_set", "V2_rev_order"):
        report["arms"][arm] = run_arm(targets, refs[arm], cells, arm=arm)

    # ---- baseline reproduction attestation
    base = report["arms"]["baseline"]
    rep = {}
    for key, sub in (("rps", "summary_rps"), ("w1", "summary_w1")):
        got, want = base[sub], RECORD[key]
        rep[key] = {
            "recomputed": {k: got.get(k) for k in ("n_valid", "median", "range")},
            "of_record": want,
            "match": bool(got.get("n_valid") == want["n_valid"]
                          and abs(got["median"] - want["median"]) < TOL
                          and abs(got["range"][0] - want["range"][0]) < TOL
                          and abs(got["range"][1] - want["range"][1]) < TOL),
        }
    rep["all_four_targets_pass_F7_rps"] = base["f7_all_targets_corroborate_rps"]
    rep["reproduces"] = bool(rep["rps"]["match"] and rep["w1"]["match"]
                             and base["f7_all_targets_corroborate_rps"])
    report["baseline_reproduction"] = rep

    if not rep["reproduces"]:
        report["STOP"] = ("baseline did NOT reproduce -- variant arms are reported but must "
                          "not be read")

    # ---- shifts vs baseline
    shifts = {}
    for arm in ("V1_alt_set", "V2_rev_order"):
        shifts[arm] = {}
        for key in ("rps", "w1"):
            b = base[f"summary_{key}"]["median"]
            v = report["arms"][arm][f"summary_{key}"]["median"]
            shifts[arm][key] = {"baseline_median": b, "variant_median": v,
                                "delta_ln": math.log(v / b), "ratio": v / b}
        # reference-side sharpening check: mean normalized entropy per reference leg
        shifts[arm]["reference_mean_norm_entropy"] = {
            f: {"baseline": base["reference_entropy"][f],
                "variant": report["arms"][arm]["reference_entropy"][f],
                "delta": report["arms"][arm]["reference_entropy"][f] - base["reference_entropy"][f]}
            for f in PANEL}
        # agreement-rate re-composition
        shifts[arm]["agreement_rate_change"] = {}
        for tgt in PANEL:
            for ref in PANEL:
                if ref == tgt:
                    continue
                b = base["per_target"][tgt]["fits"][ref]["rps_of_record"]
                v = report["arms"][arm]["per_target"][tgt]["fits"][ref]["rps_of_record"]
                shifts[arm]["agreement_rate_change"][f"{tgt}<-{ref}"] = {
                    "baseline": b["agreement_rate"], "variant": v["agreement_rate"],
                    "delta": v["agreement_rate"] - b["agreement_rate"],
                    "n_agree_baseline": b["n_agreement"], "n_agree_variant": v["n_agreement"],
                    "tau_baseline": b["tau"], "tau_variant": v["tau"],
                    "d_ln_tau": math.log(v["tau"] / b["tau"]),
                }
    report["shift_vs_baseline"] = shifts

    # ---- SENSITIVITY 1: gemma31 target from the matched-transport baseline post
    report["sensitivity_gemma31_matched_transport_target"] = {
        arm: run_arm(targets_mt, refs[arm], cells, arm=f"{arm}+gemma31_matched_transport")
        for arm in ("baseline", "V1_alt_set", "V2_rev_order")}

    # ---- SENSITIVITY 2: BOTH sides swapped (variant targets AND variant references).
    # This is where the two B-Q1 gate blemishes (qwen k5v2 post 0.80, glm k5v1 post 0.8833)
    # actually appear, so it is reported with and without the blemished targets.
    both = {}
    for arm in ("V1_alt_set", "V2_rev_order"):
        r = run_arm(variant_posts[arm], refs[arm], cells, arm=f"{arm}_both_sides")
        blem = [f for (a, f, leg) in GATE_BLEMISHES if a == arm and leg == "post"]
        for key, sub in (("rps", "rps_of_record"), ("w1", "w1_grid")):
            taus = [v[sub]["tau"] for t, tv in r["per_target"].items() if t not in blem
                    for v in tv["fits"].values() if not v["own_family"] and v[sub]["valid"]]
            r[f"summary_{key}_excl_gate_fail_targets"] = summarize(taus)
        r["gate_fail_targets_excluded"] = blem
        r["f7_all_targets_corroborate_rps_excl_gate_fail"] = bool(all(
            t["f7_cross_family_rps"]["corroborates"]
            for f, t in r["per_target"].items() if f not in blem))
        both[arm] = r
    report["sensitivity_both_sides_swapped"] = both

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tau_daca_scaffold_variants.json"
    path.write_text(json.dumps(report, indent=1, default=float))

    # ------------------------------------------------------------------ printout
    print("=" * 78)
    print("BASELINE REPRODUCTION")
    print("=" * 78)
    for key in ("rps", "w1"):
        r = rep[key]
        print(f"  {key.upper():3s} n_valid={r['recomputed']['n_valid']} "
              f"median={r['recomputed']['median']:.10f} "
              f"range=[{r['recomputed']['range'][0]:.7f}, {r['recomputed']['range'][1]:.7f}] "
              f"-> match={r['match']}")
    print(f"  all four targets pass F7 (RPS): {rep['all_four_targets_pass_F7_rps']}")
    print(f"  REPRODUCES: {rep['reproduces']}")

    for arm in ("baseline", "V1_alt_set", "V2_rev_order"):
        a = report["arms"][arm]
        print("\n" + "=" * 78)
        print(f"ARM: references = {arm}")
        print("=" * 78)
        print(f"  {'target':10s} " + " ".join(f"{r:>12s}" for r in PANEL) + "   F7(RPS)")
        for tgt in PANEL:
            row = a["per_target"][tgt]["fits"]
            cells_s = []
            for ref in PANEL:
                t = row[ref]["rps_of_record"]["tau"]
                cells_s.append(f"{t:9.3f}{'(o)' if ref == tgt else '   '}")
            v = a["per_target"][tgt]["f7_cross_family_rps"]
            print(f"  {tgt:10s} " + " ".join(cells_s) +
                  f"   {'PASS' if v['corroborates'] else 'FAIL'} (n_valid={v['n_valid']})")
        for key in ("rps", "w1"):
            s = a[f"summary_{key}"]
            print(f"  cross-family {key.upper():3s}: n={s['n_valid']} "
                  f"median={s['median']:.4f} range=[{s['min']:.4f}, {s['max']:.4f}] "
                  f"all_inside_window={s['all_inside_window']}")
        print(f"  reference mean normalized entropy: " +
              ", ".join(f"{f}={a['reference_entropy'][f]:.4f}" for f in PANEL))
        print(f"  ALL FOUR TARGETS CORROBORATE (RPS): {a['f7_all_targets_corroborate_rps']} | "
              f"(W1): {a['f7_all_targets_corroborate_w1']}")

    print("\n" + "=" * 78)
    print("SHIFT VS BASELINE")
    print("=" * 78)
    for arm, s in shifts.items():
        print(f"  {arm}: d ln median RPS = {s['rps']['delta_ln']:+.4f} "
              f"({s['rps']['baseline_median']:.4f} -> {s['rps']['variant_median']:.4f}) | "
              f"W1 = {s['w1']['delta_ln']:+.4f}")
        for f in PANEL:
            e = s["reference_mean_norm_entropy"][f]
            print(f"      ref {f:9s} h̄ {e['baseline']:.4f} -> {e['variant']:.4f} "
                  f"({e['delta']:+.4f})")

    print("\n" + "=" * 78)
    print("SENSITIVITY: both sides swapped (variant targets AND references)")
    print("=" * 78)
    for arm, r in both.items():
        s, se = r["summary_rps"], r["summary_rps_excl_gate_fail_targets"]
        print(f"  {arm}: all targets n={s['n_valid']} median={s['median']:.4f} "
              f"range=[{s['min']:.4f}, {s['max']:.4f}] inside={s['all_inside_window']} "
              f"| F7 all={r['f7_all_targets_corroborate_rps']}")
        print(f"      excl gate-fail targets {r['gate_fail_targets_excluded']}: "
              f"n={se['n_valid']} median={se['median']:.4f} "
              f"range=[{se['min']:.4f}, {se['max']:.4f}] inside={se['all_inside_window']} "
              f"| F7={r['f7_all_targets_corroborate_rps_excl_gate_fail']}")

    print("\n" + "=" * 78)
    print("SENSITIVITY: gemma31 target = matched-transport vllmchat baseline post")
    print("=" * 78)
    for arm, r in report["sensitivity_gemma31_matched_transport_target"].items():
        s = r["summary_rps"]
        print(f"  {arm}: n={s['n_valid']} median={s['median']:.4f} "
              f"range=[{s['min']:.4f}, {s['max']:.4f}] inside={s['all_inside_window']} "
              f"| F7 all={r['f7_all_targets_corroborate_rps']}")

    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
