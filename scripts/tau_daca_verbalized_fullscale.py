#!/usr/bin/env python3
"""Full-scale τ_DACA study on verbalized Study B data (120 cells).

Each panel POST leg plays the role of the closed evaluator; each panel BASE
leg is a reference.  Removes the 15-item scale caveat of the R0 prototype.

Primary question: at 120 cells with panel references, does τ_DACA meet F7
(≥3 valid fits in [τ*/2, 2τ*] = [0.577, 2.306])?

Diagnosis: reference non-exchangeability, agreement-filter composition,
filter-threshold sweeps, pegged-fit anatomy, structural-vs-repairable verdict.

Run:
  PYTHONPATH=src python scripts/tau_daca_verbalized_fullscale.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from judex_calibration import aireg
from judex_calibration.elicit_verbalized import compliance_view, floor_and_renormalize, EPSILON
from judex_calibration import study_a
from judex_calibration.study_a import T_BOUNDS, GRID, saturated
from judex_calibration.study_b import kendall_tau_b

from judex.core.distributions import ComplianceDistribution
from judex.core.metrics import wasserstein_1, ranked_probability_score
from judex.calibration import apply_temperature, fit_temperature

LABELS = ("very_low", "low", "moderate", "high", "very_high")

# Panel (user decision 2026-07-21)
PANEL = ["qwen", "gemma31_api", "glm", "maverick"]
LEGS = {
    "qwen":        "runs/study_b_qwen",
    "gemma31_api": "runs/study_b_gemma31_api",
    "glm":         "runs/study_b_glm",
    "maverick":    "runs/study_b_maverick",
    "llama31":     "runs/study_b_llama31",
}

# r3 frozen constants
T_J = 1.153
BAND = (1.0251785151221313, 1.379820350674421)
DACA_MIN_VALID = 3
DACA_RANGE = (T_J / 2.0, 2.0 * T_J)


def load_leg(run_dir: Path):
    pre = json.loads((run_dir / "pre_verbalized.json").read_text())
    post = json.loads((run_dir / "post_verbalized.json").read_text())
    return pre, post


def floored_view(recs) -> dict:
    return {k: floor_and_renormalize(v, EPSILON) for k, v in compliance_view(recs).items()}


def dist(probs, labels=LABELS) -> ComplianceDistribution:
    return ComplianceDistribution.from_values(list(probs), labels)


def norm_entropy(probs) -> float:
    return dist(probs).normalized_entropy()


# ──────────────────────────────────────────────────────────────────────
# 1. Core τ_DACA fit with rich diagnostics
# ──────────────────────────────────────────────────────────────────────

def fit_daca_detailed(closed: dict, reference: dict, cells, *, label=""):
    """Like study_a.fit_tau_daca but with per-cell diagnostic vectors."""
    by_label = {c.item_label: c for c in cells}

    overlap_labels = sorted(closed.keys() & reference.keys() & by_label.keys())
    pairs = []
    diagnostics = []

    for lb in overlap_labels:
        c = by_label[lb]
        p_closed = dist(closed[lb])
        p_ref = dist(reference[lb])
        p_gt = ComplianceDistribution.from_values(c.gt_probs, c.gt_labels)

        argmax_agree = p_closed.argmax_index() == p_ref.argmax_index()
        h_closed = p_closed.normalized_entropy()
        h_ref = p_ref.normalized_entropy()
        h_gt = p_gt.normalized_entropy()
        w1_closed_gt = wasserstein_1(p_closed, p_gt)
        w1_ref_gt = wasserstein_1(p_ref, p_gt)
        mode_agree_with_gt_closed = p_closed.argmax_index() == p_gt.argmax_index()
        mode_agree_with_gt_ref = p_ref.argmax_index() == p_gt.argmax_index()
        rps_closed_ref = ranked_probability_score(p_closed, p_ref)

        row = {
            "label": lb, "doc": c.document_id, "article": c.article,
            "argmax_agree": argmax_agree,
            "closed_argmax": p_closed.argmax_index(),
            "ref_argmax": p_ref.argmax_index(),
            "gt_argmax": p_gt.argmax_index(),
            "h_closed": h_closed, "h_ref": h_ref, "h_gt": h_gt,
            "h_gap_ref_minus_closed": h_ref - h_closed,
            "w1_closed_gt": w1_closed_gt, "w1_ref_gt": w1_ref_gt,
            "mode_agree_closed_gt": mode_agree_with_gt_closed,
            "mode_agree_ref_gt": mode_agree_with_gt_ref,
            "rps_closed_ref": rps_closed_ref,
        }
        diagnostics.append(row)

        if argmax_agree:
            pairs.append((p_closed, p_ref))

    n_overlap = len(overlap_labels)
    n_agree = len(pairs)

    if not pairs:
        tau = float("nan")
        sat = True
    else:
        tau = fit_temperature(pairs, bounds=T_BOUNDS).temperature
        sat = saturated(tau)

    ref_correct = sum(1 for d in diagnostics if d["mode_agree_ref_gt"])
    closed_correct = sum(1 for d in diagnostics if d["mode_agree_closed_gt"])

    agree_cells = [d for d in diagnostics if d["argmax_agree"]]
    disagree_cells = [d for d in diagnostics if not d["argmax_agree"]]

    return {
        "label": label,
        "n_overlap": n_overlap,
        "n_agreement": n_agree,
        "agreement_rate": n_agree / n_overlap if n_overlap else 0.0,
        "tau_daca": tau,
        "tau_daca_saturated": sat,
        "reference_argmax_acc": ref_correct / n_overlap if n_overlap else 0.0,
        "closed_argmax_acc": closed_correct / n_overlap if n_overlap else 0.0,
        "reference_below_chance": bool(n_overlap > 0 and ref_correct / n_overlap < 0.2),
        "in_daca_range": bool(not sat and DACA_RANGE[0] <= tau <= DACA_RANGE[1]),
        "entropy": {
            "mean_h_closed_agree": (statistics.mean(d["h_closed"] for d in agree_cells)
                                    if agree_cells else None),
            "mean_h_ref_agree": (statistics.mean(d["h_ref"] for d in agree_cells)
                                 if agree_cells else None),
            "mean_h_closed_disagree": (statistics.mean(d["h_closed"] for d in disagree_cells)
                                       if disagree_cells else None),
            "mean_h_ref_disagree": (statistics.mean(d["h_ref"] for d in disagree_cells)
                                    if disagree_cells else None),
            "mean_h_gap_agree": (statistics.mean(d["h_gap_ref_minus_closed"] for d in agree_cells)
                                 if agree_cells else None),
            "mean_h_gap_disagree": (statistics.mean(d["h_gap_ref_minus_closed"] for d in disagree_cells)
                                    if disagree_cells else None),
            "mean_h_ref_all": statistics.mean(d["h_ref"] for d in diagnostics) if diagnostics else None,
            "mean_h_closed_all": statistics.mean(d["h_closed"] for d in diagnostics) if diagnostics else None,
        },
        "w1": {
            "mean_w1_closed_gt_agree": (statistics.mean(d["w1_closed_gt"] for d in agree_cells)
                                        if agree_cells else None),
            "mean_w1_ref_gt_agree": (statistics.mean(d["w1_ref_gt"] for d in agree_cells)
                                     if agree_cells else None),
            "mean_w1_closed_gt_disagree": (statistics.mean(d["w1_closed_gt"] for d in disagree_cells)
                                           if disagree_cells else None),
            "mean_w1_ref_gt_disagree": (statistics.mean(d["w1_ref_gt"] for d in disagree_cells)
                                        if disagree_cells else None),
        },
        "per_cell": diagnostics,
    }


# ──────────────────────────────────────────────────────────────────────
# 2. F7 verdict per POST target
# ──────────────────────────────────────────────────────────────────────

def f7_verdict(per_ref_fits: dict) -> dict:
    valid = []
    excluded = []
    for fam, r in per_ref_fits.items():
        if r["tau_daca_saturated"] or r.get("reference_below_chance", False):
            excluded.append(fam)
        else:
            valid.append(r["tau_daca"])
    in_range = [v for v in valid if DACA_RANGE[0] <= v <= DACA_RANGE[1]]
    corroborates = len(valid) >= DACA_MIN_VALID and len(in_range) == len(valid)
    return {
        "n_valid": len(valid),
        "valid_taus": sorted(valid),
        "n_in_range": len(in_range),
        "min_valid": DACA_MIN_VALID,
        "range": list(DACA_RANGE),
        "excluded": excluded,
        "corroborates": corroborates,
    }


# ──────────────────────────────────────────────────────────────────────
# 3. Agreement-filter sweep (relax/tighten the agreement criterion)
# ──────────────────────────────────────────────────────────────────────

def agreement_filter_sweep(closed: dict, reference: dict, cells) -> list:
    """Sweep agreement-filter variants: strict argmax, top-2, top-3, no filter."""
    by_label = {c.item_label: c for c in cells}
    overlap_labels = sorted(closed.keys() & reference.keys() & by_label.keys())

    def fit_with_filter(filter_fn, name):
        pairs = []
        n_pass = 0
        for lb in overlap_labels:
            p_closed = dist(closed[lb])
            p_ref = dist(reference[lb])
            if filter_fn(p_closed, p_ref):
                n_pass += 1
                pairs.append((p_closed, p_ref))
        if not pairs:
            return {"filter": name, "n_pass": n_pass, "tau": float("nan"), "saturated": True}
        tau = fit_temperature(pairs, bounds=T_BOUNDS).temperature
        return {"filter": name, "n_pass": n_pass, "pass_rate": n_pass / len(overlap_labels),
                "tau": tau, "saturated": saturated(tau)}

    def top_k_agree(p1, p2, k):
        s1 = set(sorted(range(5), key=lambda i: -p1.probabilities[i])[:k])
        s2 = set(sorted(range(5), key=lambda i: -p2.probabilities[i])[:k])
        return bool(s1 & s2)

    return [
        fit_with_filter(lambda p, r: p.argmax_index() == r.argmax_index(), "argmax_agree"),
        fit_with_filter(lambda p, r: top_k_agree(p, r, 2), "top2_overlap"),
        fit_with_filter(lambda p, r: top_k_agree(p, r, 3), "top3_overlap"),
        fit_with_filter(lambda p, r: True, "no_filter"),
    ]


# ──────────────────────────────────────────────────────────────────────
# 4. Entropy-matched reference selection (repair prototype)
# ──────────────────────────────────────────────────────────────────────

def entropy_matched_daca(closed: dict, references: dict, cells, *, max_gap=0.15):
    """Prototype repair: per-cell, pick the reference family whose BASE entropy
    is closest to the closed (POST) entropy. Only use references within max_gap
    of the closed entropy. Fit a single τ on these entropy-matched pairs."""
    by_label = {c.item_label: c for c in cells}
    overlap = sorted(set.intersection(set(closed.keys()), *[set(r.keys()) for r in references.values()])
                     & by_label.keys())

    pairs = []
    selections = []
    for lb in overlap:
        h_closed = norm_entropy(closed[lb])
        best_fam, best_gap = None, float("inf")
        for fam, refs in references.items():
            h_ref = norm_entropy(refs[lb])
            gap = abs(h_ref - h_closed)
            if gap < best_gap:
                best_gap = gap
                best_fam = fam
        if best_gap > max_gap:
            selections.append({"label": lb, "selected": None, "gap": best_gap, "filtered": True})
            continue
        p_closed = dist(closed[lb])
        p_ref = dist(references[best_fam][lb])
        if p_closed.argmax_index() != p_ref.argmax_index():
            selections.append({"label": lb, "selected": best_fam, "gap": best_gap,
                               "filtered_by_argmax": True})
            continue
        pairs.append((p_closed, p_ref))
        selections.append({"label": lb, "selected": best_fam, "gap": best_gap, "used": True})

    if not pairs:
        tau = float("nan")
        sat = True
    else:
        tau = fit_temperature(pairs, bounds=T_BOUNDS).temperature
        sat = saturated(tau)

    fam_counts = {}
    for s in selections:
        if s.get("used"):
            fam_counts[s["selected"]] = fam_counts.get(s["selected"], 0) + 1

    return {
        "estimator": "entropy_matched_daca",
        "max_gap": max_gap,
        "n_overlap": len(overlap),
        "n_entropy_filtered": sum(1 for s in selections if s.get("filtered")),
        "n_argmax_filtered": sum(1 for s in selections if s.get("filtered_by_argmax")),
        "n_used": len(pairs),
        "tau": tau, "saturated": sat,
        "in_daca_range": bool(not sat and DACA_RANGE[0] <= tau <= DACA_RANGE[1]),
        "family_selection_counts": fam_counts,
        "mean_gap_used": (statistics.mean(s["gap"] for s in selections if s.get("used"))
                          if pairs else None),
    }


# ──────────────────────────────────────────────────────────────────────
# 5. Mode-agreement stratum decomposition
# ──────────────────────────────────────────────────────────────────────

def stratum_decomposition(closed: dict, reference: dict, cells) -> dict:
    """Decompose agreement-filtered cells by R0 analysis A strata:
    mode-agree (closed vs GT) and mode-disagree."""
    by_label = {c.item_label: c for c in cells}
    overlap = sorted(closed.keys() & reference.keys() & by_label.keys())

    agree_mode_gt = []
    disagree_mode_gt = []
    for lb in overlap:
        c = by_label[lb]
        p_closed = dist(closed[lb])
        p_ref = dist(reference[lb])
        p_gt = ComplianceDistribution.from_values(c.gt_probs, c.gt_labels)
        if p_closed.argmax_index() != p_ref.argmax_index():
            continue
        row = (p_closed, p_ref)
        if p_closed.argmax_index() == p_gt.argmax_index():
            agree_mode_gt.append(row)
        else:
            disagree_mode_gt.append(row)

    def fit_stratum(pairs, name):
        if len(pairs) < 2:
            return {"stratum": name, "n": len(pairs), "tau": float("nan")}
        tau = fit_temperature(pairs, bounds=T_BOUNDS).temperature
        return {"stratum": name, "n": len(pairs), "tau": tau, "saturated": saturated(tau)}

    return {
        "mode_agree_with_gt": fit_stratum(agree_mode_gt, "mode_agree_with_gt"),
        "mode_disagree_with_gt": fit_stratum(disagree_mode_gt, "mode_disagree_with_gt"),
    }


# ──────────────────────────────────────────────────────────────────────
# 6. Cross-reference τ_v comparison
# ──────────────────────────────────────────────────────────────────────

def tau_v_comparison(views, cells) -> dict:
    """Fit τ_v (post→pre alignment within family) for context."""
    out = {}
    for fam, (pre_fl, post_fl) in views.items():
        tau = study_a.fit_tau_oc(post_fl, pre_fl, cells)
        out[fam] = {"tau_v": tau, "tau_v_saturated": saturated(tau)}
    return out


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    cells = aireg.load_cells()
    by_label = {c.item_label: c for c in cells}

    views = {}
    for fam, rel in LEGS.items():
        pre, post = load_leg(REPO / rel)
        views[fam] = (floored_view(pre), floored_view(post))

    # τ_v context
    tau_v = tau_v_comparison(views, cells)
    print("== τ_v (post→pre) for context ==")
    for fam, r in tau_v.items():
        print(f"  {fam:12s} τ_v={r['tau_v']:.3f} sat={r['tau_v_saturated']}")

    # ── PRIMARY: per-POST-target, per-BASE-reference τ_DACA ──
    print("\n== Per-target τ_DACA at 120-cell scale ==")
    all_fits = {}
    for target_fam in PANEL:
        post_fl = views[target_fam][1]  # POST leg = "closed"
        fits = {}
        for ref_fam in PANEL:
            pre_fl = views[ref_fam][0]  # BASE leg = reference
            tag = "own" if ref_fam == target_fam else "cross"
            r = fit_daca_detailed(post_fl, pre_fl, cells,
                                  label=f"{target_fam}_post→{ref_fam}_base ({tag})")
            fits[ref_fam] = r
            print(f"  {target_fam:12s} → {ref_fam:12s} [{tag:5s}] "
                  f"τ_DACA={r['tau_daca']:.3f} agree={r['agreement_rate']:.2f} "
                  f"n_agree={r['n_agreement']}/{r['n_overlap']} "
                  f"sat={r['tau_daca_saturated']} "
                  f"h_gap_agree={r['entropy']['mean_h_gap_agree']}")
        all_fits[target_fam] = fits

    # ── F7 verdicts ──
    print("\n== F7 verdicts ==")
    f7_all = {}
    for target_fam in PANEL:
        cross_fits = {rf: all_fits[target_fam][rf] for rf in PANEL if rf != target_fam}
        v = f7_verdict(cross_fits)
        f7_all[target_fam] = v
        print(f"  {target_fam:12s} cross-family: n_valid={v['n_valid']} "
              f"taus={[round(t, 3) for t in v['valid_taus']]} "
              f"corroborates={v['corroborates']}")

    # all-reference F7 (including own-family, matching arm3a structure)
    f7_all_ref = {}
    for target_fam in PANEL:
        v = f7_verdict(all_fits[target_fam])
        f7_all_ref[target_fam] = v
        print(f"  {target_fam:12s} all-ref:      n_valid={v['n_valid']} "
              f"taus={[round(t, 3) for t in v['valid_taus']]} "
              f"corroborates={v['corroborates']}")

    # ── llama31 as additional diagnostic target ──
    print("\n== llama31 (excluded, diagnostic only) ==")
    post_llama = views["llama31"][1]
    llama_fits = {}
    for ref_fam in PANEL:
        pre_fl = views[ref_fam][0]
        r = fit_daca_detailed(post_llama, pre_fl, cells,
                              label=f"llama31_post→{ref_fam}_base")
        llama_fits[ref_fam] = r
        print(f"  llama31      → {ref_fam:12s} τ_DACA={r['tau_daca']:.3f} "
              f"agree={r['agreement_rate']:.2f} sat={r['tau_daca_saturated']}")
    v_llama = f7_verdict(llama_fits)
    print(f"  llama31 F7: n_valid={v_llama['n_valid']} "
          f"taus={[round(t, 3) for t in v_llama['valid_taus']]} "
          f"corroborates={v_llama['corroborates']}")

    # ── Agreement-filter sweeps ──
    print("\n== Agreement-filter sweeps (sample: qwen post → maverick base) ==")
    sweep_results = {}
    for target_fam in PANEL:
        sweep_results[target_fam] = {}
        for ref_fam in PANEL:
            if ref_fam == target_fam:
                continue
            sw = agreement_filter_sweep(views[target_fam][1], views[ref_fam][0], cells)
            sweep_results[target_fam][ref_fam] = sw
            if target_fam == "qwen" and ref_fam == "maverick":
                for s in sw:
                    print(f"  {s['filter']:15s} n_pass={s['n_pass']:3d} "
                          f"τ={s['tau']:.3f} sat={s.get('saturated', False)}")

    # ── Mode-agreement stratum decomposition ──
    print("\n== Mode-agreement stratum decomposition ==")
    strata_results = {}
    for target_fam in PANEL:
        strata_results[target_fam] = {}
        for ref_fam in PANEL:
            if ref_fam == target_fam:
                continue
            sd = stratum_decomposition(views[target_fam][1], views[ref_fam][0], cells)
            strata_results[target_fam][ref_fam] = sd
            ma = sd["mode_agree_with_gt"]
            md = sd["mode_disagree_with_gt"]
            print(f"  {target_fam:12s}→{ref_fam:12s} "
                  f"agree_gt: n={ma['n']} τ={ma.get('tau', float('nan')):.3f} | "
                  f"disagree_gt: n={md['n']} τ={md.get('tau', float('nan')):.3f}")

    # ── Entropy-matched repair prototype ──
    print("\n== Entropy-matched τ_DACA (repair prototype) ==")
    repair_results = {}
    for target_fam in PANEL:
        cross_refs = {rf: views[rf][0] for rf in PANEL if rf != target_fam}
        for max_gap in [0.05, 0.10, 0.15, 0.25, 0.50, 1.0]:
            rep = entropy_matched_daca(views[target_fam][1], cross_refs, cells, max_gap=max_gap)
            repair_results.setdefault(target_fam, []).append(rep)
            if max_gap in (0.10, 0.25):
                print(f"  {target_fam:12s} gap≤{max_gap:.2f}: n_used={rep['n_used']} "
                      f"τ={rep['tau']:.3f} sat={rep['saturated']} "
                      f"in_range={rep['in_daca_range']}")

    # ── Per-reference entropy profile ──
    print("\n== Per-reference entropy profile ==")
    for target_fam in PANEL:
        post_fl = views[target_fam][1]
        overlap_all = sorted(set.intersection(*[set(views[rf][0].keys()) for rf in PANEL])
                             & post_fl.keys() & by_label.keys())
        h_target = statistics.mean(norm_entropy(post_fl[lb]) for lb in overlap_all)
        print(f"  {target_fam:12s} POST h̄={h_target:.3f}")
        for ref_fam in PANEL:
            pre_fl = views[ref_fam][0]
            h_ref = statistics.mean(norm_entropy(pre_fl[lb]) for lb in overlap_all if lb in pre_fl)
            print(f"    {ref_fam:12s} BASE h̄={h_ref:.3f}  Δ(ref-target)={h_ref - h_target:+.3f}")

    # ── Assemble full report ──
    report = {
        "mandate": "full-scale τ_DACA-verbalized study (120 cells), F7 verdict + diagnosis",
        "protocol": "r3 (docs/e6_onpair_decision_protocol_r3.md, ADOPTED 2026-07-21)",
        "epsilon": EPSILON,
        "T_bounds": list(T_BOUNDS),
        "frozen": {
            "T_J": T_J, "band": list(BAND),
            "DACA_MIN_VALID": DACA_MIN_VALID,
            "DACA_RANGE": list(DACA_RANGE),
        },
        "panel": PANEL,
        "tau_v_context": tau_v,
        "per_target": {},
    }

    for target_fam in PANEL:
        cross_fits_no_cells = {}
        for rf, r in all_fits[target_fam].items():
            row = {k: v for k, v in r.items() if k != "per_cell"}
            cross_fits_no_cells[rf] = row

        report["per_target"][target_fam] = {
            "per_reference_fits": cross_fits_no_cells,
            "f7_cross_family": f7_all[target_fam],
            "f7_all_reference": f7_all_ref[target_fam],
            "agreement_filter_sweeps": sweep_results[target_fam],
            "mode_agreement_strata": strata_results[target_fam],
            "entropy_matched_repair": repair_results[target_fam],
        }

    # llama31 diagnostic
    llama_no_cells = {rf: {k: v for k, v in r.items() if k != "per_cell"}
                      for rf, r in llama_fits.items()}
    report["llama31_diagnostic"] = {
        "per_reference_fits": llama_no_cells,
        "f7": v_llama,
    }

    # ── Summary table ──
    all_cross_taus = []
    for target_fam in PANEL:
        for rf in PANEL:
            if rf == target_fam:
                continue
            r = all_fits[target_fam][rf]
            if not r["tau_daca_saturated"] and not r.get("reference_below_chance", False):
                all_cross_taus.append(r["tau_daca"])

    summary = {
        "total_cross_family_fits": len(all_cross_taus),
        "all_cross_taus": sorted(all_cross_taus),
        "median_cross_tau": (statistics.median(all_cross_taus) if all_cross_taus else None),
        "range_cross_tau": ([min(all_cross_taus), max(all_cross_taus)] if all_cross_taus else None),
        "any_target_corroborates_cross": any(v["corroborates"] for v in f7_all.values()),
        "any_target_corroborates_all_ref": any(v["corroborates"] for v in f7_all_ref.values()),
        "scale_caveat_removed": True,
        "prototype_comparison": {
            "r0_prototype": "15 items, 2 valid fits {0.284, 0.870}, NON-CORROBORATING",
            "note": "R0 used the actual closed pair (stage9-claude-gpt-medium); "
                    "this study uses POST legs as closed analogs at 120-cell scale",
        },
    }
    report["summary"] = summary

    out_path = REPO / "runs" / "tau_daca_fullscale" / "tau_daca_fullscale.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nWrote {out_path}")

    # ── Verdict ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"F7 range: [{DACA_RANGE[0]:.3f}, {DACA_RANGE[1]:.3f}]")
    print(f"Total cross-family valid fits: {len(all_cross_taus)}")
    if all_cross_taus:
        print(f"Cross-family τ range: [{min(all_cross_taus):.3f}, {max(all_cross_taus):.3f}]")
        print(f"Cross-family τ median: {statistics.median(all_cross_taus):.3f}")
    for tf in PANEL:
        v = f7_all[tf]
        print(f"  {tf:12s}: F7={'PASS' if v['corroborates'] else 'FAIL'} "
              f"(n_valid={v['n_valid']}, taus={[round(t, 3) for t in v['valid_taus']]})")
    if summary["any_target_corroborates_cross"]:
        print("\n→ At least one target CORROBORATES under F7 (cross-family)")
    else:
        print("\n→ NO target corroborates under F7 — τ_DACA NON-CORROBORATING at full scale")
        print("  The R0 prototype's scale caveat is REMOVED: the failure is CONFIRMED at 120 cells.")


if __name__ == "__main__":
    main()
