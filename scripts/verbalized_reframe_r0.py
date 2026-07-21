#!/usr/bin/env python3
"""R0 free analyses for the verbalized-first reframe (user mandate 2026-07-21).

Four $0 analyses on the on-disk Study B legs + the one existing on-pair closed run:

  A. tau_v-vs-entropy reconciliation — why the W1-aligned temperature detects
     post-training sharpening in every family while the scalar mean-entropy
     diagnostic flips sign in two (llama31, maverick).
  B. Leave-one-family-out transfer check on median(tau_v) — the free analog of
     the arm-1 adoption question.
  C. tau_DACA-verbalized PROTOTYPE — GT-free alignment of the closed pair to the
     verbalized base references. Closed side = stage9-claude-gpt-medium
     (15 on-pair contract items) — SMOKE-GRADE n, prototype-labeled, never
     adoptable; the real version rides the E6 on-pair sweep.
  D. Verbalized dispersion-pool re-derivation — pool diagnostics over all 120
     cells from the verbalized BASE legs, plus a prototype T_raw fit on the
     15-item closed run. No shrinkage is applied: the logit-era anchor/lambda
     were frozen against the logit band and must be re-derived in-channel (an
     R1 decision), so only the raw fit is reported.

Discipline (inherited): epsilon = 0.005 floors before ANY inverse-softmax op;
T_BOUNDS = (0.25, 20.0) passed explicitly; boundary = peg, never a fit;
cross-mode legs (gemma*_api) carry channel labels everywhere; gemma26's base is
capability-excluded (resolution 0.0134) and never enters a gate-passing set.

Run:  PYTHONPATH=src python scripts/verbalized_reframe_r0.py [--out runs/verbalized_r0]
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

from judex_calibration import aireg  # noqa: E402
from judex_calibration.elicit_verbalized import compliance_view, floor_and_renormalize, EPSILON  # noqa: E402
from judex_calibration import study_a  # noqa: E402
from judex_calibration.study_a import T_BOUNDS, GRID, saturated  # noqa: E402
from judex_calibration.study_b import kendall_tau_b  # noqa: E402

from judex.core.distributions import ComplianceDistribution  # noqa: E402
from judex.core.metrics import wasserstein_1, total_variation_distance  # noqa: E402
from judex.calibration import apply_temperature, fit_temperature, fit_dispersion_temperature, realized_dispersion  # noqa: E402
from judex.experiments import murphy_decomposition, replicates_by_level_from_cross_family  # noqa: E402

LABELS = ("very_low", "low", "moderate", "high", "very_high")

# Leg inventory: family -> (run dir, channel label). Channel labels are load-bearing:
# cross_mode = chat-API re-collection (16-bit pinned), in_mode = vLLM /v1/completions.
LEGS = {
    "qwen":        ("runs/study_b_qwen", "in_mode"),
    "gemma31":     ("runs/study_b_gemma31", "in_mode"),
    "gemma31_api": ("runs/study_b_gemma31_api", "cross_mode"),
    "gemma26":     ("runs/study_b_gemma26", "in_mode"),
    "gemma26_api": ("runs/study_b_gemma26_api", "cross_mode"),
    "llama31":     ("runs/study_b_llama31", "in_mode"),
    "glm":         ("runs/study_b_glm", "in_mode"),
    "maverick":    ("runs/study_b_maverick", "in_mode"),
}
# The arm-1 adoption panel (USER DECISION 2026-07-21): {qwen, gemma31, glm, maverick}.
# gemma31's measurement is the chat-template collection (study_b_gemma31_api) — the
# vLLM greedy-continuation collapse is a serving pathology, not a property of the
# weights, so the chat-template leg IS the family's tau_v; the serving transport is
# provenance (leg meta sidecars), not a headline caveat. llama31 is EXCLUDED from
# every adoption-relevant set by the same decision (accumulated instrument anomalies
# — sole wrong-sign B-Q4 association with a pegged T_c — carry caveat burden
# disproportionate to its contribution). The exclusion is numerically inert: its
# tau_v (1.025) duplicates gemma31's grid point, so the multiset {1.025, 1.025,
# 1.281, 1.380} — and with it the median, band, and cluster ratio — is unchanged.
# Decided before any on-pair data exists (pre-hoc for adoption).
PANEL = ["qwen", "gemma31_api", "glm", "maverick"]
PANEL_SENSITIVITY_PLUS_LLAMA31 = PANEL + ["llama31"]  # sensitivity read only
# gemma26_api passes the contract gates but its BASE fails the capability floor
# (resolution 0.0134) => weak-reference, excluded from every adoption set.
CAPABILITY_EXCLUDED = ["gemma26", "gemma26_api"]

CLOSED_RUN = "stage9-claude-gpt-medium"  # 3 graded docs / 15 items — PROTOTYPE ONLY


def load_leg(run_dir: Path):
    pre = json.loads((run_dir / "pre_verbalized.json").read_text())
    post = json.loads((run_dir / "post_verbalized.json").read_text())
    return pre, post


def floored_view(recs) -> dict:
    return {k: floor_and_renormalize(v, EPSILON) for k, v in compliance_view(recs).items()}


def dist(probs, cells_by_label, label) -> ComplianceDistribution:
    return ComplianceDistribution.from_values(list(probs), cells_by_label[label].gt_labels)


def norm_entropy(probs) -> float:
    return ComplianceDistribution.from_values(list(probs), LABELS).normalized_entropy()


def temper_map(preds: dict, T: float, cells_by_label) -> dict:
    out = {}
    for label, probs in preds.items():
        d = apply_temperature(dist(probs, cells_by_label, label), T)
        out[label] = list(d.probabilities)
    return out


# ---------------------------------------------------------------- A. entropy
def entropy_reconciliation(views, cells, cells_by_label) -> dict:
    """Per family: tau_v vs the scalar entropy summary, decomposed.

    Mechanism probes:
      * subset composition — parsed sets differ pre vs post; means are reported on
        each leg's own parsed set AND on the paired overlap;
      * tau_H — the temperature equating mean post entropy to mean pre entropy on
        the overlap (a pure-spread analog of tau_v; sign disagreement between
        tau_H and tau_v is the reconciliation target);
      * dH strata — tau_v refit on the post-sharper (dH>0) and post-flatter
        (dH<=0) cell strata separately;
      * mode agreement — entropy and W1-gain-at-tau_v means split by whether
        post and pre argmax agree (W1 is location-sensitive, entropy is not).
    """
    out = {}
    for fam, (pre_fl, post_fl, channel) in views.items():
        overlap = sorted(pre_fl.keys() & post_fl.keys())
        if not overlap:
            continue
        tau_v = study_a.fit_tau_oc(post_fl, pre_fl, cells)
        h_pre_own = [norm_entropy(p) for p in pre_fl.values()]
        h_post_own = [norm_entropy(p) for p in post_fl.values()]
        h_pre = {k: norm_entropy(pre_fl[k]) for k in overlap}
        h_post = {k: norm_entropy(post_fl[k]) for k in overlap}

        # tau_H: grid T equating mean tempered-post entropy to mean pre entropy (overlap)
        target = statistics.mean(h_pre.values())

        def mean_h_at(T):
            return statistics.mean(
                norm_entropy(apply_temperature(dist(post_fl[k], cells_by_label, k), T).probabilities)
                for k in overlap)

        tau_H = min(GRID, key=lambda T: abs(mean_h_at(T) - target))

        # per-cell quantities on the overlap
        d_h = {k: h_pre[k] - h_post[k] for k in overlap}  # >0 = post sharper than pre
        w1_raw = {k: wasserstein_1(dist(post_fl[k], cells_by_label, k), dist(pre_fl[k], cells_by_label, k))
                  for k in overlap}
        w1_at_tau = {k: wasserstein_1(apply_temperature(dist(post_fl[k], cells_by_label, k), tau_v),
                                      dist(pre_fl[k], cells_by_label, k)) for k in overlap}
        gain = {k: w1_raw[k] - w1_at_tau[k] for k in overlap}  # >0 = flattening helped this cell
        mode_agree = {k: dist(post_fl[k], cells_by_label, k).argmax_index()
                         == dist(pre_fl[k], cells_by_label, k).argmax_index() for k in overlap}

        def stratum(keys):
            keys = list(keys)
            if len(keys) < 2:
                return {"n": len(keys)}
            t = study_a.fit_tau_oc({k: post_fl[k] for k in keys}, {k: pre_fl[k] for k in keys}, cells)
            return {"n": len(keys), "tau_v": t, "tau_v_saturated": saturated(t),
                    "mean_w1_gain_at_family_tau": statistics.mean(gain[k] for k in keys),
                    "mean_dH": statistics.mean(d_h[k] for k in keys)}

        sharper = [k for k in overlap if d_h[k] > 0]
        flatter = [k for k in overlap if d_h[k] <= 0]
        agree = [k for k in overlap if mode_agree[k]]
        disagree = [k for k in overlap if not mode_agree[k]]

        out[fam] = {
            "channel": channel,
            "n_pre_parsed": len(pre_fl), "n_post_parsed": len(post_fl), "n_overlap": len(overlap),
            "tau_v": tau_v, "tau_v_saturated": saturated(tau_v),
            "tau_H_mean_entropy_match": tau_H, "tau_H_saturated": saturated(tau_H),
            "tau_H_below_1": tau_H < 1.0,
            "mean_norm_entropy": {
                "pre_own_parsed_set": statistics.mean(h_pre_own),
                "post_own_parsed_set": statistics.mean(h_post_own),
                "pre_overlap": statistics.mean(h_pre.values()),
                "post_overlap": statistics.mean(h_post.values()),
            },
            "median_norm_entropy": {
                "pre_overlap": statistics.median(h_pre.values()),
                "post_overlap": statistics.median(h_post.values()),
            },
            "share_post_sharper": len(sharper) / len(overlap),
            "strata_by_dH": {"post_sharper": stratum(sharper), "post_flatter": stratum(flatter)},
            "strata_by_mode_agreement": {
                "agree": {**stratum(agree),
                          "mean_H_post": statistics.mean(h_post[k] for k in agree) if agree else None},
                "disagree": {**stratum(disagree),
                             "mean_H_post": statistics.mean(h_post[k] for k in disagree) if disagree else None},
            },
            "kendall_dH_vs_w1_gain": kendall_tau_b([d_h[k] for k in overlap], [gain[k] for k in overlap]),
        }
    return out


# ------------------------------------------------------------------- B. LOFO
def lofo_transfer(views, cells, cells_by_label, gate_sets) -> dict:
    """Leave-one-family-out transfer of median(tau_v) — the free adoption analog.

    Two reads per held-out family:
      * in-channel (GT-free, matches the tau_v objective): does tempering the
        held-out POST leg by the others' median recover the W1(post->pre)
        alignment its own tau_v achieves? recovered_fraction = transfer gain /
        oracle gain.
      * GT-side (labeled supervised read, reported not gated): mean RPS/W1 and
        Murphy reliability against AIReg GT at T=1 / transferred T / own tau_v.
    Median convention is a live protocol-r3 decision: both the interpolated
    median and study_a's upper-median (sorted[n//2]) are reported.
    """
    tau = {}
    for fam, (pre_fl, post_fl, _) in views.items():
        tau[fam] = study_a.fit_tau_oc(post_fl, pre_fl, cells)

    def gt_scores(preds):
        by_label = {c.item_label: c for c in cells}
        items = [study_a._metric_item(lb, dist(p, by_label, lb),
                                      ComplianceDistribution.from_values(by_label[lb].gt_probs,
                                                                         by_label[lb].gt_labels))
                 for lb, p in preds.items() if lb in by_label]
        m = murphy_decomposition(items)
        return {"mean_rps": statistics.mean(i.rps for i in items),
                "mean_w1": statistics.mean(i.w1 for i in items),
                "reliability": m["reliability"], "resolution": m["resolution"]}

    out = {}
    for set_name, fams in gate_sets.items():
        rows = {}
        for f in fams:
            others = [tau[g] for g in fams if g != f]
            med = statistics.median(others)
            pre_fl, post_fl, channel = views[f]
            overlap = sorted(pre_fl.keys() & post_fl.keys())
            pre_o = {k: pre_fl[k] for k in overlap}
            post_o = {k: post_fl[k] for k in overlap}

            def mean_w1_to_pre(preds):
                return statistics.mean(
                    wasserstein_1(dist(preds[k], cells_by_label, k), dist(pre_o[k], cells_by_label, k))
                    for k in overlap)

            raw = mean_w1_to_pre(post_o)
            at_own = mean_w1_to_pre(temper_map(post_o, tau[f], cells_by_label))
            at_transfer = mean_w1_to_pre(temper_map(post_o, med, cells_by_label))
            oracle_gain = raw - at_own
            rows[f] = {
                "channel": channel, "tau_v_own": tau[f], "tau_transfer_lofo_median": med,
                "log_ratio_transfer_vs_own": math.log(med / tau[f]),
                "in_channel_w1_to_pre": {"raw": raw, "at_own_tau": at_own, "at_transfer": at_transfer,
                                         "recovered_fraction": ((raw - at_transfer) / oracle_gain)
                                         if oracle_gain > 1e-12 else None},
                "gt_side_supervised_read": {
                    "T1": gt_scores(post_o),
                    "at_transfer": gt_scores(temper_map(post_o, med, cells_by_label)),
                    "at_own_tau": gt_scores(temper_map(post_o, tau[f], cells_by_label)),
                },
            }
        taus_sorted = sorted(tau[f] for f in fams)
        out[set_name] = {
            "families": fams,
            "tau_v": {f: tau[f] for f in fams},
            "median_interpolated": statistics.median(taus_sorted),
            "median_upper_study_a_convention": taus_sorted[len(taus_sorted) // 2],
            "band": [min(taus_sorted), max(taus_sorted)],
            "cluster_ratio": taus_sorted[-1] / taus_sorted[0],
            "cluster_rule_ok": taus_sorted[-1] / taus_sorted[0] <= study_a.CLUSTER_RULE_MAX_RATIO,
            "lofo": rows,
            "max_abs_log_ratio_transfer_vs_own": max(abs(r["log_ratio_transfer_vs_own"])
                                                     for r in rows.values()),
        }
    return out


# ------------------------------------------------------- closed-run loading
def load_closed_predictions(eval_runs: Path) -> dict:
    """Reconciled final distributions of the on-pair prototype run, canonical order."""
    mr = json.loads((eval_runs / CLOSED_RUN / "metrics_report.json").read_text())
    preds = {}
    for it in mr["items"]:
        p = it["prediction"]
        by = dict(zip(p["labels"], p["probabilities"]))
        preds[it["item_label"]] = [float(by[full]) for full in FULL_TO_SHORT]
    return preds


FULL_TO_SHORT = {
    "Very low probability of compliance": "very_low",
    "Low probability of compliance": "low",
    "Moderate probability of compliance": "moderate",
    "High probability of compliance": "high",
    "Very high probability of compliance": "very_high",
}


def load_closed_replicates(eval_runs: Path, cells) -> dict:
    """{item_label: [cfp1 replicate dists]} from the prototype run's per-doc payloads."""
    by_doc_article = {(c.document_id, str(c.article)): c for c in cells}
    reps: dict = {}
    run_dir = eval_runs / CLOSED_RUN
    for doc_dir in sorted((run_dir / "documents").iterdir()):
        payload_path = doc_dir / "cross_family_evaluation.json"
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text())
        per_level = replicates_by_level_from_cross_family(payload, source="cross_family_phase1")
        phase4 = payload.get("phase4") or {}
        for level in phase4.get("reconciled_levels", []):
            if level.get("hierarchy_level") != "dimension":
                continue
            article = level["level_id"].split(":")[0].replace("article_", "")
            cell = by_doc_article.get((payload["document_id"], article))
            if cell is None:
                continue
            rl = []
            for r in per_level.get(level["level_id"], []):
                by = dict(zip(r.labels, r.probabilities))
                rl.append(ComplianceDistribution.from_values([by[full] for full in FULL_TO_SHORT], LABELS))
            reps[cell.item_label] = rl
    return reps


# --------------------------------------------------------------- C. tau_DACA
def daca_prototype(views, closed_preds, cells, tau_transfer: float) -> dict:
    refs = {f: views[f][0] for f in PANEL}                   # floored BASE legs
    proto_pairs = [(dist(p, {c.item_label: c for c in cells}, lb),
                    ComplianceDistribution.from_values(
                        next(c for c in cells if c.item_label == lb).gt_probs,
                        next(c for c in cells if c.item_label == lb).gt_labels))
                   for lb, p in closed_preds.items()
                   if any(c.item_label == lb for c in cells)]
    T_sup_proto = fit_temperature(proto_pairs, bounds=T_BOUNDS).temperature
    tri = study_a.daca_triangulation(closed_preds, refs, cells,
                                     tau_transfer=tau_transfer, T_supervised=T_sup_proto)
    tri["prototype"] = {
        "closed_run": CLOSED_RUN, "n_closed_items": len(closed_preds),
        "smoke_grade": True,
        "note": ("PROTOTYPE: closed side is the 15-item on-pair trial run — smoke discipline "
                 "(<120 cells) applies; nothing here is adoptable. The real tau_DACA-verbalized "
                 "rides the E6 on-pair sweep."),
        "T_supervised_prototype_15_items": T_sup_proto,
        "T_supervised_prototype_saturated": saturated(T_sup_proto),
        "reference_serving_provenance": {f: views[f][2] for f in PANEL},
    }
    return tri


# ----------------------------------------------------------------- D. pool
def dispersion_pool(views, cells, cells_by_label, closed_preds, closed_reps) -> dict:
    """Verbalized dispersion-pool re-derivation.

    Pool diagnostics run on ALL 120 cells (bases only, $0). The T_raw fit needs a
    closed-run target, so it runs on the 15-item prototype — smoke-grade, labeled.
    No shrinkage: the logit-era anchor (2.794) and lambda (0.35) were frozen
    against the logit band [1.60, 4.88]; carrying them into this channel would be
    an unfrozen retune. Raw fits only; the anchor/lambda re-derivation is R1.
    """
    pools = {"P4_panel_bases": PANEL, "P5_sensitivity_plus_llama31": PANEL_SENSITIVITY_PLUS_LLAMA31}
    out = {}
    for name, fams in pools.items():
        bases = {f: views[f][0] for f in fams}
        common = sorted(set.intersection(*(set(b.keys()) for b in bases.values())))

        def pool_entropy(members, labels):
            return statistics.mean(
                realized_dispersion([dist(bases[f][lb], cells_by_label, lb) for f in members]).mixture_entropy
                for lb in labels)

        h_all = pool_entropy(fams, common)
        lobo = {f: pool_entropy([g for g in fams if g != f], common) for f in fams}
        # decorrelation check: mean pairwise TVD between member base distributions
        pair_tvd = {}
        for i, f in enumerate(fams):
            for g in fams[i + 1:]:
                pair_tvd[f + "|" + g] = statistics.mean(
                    total_variation_distance(dist(bases[f][lb], cells_by_label, lb),
                                             dist(bases[g][lb], cells_by_label, lb))
                    for lb in common)

        # prototype closed-run fits (15 items)
        closed_labels = [lb for lb in closed_preds if lb in cells_by_label and
                         all(lb in bases[f] for f in fams)]
        finals = {lb: dist(closed_preds[lb], cells_by_label, lb) for lb in closed_labels}
        pool_bases_only = {lb: [dist(bases[f][lb], cells_by_label, lb) for f in fams]
                           for lb in closed_labels}
        pool_aug = {lb: list(closed_reps.get(lb, [])) + pool_bases_only[lb] for lb in closed_labels}
        fit_b = fit_dispersion_temperature([(finals[lb], pool_bases_only[lb]) for lb in closed_labels],
                                           bounds=T_BOUNDS)
        fit_a = fit_dispersion_temperature([(finals[lb], pool_aug[lb]) for lb in closed_labels],
                                           bounds=T_BOUNDS)
        out[name] = {
            "members": fams,
            "serving_provenance": {f: views[f][2] for f in fams},
            "n_common_cells": len(common),
            "pool_mixture_entropy_mean_nats": h_all,
            "pool_entropy_ok_vs_logit_era_floor_1.00": h_all >= 1.00,
            "lobo_pool_entropy": lobo,
            "lobo_entropy_min": min(lobo.values()),
            "mean_pairwise_tvd": pair_tvd,
            "min_pairwise_tvd": min(pair_tvd.values()),
            "prototype_closed_fit": {
                "closed_run": CLOSED_RUN, "n_items": len(closed_labels), "smoke_grade": True,
                "T_raw_bases_only": fit_b.temperature,
                "T_raw_bases_only_saturated": saturated(fit_b.temperature),
                "T_raw_cfp1_plus_bases": fit_a.temperature,
                "T_raw_cfp1_plus_bases_saturated": saturated(fit_a.temperature),
                "realized_entropy_mean_bases_only": fit_b.realized_entropy_mean,
                "claimed_entropy_at_unit": fit_b.claimed_entropy_at_unit,
                "note": ("raw fits only — the logit-era shrinkage (anchor 2.794, lambda 0.35) is "
                         "NOT applied; re-deriving anchor/lambda against the verbalized band is an "
                         "R1 decision"),
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "verbalized_r0")
    args = ap.parse_args()

    cells = aireg.load_cells()
    cells_by_label = {c.item_label: c for c in cells}
    eval_runs = REPO.parent / "judex-evaluator" / "runs"

    views = {}
    for fam, (rel, channel) in LEGS.items():
        pre, post = load_leg(REPO / rel)
        views[fam] = (floored_view(pre), floored_view(post), channel)

    gate_sets = {"panel_n4": PANEL, "sensitivity_plus_llama31_n5": PANEL_SENSITIVITY_PLUS_LLAMA31}
    lofo = lofo_transfer(views, cells, cells_by_label, gate_sets)
    tau_transfer = lofo["panel_n4"]["median_interpolated"]

    closed_preds = load_closed_predictions(eval_runs)
    closed_reps = load_closed_replicates(eval_runs, cells)

    report = {
        "mandate": "verbalized-first reframe R0 (2026-07-21); all analyses $0, prototype-labeled where closed-side n=15",
        "epsilon": EPSILON, "T_bounds": list(T_BOUNDS),
        "capability_excluded_families": CAPABILITY_EXCLUDED,
        "A_entropy_reconciliation": entropy_reconciliation(views, cells, cells_by_label),
        "B_lofo_transfer": lofo,
        "C_tau_daca_verbalized_prototype": daca_prototype(views, closed_preds, cells, tau_transfer),
        "D_dispersion_pool_verbalized": dispersion_pool(views, cells, cells_by_label,
                                                        closed_preds, closed_reps),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "r0_analyses.json"
    out_path.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out_path}")

    # compact console summary
    A = report["A_entropy_reconciliation"]
    print("\n== A. entropy reconciliation ==")
    for fam, r in A.items():
        m = r["mean_norm_entropy"]
        print(f"  {fam:12s} [{r['channel']:10s}] tau_v={r['tau_v']:.3f} tau_H={r['tau_H_mean_entropy_match']:.3f} "
              f"H pre->post (overlap) {m['pre_overlap']:.3f}->{m['post_overlap']:.3f} "
              f"share_post_sharper={r['share_post_sharper']:.2f} "
              f"tau_v|sharper={r['strata_by_dH']['post_sharper'].get('tau_v', float('nan')):.3f} "
              f"tau_v|flatter={r['strata_by_dH']['post_flatter'].get('tau_v', float('nan')):.3f}")
    print("\n== B. LOFO transfer ==")
    for name, s in report["B_lofo_transfer"].items():
        print(f"  {name}: median_interp={s['median_interpolated']:.3f} "
              f"median_upper={s['median_upper_study_a_convention']:.3f} "
              f"ratio={s['cluster_ratio']:.3f} max|log(transfer/own)|={s['max_abs_log_ratio_transfer_vs_own']:.3f}")
        for f, r in s["lofo"].items():
            ic = r["in_channel_w1_to_pre"]
            rec = ic["recovered_fraction"]
            print(f"    {f:12s} own={r['tau_v_own']:.3f} transfer={r['tau_transfer_lofo_median']:.3f} "
                  f"W1 raw/own/transfer {ic['raw']:.4f}/{ic['at_own_tau']:.4f}/{ic['at_transfer']:.4f} "
                  f"recovered={rec if rec is None else round(rec, 3)}")
    C = report["C_tau_daca_verbalized_prototype"]
    print("\n== C. tau_DACA-verbalized PROTOTYPE (15 items, smoke-grade) ==")
    print(f"  summary: {json.dumps(C['summary'], default=float)}")
    for f, r in C["references"].items():
        print(f"    ref {f:12s} tau_daca={r['tau_daca']:.3f} sat={r['tau_daca_saturated']} "
              f"agree={r['agreement_rate']:.2f} ref_acc(gate)={r['reference_argmax_acc_gate_only']:.2f}")
    D = report["D_dispersion_pool_verbalized"]
    print("\n== D. verbalized dispersion pool ==")
    for name, r in D.items():
        pf = r["prototype_closed_fit"]
        print(f"  {name}: H_pool={r['pool_mixture_entropy_mean_nats']:.3f} "
              f"(floor 1.00 ok={r['pool_entropy_ok_vs_logit_era_floor_1.00']}) "
              f"LOBO_min={r['lobo_entropy_min']:.3f} minTVD={r['min_pairwise_tvd']:.3f} "
              f"T_raw bases/aug {pf['T_raw_bases_only']:.3f}/{pf['T_raw_cfp1_plus_bases']:.3f} "
              f"(n={pf['n_items']}, smoke)")


if __name__ == "__main__":
    main()
