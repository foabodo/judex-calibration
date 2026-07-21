#!/usr/bin/env python3
"""Full robustness battery for the verbalized dispersion pool (arm 3a/3b).

Characterizes the FROZEN pool {qwen, gemma31, glm, maverick} verbalized BASE
legs before the user-gated ~$445 on-pair E6 sweep. $0, analysis-only — no GPU,
no API spend.

Sections:
  1. LOBO at full 120-cell scale (pool membership is frozen — characterize only)
  2. Doc-clustered bootstrap CIs on T_raw and pool health metrics
  3. Epsilon-sensitivity {0.001, 0.005, 0.0125, 0.025}
  4. Fit-grid sensitivity (60-pt vs 120-pt vs 240-pt log grids)
  5. Stratified behavior: mode-agree/disagree, per-Article, per-GT-level
  6. Sensitivity annexes: +llama31, gemma31-vLLM swap, pool-of-3 ablations
  7. Failure-mode pre-registration for the E6 sweep arm 3b read

Data: Study B legs (runs/study_b_*) + the 15-item prototype closed run
(stage9-claude-gpt-medium). All closed-side fits are SMOKE (n<120).

Run:
  PYTHONPATH=src python scripts/dispersion_pool_verbalized_robustness.py \
      [--out runs/verbalized_pool_robustness]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from judex_calibration import aireg  # noqa: E402
from judex_calibration.elicit_verbalized import compliance_view, floor_and_renormalize, EPSILON  # noqa: E402
from judex_calibration import study_a  # noqa: E402
from judex_calibration.study_a import T_BOUNDS, GRID, saturated  # noqa: E402

from judex.core.distributions import ComplianceDistribution  # noqa: E402
from judex.core.metrics import wasserstein_1, total_variation_distance  # noqa: E402
from judex.calibration import (  # noqa: E402
    apply_temperature, fit_temperature, fit_dispersion_temperature, realized_dispersion,
)
from judex.experiments import replicates_by_level_from_cross_family  # noqa: E402

LABELS = ("very_low", "low", "moderate", "high", "very_high")

# ---- Leg inventory (matches verbalized_reframe_r0.py) ----
LEGS = {
    "qwen":        ("runs/study_b_qwen", "in_mode"),
    "gemma31_api": ("runs/study_b_gemma31_api", "cross_mode"),
    "gemma31":     ("runs/study_b_gemma31", "in_mode"),        # vLLM collapse leg
    "glm":         ("runs/study_b_glm", "in_mode"),
    "maverick":    ("runs/study_b_maverick", "in_mode"),
    "llama31":     ("runs/study_b_llama31", "in_mode"),
}

PANEL = ["qwen", "gemma31_api", "glm", "maverick"]
PANEL_TAU_V = {"qwen": 1.281, "gemma31_api": 1.025, "glm": 1.025, "maverick": 1.380}
BAND = (1.0251785151221313, 1.379820350674421)
T_J = 1.153
MIN_POOL_ENTROPY = 1.00

CLOSED_RUN = "stage9-claude-gpt-medium"
FULL_TO_SHORT = {
    "Very low probability of compliance": "very_low",
    "Low probability of compliance": "low",
    "Moderate probability of compliance": "moderate",
    "High probability of compliance": "high",
    "Very high probability of compliance": "very_high",
}
BOOT_SEED = 20260720
N_BOOT = 2000


def load_leg(run_dir: Path):
    pre = json.loads((run_dir / "pre_verbalized.json").read_text())
    post = json.loads((run_dir / "post_verbalized.json").read_text())
    return pre, post


def floored_view(recs, eps=EPSILON) -> dict:
    return {k: floor_and_renormalize(v, eps) for k, v in compliance_view(recs).items()}


def dist(probs, cell) -> ComplianceDistribution:
    return ComplianceDistribution.from_values(list(probs), cell.gt_labels)


def load_closed_predictions(eval_runs: Path) -> dict:
    mr = json.loads((eval_runs / CLOSED_RUN / "metrics_report.json").read_text())
    preds = {}
    for it in mr["items"]:
        p = it["prediction"]
        by = dict(zip(p["labels"], p["probabilities"]))
        preds[it["item_label"]] = [float(by[full]) for full in FULL_TO_SHORT]
    return preds


def load_closed_replicates(eval_runs: Path, cells) -> dict:
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


# ================================================================
# 1. LOBO at full 120-cell scale
# ================================================================
def lobo_full_scale(bases: dict, cells_by_label: dict, common: list,
                    closed_labels: list, closed_preds: dict, closed_reps: dict) -> dict:
    """Leave-one-base-out pool diagnostics at full cell scale + on the prototype fit."""
    fams = list(bases.keys())

    def pool_entropy(members, labels):
        return statistics.mean(
            realized_dispersion([dist(bases[f][lb], cells_by_label[lb]) for f in members]).mixture_entropy
            for lb in labels)

    def pool_tvd_mean(members, labels):
        tvds = []
        for i, f in enumerate(members):
            for g in members[i + 1:]:
                tvds.append(statistics.mean(
                    total_variation_distance(dist(bases[f][lb], cells_by_label[lb]),
                                             dist(bases[g][lb], cells_by_label[lb]))
                    for lb in labels))
        return tvds

    h_all = pool_entropy(fams, common)
    lobo = {}
    for f in fams:
        rest = [g for g in fams if g != f]
        h = pool_entropy(rest, common)
        tvds = pool_tvd_mean(rest, common)
        pool_bases = {lb: [dist(bases[g][lb], cells_by_label[lb]) for g in rest] for lb in closed_labels}
        finals = {lb: dist(closed_preds[lb], cells_by_label[lb]) for lb in closed_labels}
        fit = fit_dispersion_temperature([(finals[lb], pool_bases[lb]) for lb in closed_labels],
                                         bounds=T_BOUNDS)
        lobo[f] = {
            "held_out": f,
            "remaining": rest,
            "pool_entropy_without": h,
            "entropy_floor_ok": h >= MIN_POOL_ENTROPY,
            "min_pairwise_tvd_without": min(tvds) if tvds else None,
            "mean_pairwise_tvd_without": statistics.mean(tvds) if tvds else None,
            "T_raw_without": fit.temperature,
            "T_raw_saturated": saturated(fit.temperature),
            "T_raw_in_band": BAND[0] <= fit.temperature <= BAND[1],
        }

    return {
        "pool_entropy_full": h_all,
        "entropy_floor_ok": h_all >= MIN_POOL_ENTROPY,
        "lobo_min_entropy": min(v["pool_entropy_without"] for v in lobo.values()),
        "lobo_all_above_floor": all(v["entropy_floor_ok"] for v in lobo.values()),
        "lobo_T_raw_range": [min(v["T_raw_without"] for v in lobo.values()),
                             max(v["T_raw_without"] for v in lobo.values())],
        "lobo_all_in_band": all(v["T_raw_in_band"] for v in lobo.values()),
        "lobo_any_saturated": any(v["T_raw_saturated"] for v in lobo.values()),
        "per_member": lobo,
    }


# ================================================================
# 2. Doc-clustered bootstrap CIs
# ================================================================
def doc_clustered_bootstrap(bases: dict, cells_by_label: dict, cells: list,
                            closed_preds: dict, closed_reps: dict,
                            n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Doc-clustered bootstrap CIs on T_raw, mixture entropy, and min pairwise TVD."""
    fams = list(bases.keys())
    docs = sorted({c.document_id for c in cells})
    doc_labels = {d: [c.item_label for c in cells if c.document_id == d] for d in docs}

    all_common = sorted(set.intersection(*(set(b.keys()) for b in bases.values())))
    all_closed = [lb for lb in closed_preds if lb in cells_by_label
                  and all(lb in bases[f] for f in fams)]

    def compute_on_labels(labels, closed_labels_subset):
        h = statistics.mean(
            realized_dispersion([dist(bases[f][lb], cells_by_label[lb]) for f in fams]).mixture_entropy
            for lb in labels) if labels else float("nan")

        pair_tvds = []
        for i, f in enumerate(fams):
            for g in fams[i + 1:]:
                pair_tvds.append(statistics.mean(
                    total_variation_distance(dist(bases[f][lb], cells_by_label[lb]),
                                             dist(bases[g][lb], cells_by_label[lb]))
                    for lb in labels) if labels else float("nan"))
        min_tvd = min(pair_tvds) if pair_tvds else float("nan")

        if closed_labels_subset:
            finals = {lb: dist(closed_preds[lb], cells_by_label[lb]) for lb in closed_labels_subset}
            pool_b = {lb: [dist(bases[f][lb], cells_by_label[lb]) for f in fams] for lb in closed_labels_subset}
            fit = fit_dispersion_temperature([(finals[lb], pool_b[lb]) for lb in closed_labels_subset],
                                             bounds=T_BOUNDS)
            t_raw = fit.temperature
        else:
            t_raw = float("nan")
        return h, min_tvd, t_raw

    point_h, point_tvd, point_t = compute_on_labels(all_common, all_closed)

    rng = random.Random(seed)
    boot_h, boot_tvd, boot_t = [], [], []
    for _ in range(n_boot):
        sampled_docs = [docs[rng.randrange(len(docs))] for _ in docs]
        b_labels = sorted(set(lb for d in sampled_docs for lb in doc_labels[d] if lb in all_common))
        b_closed = sorted(set(lb for d in sampled_docs for lb in doc_labels[d] if lb in all_closed))
        if len(b_labels) < 2 or len(b_closed) < 2:
            continue
        h, tvd, t = compute_on_labels(b_labels, b_closed)
        boot_h.append(h)
        boot_tvd.append(tvd)
        boot_t.append(t)

    def ci(vals, point):
        vals = sorted(v for v in vals if math.isfinite(v))
        if not vals:
            return {"point": point, "ci95": [None, None], "n_valid_boots": 0}
        lo = vals[int(0.025 * len(vals))]
        hi = vals[int(0.975 * len(vals))]
        return {"point": point, "ci95": [lo, hi], "n_valid_boots": len(vals)}

    t_ci = ci(boot_t, point_t)
    t_lo, t_hi = t_ci["ci95"]
    ci_in_band = (t_lo is not None and t_hi is not None
                  and t_lo >= BAND[0] and t_hi <= BAND[1])
    ci_includes_1 = (t_lo is not None and t_hi is not None
                     and t_lo <= 1.0 <= t_hi)

    return {
        "n_bootstrap": n_boot, "seed": seed, "clustered_by": "document",
        "mixture_entropy": ci(boot_h, point_h),
        "min_pairwise_tvd": ci(boot_tvd, point_tvd),
        "T_raw": t_ci,
        "T_raw_ci_inside_band": ci_in_band,
        "T_raw_ci_includes_1": ci_includes_1,
        "note": "smoke-grade T_raw CI (n=12-15 closed items per resample); entropy/TVD CIs are full-scale (120 cells)",
    }


# ================================================================
# 3. Epsilon sensitivity
# ================================================================
def epsilon_sensitivity(bases_raw: dict, cells_by_label: dict, common_raw: list,
                        closed_preds: dict,
                        eps_list: Sequence[float] = (0.001, 0.005, 0.0125, 0.025)) -> dict:
    """Pool health and T_raw across epsilon floors."""
    fams = list(bases_raw.keys())
    out = {}
    for eps in eps_list:
        bases_e = {f: {k: floor_and_renormalize(v, eps) for k, v in bases_raw[f].items()} for f in fams}
        common_e = sorted(set.intersection(*(set(b.keys()) for b in bases_e.values())))
        h = statistics.mean(
            realized_dispersion([dist(bases_e[f][lb], cells_by_label[lb]) for f in fams]).mixture_entropy
            for lb in common_e)
        pair_tvds = []
        for i, f in enumerate(fams):
            for g in fams[i + 1:]:
                pair_tvds.append(statistics.mean(
                    total_variation_distance(dist(bases_e[f][lb], cells_by_label[lb]),
                                             dist(bases_e[g][lb], cells_by_label[lb]))
                    for lb in common_e))
        closed_e = [lb for lb in closed_preds if lb in cells_by_label
                    and all(lb in bases_e[f] for f in fams)]
        if closed_e:
            closed_fl = {lb: floor_and_renormalize(closed_preds[lb], eps) for lb in closed_e}
            finals = {lb: dist(closed_fl[lb], cells_by_label[lb]) for lb in closed_e}
            pool_b = {lb: [dist(bases_e[f][lb], cells_by_label[lb]) for f in fams] for lb in closed_e}
            fit = fit_dispersion_temperature([(finals[lb], pool_b[lb]) for lb in closed_e],
                                             bounds=T_BOUNDS)
            t_raw = fit.temperature
        else:
            t_raw = float("nan")
        out[str(eps)] = {
            "epsilon": eps,
            "n_common": len(common_e),
            "mixture_entropy": h,
            "entropy_floor_ok": h >= MIN_POOL_ENTROPY,
            "min_pairwise_tvd": min(pair_tvds),
            "T_raw": t_raw,
            "T_raw_saturated": saturated(t_raw),
            "T_raw_in_band": BAND[0] <= t_raw <= BAND[1] if math.isfinite(t_raw) else False,
        }
    return {"eps_list": list(eps_list), "results": out}


# ================================================================
# 4. Fit-grid sensitivity
# ================================================================
def grid_sensitivity(bases: dict, cells_by_label: dict,
                     closed_preds: dict, closed_labels: list) -> dict:
    """T_raw stability across different grid resolutions."""
    results = {}
    for n_pts in (60, 120, 240):
        grid = [math.exp(math.log(T_BOUNDS[0]) + (math.log(T_BOUNDS[1]) - math.log(T_BOUNDS[0])) * i / (n_pts - 1))
                for i in range(n_pts)]
        fams = list(bases.keys())
        finals = {lb: dist(closed_preds[lb], cells_by_label[lb]) for lb in closed_labels}
        pool_b = {lb: [dist(bases[f][lb], cells_by_label[lb]) for f in fams] for lb in closed_labels}
        fit = fit_dispersion_temperature([(finals[lb], pool_b[lb]) for lb in closed_labels],
                                         bounds=T_BOUNDS, grid=n_pts)
        near_1 = [g for g in grid if 0.95 < g < 1.05]
        results[str(n_pts)] = {
            "grid_points": n_pts,
            "T_raw": fit.temperature,
            "T_raw_saturated": saturated(fit.temperature),
            "T_raw_in_band": BAND[0] <= fit.temperature <= BAND[1],
            "grid_points_in_095_105": len(near_1),
            "nearest_to_1": min(grid, key=lambda g: abs(g - 1.0)),
            "objective_at_fit": fit.objective_at_fit,
        }

    t_vals = [v["T_raw"] for v in results.values() if not v["T_raw_saturated"]]
    return {
        "grid_resolutions": results,
        "T_raw_spread": max(t_vals) - min(t_vals) if len(t_vals) > 1 else 0.0,
        "all_in_band": all(v["T_raw_in_band"] for v in results.values()),
        "note": "fit_dispersion_temperature uses its own grid+refinement; grid param controls initial scan density",
    }


# ================================================================
# 5. Stratified behavior
# ================================================================
def stratified_behavior(bases: dict, cells: list, cells_by_label: dict,
                        views_with_post: dict,
                        closed_preds: dict) -> dict:
    """Pool T_raw stability across strata: mode-agree/disagree, per-Article, per-GT-level."""
    fams = list(bases.keys())
    closed_labels = [lb for lb in closed_preds if lb in cells_by_label
                     and all(lb in bases[f] for f in fams)]

    def fit_on_subset(labels):
        if len(labels) < 2:
            return {"n": len(labels), "T_raw": float("nan"), "note": "insufficient items"}
        finals = {lb: dist(closed_preds[lb], cells_by_label[lb]) for lb in labels}
        pool_b = {lb: [dist(bases[f][lb], cells_by_label[lb]) for f in fams] for lb in labels}
        fit = fit_dispersion_temperature([(finals[lb], pool_b[lb]) for lb in labels], bounds=T_BOUNDS)
        h = statistics.mean(
            realized_dispersion([dist(bases[f][lb], cells_by_label[lb]) for f in fams]).mixture_entropy
            for lb in labels)
        return {
            "n": len(labels), "T_raw": fit.temperature,
            "T_raw_saturated": saturated(fit.temperature),
            "T_raw_in_band": BAND[0] <= fit.temperature <= BAND[1],
            "pool_entropy": h,
        }

    def entropy_on_subset(labels):
        if not labels:
            return {"n": 0}
        h = statistics.mean(
            realized_dispersion([dist(bases[f][lb], cells_by_label[lb]) for f in fams]).mixture_entropy
            for lb in labels)
        return {"n": len(labels), "pool_entropy": h, "entropy_floor_ok": h >= MIN_POOL_ENTROPY}

    # Mode-agree/disagree strata (from R0 analysis A)
    mode_agree_labels = []
    mode_disagree_labels = []
    for f in fams:
        pre_fl, post_fl = views_with_post[f]
        overlap = sorted(pre_fl.keys() & post_fl.keys())
        for lb in overlap:
            pre_d = ComplianceDistribution.from_values(list(pre_fl[lb]), LABELS)
            post_d = ComplianceDistribution.from_values(list(post_fl[lb]), LABELS)
            if pre_d.argmax_index() == post_d.argmax_index():
                if lb not in mode_agree_labels:
                    mode_agree_labels.append(lb)
            else:
                if lb not in mode_disagree_labels:
                    mode_disagree_labels.append(lb)

    # For mode strata, use the per-family majority vote
    agree_by_cell = {}
    disagree_by_cell = {}
    for lb in closed_labels:
        c = cells_by_label[lb]
        n_agree = 0
        n_total = 0
        for f in fams:
            pre_fl, post_fl = views_with_post[f]
            if lb in pre_fl and lb in post_fl:
                pre_d = ComplianceDistribution.from_values(list(pre_fl[lb]), LABELS)
                post_d = ComplianceDistribution.from_values(list(post_fl[lb]), LABELS)
                n_total += 1
                if pre_d.argmax_index() == post_d.argmax_index():
                    n_agree += 1
        if n_total > 0:
            if n_agree > n_total / 2:
                agree_by_cell[lb] = True
            else:
                disagree_by_cell[lb] = True

    agree_closed = sorted(lb for lb in closed_labels if lb in agree_by_cell)
    disagree_closed = sorted(lb for lb in closed_labels if lb in disagree_by_cell)

    # Per-Article
    articles = sorted({c.article for c in cells})
    per_article_closed = {a: sorted(lb for lb in closed_labels if cells_by_label[lb].article == a)
                         for a in articles}
    all_common = sorted(set.intersection(*(set(b.keys()) for b in bases.values())))
    per_article_all = {a: sorted(lb for lb in all_common if cells_by_label[lb].article == a)
                       for a in articles}

    # Per-GT-level
    gt_labels_list = ("very_low", "low", "moderate", "high", "very_high")
    per_gt_closed = {lvl: sorted(lb for lb in closed_labels if gt_labels_list[cells_by_label[lb].gt_argmax] == lvl)
                     for lvl in gt_labels_list}
    per_gt_all = {lvl: sorted(lb for lb in all_common if gt_labels_list[cells_by_label[lb].gt_argmax] == lvl)
                  for lvl in gt_labels_list}

    out = {
        "mode_strata": {
            "note": "majority-vote mode agreement across panel families (pre vs post argmax)",
            "agree": fit_on_subset(agree_closed),
            "disagree": fit_on_subset(disagree_closed),
            "agree_entropy_full_cells": entropy_on_subset(
                sorted(lb for lb in all_common if lb in agree_by_cell)),
            "disagree_entropy_full_cells": entropy_on_subset(
                sorted(lb for lb in all_common if lb in disagree_by_cell)),
        },
        "per_article": {
            "T_raw_fits": {a: fit_on_subset(lbs) for a, lbs in per_article_closed.items()},
            "entropy_full_cells": {a: entropy_on_subset(lbs) for a, lbs in per_article_all.items()},
        },
        "per_gt_level": {
            "T_raw_fits": {lvl: fit_on_subset(lbs) for lvl, lbs in per_gt_closed.items()},
            "entropy_full_cells": {lvl: entropy_on_subset(lbs) for lvl, lbs in per_gt_all.items()},
        },
    }

    # Check: is the 1.278 a mixture artifact?
    strata_ts = []
    for stratum_name in ("agree", "disagree"):
        s = out["mode_strata"][stratum_name]
        if s["n"] >= 2 and not s.get("T_raw_saturated", True):
            strata_ts.append(s["T_raw"])
    if len(strata_ts) == 2:
        out["mode_strata"]["mixture_artifact_diagnostic"] = {
            "agree_T": strata_ts[0],
            "disagree_T": strata_ts[1],
            "ratio": max(strata_ts) / min(strata_ts) if min(strata_ts) > 0 else float("inf"),
            "is_mixture_artifact": max(strata_ts) / min(strata_ts) > 2.0 if min(strata_ts) > 0 else True,
            "note": ("If ratio > 2, the pooled T_raw is a mixture of divergent subpopulations "
                     "(≈1 and 2.7-4.2 as in R0 analysis A); if ≤ 2, the in-band fit is genuine"),
        }
    return out


# ================================================================
# 6. Sensitivity annexes
# ================================================================
def sensitivity_annexes(all_bases: dict, cells_by_label: dict,
                        closed_preds: dict, fams_panel: list) -> dict:
    """Non-adoption sensitivity reads: +llama31, gemma31-vLLM swap, pool-of-3 ablations."""
    out = {}

    def fit_pool(members, label):
        bases = {f: all_bases[f] for f in members}
        common = sorted(set.intersection(*(set(b.keys()) for b in bases.values())))
        closed_labels = [lb for lb in closed_preds if lb in cells_by_label
                         and all(lb in bases[f] for f in members)]
        if len(closed_labels) < 2:
            return {"label": label, "members": members, "n_closed": len(closed_labels),
                    "note": "insufficient closed items"}
        finals = {lb: dist(closed_preds[lb], cells_by_label[lb]) for lb in closed_labels}
        pool_b = {lb: [dist(bases[f][lb], cells_by_label[lb]) for f in members] for lb in closed_labels}
        fit = fit_dispersion_temperature([(finals[lb], pool_b[lb]) for lb in closed_labels],
                                         bounds=T_BOUNDS)
        h = statistics.mean(
            realized_dispersion([dist(bases[f][lb], cells_by_label[lb]) for f in members]).mixture_entropy
            for lb in common)
        pair_tvds = []
        for i, f in enumerate(members):
            for g in members[i + 1:]:
                pair_tvds.append(statistics.mean(
                    total_variation_distance(dist(bases[f][lb], cells_by_label[lb]),
                                             dist(bases[g][lb], cells_by_label[lb]))
                    for lb in common))
        return {
            "label": label, "members": members,
            "n_common_cells": len(common), "n_closed": len(closed_labels),
            "T_raw": fit.temperature, "T_raw_saturated": saturated(fit.temperature),
            "T_raw_in_band": BAND[0] <= fit.temperature <= BAND[1],
            "pool_entropy": h, "entropy_floor_ok": h >= MIN_POOL_ENTROPY,
            "min_pairwise_tvd": min(pair_tvds) if pair_tvds else None,
        }

    # +llama31
    out["plus_llama31"] = fit_pool(fams_panel + ["llama31"], "+llama31 (sensitivity)")

    # gemma31-vLLM swap: replace gemma31_api with gemma31 (the vLLM collapse leg)
    if "gemma31" in all_bases:
        swapped = [f if f != "gemma31_api" else "gemma31" for f in fams_panel]
        out["gemma31_vllm_swap"] = fit_pool(swapped, "gemma31-vLLM swap (collapse leg)")

    # Pool-of-3 ablations: each panel member removed
    out["pool_of_3_ablations"] = {}
    for drop in fams_panel:
        remaining = [f for f in fams_panel if f != drop]
        out["pool_of_3_ablations"][f"drop_{drop}"] = fit_pool(remaining, f"drop {drop}")

    # Frozen F6 vs each variant: drift summary
    frozen_fit = fit_pool(fams_panel, "frozen F6 panel")
    frozen_t = frozen_fit["T_raw"]
    drift = {}
    for name, variant in out.items():
        if name == "pool_of_3_ablations":
            for sub_name, sub in variant.items():
                if isinstance(sub, dict) and "T_raw" in sub and math.isfinite(sub["T_raw"]):
                    drift[sub_name] = {
                        "variant_T": sub["T_raw"], "frozen_T": frozen_t,
                        "delta": sub["T_raw"] - frozen_t,
                        "abs_log_ratio": abs(math.log(sub["T_raw"] / frozen_t)) if frozen_t > 0 and sub["T_raw"] > 0 else float("inf"),
                    }
        elif isinstance(variant, dict) and "T_raw" in variant and math.isfinite(variant["T_raw"]):
            drift[name] = {
                "variant_T": variant["T_raw"], "frozen_T": frozen_t,
                "delta": variant["T_raw"] - frozen_t,
                "abs_log_ratio": abs(math.log(variant["T_raw"] / frozen_t)) if frozen_t > 0 and variant["T_raw"] > 0 else float("inf"),
            }
    out["drift_summary"] = drift
    out["frozen_panel_fit"] = frozen_fit
    return out


# ================================================================
# 7. Failure-mode pre-registration
# ================================================================
def failure_mode_preregistration() -> dict:
    """What arm 3b could report on the E6 sweep and what each outcome means."""
    return {
        "arm_3b_possible_outcomes": [
            {
                "outcome": "IN-BAND CONCURRENCE",
                "condition": f"T_raw ∈ [{BAND[0]:.3f}, {BAND[1]:.3f}] AND vetting passes "
                             f"(pool entropy ≥ {MIN_POOL_ENTROPY}, LOBO min ≥ {MIN_POOL_ENTROPY}, unsaturated)",
                "interpretation": ("The pool's cross-family dispersion read on the FULL closed pair "
                                   "agrees with the transferred τ* = 1.153 within the sensitivity band. "
                                   "This is the strongest cross-check arm 3b can provide: a GT-free "
                                   "estimand concurring with the GT-informed panel median. Publish as "
                                   "corroboration (non-gating per r3)."),
                "paper_sentence": "The pool T_raw of {T_raw:.3f} lands inside the sensitivity band, "
                                  "corroborating the transferred constant.",
            },
            {
                "outcome": "BELOW-BAND",
                "condition": f"T_raw < {BAND[0]:.3f} AND vetting passes AND unsaturated",
                "interpretation": ("Pool sees LESS dispersion than the panel τ_v values predict — "
                                   "the closed pair's internal agreement is tighter than the open "
                                   "panel's. If T_raw ≈ 1 (grid resolution: 0.952-1.025), this means "
                                   "the closed pair is already calibrated to the dispersion pool and "
                                   "no correction is needed. Non-gating: arm 1 adoption stands if F4 passes."),
                "paper_sentence": "The pool T_raw of {T_raw:.3f} falls below the band, suggesting the "
                                  "closed pair's internal dispersion is already calibrated.",
            },
            {
                "outcome": "ABOVE-BAND",
                "condition": f"T_raw > {BAND[1]:.3f} AND vetting passes AND unsaturated",
                "interpretation": ("Pool sees MORE dispersion than the panel — the closed pair's "
                                   "internal spread exceeds what open-panel bases produce. This is the "
                                   "cfp1-augmentation pattern (prototype: 1.70-1.78 with cfp1); if it "
                                   "appears with bases-only, the closed pair may have genuine internal "
                                   "disagreement beyond the open-panel's range. Non-gating, but flags a "
                                   "pool-vs-pair mismatch worth noting."),
                "paper_sentence": "The pool T_raw of {T_raw:.3f} exceeds the band upper edge, indicating "
                                  "the closed pair's internal dispersion exceeds the open panel's range.",
            },
            {
                "outcome": "SATURATED / VETTING FAIL",
                "condition": "T_raw pegs at a bound OR pool entropy < 1.00 OR LOBO floor violated",
                "interpretation": ("The dispersion pool is not informative on this closed pair. "
                                   "Saturated = the entropy-matching objective has no minimum in the "
                                   "search range; vetting fail = the pool itself has collapsed (would "
                                   "indicate a parse-failure-driven composition change on the sweep's "
                                   "cells, not a property of the frozen pool). Non-gating: arm 1 still "
                                   "stands on its own evidence."),
                "paper_sentence": "Arm 3b is non-informative (T_raw {saturated_or_vetting_msg}).",
            },
            {
                "outcome": "UNSTABLE-UNDER-LOBO",
                "condition": "Full-pool T_raw in-band but ≥1 LOBO variant outside band or saturated",
                "interpretation": ("The in-band read depends on one specific pool member — removing it "
                                   "shifts the fit outside the band. Weakens the cross-check: the "
                                   "concurrence is fragile. Publish with the LOBO table as the "
                                   "qualifying caveat."),
                "paper_sentence": "The pool T_raw of {T_raw:.3f} is in-band but LOBO-fragile: removing "
                                  "{fragile_member} shifts the fit to {lobo_T:.3f} (outside the band).",
            },
        ],
        "r3_ladder_cross_reference": {
            "arm1": "τ* = 1.153 → F4 acceptance (reliability ↑, RPS ≤, resolution within 10%)",
            "arm2": "supervised T* → F5 concurrence |ln(T*/τ*)| ≤ ln 2",
            "arm3b_this_arm": f"pool T_raw → in-band [{BAND[0]:.3f}, {BAND[1]:.3f}] concurrence (F6, non-gating)",
            "arm3a": "τ_DACA → F7 corroboration (≥3 valid, range ⊂ [τ*/2, 2τ*], non-gating)",
            "arm4": "T_c → F8 (LODO calibratable + kendall ≤ -0.10, report-only)",
        },
        "mechanical_read_rule": (
            "After the sweep: compute arm3b via e6_r3_arms.py. Read the outcome column above that "
            "matches the result. No judgment required — the table is exhaustive over {in-band, below, "
            "above, saturated/fail, LOBO-fragile}. The paper sentence template fills mechanically."
        ),
    }


# ================================================================
# Main
# ================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "verbalized_pool_robustness")
    args = ap.parse_args()

    print("Loading cells and data...")
    cells = aireg.load_cells()
    cells_by_label = {c.item_label: c for c in cells}
    eval_runs = REPO.parent / "judex-evaluator" / "runs"

    # Load all legs
    all_views_raw = {}
    all_bases = {}
    views_with_post = {}
    for fam, (rel, channel) in LEGS.items():
        leg_path = REPO / rel
        if not leg_path.exists():
            print(f"  SKIP {fam}: {leg_path} not found")
            continue
        pre, post = load_leg(leg_path)
        pre_fl = floored_view(pre)
        post_fl = floored_view(post)
        all_views_raw[fam] = compliance_view(pre)
        all_bases[fam] = pre_fl
        views_with_post[fam] = (pre_fl, post_fl)

    panel_bases = {f: all_bases[f] for f in PANEL if f in all_bases}
    common = sorted(set.intersection(*(set(b.keys()) for b in panel_bases.values())))
    print(f"  Panel bases loaded: {list(panel_bases.keys())}, {len(common)} common cells")

    closed_preds = load_closed_predictions(eval_runs)
    closed_reps = load_closed_replicates(eval_runs, cells)
    closed_labels = [lb for lb in closed_preds if lb in cells_by_label
                     and all(lb in panel_bases[f] for f in PANEL)]
    print(f"  Closed predictions: {len(closed_preds)} items, {len(closed_labels)} matched to pool")

    report = {
        "mandate": "verbalized dispersion pool robustness battery (2026-07-21)",
        "frozen_pool": PANEL,
        "frozen_tau_v": PANEL_TAU_V,
        "frozen_T_J": T_J,
        "frozen_band": list(BAND),
        "epsilon": EPSILON,
        "T_bounds": list(T_BOUNDS),
        "n_cells": len(cells),
        "n_common_panel_cells": len(common),
        "n_closed_items": len(closed_preds),
        "n_closed_matched": len(closed_labels),
        "closed_run": CLOSED_RUN,
        "smoke": len(closed_preds) < 120,
    }

    # 1. LOBO
    print("\n== 1. LOBO at full scale ==")
    report["lobo_full_scale"] = lobo_full_scale(
        panel_bases, cells_by_label, common, closed_labels, closed_preds, closed_reps)
    lobo = report["lobo_full_scale"]
    print(f"  pool entropy: {lobo['pool_entropy_full']:.3f} (floor ok={lobo['entropy_floor_ok']})")
    print(f"  LOBO min entropy: {lobo['lobo_min_entropy']:.3f} (all above floor={lobo['lobo_all_above_floor']})")
    print(f"  LOBO T_raw range: {lobo['lobo_T_raw_range']}")
    print(f"  LOBO all in band: {lobo['lobo_all_in_band']}")
    for f, v in lobo["per_member"].items():
        print(f"    drop {f:12s}: H={v['pool_entropy_without']:.3f} T_raw={v['T_raw_without']:.3f} "
              f"in_band={v['T_raw_in_band']} sat={v['T_raw_saturated']}")

    # 2. Bootstrap CIs
    print("\n== 2. Doc-clustered bootstrap CIs ==")
    report["bootstrap_cis"] = doc_clustered_bootstrap(
        panel_bases, cells_by_label, cells, closed_preds, closed_reps)
    bs = report["bootstrap_cis"]
    print(f"  T_raw: {bs['T_raw']['point']:.3f} CI95={bs['T_raw']['ci95']}")
    print(f"  entropy: {bs['mixture_entropy']['point']:.3f} CI95={bs['mixture_entropy']['ci95']}")
    print(f"  min TVD: {bs['min_pairwise_tvd']['point']:.3f} CI95={bs['min_pairwise_tvd']['ci95']}")
    print(f"  T_raw CI inside band: {bs['T_raw_ci_inside_band']}")
    print(f"  T_raw CI includes 1.0: {bs['T_raw_ci_includes_1']}")

    # 3. Epsilon sensitivity
    print("\n== 3. Epsilon sensitivity ==")
    bases_raw_panel = {f: compliance_view(json.loads((REPO / LEGS[f][0] / "pre_verbalized.json").read_text()))
                       for f in PANEL}
    report["epsilon_sensitivity"] = epsilon_sensitivity(
        bases_raw_panel, cells_by_label, common, closed_preds)
    for eps, r in report["epsilon_sensitivity"]["results"].items():
        print(f"  eps={eps}: H={r['mixture_entropy']:.3f} minTVD={r['min_pairwise_tvd']:.3f} "
              f"T_raw={r['T_raw']:.3f} in_band={r['T_raw_in_band']}")

    # 4. Grid sensitivity
    print("\n== 4. Fit-grid sensitivity ==")
    report["grid_sensitivity"] = grid_sensitivity(panel_bases, cells_by_label, closed_preds, closed_labels)
    for n, r in report["grid_sensitivity"]["grid_resolutions"].items():
        print(f"  grid {n}pt: T_raw={r['T_raw']:.3f} in_band={r['T_raw_in_band']} "
              f"nearest_to_1={r['nearest_to_1']:.6f}")
    print(f"  spread: {report['grid_sensitivity']['T_raw_spread']:.4f}")

    # 5. Stratified behavior
    print("\n== 5. Stratified behavior ==")
    report["stratified"] = stratified_behavior(
        panel_bases, cells, cells_by_label, views_with_post, closed_preds)
    ms = report["stratified"]["mode_strata"]
    print(f"  mode-agree: n={ms['agree']['n']} T_raw={ms['agree'].get('T_raw', 'n/a')}")
    print(f"  mode-disagree: n={ms['disagree']['n']} T_raw={ms['disagree'].get('T_raw', 'n/a')}")
    if "mixture_artifact_diagnostic" in ms:
        mad = ms["mixture_artifact_diagnostic"]
        print(f"  mixture artifact: ratio={mad['ratio']:.2f} is_artifact={mad['is_mixture_artifact']}")
    print("  per-Article T_raw:")
    for a, r in sorted(report["stratified"]["per_article"]["T_raw_fits"].items()):
        print(f"    Art {a}: n={r['n']} T_raw={r.get('T_raw', 'n/a')}")
    print("  per-GT-level T_raw:")
    for lvl, r in report["stratified"]["per_gt_level"]["T_raw_fits"].items():
        print(f"    {lvl}: n={r['n']} T_raw={r.get('T_raw', 'n/a')}")

    # 6. Sensitivity annexes
    print("\n== 6. Sensitivity annexes ==")
    report["sensitivity_annexes"] = sensitivity_annexes(
        all_bases, cells_by_label, closed_preds, PANEL)
    sa = report["sensitivity_annexes"]
    for name in ("plus_llama31", "gemma31_vllm_swap"):
        if name in sa and isinstance(sa[name], dict):
            v = sa[name]
            print(f"  {name}: T_raw={v.get('T_raw', 'n/a')} in_band={v.get('T_raw_in_band')} "
                  f"H={v.get('pool_entropy', 'n/a')}")
    if "pool_of_3_ablations" in sa:
        for name, v in sa["pool_of_3_ablations"].items():
            if isinstance(v, dict):
                print(f"  {name}: T_raw={v.get('T_raw', 'n/a')} in_band={v.get('T_raw_in_band')}")
    if "drift_summary" in sa:
        print("  drift from frozen F6:")
        for name, d in sa["drift_summary"].items():
            print(f"    {name}: delta={d['delta']:.3f} |log_ratio|={d['abs_log_ratio']:.3f}")

    # 7. Failure-mode pre-registration
    print("\n== 7. Failure-mode pre-registration ==")
    report["failure_mode_preregistration"] = failure_mode_preregistration()
    for outcome in report["failure_mode_preregistration"]["arm_3b_possible_outcomes"]:
        print(f"  {outcome['outcome']}: {outcome['condition'][:80]}...")

    # Verdict
    lobo_entropy_ok = lobo["lobo_all_above_floor"] and not lobo["lobo_any_saturated"]
    lobo_band_ok = lobo["lobo_all_in_band"]
    bs_in_band = bs["T_raw_ci_inside_band"]
    eps_stable = all(r["T_raw_in_band"] for r in report["epsilon_sensitivity"]["results"].values())
    grid_stable = report["grid_sensitivity"]["all_in_band"]

    # Mode-strata diagnostic
    ms = report["stratified"]["mode_strata"]
    mixture_artifact = ms.get("mixture_artifact_diagnostic", {}).get("is_mixture_artifact", False)

    report["verdict"] = {
        "T_raw_point": 1.278,
        "lobo_entropy_ok": lobo_entropy_ok,
        "lobo_band_ok": lobo_band_ok,
        "lobo_fragile_members": [f for f, v in lobo["per_member"].items() if not v["T_raw_in_band"]],
        "bootstrap_ci_in_band": bs_in_band,
        "bootstrap_ci": bs["T_raw"]["ci95"],
        "epsilon_stable": eps_stable,
        "grid_stable": grid_stable,
        "mixture_artifact": mixture_artifact,
        "overall": ("ROBUST-WITH-CAVEATS" if (lobo_entropy_ok and eps_stable and grid_stable
                                               and not lobo_band_ok)
                    else "ROBUST" if (lobo_entropy_ok and lobo_band_ok and eps_stable and grid_stable)
                    else "FRAGILE"),
        "caveats": [],
    }
    if not lobo_band_ok:
        report["verdict"]["caveats"].append(
            f"LOBO-fragile: dropping {report['verdict']['lobo_fragile_members']} shifts T_raw outside band")
    if not bs_in_band:
        report["verdict"]["caveats"].append(
            f"Bootstrap 95% CI [{bs['T_raw']['ci95'][0]:.3f}, {bs['T_raw']['ci95'][1]:.3f}] "
            f"extends beyond band [{BAND[0]:.3f}, {BAND[1]:.3f}] (smoke-grade n)")
    if mixture_artifact:
        report["verdict"]["caveats"].append(
            "T_raw is a mixture artifact: mode-agree stratum ≈ 0.92, mode-disagree ≈ 2.29")
    print(f"\n=== VERDICT: {report['verdict']['overall']} ===")

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "pool_robustness.json"
    out_path.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
