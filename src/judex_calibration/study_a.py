"""Study A analysis — per-family/variant calibration against AIReg human GT.

Channel-agnostic: operates on elicited 5-way distributions (very_low..very_high)
keyed by item_label, however they were produced (vLLM token-slice for base/post,
or verbalized). Reuses judex-evaluator calibration tooling verbatim.

Per variant it reports: argmax accuracy (the gate), T*(RPS-min), T*(Reliability-
min), the Murphy decomposition, and mean RPS/W1 (uncalibrated and at T*). For a
pre/post family it also fits tau_oc = the temperature aligning post -> pre (the
clean post-training overconfidence). cross_family() tabulates and tests whether
tau_oc clusters (Q2: is a transferred constant T for the closed evaluators
justified?).
"""
from __future__ import annotations

import json, math, os, sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Sibling-repo layout: .../judex/{judex-calibration,judex-evaluator}. Prefer an editable
# install (pip install -e ../judex-evaluator); fall back to the sibling src on sys.path.
# JUDEX_UMBRELLA overrides the parent-dir default (worktrees live under <umbrella>/worktrees/).
_UMBRELLA = Path(os.environ.get("JUDEX_UMBRELLA") or Path(__file__).resolve().parents[3])
EVAL_SRC = _UMBRELLA / "judex-evaluator" / "src"
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))

from judex.core.distributions import ComplianceDistribution          # noqa: E402
from judex.core.metrics import ranked_probability_score, wasserstein_1  # noqa: E402
from judex.calibration import fit_temperature, apply_temperature      # noqa: E402
from judex.experiments import murphy_decomposition                     # noqa: E402
from judex.metrics_report import MetricItemResult                      # noqa: E402
from judex.core.metrics import signed_delta, total_variation_distance  # noqa: E402

from .aireg import Cell, load_cells

# Study A's OWN temperature search range — passed explicitly to every evaluator fitter.
# It must reach well past judex-evaluator's ``DEFAULT_TEMPERATURE_BOUNDS = (0.25, 4.0)``
# (`judex.calibration`), because the failure this study characterises pegs T at ~19 (§0).
# Passing ``bounds`` is deliberate and load-bearing: inheriting the evaluator default silently
# censored ``T_rps`` at 4.0 while ``T_rel`` and ``tau_oc`` — both searched on GRID — ran to 20,
# so the three temperatures were not on the same scale and a pegged T_rps read as a real fit.
# Owning the range here also means an evaluator-side default change cannot move our numbers.
T_BOUNDS = (0.25, 20.0)
GRID = [math.exp(math.log(T_BOUNDS[0]) + (math.log(T_BOUNDS[1]) - math.log(T_BOUNDS[0])) * i / 59)
        for i in range(60)]


def saturated(T: float, tol: float = 1e-6) -> bool:
    """True when a fitted temperature sits on the search boundary — a peg, not a fit.

    A saturated T means the objective flattened to the marginal (the accuracy-deficit failure
    mode of §0): it is a boundary artefact, not a measurement, and must never be adopted as the
    transferred constant. Reported alongside every temperature so a peg is never silently read
    as a result. NaN (no overlapping cells) counts as not-a-valid-fit.
    """
    return (not math.isfinite(T)) or T <= T_BOUNDS[0] * (1 + tol) or T >= T_BOUNDS[1] * (1 - tol)


def load_predictions(path: str | Path) -> Dict[str, List[float]]:
    """{item_label: [p_very_low..p_very_high]} from a JSON file."""
    return json.loads(Path(path).read_text())


def _pred_dist(probs: Sequence[float], gt_labels: Tuple[str, ...]) -> ComplianceDistribution:
    # predicted order very_low..very_high aligns with gt_labels index 0..4
    return ComplianceDistribution.from_values(list(probs), gt_labels)


def _metric_item(label, pred, gt):
    pa, ga = pred.argmax_index(), gt.argmax_index()
    return MetricItemResult(
        item_id=label, item_label=label, prediction=pred, ground_truth=gt,
        rps=ranked_probability_score(pred, gt), w1=wasserstein_1(pred, gt),
        tvd=total_variation_distance(pred, gt), argmax_agreement=pa == ga,
        predicted_argmax_index=pa, predicted_argmax_label=pred.labels[pa],
        ground_truth_argmax_index=ga, ground_truth_argmax_label=gt.labels[ga],
        prediction_entropy=pred.entropy(), prediction_normalized_entropy=pred.normalized_entropy(),
        ground_truth_entropy=gt.entropy(), ground_truth_normalized_entropy=gt.normalized_entropy(),
        entropy_review_flag=pred.normalized_entropy() > 0.75, signed_delta=signed_delta(pred, gt))


def _reliability(items, T, bins=3) -> float:
    """Murphy reliability of temperature-scaled predictions (grid target for T_rel)."""
    scaled = [_metric_item(it.item_label, apply_temperature(it.prediction, T), it.ground_truth) for it in items]
    return murphy_decomposition(scaled, bins=bins)["reliability"]


def score_variant(preds: Dict[str, List[float]], cells: List[Cell]) -> dict:
    by_label = {c.item_label: c for c in cells}
    items = []
    for label, probs in preds.items():
        c = by_label.get(label)
        if c is None:
            continue
        items.append(_metric_item(label, _pred_dist(probs, c.gt_labels),
                                  ComplianceDistribution.from_values(c.gt_probs, c.gt_labels)))
    if not items:
        return {"n": 0}
    pairs = [(it.prediction, it.ground_truth) for it in items]
    T_rps = fit_temperature(pairs, bounds=T_BOUNDS).temperature
    T_rel = min(GRID, key=lambda T: _reliability(items, T))
    murphy = murphy_decomposition(items)
    return {
        "n": len(items),
        "argmax_acc": sum(it.argmax_agreement for it in items) / len(items),
        "mean_rps": sum(it.rps for it in items) / len(items),
        "mean_w1": sum(it.w1 for it in items) / len(items),
        "T_rps": T_rps, "T_rel": T_rel,
        "T_rps_saturated": saturated(T_rps), "T_rel_saturated": saturated(T_rel),
        "T_bounds": list(T_BOUNDS),
        "murphy": {k: round(murphy[k], 5) for k in ("uncertainty", "resolution", "reliability", "mean_rps")},
        "mean_pred_norm_entropy": sum(it.prediction_normalized_entropy for it in items) / len(items),
    }


def fit_tau_oc(post: Dict[str, List[float]], pre: Dict[str, List[float]], cells: List[Cell]) -> float:
    """Temperature aligning POST -> PRE (clean post-training overconfidence)."""
    by_label = {c.item_label: c for c in cells}
    pairs = []
    for label in post.keys() & pre.keys():
        c = by_label.get(label)
        if c:
            pairs.append((_pred_dist(post[label], c.gt_labels), _pred_dist(pre[label], c.gt_labels)))
    if not pairs:
        return float("nan")
    return min(GRID, key=lambda T: sum(wasserstein_1(apply_temperature(p, T), q) for p, q in pairs) / len(pairs))


def closed_side_check(preds: Dict[str, List[float]], cells: List[Cell], T: float, bins: int = 10) -> dict:
    """Q4: apply the open-derived constant ``T`` to closed-evaluator (Claude/GPT)
    predictions on AIReg. The transfer is legitimate iff Murphy **Reliability**
    improves **without** destroying Resolution or RPS (temperature preserves
    argmax, so accuracy is unchanged by construction)."""
    by_label = {c.item_label: c for c in cells}
    base, cal = [], []
    for label, probs in preds.items():
        c = by_label.get(label)
        if c is None:
            continue
        gt = ComplianceDistribution.from_values(c.gt_probs, c.gt_labels)
        pred = _pred_dist(probs, c.gt_labels)
        base.append(_metric_item(label, pred, gt))
        cal.append(_metric_item(label, apply_temperature(pred, T), gt))
    if not base:
        return {"n": 0}
    mb, mc = murphy_decomposition(base), murphy_decomposition(cal)

    def agg(items):
        return {"mean_rps": sum(i.rps for i in items) / len(items),
                "mean_w1": sum(i.w1 for i in items) / len(items),
                "argmax_acc": sum(i.argmax_agreement for i in items) / len(items)}

    ab, ac = agg(base), agg(cal)
    return {
        "n": len(base), "T": T,
        "uncalibrated": {**{k: round(mb[k], 5) for k in ("reliability", "resolution", "uncertainty")},
                         **{k: round(v, 5) for k, v in ab.items()}},
        "calibrated": {**{k: round(mc[k], 5) for k in ("reliability", "resolution", "uncertainty")},
                       **{k: round(v, 5) for k, v in ac.items()}},
        "reliability_improvement": round(mb["reliability"] - mc["reliability"], 5),
        "resolution_change": round(mc["resolution"] - mb["resolution"], 5),
        "rps_change": round(ac["mean_rps"] - ab["mean_rps"], 5),
        # transfer is defensible iff reliability drops and RPS does not get worse
        "verdict_ok": bool(mc["reliability"] < mb["reliability"] and ac["mean_rps"] <= ab["mean_rps"] + 1e-9),
    }


def cross_family(families: Dict[str, Dict[str, Dict[str, List[float]]]], cells: List[Cell]) -> dict:
    """families = {family: {'pre': preds, 'post': preds}}. Returns per-family scores + tau_oc spread."""
    rows = {}
    for fam, variants in families.items():
        row = {v: score_variant(p, cells) for v, p in variants.items()}
        if "pre" in variants and "post" in variants:
            row["tau_oc"] = fit_tau_oc(variants["post"], variants["pre"], cells)
            row["tau_oc_saturated"] = saturated(row["tau_oc"])
            # tau_oc can be a perfectly well-behaved number while being meaningless: it aligns
            # post -> pre, so if the PRE leg's own T* pegged, the reference it aligns to is a
            # distribution whose calibration could not be fit at all. The fp16 4B pilot is the
            # worked example — gemma3-4b's pre leg pegs on both objectives while its tau_oc
            # (4.877) sits mid-range and looks like a measurement. Flag the reference, not just
            # the value. (The §0 accuracy gate is the primary guard; this is the second line.)
            row["tau_oc_reference_degenerate"] = bool(
                row.get("pre", {}).get("T_rps_saturated") or row.get("pre", {}).get("T_rel_saturated"))
        rows[fam] = row
    taus = [r["tau_oc"] for r in rows.values() if isinstance(r.get("tau_oc"), float) and not math.isnan(r["tau_oc"])]
    summary = {}
    if taus:
        taus_sorted = sorted(taus)
        pegged = sorted(f for f, r in rows.items() if r.get("tau_oc_saturated"))
        degenerate = sorted(f for f, r in rows.items() if r.get("tau_oc_reference_degenerate"))
        summary = {"tau_oc_median": taus_sorted[len(taus) // 2], "tau_oc_min": min(taus),
                   "tau_oc_max": max(taus), "tau_oc_spread": max(taus) - min(taus), "n_families": len(taus),
                   # A pegged tau_oc is a boundary artefact, not a measurement (see `saturated`).
                   # Surfaced here so Q3/Q4 can never adopt a constant that was never actually fit.
                   "tau_oc_saturated_families": pegged,
                   "tau_oc_any_saturated": bool(pegged),
                   # Families whose PRE reference itself pegged — tau_oc is finite but not a
                   # measurement of post-training overconfidence. Q3 must exclude these.
                   "tau_oc_degenerate_reference_families": degenerate,
                   "tau_oc_any_degenerate_reference": bool(degenerate),
                   "T_bounds": list(T_BOUNDS)}
    return {"families": rows, "tau_oc_summary": summary}


def calibration_block(report: dict, *, accuracy_gate=None, bootstrap_ci=None) -> dict | None:
    """Turn a cross_family() report into a drop-in judex-evaluator calibration block.

    Mirrors the shape of ``judex.experiments.dispersion_calibration_config`` so it can be
    pasted verbatim under ``pipeline.yaml`` -> ``calibration``. Uses ``mode: "temperature"``
    (a fixed transferred constant is a supervised-derived scalar) — NOT ``mode: "dispersion"``,
    which stamps ``gt_free_dispersion_fit`` provenance and would be a false audit trail.

    NB (see guide §4.6 + docs/integration_remediation_2026_07_01.md): the merged pipeline seam is
    GLOBAL, but at evaluation time it only ever sees the closed evaluator pair (Claude/GPT) — the
    open raters run at construction, not here. So for two closed families + one clustered constant,
    this block drops into the global ``calibration`` key as-is (in
    ``judex-evaluator/configs/pipeline.yaml`` — and in ``configs_v2exemplars/pipeline.yaml`` too if
    the run uses that bundle). Family-scoping is an OPTIONAL refinement (per-family T if tau_oc does
    not cluster, a different evaluator pair, or Phase-3).

    ⚠ PROVENANCE IS NOT PERSISTED BY THE EVALUATOR (verified 2026-07-18 against evaluator
    ``2b6322b``). ``judex.calibration.calibrate_distribution`` reads ``provenance`` **only** on the
    ``dispersion`` branch; ``mode: temperature`` routes to ``calibrate_invert_softmax(distribution,
    temperature=...)``, which records ``method: "invert_softmax"`` and drops this whole dict —
    including ``smoke``. So the block below is accepted verbatim, but once pasted, nothing in the
    evaluator run records where the constant came from or that it was a smoke value. Keep
    ``pipeline_calibration_block.json`` as the audit trail alongside the run, and never rely on the
    evaluator to carry it. (Changing this is an evaluator-side change, out of Study A's scope.)
    """
    summary = report.get("tau_oc_summary", {})
    temperature = summary.get("tau_oc_median")
    if temperature is None:
        return None
    return {
        "mode": "temperature",
        "temperature": temperature,
        "provenance": {
            "source": "study_a_tau_oc",
            "replicate_source": "aireg_bench_post_to_pre",
            "tau_oc_median": temperature,
            "tau_oc_min": summary.get("tau_oc_min"),
            "tau_oc_max": summary.get("tau_oc_max"),
            "tau_oc_spread": summary.get("tau_oc_spread"),
            "n_families": summary.get("n_families"),
            # A pegged tau_oc is a boundary artefact, never adoptable — see `saturated`.
            "tau_oc_any_saturated": summary.get("tau_oc_any_saturated"),
            "tau_oc_saturated_families": summary.get("tau_oc_saturated_families"),
            "T_bounds": summary.get("T_bounds"),
            "accuracy_gate": accuracy_gate,
            "bootstrap_ci": bootstrap_ci,
            "target_families": "closed_evaluators (Claude/GPT); global seam adequate for a clustered constant, else family-scoped",
        },
    }


if __name__ == "__main__":
    # Smoke: score an existing JUDEX run's reconciled predictions as one variant, to validate the
    # analysis path end-to-end on real distributions (no API). Defaults to the LEGACY Gemini/GPT
    # run — kept only because it is the one 120-cell run on disk; the current closed pair is
    # Claude + GPT, so these numbers are NOT a Q4 result. Override with STUDY_A_SMOKE_RUN.
    # `judex-evaluator/runs/` is gitignored, so this smoke is Mac-only by construction (it is
    # absent on a fresh clone / vast box) — hence the explicit guard rather than a stack trace.
    run_id = os.environ.get("STUDY_A_SMOKE_RUN", "stage9-gemini-gpt-medium")
    path = _UMBRELLA / "judex-evaluator" / "runs" / run_id / "metrics_report.json"
    if not path.exists():
        raise SystemExit(
            f"no metrics_report.json at {path}\n"
            "judex-evaluator/runs/ is gitignored — this smoke only runs where the run exists.\n"
            "Set STUDY_A_SMOKE_RUN=<run-id> to point at a run you have.")
    cells = load_cells()
    preds = {it["item_label"]: it["prediction"]["probabilities"]
             for it in json.loads(path.read_text())["items"]}
    s = score_variant(preds, cells)
    print(f"SMOKE score_variant on {run_id} reconciled predictions "
          f"({len(preds)} cells; T bounds {T_BOUNDS}):")
    for k, v in s.items():
        print(f"  {k}: {v}")
