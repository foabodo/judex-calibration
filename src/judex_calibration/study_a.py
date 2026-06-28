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

import json, math, sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

EVAL_SRC = Path("/Users/fabodo/Downloads/Projects/judex/judex-evaluator/src")
if str(EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(EVAL_SRC))

from judex.core.distributions import ComplianceDistribution          # noqa: E402
from judex.core.metrics import ranked_probability_score, wasserstein_1  # noqa: E402
from judex.calibration import fit_temperature, apply_temperature      # noqa: E402
from judex.experiments import murphy_decomposition                     # noqa: E402
from judex.metrics_report import MetricItemResult                      # noqa: E402
from judex.core.metrics import signed_delta, total_variation_distance  # noqa: E402

from .aireg import Cell, load_cells

GRID = [math.exp(math.log(0.25) + (math.log(20.0) - math.log(0.25)) * i / 59) for i in range(60)]


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
    T_rps = fit_temperature(pairs).temperature
    T_rel = min(GRID, key=lambda T: _reliability(items, T))
    murphy = murphy_decomposition(items)
    return {
        "n": len(items),
        "argmax_acc": sum(it.argmax_agreement for it in items) / len(items),
        "mean_rps": sum(it.rps for it in items) / len(items),
        "mean_w1": sum(it.w1 for it in items) / len(items),
        "T_rps": T_rps, "T_rel": T_rel,
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
    """Q4: apply the open-derived constant ``T`` to closed-evaluator (Gemini/GPT)
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
        rows[fam] = row
    taus = [r["tau_oc"] for r in rows.values() if isinstance(r.get("tau_oc"), float) and not math.isnan(r["tau_oc"])]
    summary = {}
    if taus:
        taus_sorted = sorted(taus)
        summary = {"tau_oc_median": taus_sorted[len(taus) // 2], "tau_oc_min": min(taus),
                   "tau_oc_max": max(taus), "tau_oc_spread": max(taus) - min(taus), "n_families": len(taus)}
    return {"families": rows, "tau_oc_summary": summary}


if __name__ == "__main__":
    # Smoke: treat the existing Gemini/GPT reconciled predictions as one variant to
    # validate the analysis path end-to-end on real distributions (no API).
    cells = load_cells()
    m = json.loads((Path(__file__).resolve().parents[3]
                    / "judex-evaluator/runs/stage9-gemini-gpt-medium/metrics_report.json").read_text())
    preds = {it["item_label"]: it["prediction"]["probabilities"] for it in m["items"]}
    s = score_variant(preds, cells)
    print("SMOKE score_variant on gemini-gpt reconciled predictions:")
    for k, v in s.items():
        print(f"  {k}: {v}")
