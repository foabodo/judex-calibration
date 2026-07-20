"""Study B analysis — tau_v (via study_a machinery) + the B-Q4 confidence-calibration fits.

B-Q4 (design doc §6, pre-registered): does the verbalized confidence_distribution track
realized per-cell error, and does a learned one-parameter temperature improve that
tracking out of sample?

- Event C per cell = argmax agreement of the parsed compliance distribution with GT.
- Fixed link (pre-registered): p_hat(C) = 0*p(low) + 0.5*p(medium) + 1*p(high) on the
  epsilon-floored confidence 3-vector.
- Learned map: one temperature T_c via inverse-softmax on the 3-vector, fit by
  minimizing the Brier score of C under the link; Study A's T_BOUNDS + log grid.
- Out-of-sample: doc-clustered LODO over the 24 documents; verdict "calibratable" iff
  mean OOS dBrier < 0 with a 95% doc-clustered bootstrap CI excluding 0 (seeded).
- Association diagnostic: Kendall tau_b between s = p_hat(C) and realized W1.
- Secondary: risk-coverage AURC (rank by s desc) before/after T_c.

Everything here consumes the {item_label: cell record} maps written by
elicit_verbalized.run_variant and the cells from aireg.load_cells() — never a
runs/ metrics_report.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

from .elicit_verbalized import floor_and_renormalize, EPSILON
from .study_a import T_BOUNDS, GRID, saturated

CONF_WEIGHTS = (0.0, 0.5, 1.0)  # low, medium, high — pre-registered fixed ordinal link


def temper(probs: Sequence[float], T: float, eps: float = EPSILON) -> List[float]:
    """softmax(ln p / T) on an epsilon-floored vector — the inverse-softmax temperature op."""
    p = floor_and_renormalize(probs, eps)
    logits = [math.log(x) / T for x in p]
    m = max(logits)
    ex = [math.exp(l - m) for l in logits]
    z = sum(ex)
    return [e / z for e in ex]


def link(conf3: Sequence[float]) -> float:
    """p_hat(C) under the pre-registered fixed link."""
    return sum(w * p for w, p in zip(CONF_WEIGHTS, conf3))


def _rows(recs: Dict[str, dict], cells) -> List[dict]:
    """Join parsed cells to GT: [{label, doc, conf, correct, w1}]. parse_ok only."""
    from judex.core.distributions import ComplianceDistribution
    from judex.core.metrics import wasserstein_1
    by_label = {c.item_label: c for c in cells}
    rows = []
    for label, r in recs.items():
        c = by_label.get(label)
        if c is None or not r.get("parse_ok"):
            continue
        comp = floor_and_renormalize(r["compliance"])
        pred = ComplianceDistribution.from_values(comp, c.gt_labels)
        gt = ComplianceDistribution.from_values(c.gt_probs, c.gt_labels)
        rows.append({
            "label": label, "doc": c.document_id,
            "conf": list(r["confidence"]),
            "correct": pred.argmax_index() == gt.argmax_index(),
            "w1": wasserstein_1(pred, gt),
        })
    return rows


def brier(rows: Sequence[dict], T: float, eps: float = EPSILON) -> float:
    """Mean Brier of the correctness event under the tempered link."""
    if not rows:
        return float("nan")
    tot = 0.0
    for r in rows:
        p = link(temper(r["conf"], T, eps))
        tot += (p - (1.0 if r["correct"] else 0.0)) ** 2
    return tot / len(rows)


def fit_Tc(rows: Sequence[dict], eps: float = EPSILON) -> float:
    """Grid-fit the confidence temperature (Study A's 60-pt log grid over T_BOUNDS)."""
    if not rows:
        return float("nan")
    return min(GRID, key=lambda T: brier(rows, T, eps))


def kendall_tau_b(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Kendall tau-b (tie-corrected); O(n^2), fine at n<=120."""
    n = len(xs)
    if n < 2:
        return float("nan")
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = xs[i] - xs[j], ys[i] - ys[j]
            if a == 0 and b == 0:
                tx += 1; ty += 1
            elif a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif a * b > 0:
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - tx) * (n0 - ty))
    return (conc - disc) / denom if denom > 0 else float("nan")


def aurc(rows: Sequence[dict], T: float | None, eps: float = EPSILON) -> float:
    """Area under the risk-coverage curve, ranking cells by s = link(tempered conf) desc.

    Risk = mean W1 of the covered prefix. Lower is better. T=None ranks by the raw link.
    """
    if not rows:
        return float("nan")
    scored = sorted(rows, key=lambda r: -link(temper(r["conf"], T, eps) if T else
                                              floor_and_renormalize(r["conf"], eps)))
    area = run = 0.0
    for k, r in enumerate(scored, 1):
        run += r["w1"]
        area += run / k
    return area / len(scored)


def lodo(rows: Sequence[dict], eps: float = EPSILON, seed: int = 20260720,
         n_boot: int = 2000) -> dict:
    """Doc-clustered leave-one-doc-out for the T_c fit (the pre-registered OOS check).

    Per held-out doc: fit T_c on the remaining docs, record the held-out mean Brier
    delta (tempered minus untempered, T=1). Verdict "calibratable" iff the mean OOS
    delta < 0 AND the 95% doc-clustered bootstrap CI (resampling docs, seeded) excludes 0.
    """
    docs = sorted({r["doc"] for r in rows})
    per_doc: Dict[str, dict] = {}
    for d in docs:
        train = [r for r in rows if r["doc"] != d]
        test = [r for r in rows if r["doc"] == d]
        if not train or not test:
            continue
        T = fit_Tc(train, eps)
        per_doc[d] = {"T_c": T, "T_c_saturated": saturated(T), "n": len(test),
                      "delta_brier": brier(test, T, eps) - brier(test, 1.0, eps)}
    deltas = [v["delta_brier"] for v in per_doc.values()]
    if not deltas:
        return {"n_docs": 0}
    mean_delta = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    return {
        "n_docs": len(per_doc), "per_doc": per_doc,
        "mean_oos_delta_brier": mean_delta,
        "ci95": [lo, hi], "bootstrap": {"n": n_boot, "seed": seed, "clustered_by": "document"},
        "calibratable": bool(mean_delta < 0 and hi < 0),
        "Tc_range_over_folds": [min(v["T_c"] for v in per_doc.values()),
                                max(v["T_c"] for v in per_doc.values())],
        "any_fold_saturated": any(v["T_c_saturated"] for v in per_doc.values()),
    }


def analyze_confidence(recs: Dict[str, dict], cells, eps: float = EPSILON) -> dict:
    """The full B-Q4 block for one leg's cell-record map."""
    rows = _rows(recs, cells)
    if not rows:
        return {"n": 0}
    T_full = fit_Tc(rows, eps)
    s_raw = [link(floor_and_renormalize(r["conf"], eps)) for r in rows]
    return {
        "n": len(rows), "epsilon": eps,
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "mean_link_phat": sum(s_raw) / len(rows),
        "brier_T1": brier(rows, 1.0, eps),
        "T_c_full_sample": T_full, "T_c_saturated": saturated(T_full),
        "brier_at_Tc": brier(rows, T_full, eps),
        "kendall_tau_b_s_vs_w1": kendall_tau_b(s_raw, [r["w1"] for r in rows]),
        "aurc_raw": aurc(rows, None, eps),
        "aurc_at_Tc": aurc(rows, T_full, eps),
        "lodo": lodo(rows, eps),
        "link_weights": list(CONF_WEIGHTS), "T_bounds": list(T_BOUNDS),
    }


def epsilon_sensitivity(pre_recs: Dict[str, dict], post_recs: Dict[str, dict], cells,
                        eps_list: Sequence[float] = (0.001, 0.005, 0.0125, 0.025)) -> dict:
    """The once-only pre-registered epsilon check on B1 data: tau_v + T_c across floors."""
    from .elicit_verbalized import compliance_view
    from . import study_a
    out = {}
    for eps in eps_list:
        pre = {k: floor_and_renormalize(v, eps) for k, v in compliance_view(pre_recs).items()}
        post = {k: floor_and_renormalize(v, eps) for k, v in compliance_view(post_recs).items()}
        tau_v = study_a.fit_tau_oc(post, pre, cells)
        post_rows = _rows(post_recs, cells)
        out[str(eps)] = {"tau_v": tau_v, "tau_v_saturated": study_a.saturated(tau_v),
                         "T_c_post": fit_Tc(post_rows, eps)}
    return out
