#!/usr/bin/env python3
"""Estimand test: is the T*-vs-T_J gap a SIZE effect or a RATIO-vs-ABSOLUTE effect?

Motivating observation (2026-07-21): the full 24-doc stage-9 closed-pair sweep
fits a supervised T* = 2.545, against the frozen transferred constant
T_J = 1.153. Hypothesis under test (user): the open panel lacked giant-scale
families, and giants -- being closer in capacity to the closed pair -- would
have formed a second, higher tau_v cluster nearer T*.

Competing explanation (this script): tau_v and T* are DIFFERENT ESTIMANDS.

    tau_v          : post -> own pretrained base      (a RATIO)
    T_abs(post)    : post -> distributional GT        (ABSOLUTE, == T*'s estimand)
    T_abs(pre)     : base -> distributional GT        (ABSOLUTE)

They coincide only if bases are themselves GT-calibrated, i.e. T_abs(pre) ~ 1.
If instead T_abs(pre) >> 1, then a family can be barely sharper than its base
(tau_v ~ 1) while still needing a large absolute correction -- with no size
story required.

Tests
  T1  absolute verbalized temperatures per panel leg vs GT (harvested from the
      Study B reports of record, re-verified here by recomputation), compared
      against T* = 2.545 and T_J = 1.153; doc-clustered bootstrap CIs on the
      post-leg absolute fits (the direct analog of the closed pair's T*).
  T2  size association: Spearman rho of parameter count (total and active)
      against tau_v and against T_abs(post).
  T3  within-lineage size ladder (Gemma 26B-A4B vs 31B), the only
      recipe-controlled size contrast on disk.

$0 -- analysis-only on existing artifacts. Report-only: r3 F1-F8 are frozen;
nothing here adopts anything.

Run from judex-calibration:
  JUDEX_UMBRELLA=/path/to/judex PYTHONPATH=src \
      python scripts/absolute_vs_ratio_estimand.py [--out runs/estimand_test]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from judex_calibration import aireg, study_a  # noqa: E402
from judex.calibration import fit_temperature  # noqa: E402
from judex.core.distributions import ComplianceDistribution  # noqa: E402

EPSILON = 0.005          # Study B analysis convention
T_J = 1.153              # r3 F2 (frozen)
BAND = (1.025, 1.380)    # r3 F3 (frozen)
T_STAR_SWEEP = 2.545     # observed supervised fit, full 24-doc closed-pair sweep
BOOT_SEED = 20260720
BOOT_N = 2000

# (key, run dir, label, total params B, active params B, in adoption panel)
FAMILIES = [
    ("gemma31", "study_b_gemma31_api", "Gemma-4-31B",        31.3,  31.3, True),
    ("qwen",    "study_b_qwen",        "Qwen3.5-35B-A3B",    35.0,   3.0, True),
    ("glm",     "study_b_glm",         "GLM-4.5",           355.0,  32.0, True),
    ("maverick","study_b_maverick",    "Llama-4-Maverick",  400.0,  17.0, True),
    ("llama31", "study_b_llama31",     "Llama-3.1-405B",    405.9, 405.9, False),
    ("gemma26", "study_b_gemma26_api", "Gemma-4-26B-A4B",    25.2,   3.8, False),
]


def floor_renorm(v, eps=EPSILON):
    w = [max(float(x), eps) for x in v]
    z = sum(w)
    return [x / z for x in w]


def load_leg(run_dir: Path, leg: str):
    """{item_label: floored compliance vector} for parse_ok cells."""
    p = run_dir / f"{leg}_verbalized.json"
    if not p.exists():
        return None
    recs = json.loads(p.read_text())
    out = {}
    for label, r in recs.items():
        if r.get("parse_ok") and r.get("compliance"):
            out[label] = floor_renorm(r["compliance"])
    return out or None


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def _pairs_by_doc(preds, cells):
    """[(document_id, (pred_dist, gt_dist))] — the exact objects score_variant fits on."""
    by_label = {c.item_label: c for c in cells}
    out = []
    for label, probs in preds.items():
        c = by_label.get(label)
        if c is None:
            continue
        pred = study_a._pred_dist(probs, c.gt_labels)
        gt = ComplianceDistribution.from_values(c.gt_probs, c.gt_labels)
        out.append((c.document_id, (pred, gt)))
    return out


def boot_T_abs(preds, cells, n=BOOT_N, seed=BOOT_SEED):
    """Doc-clustered bootstrap CI on the absolute T_rps fit (leg -> GT).

    Resamples DOCUMENTS with replacement (the real cluster: 24 docs x 5
    Articles), refitting the same RPS objective on the resampled pair list.
    """
    tagged = _pairs_by_doc(preds, cells)
    by_doc = {}
    for doc, pair in tagged:
        by_doc.setdefault(doc, []).append(pair)
    docs = sorted(by_doc)
    if len(docs) < 3:
        return None
    rng = random.Random(seed)
    fits = []
    for _ in range(n):
        pairs = []
        for _ in docs:
            pairs.extend(by_doc[docs[rng.randrange(len(docs))]])
        if not pairs:
            continue
        fits.append(fit_temperature(pairs, bounds=study_a.T_BOUNDS).temperature)
    if len(fits) < 100:
        return None
    fits.sort()
    return {"lo": fits[int(0.025 * len(fits))],
            "hi": fits[int(0.975 * len(fits)) - 1],
            "median": statistics.median(fits),
            "n_docs": len(docs), "n_reps": len(fits)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/estimand_test")
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()

    cells = aireg.load_cells()
    runs = REPO / "runs"
    report = {
        "frozen": {"T_J": T_J, "band": list(BAND), "epsilon": EPSILON,
                   "T_bounds": list(study_a.T_BOUNDS)},
        "observed_closed_sweep_T_star": T_STAR_SWEEP,
        "families": {},
    }

    for key, d, label, tot, act, panel in FAMILIES:
        rd = runs / d
        if not rd.exists():
            continue
        pre, post = load_leg(rd, "pre"), load_leg(rd, "post")
        if not pre or not post:
            continue
        entry = {"label": label, "params_total_B": tot, "params_active_B": act,
                 "in_adoption_panel": panel, "n_pre": len(pre), "n_post": len(post)}

        s_pre = study_a.score_variant(pre, cells)
        s_post = study_a.score_variant(post, cells)
        entry["T_abs_pre"] = s_pre["T_rps"]
        entry["T_abs_pre_saturated"] = s_pre["T_rps_saturated"]
        entry["T_abs_post"] = s_post["T_rps"]
        entry["T_abs_post_saturated"] = s_post["T_rps_saturated"]
        entry["argmax_pre"] = s_pre["argmax_acc"]
        entry["argmax_post"] = s_post["argmax_acc"]
        entry["entropy_pre"] = s_pre["mean_pred_norm_entropy"]
        entry["entropy_post"] = s_post["mean_pred_norm_entropy"]

        # ratio estimand, recomputed on the same floored vectors
        entry["tau_v_recomputed"] = study_a.fit_tau_oc(post, pre, cells)

        # the algebraic decomposition the hypothesis turns on
        entry["ratio_T_abs_post_over_tau_v"] = (
            entry["T_abs_post"] / entry["tau_v_recomputed"]
            if entry["tau_v_recomputed"] else float("nan"))

        if not args.no_bootstrap:
            entry["T_abs_post_boot"] = boot_T_abs(post, cells)

        report["families"][key] = entry

    # ---- T2: size association
    panel = [(k, v) for k, v in report["families"].items() if v["in_adoption_panel"]]
    allf = list(report["families"].items())
    def assoc(sel, xkey, ykey):
        xs = [v[xkey] for _, v in sel]
        ys = [v[ykey] for _, v in sel]
        return {"n": len(sel), "spearman_rho": spearman(xs, ys),
                "x": dict((k, v[xkey]) for k, v in sel),
                "y": dict((k, v[ykey]) for k, v in sel)}
    report["size_association"] = {
        "panel_total_vs_tau_v":  assoc(panel, "params_total_B", "tau_v_recomputed"),
        "panel_total_vs_T_abs_post": assoc(panel, "params_total_B", "T_abs_post"),
        "panel_active_vs_tau_v": assoc(panel, "params_active_B", "tau_v_recomputed"),
        "all_total_vs_tau_v":    assoc(allf, "params_total_B", "tau_v_recomputed"),
        "all_total_vs_T_abs_post": assoc(allf, "params_total_B", "T_abs_post"),
    }

    # ---- T3: within-lineage ladder (Gemma), recipe controlled
    fams = report["families"]
    if "gemma26" in fams and "gemma31" in fams:
        a, b = fams["gemma26"], fams["gemma31"]
        report["size_ladder_gemma"] = {
            "smaller": {"label": a["label"], "params_total_B": a["params_total_B"],
                        "tau_v": a["tau_v_recomputed"], "T_abs_post": a["T_abs_post"],
                        "T_abs_pre": a["T_abs_pre"]},
            "larger": {"label": b["label"], "params_total_B": b["params_total_B"],
                       "tau_v": b["tau_v_recomputed"], "T_abs_post": b["T_abs_post"],
                       "T_abs_pre": b["T_abs_pre"]},
            "note": "gemma26 is capability-gate FAIL (weak reference) — ladder is directional only",
        }

    # ---- headline summary
    pT = [v["T_abs_post"] for _, v in panel]
    preT = [v["T_abs_pre"] for _, v in panel]
    tv = [v["tau_v_recomputed"] for _, v in panel]
    report["summary"] = {
        "panel_T_abs_post": {"min": min(pT), "max": max(pT), "median": statistics.median(pT)},
        "panel_T_abs_pre": {"min": min(preT), "max": max(preT), "median": statistics.median(preT)},
        "panel_tau_v": {"min": min(tv), "max": max(tv), "median": statistics.median(tv)},
        "T_star_sweep_inside_panel_T_abs_post_range": min(pT) <= T_STAR_SWEEP <= max(pT),
        "T_J_inside_panel_T_abs_post_range": min(pT) <= T_J <= max(pT),
        "f5_ln_ratio_T_star_vs_T_J": abs(math.log(T_STAR_SWEEP / T_J)),
        "f5_threshold_ln2": math.log(2),
        "f5_concurrent": abs(math.log(T_STAR_SWEEP / T_J)) <= math.log(2),
    }

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "estimand_test.json").write_text(json.dumps(report, indent=2))

    # ---- console
    print(f"\n{'family':22} {'tot B':>7} {'tau_v':>7} {'T_abs(pre)':>11} {'T_abs(post)':>12} {'panel':>6}")
    for k, v in report["families"].items():
        print(f"{v['label']:22} {v['params_total_B']:7.1f} {v['tau_v_recomputed']:7.3f} "
              f"{v['T_abs_pre']:11.3f} {v['T_abs_post']:12.3f} {str(v['in_adoption_panel']):>6}")
    s = report["summary"]
    print(f"\npanel T_abs(post) range [{s['panel_T_abs_post']['min']:.3f}, "
          f"{s['panel_T_abs_post']['max']:.3f}] median {s['panel_T_abs_post']['median']:.3f}")
    print(f"panel T_abs(pre)  range [{s['panel_T_abs_pre']['min']:.3f}, "
          f"{s['panel_T_abs_pre']['max']:.3f}] median {s['panel_T_abs_pre']['median']:.3f}")
    print(f"observed closed-pair T* = {T_STAR_SWEEP} -> inside panel T_abs(post) range: "
          f"{s['T_star_sweep_inside_panel_T_abs_post_range']}")
    print(f"T_J = {T_J} -> inside panel T_abs(post) range: {s['T_J_inside_panel_T_abs_post_range']}")
    print(f"F5 |ln(T*/T_J)| = {s['f5_ln_ratio_T_star_vs_T_J']:.4f} vs ln2 = {s['f5_threshold_ln2']:.4f} "
          f"-> concurrent: {s['f5_concurrent']}")
    print("\nsize association (Spearman rho):")
    for k, v in report["size_association"].items():
        print(f"  {k:28} n={v['n']}  rho={v['spearman_rho']:+.3f}")
    print(f"\nwrote {out / 'estimand_test.json'}")


if __name__ == "__main__":
    main()
