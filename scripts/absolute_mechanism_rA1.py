#!/usr/bin/env python3
"""rA1 confirmation driver — the absolute-mechanism protocol read.

Tests A1-A8 of docs/absolute_mechanism_protocol_rA1.md (frozen 2026-07-24,
commit 1f8856d) against the executed E6 on-pair sweep. Report-only: adopts
nothing, touches no config, evaluator seam stays `mode: noop`.

Ground truth is `aireg.load_cells()` and nothing else -- the AIReg-Bench
human reference reconciled over THREE legal-expert annotators, NOT the 7-seat
LLM exemplar panel in judex-corpus.

Instrument reuse (unforked): e6_r3_arms.load_closed for the deployed pair's
predictions, study_a.closed_side_check for the A7 acceptance read.

    PYTHONPATH=src python scripts/absolute_mechanism_rA1.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from judex_calibration import aireg, study_a  # noqa: E402
from judex.core.distributions import ComplianceDistribution  # noqa: E402
from judex.metrics_report import murphy_decomposition  # noqa: E402

# ---- reuse e6_r3_arms.load_closed unforked (file has a hyphen-free name but no package)
_spec = importlib.util.spec_from_file_location("e6_r3_arms", REPO / "scripts" / "e6_r3_arms.py")
_e6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e6)
load_closed = _e6.load_closed

# ---------------------------------------------------------------- frozen constants (rA1 §0)
A1_PANEL = ("qwen", "gemma31", "glm", "maverick")          # r3 F1, carried over verbatim
A1_SENSITIVITY = A1_PANEL + ("llama31", "gemma26")
A2_BINS = (3, 5, 10, 15, 20)
A2_MIN_RESOLUTION = 0.03
A2_ARGMAX_FLOOR = 1.0 / 3.0                                 # GT majority-class marginal
A3_MAX_RATIO = 2.0                                          # Study A Q3 factor-2 rule
A7_RESOLUTION_TOL = 0.10                                    # r3 F4, unchanged
A7_RPS_SLACK = 1e-9

EPSILON = 0.005                                             # Study B analysis convention

RUN_DIRS = {
    "qwen":     "study_b_qwen",
    "gemma31":  "study_b_gemma31_api",
    "glm":      "study_b_glm",
    "maverick": "study_b_maverick",
    "llama31":  "study_b_llama31",
    "gemma26":  "study_b_gemma26_api",
}
LABELS = {
    "qwen": "Qwen3.5-35B-A3B", "gemma31": "Gemma-4-31B", "glm": "GLM-4.5",
    "maverick": "Llama-4-Maverick", "llama31": "Llama-3.1-405B", "gemma26": "Gemma-4-26B-A4B",
}
CLOSED_RUN = Path("/Users/fabodo/Downloads/Projects/judex/judex-evaluator/runs/"
                  "stage9-onpair-e6-20260721")


def floor_renorm(v, eps=EPSILON):
    w = [max(float(x), eps) for x in v]
    z = sum(w)
    return [x / z for x in w]


def load_leg(run_dir: Path, leg: str):
    p = run_dir / f"{leg}_verbalized.json"
    if not p.exists():
        return None
    recs = json.loads(p.read_text())
    out = {lb: floor_renorm(r["compliance"]) for lb, r in recs.items()
           if r.get("parse_ok") and r.get("compliance")}
    return out or None


def _items(preds, cells):
    by = {c.item_label: c for c in cells}
    return [study_a._metric_item(
                lb,
                ComplianceDistribution.from_values(list(preds[lb]), by[lb].gt_labels),
                ComplianceDistribution.from_values(list(by[lb].gt_probs), by[lb].gt_labels))
            for lb in preds if lb in by]


def panel_side(cells, keys):
    """A2/A3/A4/A5/A6 on the open panel's verbalized POST legs."""
    fam = {}
    for k in keys:
        post = load_leg(REPO / "runs" / RUN_DIRS[k], "post")
        if not post:
            continue
        s = study_a.score_variant(post, cells)
        its = _items(post, cells)
        res = {}
        for b in A2_BINS:
            md = murphy_decomposition(its, bins=b)
            res[b] = md.resolution if hasattr(md, "resolution") else md["resolution"]
        fam[k] = {
            "label": LABELS[k], "n": len(its),
            "T_abs_post_rps": s["T_rps"], "saturated": bool(s["T_rps_saturated"]),
            "argmax": s["argmax_acc"], "argmax_above_floor": s["argmax_acc"] > A2_ARGMAX_FLOOR,
            "resolution_by_bins": res, "resolution_min": min(res.values()),
        }
    T = [v["T_abs_post_rps"] for v in fam.values()]
    res_min = min(v["resolution_min"] for v in fam.values())
    ratio = max(T) / min(T)
    return {
        "families": fam,
        "A2_min_resolution": res_min,
        "A2_pass": bool(res_min >= A2_MIN_RESOLUTION
                        and all(v["resolution_min"] > 0 for v in fam.values())),
        "A2_argmax_diagnostic_all_above_floor": all(v["argmax_above_floor"] for v in fam.values()),
        "A3_ratio_max_over_min": ratio,
        "A3_pass": bool(ratio <= A3_MAX_RATIO),
        "A4_T_A": statistics.median(T),
        "A5_band": [min(T), max(T)],
        "A6_pass": bool(not any(v["saturated"] for v in fam.values())
                        and study_a.T_BOUNDS[0] < statistics.median(T) < study_a.T_BOUNDS[1]),
    }


def a7_read(preds, cells, T):
    """r3 F4 acceptance, carried over unchanged."""
    chk = study_a.closed_side_check(preds, cells, T)
    res_un = chk["uncalibrated"]["resolution"]
    within = (abs(chk["resolution_change"]) <= A7_RESOLUTION_TOL * res_un) if res_un > 0 else False
    chk["resolution_within_tol"] = bool(within)
    chk["a7_pass"] = bool(chk["verdict_ok"] and within)
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/absolute_mechanism_rA1")
    args = ap.parse_args()

    cells = aireg.load_cells()
    report = {
        "protocol": "rA1 (docs/absolute_mechanism_protocol_rA1.md, frozen 2026-07-24)",
        "reference": ("AIReg-Bench GT via aireg.load_cells() -- MG-MFRM reconciliation over "
                      "THREE human legal-expert annotators, readout tau=0.675. NOT the 7-seat "
                      "LLM exemplar panel."),
        "n_cells": len(cells),
        "frozen": {"A1_panel": list(A1_PANEL), "A2_min_resolution": A2_MIN_RESOLUTION,
                   "A3_max_ratio": A3_MAX_RATIO, "A7_resolution_tol": A7_RESOLUTION_TOL,
                   "epsilon": EPSILON, "T_bounds": list(study_a.T_BOUNDS),
                   "murphy_bins": study_a.MURPHY_BINS},
    }

    panel = panel_side(cells, A1_PANEL)
    report["panel"] = panel
    report["panel_sensitivity_all_six"] = panel_side(cells, A1_SENSITIVITY)

    T_A = panel["A4_T_A"]
    preds = load_closed(CLOSED_RUN)
    report["closed"] = {"run": str(CLOSED_RUN), "n_predictions": len(preds)}

    report["A7_at_T_A"] = a7_read(preds, cells, T_A)
    report["A7_at_band_low"] = a7_read(preds, cells, panel["A5_band"][0])
    report["A7_at_band_high"] = a7_read(preds, cells, panel["A5_band"][1])
    # r3's T_J, same instrument, for the side-by-side the paper needs
    report["reference_read_at_T_J_1.153"] = a7_read(preds, cells, 1.153)

    # ---- SUPPLEMENTARY (not part of the rA1 verdict): held-out document split.
    # rA1 sec.3 names this as the $0 out-of-sample option. It removes item-level
    # in-sampleness: T_A is re-derived from the panel using ONLY fold-A documents,
    # then the closed A7 read runs on fold-B cells only, and vice versa. The
    # reference and rubric stay shared, so this is the weakest of the three options.
    docs = sorted({c.document_id for c in cells})
    folds = {"A": set(docs[0::2]), "B": set(docs[1::2])}
    split = {"n_documents": len(docs), "folds": {k: sorted(v) for k, v in folds.items()}}
    for fit_on, read_on in (("A", "B"), ("B", "A")):
        fit_cells = [c for c in cells if c.document_id in folds[fit_on]]
        read_cells = [c for c in cells if c.document_id in folds[read_on]]
        p = panel_side(fit_cells, A1_PANEL)
        chk = a7_read(preds, read_cells, p["A4_T_A"])
        split[f"fit_{fit_on}_read_{read_on}"] = {
            "T_A_from_fold": p["A4_T_A"], "band": p["A5_band"],
            "A3_ratio": p["A3_ratio_max_over_min"], "A3_pass": p["A3_pass"],
            "A2_pass": p["A2_pass"], "n_read_cells": chk["n"],
            "reliability_improvement": chk["reliability_improvement"],
            "rps_change": chk["rps_change"], "resolution_change": chk["resolution_change"],
            "resolution_within_tol": chk["resolution_within_tol"], "a7_pass": chk["a7_pass"],
        }
    split["both_folds_pass"] = bool(split["fit_A_read_B"]["a7_pass"]
                                    and split["fit_B_read_A"]["a7_pass"])
    report["supplementary_heldout_document_split"] = split

    verdict = bool(panel["A2_pass"] and panel["A3_pass"] and panel["A6_pass"]
                   and report["A7_at_T_A"]["a7_pass"])
    report["VERDICT"] = "CONFIRMED" if verdict else "NOT CONFIRMED"
    report["verdict_components"] = {
        "A2_capability_floor": panel["A2_pass"], "A3_clustering": panel["A3_pass"],
        "A6_no_saturation": panel["A6_pass"], "A7_closed_acceptance": report["A7_at_T_A"]["a7_pass"],
    }
    report["scope_reminder"] = (
        "Registered-form, NOT blind (rA1 sec.3): the analyst had prior exposure to a closed read "
        "before the freeze. In-sample w.r.t. task, documents and reference; out-of-sample only "
        "w.r.t. the model population. Never describe as pre-registered in the sense T_J was.")

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "absolute_mechanism_rA1.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\nrA1 panel   A2 res_min={panel['A2_min_resolution']:.4f} pass={panel['A2_pass']}  "
          f"A3 ratio={panel['A3_ratio_max_over_min']:.4f} pass={panel['A3_pass']}  "
          f"A6 pass={panel['A6_pass']}")
    print(f"rA1 constant T_A = {T_A:.4f}   band {panel['A5_band'][0]:.4f}-{panel['A5_band'][1]:.4f}")
    for name, key in (("T_A", "A7_at_T_A"), ("band_lo", "A7_at_band_low"),
                      ("band_hi", "A7_at_band_high"), ("T_J=1.153", "reference_read_at_T_J_1.153")):
        c = report[key]
        print(f"  {name:>10}: relGain {c['reliability_improvement']:+.5f}  dRPS {c['rps_change']:+.5f}  "
              f"dRes {c['resolution_change']:+.5f}  within_tol={c['resolution_within_tol']}  "
              f"pass={c['a7_pass']}")
    print("\n  supplementary held-out document split (not part of the verdict):")
    for k in ("fit_A_read_B", "fit_B_read_A"):
        s = report["supplementary_heldout_document_split"][k]
        print(f"    {k}: T_A={s['T_A_from_fold']:.4f}  n={s['n_read_cells']}  "
              f"relGain {s['reliability_improvement']:+.5f}  dRPS {s['rps_change']:+.5f}  "
              f"dRes {s['resolution_change']:+.5f}  pass={s['a7_pass']}")

    print(f"\nVERDICT: {report['VERDICT']}  {report['verdict_components']}")
    print(f"written: {out / 'absolute_mechanism_rA1.json'}")


if __name__ == "__main__":
    main()
