#!/usr/bin/env python3
"""Phase-1a/1b variant read: the estimands under the k=5 scaffold variants.

LABELED VARIANT COPY of ``scripts/absolute_vs_ratio_estimand.py``. That script is the
FROZEN rA1-adjacent driver whose family->dir map points at the shipped baseline legs and
whose numbers are the values of record; it is NEVER edited. This file exists so the
coverage-preserving scaffold variants (V1 ``alt_set``, V2 ``rev_order``) can be read on
the SAME machinery without touching it.

Differences from the original, and only these:
  * the family->dir map carries THREE dirs per family (baseline, k5v1, k5v2) instead of
    the six-family single-dir panel; Phase 1a supplied qwen + gemma31, Phase 1b extends
    it to the full adoption panel {qwen, gemma31, glm, maverick};
  * the T2 size-association and T3 gemma size-ladder blocks are dropped (they are
    panel-composition reads, meaningless on a scaffold perturbation);
  * the report adds the pre-specified per-leg deltas -- Delta argmax in percentage
    points, Delta ln tau_v, Delta ln T_abs(post) -- against the shipped k=5 baseline,
    plus the memo's bound criterion B = max{|Delta ln tau_v|, |Delta ln T_abs|} compared
    against the leave-one-family-out worst case 0.191 and the tau_v grid step
    ln 1.0771 = 0.0743;
  * PHASE 1b: a PANEL-LEVEL recomputation. For each scaffold (baseline, alt_set,
    rev_order) the four-family panel is rebuilt from scratch and every published panel
    quantity is recomputed on it -- the tau_v set with its max/min cluster ratio and the
    <=2 rule, the band, the interpolated panel median (the T_J analog), the T_abs(post)
    set with its own cluster ratio and median (the T_A analog), and the
    leave-one-family-out transfer error max |ln(transferred/own)| under the exact
    ``verbalized_reframe_r0.lofo_transfer`` convention (transferred = median of the other
    three). A panel is only computed when all four families are present for that
    scaffold; otherwise it is reported as unavailable with the missing families named.

This is a variant read, NOT the frozen rA1 driver. Nothing here adopts anything; r3
F1-F8 and rA1 A1-A8 are frozen and untouched.

$0 -- analysis-only on existing artifacts.

Run from judex-calibration:
  python scripts/absolute_estimand_k5variants.py [--out runs/estimand_k5variants]
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

from judex_calibration import aireg, study_a  # noqa: E402

EPSILON = 0.005            # Study B analysis convention (unchanged)
GRID_STEP_LN = math.log(1.0771)   # one tau_v grid step, 0.0743 (memo A.2)
LOFO_WORST = 0.191                # leave-one-family-out worst case (memo A.3)

# Values of record, quoted from the memo's A.2 baseline table. Recomputed below from the
# legacy dirs as a cross-check; a mismatch is reported, never silently absorbed.
# NOTE for gemma31: 1.0252 / 2.2417 were fitted on the OpenRouter-transport post leg. Under
# the 2026-08-09 quantization-matching directive that leg is no longer the reference for a
# scaffold delta -- the matched-transport baseline below is. The legacy pair is still
# recomputed and printed, as context and as an instrument check, never as the denominator.
BASELINE_OF_RECORD = {
    "qwen":     {"tau_v": 1.281052072726771,  "T_abs_post": 2.6365430791635713,
                 "is_delta_reference": True,  "dir": "study_b_qwen"},
    "gemma31":  {"tau_v": 1.0251785151221313, "T_abs_post": 2.2416624655619084,
                 "is_delta_reference": False, "dir": "study_b_gemma31_api"},
    "glm":      {"tau_v": 1.0251785151221313, "T_abs_post": 2.4639456555374104,
                 "is_delta_reference": True,  "dir": "study_b_glm"},
    "maverick": {"tau_v": 1.379820350674421,  "T_abs_post": 2.0350699223438102,
                 "is_delta_reference": True,  "dir": "study_b_maverick"},
}

# ---------------------------------------------------------------- panel constants
# The adoption panel of record (r3 F1, carried into rA1 A1 verbatim). Panel order is
# irrelevant to every statistic below; the tuple fixes the reporting order only.
ADOPTION_PANEL = ("qwen", "gemma31", "glm", "maverick")
CLUSTER_RULE_MAX_RATIO = 2.0       # the <=2 rule (study_a.CLUSTER_RULE_MAX_RATIO)

# Published panel quantities, recomputed below from the baseline run dirs and asserted
# against these. Every one of them reproduces to floating-point identity from
# runs/study_b_{qwen,gemma31_api,glm,maverick}.
PANEL_OF_RECORD = {
    "tau_v_cluster_ratio":     1.345931786826454,     # r3 F1, printed as 1.346
    "tau_v_band":             [1.0251785151221313, 1.379820350674421],   # [1.03, 1.38]
    "tau_v_median_interpolated": 1.1531152939244511,  # r3 F2 T_J, printed as 1.153
    "tau_v_lofo_max_abs_ln":   0.29708655150331403,
    "T_abs_cluster_ratio":     1.295554049625498,     # rA1 A3, printed as 1.296
    "T_abs_band":             [2.0350699223438102, 2.6365430791635713],
    "T_abs_median_interpolated": 2.3528040605496594,  # rA1 A4 T_A, printed as 2.353
    "T_abs_lofo_max_abs_ln":   0.1912338118671258,    # the published <=0.191
}

# family -> {variant: run dir}.
#
# The delta reference must differ from the variant ONLY in the scaffold. For qwen that is
# automatic: both legs of every dir are bf16 vLLM raw completions on our own box, so the
# shipped baseline dir is already matched.
#
# gemma31 is not automatic. Its post twin cannot be collected on raw completions at all
# (the recorded greedy-repetition collapse, 30.8% contract-complete in study_b_gemma31),
# so the post leg needs a chat template. The shipped chat post came from OpenRouter, i.e.
# a different provider stack from the bf16 base leg. Under the quantization-matching
# directive the post twin is now served from OUR box, bf16, through vLLM's chat endpoint,
# and a matched-transport BASELINE post leg is collected there too. Every gemma31 delta
# below is therefore (vast bf16 raw pre + our-box bf16 chat post) on both sides, with the
# scaffold as the only difference.
FAMILIES = {
    "qwen": {
        "label": "Qwen3.5-35B-A3B (bf16 vLLM raw completions, both legs)",
        "dirs": {"baseline": "study_b_qwen",
                 "alt_set":  "study_b_qwen_k5v1",
                 "rev_order": "study_b_qwen_k5v2"},
    },
    "gemma31": {
        "label": "Gemma-4-31B (bf16: vast raw pre x self-hosted vLLM chat post)",
        "dirs": {"baseline": "study_b_gemma31_vllmchat_base",
                 "alt_set":  "study_b_gemma31_k5v1",
                 "rev_order": "study_b_gemma31_k5v2"},
    },
    # Phase 1b. Both families were clean on raw /v1/completions for BOTH legs in the
    # shipped campaign (GLM 96.7/92.5, Maverick 98.3/98.3), so -- unlike gemma31 -- their
    # shipped baseline dirs are already serving-matched to the variants and need no
    # re-collection: bf16 self-hosted vLLM raw completions on both sides, 8xH200.
    "glm": {
        "label": "GLM-4.5 355B-A32B (bf16 vLLM raw completions, both legs)",
        "dirs": {"baseline": "study_b_glm",
                 "alt_set":  "study_b_glm_k5v1",
                 "rev_order": "study_b_glm_k5v2"},
    },
    "maverick": {
        "label": "Llama-4-Maverick 17B-128E (bf16 vLLM raw completions, both legs)",
        "dirs": {"baseline": "study_b_maverick",
                 "alt_set":  "study_b_maverick_k5v1",
                 "rev_order": "study_b_maverick_k5v2"},
    },
}

# Which dir supplies each family's PANEL membership under the baseline scaffold.
#
# Two baseline panels are reported and they differ in one cell. The PUBLISHED panel took
# gemma31's post leg from OpenRouter (study_b_gemma31_api) -- that is the pairing every
# printed constant was fitted on. The MATCHED-TRANSPORT panel takes it from the bf16
# vLLM chat baseline collected in Phase 1a (study_b_gemma31_vllmchat_base), which is the
# like-for-like denominator for a scaffold delta (same box, same dtype, same transport as
# the gemma31 variants). The two differ by ln 0.0006 in T_abs and land on the identical
# tau_v grid point, so no panel statistic turns on the choice -- but the delta reference
# is the matched-transport panel, and the published panel is carried alongside so the
# reader can see the published numbers reproduce.
PANEL_BASELINE_DIRS = {
    "published":         {"qwen": "study_b_qwen", "gemma31": "study_b_gemma31_api",
                          "glm": "study_b_glm", "maverick": "study_b_maverick"},
    "matched_transport": {"qwen": "study_b_qwen",
                          "gemma31": "study_b_gemma31_vllmchat_base",
                          "glm": "study_b_glm", "maverick": "study_b_maverick"},
}

# Legacy / annex dirs: measured, reported, never inputs to the primary read.
LEGACY_DIRS = {
    "gemma31_openrouter_baseline": "study_b_gemma31_api",
    "gemma31_openrouter_alt_set_partial": "study_b_gemma31_api_k5v1",
}

# Transport diagnostic (report-only): the SAME scaffold and the SAME weights, served two
# ways -- our bf16 vLLM chat endpoint vs OpenRouter's bf16/fp16 pool.
TRANSPORT_DIAGNOSTIC = ("study_b_gemma31_vllmchat_base", "study_b_gemma31_api")


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
    out = {label: floor_renorm(r["compliance"])
           for label, r in recs.items()
           if r.get("parse_ok") and r.get("compliance")}
    return out or None


def leg_meta(run_dir: Path, leg: str):
    p = run_dir / f"{leg}_verbalized.meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def contract_rate(run_dir: Path, leg: str, n_cells: int):
    """B-Q1 gate numbers straight off the stored records (no report file needed)."""
    p = run_dir / f"{leg}_verbalized.json"
    if not p.exists():
        return None
    recs = json.loads(p.read_text())
    n = len(recs)
    parsed = [r for r in recs.values() if r.get("parse_ok")]
    complete = [r for r in parsed if r.get("contract_complete")]
    return {"n_cells": n_cells, "n_elicited": n, "n_parsed": len(parsed),
            "parse_rate": len(parsed) / n if n else 0.0,
            "n_contract_complete": len(complete),
            "contract_complete_rate": len(complete) / n if n else 0.0,
            "parse_gate_ok": (len(complete) / n if n else 0.0) >= 0.90}


def measure(run_dir: Path, cells) -> dict:
    pre, post = load_leg(run_dir, "pre"), load_leg(run_dir, "post")
    if not pre or not post:
        return {"available": False, "dir": run_dir.name,
                "pre_present": bool(pre), "post_present": bool(post)}
    s_pre = study_a.score_variant(pre, cells)
    s_post = study_a.score_variant(post, cells)
    return {
        "available": True, "dir": run_dir.name,
        "n_pre": len(pre), "n_post": len(post),
        "gate_pre": contract_rate(run_dir, "pre", len(cells)),
        "gate_post": contract_rate(run_dir, "post", len(cells)),
        "meta_pre": leg_meta(run_dir, "pre"), "meta_post": leg_meta(run_dir, "post"),
        "argmax_pre": s_pre["argmax_acc"], "argmax_post": s_post["argmax_acc"],
        "entropy_pre": s_pre["mean_pred_norm_entropy"],
        "entropy_post": s_post["mean_pred_norm_entropy"],
        "T_abs_pre": s_pre["T_rps"], "T_abs_pre_saturated": s_pre["T_rps_saturated"],
        "T_abs_post": s_post["T_rps"], "T_abs_post_saturated": s_post["T_rps_saturated"],
        "tau_v": study_a.fit_tau_oc(post, pre, cells),
    }


def transport_diagnostic(dir_a: Path, dir_b: Path, cells) -> dict:
    """Same scaffold, same weights, two serving transports -- a standalone datum.

    This is NOT a scaffold delta and is never fed into B. It answers a different
    question: how much does the post twin's distribution move when only the serving
    stack changes? Reported on the cell intersection, plus each side's own gate.
    """
    a, b = load_leg(dir_a, "post"), load_leg(dir_b, "post")
    if not a or not b:
        return {"available": False, "a": dir_a.name, "b": dir_b.name}
    sa, sb = study_a.score_variant(a, cells), study_a.score_variant(b, cells)
    shared = sorted(set(a) & set(b))
    l1 = [sum(abs(x - y) for x, y in zip(a[k], b[k])) for k in shared]
    argmax_agree = sum(a[k].index(max(a[k])) == b[k].index(max(b[k])) for k in shared)
    return {
        "available": True, "a": dir_a.name, "b": dir_b.name,
        "gate_a": contract_rate(dir_a, "post", len(cells)),
        "gate_b": contract_rate(dir_b, "post", len(cells)),
        "n_a": len(a), "n_b": len(b), "n_shared": len(shared),
        "argmax_a": sa["argmax_acc"], "argmax_b": sb["argmax_acc"],
        "d_argmax_pp": 100 * (sa["argmax_acc"] - sb["argmax_acc"]),
        "T_abs_post_a": sa["T_rps"], "T_abs_post_b": sb["T_rps"],
        "d_ln_T_abs_post": dln(sa["T_rps"], sb["T_rps"]),
        "entropy_a": sa["mean_pred_norm_entropy"], "entropy_b": sb["mean_pred_norm_entropy"],
        "mean_L1_between_posts": (sum(l1) / len(l1)) if l1 else None,
        "max_L1_between_posts": max(l1) if l1 else None,
        "cellwise_argmax_agreement": (argmax_agree / len(shared)) if shared else None,
        "note": "TRANSPORT sensitivity (serving stack), NOT scaffold sensitivity. "
                "Report-only; excluded from the bound B.",
    }


def dln(new, old):
    if not new or not old or new <= 0 or old <= 0:
        return None
    return math.log(new / old)


# ------------------------------------------------------------------- panel read (1b)
def _estimand_panel(values: dict) -> dict:
    """Every published panel statistic for ONE estimand over the four-family panel.

    ``values`` = {family: positive float}. Contains the max/min cluster ratio and the
    <=2 rule, the band, both median conventions, and the leave-one-family-out transfer
    error under the exact ``verbalized_reframe_r0.lofo_transfer`` convention
    (transferred = median of the OTHER three, log ratio taken transferred/own).
    """
    ordered = sorted(values)
    vals = sorted(values.values())
    lofo = {}
    for f in ordered:
        own = values[f]
        transfer = statistics.median([values[g] for g in ordered if g != f])
        lofo[f] = {"own": own, "transfer_lofo_median": transfer,
                   "log_ratio_transfer_vs_own": math.log(transfer / own)}
    worst = max(lofo, key=lambda f: abs(lofo[f]["log_ratio_transfer_vs_own"]))
    return {
        "values": {f: values[f] for f in ordered},
        "band": [vals[0], vals[-1]],
        "cluster_ratio": vals[-1] / vals[0],
        "cluster_rule_max_ratio": CLUSTER_RULE_MAX_RATIO,
        "cluster_rule_ok": (vals[-1] / vals[0]) <= CLUSTER_RULE_MAX_RATIO,
        "median_interpolated": statistics.median(vals),
        "median_upper_study_a_convention": vals[len(vals) // 2],
        "lofo": lofo,
        "lofo_max_abs_log_ratio": abs(lofo[worst]["log_ratio_transfer_vs_own"]),
        "lofo_max_source": worst,
    }


def panel_from_measurements(measured: dict, label: str) -> dict:
    """Rebuild the four-family panel from per-family measurement dicts.

    ``measured`` = {family: measure() result}. A panel is only computed when ALL of
    ADOPTION_PANEL is present and available; a partial panel is meaningless (every
    statistic here -- ratio, band, median, LOFO -- is defined over the whole panel), so
    it is reported unavailable with the missing families named rather than computed on a
    subset and silently mislabelled.
    """
    missing = [f for f in ADOPTION_PANEL
               if not (measured.get(f) or {}).get("available")]
    if missing:
        return {"available": False, "label": label, "missing_families": missing,
                "present_families": [f for f in ADOPTION_PANEL if f not in missing]}
    gates = {}
    for f in ADOPTION_PANEL:
        m = measured[f]
        gates[f] = {"dir": m["dir"],
                    "pre_gate_ok": m["gate_pre"]["parse_gate_ok"],
                    "post_gate_ok": m["gate_post"]["parse_gate_ok"],
                    "pre_contract_complete_rate": m["gate_pre"]["contract_complete_rate"],
                    "post_contract_complete_rate": m["gate_post"]["contract_complete_rate"]}
    return {
        "available": True, "label": label,
        "families": list(ADOPTION_PANEL),
        "dirs": {f: measured[f]["dir"] for f in ADOPTION_PANEL},
        "b_q1_gates": gates,
        "all_legs_pass_b_q1": all(g["pre_gate_ok"] and g["post_gate_ok"]
                                  for g in gates.values()),
        "tau_v": _estimand_panel({f: measured[f]["tau_v"] for f in ADOPTION_PANEL}),
        "T_abs_post": _estimand_panel({f: measured[f]["T_abs_post"]
                                       for f in ADOPTION_PANEL}),
    }


def panel_delta(variant_panel: dict, base_panel: dict) -> dict:
    """Movement of every published panel quantity under a scaffold perturbation."""
    if not (variant_panel.get("available") and base_panel.get("available")):
        return {"available": False}
    out = {"available": True, "reference": base_panel["label"]}
    for est in ("tau_v", "T_abs_post"):
        v, b = variant_panel[est], base_panel[est]
        out[est] = {
            "cluster_ratio": v["cluster_ratio"],
            "cluster_ratio_baseline": b["cluster_ratio"],
            "d_cluster_ratio": v["cluster_ratio"] - b["cluster_ratio"],
            "cluster_rule_ok": v["cluster_rule_ok"],
            "cluster_rule_ok_baseline": b["cluster_rule_ok"],
            "cluster_rule_survives": v["cluster_rule_ok"],
            "band": v["band"], "band_baseline": b["band"],
            "d_ln_band_low": dln(v["band"][0], b["band"][0]),
            "d_ln_band_high": dln(v["band"][1], b["band"][1]),
            "median_interpolated": v["median_interpolated"],
            "median_interpolated_baseline": b["median_interpolated"],
            "d_ln_median_interpolated": dln(v["median_interpolated"],
                                            b["median_interpolated"]),
            "d_ln_median_in_grid_steps": (
                abs(dln(v["median_interpolated"], b["median_interpolated"])) / GRID_STEP_LN
                if est == "tau_v" else None),
            "lofo_max_abs_log_ratio": v["lofo_max_abs_log_ratio"],
            "lofo_max_abs_log_ratio_baseline": b["lofo_max_abs_log_ratio"],
            "d_lofo_max_abs_log_ratio": (v["lofo_max_abs_log_ratio"]
                                         - b["lofo_max_abs_log_ratio"]),
            "lofo_max_source": v["lofo_max_source"],
        }
    return out


def panel_of_record_check(published_panel: dict) -> dict:
    """The published panel constants must fall out of the baseline dirs exactly."""
    if not published_panel.get("available"):
        return {"available": False}
    got = {
        "tau_v_cluster_ratio": published_panel["tau_v"]["cluster_ratio"],
        "tau_v_band": published_panel["tau_v"]["band"],
        "tau_v_median_interpolated": published_panel["tau_v"]["median_interpolated"],
        "tau_v_lofo_max_abs_ln": published_panel["tau_v"]["lofo_max_abs_log_ratio"],
        "T_abs_cluster_ratio": published_panel["T_abs_post"]["cluster_ratio"],
        "T_abs_band": published_panel["T_abs_post"]["band"],
        "T_abs_median_interpolated": published_panel["T_abs_post"]["median_interpolated"],
        "T_abs_lofo_max_abs_ln": published_panel["T_abs_post"]["lofo_max_abs_log_ratio"],
    }
    checks = {}
    for k, want in PANEL_OF_RECORD.items():
        have = got[k]
        if isinstance(want, list):
            ok = all(abs(a - b) < 1e-9 for a, b in zip(have, want))
        else:
            ok = abs(have - want) < 1e-9
        checks[k] = {"recomputed": have, "of_record": want, "match": ok}
    return {"available": True, "checks": checks,
            "all_match": all(c["match"] for c in checks.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/estimand_k5variants")
    args = ap.parse_args()

    cells = aireg.load_cells()
    runs = REPO / "runs"
    report = {
        "what": "Phase-1a k=5 coverage-preserving scaffold-variant read "
                "(labeled variant copy of absolute_vs_ratio_estimand.py; the frozen "
                "original is untouched)",
        "epsilon": EPSILON, "T_bounds": list(study_a.T_BOUNDS),
        "grid_step_ln": GRID_STEP_LN, "lofo_worst_case": LOFO_WORST,
        "families": {}, "deltas": {},
    }

    for fam, spec in FAMILIES.items():
        report["families"][fam] = {"label": spec["label"], "variants": {}}
        for variant, d in spec["dirs"].items():
            report["families"][fam]["variants"][variant] = measure(runs / d, cells)

    # legacy / annex dirs, measured but never inputs to the primary read
    report["legacy"] = {name: measure(runs / d, cells) for name, d in LEGACY_DIRS.items()}

    # transport diagnostic (report-only)
    report["transport_diagnostic"] = transport_diagnostic(
        runs / TRANSPORT_DIAGNOSTIC[0], runs / TRANSPORT_DIAGNOSTIC[1], cells)

    # values-of-record cross-check, against whichever dir produced them
    checks = {}
    for fam, rec in BASELINE_OF_RECORD.items():
        src = report["families"][fam]["variants"].get("baseline") \
            if rec["is_delta_reference"] else report["legacy"].get(
                "gemma31_openrouter_baseline")
        if src and src.get("available"):
            checks[fam] = {
                "source_dir": src["dir"], "is_delta_reference": rec["is_delta_reference"],
                "tau_v_recomputed": src["tau_v"], "tau_v_of_record": rec["tau_v"],
                "tau_v_match": abs(src["tau_v"] - rec["tau_v"]) < 1e-9,
                "T_abs_post_recomputed": src["T_abs_post"],
                "T_abs_post_of_record": rec["T_abs_post"],
                "T_abs_post_match": abs(src["T_abs_post"] - rec["T_abs_post"]) < 1e-6,
            }
    report["baseline_of_record_check"] = checks

    # ---- PANEL-LEVEL recomputation (Phase 1b, the reason 1b exists)
    #
    # Each scaffold gets its own four-family panel, rebuilt from the legs elicited under
    # that scaffold, and every published panel quantity is recomputed on it. The question
    # is not whether a single family's estimand moves (that is the per-leg delta above)
    # but whether the PANEL-LEVEL constants -- the <=2 clustering rule, the band, the
    # interpolated median that becomes T_J / T_A, and the LOFO transfer bound -- survive
    # the perturbation.
    panels = {}
    for name, dirs in PANEL_BASELINE_DIRS.items():
        measured = {f: measure(runs / d, cells) for f, d in dirs.items()}
        panels[f"baseline_{name}"] = panel_from_measurements(
            measured, f"baseline scaffold, {name} pairing")
    for variant in ("alt_set", "rev_order"):
        measured = {f: report["families"].get(f, {}).get("variants", {}).get(variant, {})
                    for f in ADOPTION_PANEL}
        panels[variant] = panel_from_measurements(measured, f"{variant} scaffold")
    report["panels"] = panels

    # published constants must fall out of the published-pairing baseline panel exactly
    report["panel_of_record_check"] = panel_of_record_check(
        panels.get("baseline_published", {}))

    # variant panels are compared against the MATCHED-TRANSPORT baseline panel (the
    # like-for-like denominator); the published panel is reported for reference only.
    report["panel_deltas"] = {
        v: panel_delta(panels.get(v, {}), panels.get("baseline_matched_transport", {}))
        for v in ("alt_set", "rev_order")
    }

    # ---- the three pre-specified per-leg deltas (memo A.2)
    bound_terms = []
    argmax_terms = []
    for fam, spec in report["families"].items():
        base = spec["variants"].get("baseline", {})
        if not base.get("available"):
            continue
        report["deltas"][fam] = {}
        for variant in ("alt_set", "rev_order"):
            v = spec["variants"].get(variant, {})
            if not v.get("available"):
                report["deltas"][fam][variant] = {"available": False}
                continue
            d = {
                "available": True,
                "d_argmax_pre_pp": 100 * (v["argmax_pre"] - base["argmax_pre"]),
                "d_argmax_post_pp": 100 * (v["argmax_post"] - base["argmax_post"]),
                "tau_v": v["tau_v"], "tau_v_baseline": base["tau_v"],
                "d_ln_tau_v": dln(v["tau_v"], base["tau_v"]),
                "T_abs_post": v["T_abs_post"], "T_abs_post_baseline": base["T_abs_post"],
                "d_ln_T_abs_post": dln(v["T_abs_post"], base["T_abs_post"]),
                "T_abs_pre": v["T_abs_pre"], "T_abs_pre_baseline": base["T_abs_pre"],
                "d_ln_T_abs_pre": dln(v["T_abs_pre"], base["T_abs_pre"]),
            }
            d["d_ln_tau_v_grid_steps"] = (abs(d["d_ln_tau_v"]) / GRID_STEP_LN
                                          if d["d_ln_tau_v"] is not None else None)
            d["tau_v_movement_resolvable"] = (d["d_ln_tau_v"] is not None
                                              and abs(d["d_ln_tau_v"]) >= GRID_STEP_LN)
            report["deltas"][fam][variant] = d
            for key in ("d_ln_tau_v", "d_ln_T_abs_post"):
                if d[key] is not None:
                    bound_terms.append((abs(d[key]), f"{fam}/{variant}/{key}"))
            for key in ("d_argmax_pre_pp", "d_argmax_post_pp"):
                argmax_terms.append((abs(d[key]), f"{fam}/{variant}/{key}"))

    def classify(B, A):
        if B < GRID_STEP_LN and A < 3.0:
            return "bounded-small"
        return "bounded-comparable-to-LOFO" if B <= LOFO_WORST else "bounded-large"

    if bound_terms:
        B, B_src = max(bound_terms)
        A, A_src = max(argmax_terms)
        outcome = classify(B, A)
        # Robustness of the verdict to the one leg that fails the B-Q1 contract gate:
        # recompute B over gate-PASSING (family, variant) pairs only. Not a pre-specified
        # quantity -- reported so the memo row cannot turn on a single defective leg.
        gate_ok_terms, gate_ok_argmax = [], []
        for fam, dv in report["deltas"].items():
            for variant, d in dv.items():
                if not d.get("available"):
                    continue
                v = report["families"][fam]["variants"][variant]
                if not (v["gate_pre"]["parse_gate_ok"] and v["gate_post"]["parse_gate_ok"]):
                    continue
                for key in ("d_ln_tau_v", "d_ln_T_abs_post"):
                    if d[key] is not None:
                        gate_ok_terms.append((abs(d[key]), f"{fam}/{variant}/{key}"))
                for key in ("d_argmax_pre_pp", "d_argmax_post_pp"):
                    gate_ok_argmax.append((abs(d[key]), f"{fam}/{variant}/{key}"))
        report["bound_gate_passing_only"] = None
        if gate_ok_terms:
            Bg, Bg_src = max(gate_ok_terms)
            Ag, Ag_src = max(gate_ok_argmax)
            report["bound_gate_passing_only"] = {
                "B": Bg, "B_source": Bg_src, "max_abs_d_argmax_pp": Ag,
                "max_abs_d_argmax_source": Ag_src, "memo_A5_row": classify(Bg, Ag),
                "n_pairs": len(gate_ok_terms) // 2,
                "excluded": [f"{fam}/{variant}" for fam, dv in report["deltas"].items()
                             for variant, d in dv.items() if d.get("available")
                             and not (report["families"][fam]["variants"][variant]["gate_pre"]["parse_gate_ok"]
                                      and report["families"][fam]["variants"][variant]["gate_post"]["parse_gate_ok"])],
            }
        report["bound"] = {
            "B": B, "B_source": B_src,
            "max_abs_d_argmax_pp": A, "max_abs_d_argmax_source": A_src,
            "grid_step_ln": GRID_STEP_LN, "lofo_worst_case": LOFO_WORST,
            "B_in_grid_steps": B / GRID_STEP_LN,
            "B_over_lofo": B / LOFO_WORST,
            "memo_A5_row": outcome,
            "note": "memo A.5: bounded-small = B < 0.0743 AND max|d argmax| < ~3pp; "
                    "bounded-comparable = 0.0743 <= B <= 0.191; bounded-large = B > 0.191",
        }

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "estimand_k5variants.json").write_text(json.dumps(report, indent=2))

    # ---- console
    hdr = f"{'family/variant':34} {'argmax_pre':>10} {'argmax_post':>11} {'tau_v':>8} {'T_abs_pre':>10} {'T_abs_post':>11}"
    print("\n" + hdr)
    for fam, spec in report["families"].items():
        for variant, v in spec["variants"].items():
            if not v.get("available"):
                print(f"{fam+'/'+variant:34} -- not available ({v.get('dir')})")
                continue
            print(f"{fam+'/'+variant:34} {v['argmax_pre']:10.4f} {v['argmax_post']:11.4f} "
                  f"{v['tau_v']:8.4f} {v['T_abs_pre']:10.4f} {v['T_abs_post']:11.4f}")
    print(f"\n{'delta vs shipped k=5':34} {'dargmax_pre':>12} {'dargmax_post':>13} {'dln tau_v':>10} {'dln T_abs':>10} {'steps':>7}")
    for fam, dv in report["deltas"].items():
        for variant, d in dv.items():
            if not d.get("available"):
                print(f"{fam+'/'+variant:34} -- not available")
                continue
            print(f"{fam+'/'+variant:34} {d['d_argmax_pre_pp']:+12.2f} {d['d_argmax_post_pp']:+13.2f} "
                  f"{d['d_ln_tau_v']:+10.4f} {d['d_ln_T_abs_post']:+10.4f} "
                  f"{d['d_ln_tau_v_grid_steps']:7.2f}")
    if "bound" in report:
        b = report["bound"]
        print(f"\nB = max|dln| = {b['B']:.4f}  ({b['B_source']})")
        print(f"  vs one tau_v grid step {GRID_STEP_LN:.4f} -> {b['B_in_grid_steps']:.2f} steps")
        print(f"  vs LOFO worst case     {LOFO_WORST:.4f} -> {b['B_over_lofo']:.2f}x")
        print(f"  max |d argmax| = {b['max_abs_d_argmax_pp']:.2f} pp ({b['max_abs_d_argmax_source']})")
        print(f"  memo A.5 row: {b['memo_A5_row']}")
        bg = report.get("bound_gate_passing_only")
        if bg:
            print(f"  gate-passing pairs only (n={bg['n_pairs']}, excluded "
                  f"{bg['excluded'] or 'none'}): B = {bg['B']:.4f} ({bg['B_source']}), "
                  f"max |d argmax| = {bg['max_abs_d_argmax_pp']:.2f} pp "
                  f"-> memo A.5 row: {bg['memo_A5_row']}")
    # ---- panel-level console block
    print("\n=== PANEL-LEVEL recomputation (four-family adoption panel) ===")
    for name, p in report["panels"].items():
        if not p.get("available"):
            print(f"{name:28} -- not available (missing: {', '.join(p['missing_families'])};"
                  f" present: {', '.join(p['present_families']) or 'none'})")
            continue
        for est, pretty in (("tau_v", "tau_v"), ("T_abs_post", "T_abs")):
            e = p[est]
            vals = "  ".join(f"{f}={e['values'][f]:.4f}" for f in e["values"])
            print(f"{name:28} {pretty:6} {vals}")
            print(f"{'':28} {'':6} ratio={e['cluster_ratio']:.4f} "
                  f"(<=2: {e['cluster_rule_ok']})  band=[{e['band'][0]:.4f}, {e['band'][1]:.4f}]  "
                  f"median={e['median_interpolated']:.4f}  "
                  f"LOFO max|ln|={e['lofo_max_abs_log_ratio']:.4f} ({e['lofo_max_source']})")
        print(f"{'':28} all legs pass B-Q1: {p['all_legs_pass_b_q1']}")
    porc = report["panel_of_record_check"]
    if porc.get("available"):
        bad = [k for k, c in porc["checks"].items() if not c["match"]]
        print(f"\npanel constants of record: "
              f"{'ALL reproduce exactly' if porc['all_match'] else 'MISMATCH on ' + ', '.join(bad)}")
    for variant, d in report["panel_deltas"].items():
        if not d.get("available"):
            print(f"panel delta [{variant}]: not available")
            continue
        print(f"\npanel delta [{variant}] vs {d['reference']}:")
        for est in ("tau_v", "T_abs_post"):
            x = d[est]
            print(f"  {est:11} ratio {x['cluster_ratio_baseline']:.4f} -> "
                  f"{x['cluster_ratio']:.4f} ({x['d_cluster_ratio']:+.4f}); "
                  f"<=2 rule survives: {x['cluster_rule_survives']}")
            print(f"  {'':11} band [{x['band_baseline'][0]:.4f}, {x['band_baseline'][1]:.4f}] -> "
                  f"[{x['band'][0]:.4f}, {x['band'][1]:.4f}] "
                  f"(dln low {x['d_ln_band_low']:+.4f}, high {x['d_ln_band_high']:+.4f})")
            print(f"  {'':11} median {x['median_interpolated_baseline']:.4f} -> "
                  f"{x['median_interpolated']:.4f} (dln {x['d_ln_median_interpolated']:+.4f})")
            print(f"  {'':11} LOFO max|ln| {x['lofo_max_abs_log_ratio_baseline']:.4f} -> "
                  f"{x['lofo_max_abs_log_ratio']:.4f} "
                  f"({x['d_lofo_max_abs_log_ratio']:+.4f}, worst {x['lofo_max_source']})")

    td = report["transport_diagnostic"]
    if td.get("available"):
        print(f"\ntransport diagnostic (report-only, NOT scaffold): {td['a']} vs {td['b']}")
        print(f"  argmax {td['argmax_a']:.4f} vs {td['argmax_b']:.4f} "
              f"({td['d_argmax_pp']:+.2f} pp); T_abs(post) {td['T_abs_post_a']:.4f} vs "
              f"{td['T_abs_post_b']:.4f} (dln {td['d_ln_T_abs_post']:+.4f})")
        print(f"  per-cell mean L1 {td['mean_L1_between_posts']:.4f}, "
              f"argmax agreement {td['cellwise_argmax_agreement']:.4f} "
              f"over {td['n_shared']} shared cells")
    for fam, c in report["baseline_of_record_check"].items():
        tag = "delta reference" if c["is_delta_reference"] else "legacy, NOT the reference"
        ok = c["tau_v_match"] and c["T_abs_post_match"]
        print(f"\nvalues of record [{fam}, {c['source_dir']}, {tag}]: "
              f"{'reproduce exactly' if ok else 'MISMATCH — ' + json.dumps(c)}")
    print(f"\nwrote {out / 'estimand_k5variants.json'}")


if __name__ == "__main__":
    main()
