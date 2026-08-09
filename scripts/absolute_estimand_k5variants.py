#!/usr/bin/env python3
"""Phase-1a variant read: the absolute estimand T_abs on the k=5 scaffold variants.

LABELED VARIANT COPY of ``scripts/absolute_vs_ratio_estimand.py``. That script is the
FROZEN rA1-adjacent driver whose family->dir map points at the shipped baseline legs and
whose numbers are the values of record; it is NEVER edited. This file exists so the
Phase-1a coverage-preserving scaffold variants (V1 ``alt_set``, V2 ``rev_order``) can be
read on the SAME machinery without touching it.

Differences from the original, and only these:
  * the family->dir map covers qwen + gemma31 ONLY, three dirs each (baseline, k5v1,
    k5v2) instead of the six-family single-dir panel;
  * the T2 size-association and T3 gemma size-ladder blocks are dropped (they are
    panel-composition reads, meaningless on a two-family scaffold perturbation);
  * the report adds the pre-specified Phase-1a deltas -- Delta argmax in percentage
    points, Delta ln tau_v, Delta ln T_abs(post) -- against the shipped k=5 baseline,
    plus the memo's bound criterion B = max{|Delta ln tau_v|, |Delta ln T_abs|} compared
    against the leave-one-family-out worst case 0.191 and the tau_v grid step
    ln 1.0771 = 0.0743.

This is a Phase-1a variant read, NOT the frozen rA1 driver. Nothing here adopts
anything; r3 F1-F8 and rA1 A1-A8 are frozen and untouched.

$0 -- analysis-only on existing artifacts.

Run from judex-calibration:
  python scripts/absolute_estimand_k5variants.py [--out runs/estimand_k5variants]
"""
from __future__ import annotations

import argparse
import json
import math
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
    "qwen":    {"tau_v": 1.281052072726771,  "T_abs_post": 2.6365430791635713,
                "is_delta_reference": True,  "dir": "study_b_qwen"},
    "gemma31": {"tau_v": 1.0251785151221313, "T_abs_post": 2.2416624655619084,
                "is_delta_reference": False, "dir": "study_b_gemma31_api"},
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
