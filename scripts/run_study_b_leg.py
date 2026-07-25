#!/usr/bin/env python3
"""Study B driver — one VERBALIZED-channel leg per invocation, plus --analyze.

Same workflow split as Study A's driver (see run_qwen_phase1.py):
(S) MAC SMOKE — free plumbing test, numbers DISCARDED: small int4/Q4 stand-in on a local
    llama.cpp/Metal server with --limit N [--no-reason]. Auto-marked smoke.
(L) VAST LIVE — the paid experiment: a panel family's base or post twin on vLLM bf16,
    all 120 cells, reasoning ON (omit --limit/--no-reason). USER-GATED per phase.

One run dir per family (fresh --out per experiment — run-id identity; resume is crash
recovery only, enforced by the meta sidecar). Legs are named pre/post:
  run_study_b_leg.py --out runs/study_b_qwen --leg pre  --base-url http://... --model Qwen/Qwen3.5-35B-A3B-Base
  run_study_b_leg.py --out runs/study_b_qwen --leg post --base-url http://... --model Qwen/Qwen3.5-35B-A3B
  run_study_b_leg.py --out runs/study_b_qwen --analyze

--analyze is free and idempotent: per-leg B-Q1 contract-compliance summary + study_a
scoring of the parsed compliance view (argmax/RPS/W1/T*/Murphy, resolution-primary gate
numbers) + tau_v (post -> pre, W1-aligned, same machinery as tau_oc — NEVER conflate the
two) on the parse-intersection. Confidence-side analysis (B-Q4) lands with Phase B2.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from judex_calibration.aireg import load_cells
from judex_calibration import elicit_verbalized as ev
from judex_calibration import study_a, study_b

# Pre-registered B-Q1 gate (design doc §4; FULL-contract tier per the user's 2026-07-20
# approval): a leg is contract-compliant iff >= 90% of its elicited cells emit the complete
# six-field contract 0.2.0 shape. parse_rate (distributions-only) is reported alongside;
# tau_v scores the parse_ok subset (valid distributions) for data efficiency.
PARSE_RATE_GATE = 0.90


def leg_path(out_dir: Path, leg: str) -> Path:
    return out_dir / f"{leg}_verbalized.json"


def analyze(out_dir: Path, cells, eps_sensitivity: bool = False) -> dict:
    report = {"channel": "verbalized", "epsilon": ev.EPSILON,
              "parse_rate_gate": PARSE_RATE_GATE, "legs": {}}
    views = {}
    leg_recs = {}
    for leg in ("pre", "post"):
        p = leg_path(out_dir, leg)
        if not p.exists():
            continue
        recs = json.loads(p.read_text())
        leg_recs[leg] = recs
        gate = ev.contract_compliance_summary(recs, n_cells=len(cells))
        gate["parse_gate_ok"] = gate["contract_complete_rate"] >= PARSE_RATE_GATE
        # Epsilon-floor BEFORE any temperature machinery: stored vectors keep the
        # contract's exact grid zeros, but judex.calibration's inverse-softmax would
        # turn ln(1e-12 clip) into ~-27.6 logits — the registered EPSILON (0.005, one
        # tenth of the 0.05 grid step; the older "half the grid step" gloss was
        # arithmetically wrong — half is 0.025) is the channel's floor (design doc §5).
        comp = {k: ev.floor_and_renormalize(v) for k, v in ev.compliance_view(recs).items()}
        views[leg] = comp
        report["legs"][leg] = {"contract_compliance": gate,
                               "score": study_a.score_variant(comp, cells) if comp else {"n": 0},
                               "confidence_bq4": study_b.analyze_confidence(recs, cells)}
    if "pre" in views and "post" in views:
        tau_v = study_a.fit_tau_oc(views["post"], views["pre"], cells)
        report["tau_v"] = tau_v
        report["tau_v_saturated"] = study_a.saturated(tau_v)
        report["tau_v_note"] = ("verbalized-channel analog of tau_oc, measured FRESH; "
                                "never convert to/from Study A's tau_oc (channels are "
                                "incommensurable — grid-quantized, no deterministic bridge)")
        if eps_sensitivity:
            report["epsilon_sensitivity"] = study_b.epsilon_sensitivity(
                leg_recs["pre"], leg_recs["post"], cells)
    # a leg elicited on <120 cells is smoke regardless of the sentinel
    smoke = (out_dir / ".smoke").exists() or any(
        leg["contract_compliance"]["n_elicited"] < len(cells) for leg in report["legs"].values())
    report["smoke"] = smoke
    if smoke:
        report["smoke_note"] = "SMOKE run — machinery validation only; every number is DISCARDED"
    out = out_dir / "study_b_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return report


def merge(specs, cells) -> dict:
    """Cross-family B-Q3: per-family tau_v + the clustering rule over gate-passers.

    specs = {family: run_dir}. Reuses study_a.cluster_fields verbatim (the rule fields
    read 'tau_oc' keys — they are tau_v here; the report says so). Gate passage
    (contract_complete >= 0.90 both legs + resolution-primary) is decided from each
    family's study_b_report.json.
    """
    rows, smoke = {}, False
    for fam, d in specs.items():
        d = Path(d)
        rp = d / "study_b_report.json"
        if not rp.exists():
            raise SystemExit(f"--merge: no study_b_report.json in {d} (family {fam!r}) — run --analyze first")
        r = json.loads(rp.read_text())
        smoke = smoke or r.get("smoke", False)
        legs = r.get("legs", {})
        gates_ok = all(legs.get(l, {}).get("contract_compliance", {}).get("parse_gate_ok")
                       for l in ("pre", "post"))
        res_pre = legs.get("pre", {}).get("score", {}).get("murphy", {}).get("resolution")
        rows[fam] = {
            "tau_v": r.get("tau_v"), "tau_v_saturated": r.get("tau_v_saturated"),
            "contract_gates_ok_both_legs": gates_ok,
            "pre_resolution": res_pre,
            "pre_T_rps_saturated": legs.get("pre", {}).get("score", {}).get("T_rps_saturated"),
            "confidence_bq4_post": legs.get("post", {}).get("confidence_bq4"),
        }
    # study_a.cluster_fields consumes {'tau_oc': ...}-shaped rows; feed only families
    # passing BOTH Study B gates so the ratio is the pre-registered B-Q3 quantity.
    clean = {f: {"tau_oc": v["tau_v"], "tau_oc_saturated": v["tau_v_saturated"],
                 "tau_oc_reference_degenerate": bool(v["pre_T_rps_saturated"])}
             for f, v in rows.items() if v["contract_gates_ok_both_legs"]}
    cluster = study_a.cluster_fields(clean)
    out = {"families": rows, "gate_passing": sorted(clean),
           "bq3_cluster": {k.replace("tau_oc", "tau_v"): v for k, v in cluster.items()},
           "smoke": smoke}
    if smoke:
        out["smoke_note"] = "SMOKE — a merged source is smoke; numbers discarded"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run dir, e.g. runs/study_b_qwen (one per family)")
    ap.add_argument("--merge", nargs="+", metavar="FAMILY=DIR",
                    help="cross-family B-Q3 report from per-family run dirs; writes to --out")
    ap.add_argument("--leg", choices=["pre", "post"], help="which twin this serving run is")
    ap.add_argument("--base-url", help="OpenAI-compatible /v1 server root")
    ap.add_argument("--model", help="model id as served")
    ap.add_argument("--fewshot-k", type=int, default=None, help="default: models.yaml fewshot_k")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, help="SMOKE: first N cells only (numbers discarded)")
    ap.add_argument("--no-reason", action="store_true", help="SMOKE: skip the reasoning stage")
    ap.add_argument("--analyze", action="store_true", help="score existing legs; no elicitation")
    ap.add_argument("--eps-sensitivity", action="store_true",
                    help="include the once-only pre-registered epsilon check (B1 report only)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells()

    if args.merge:
        specs = {}
        for spec in args.merge:
            fam, sep, d = spec.partition("=")
            if not sep or not fam or not d or fam in specs:
                raise SystemExit(f"--merge expects unique FAMILY=DIR, got {spec!r}")
            specs[fam] = d
        report = merge(specs, cells)
        p = out_dir / "study_b_cross_family.json"
        p.write_text(json.dumps(report, indent=2))
        print(f"wrote {p}")
        bq3 = report["bq3_cluster"]
        print(f"  gate-passing: {report['gate_passing']}")
        print(f"  tau_v cluster ratio: {bq3.get('tau_v_cluster_ratio')} "
              f"(rule_ok: {bq3.get('tau_v_cluster_rule_ok')})")
        if report.get("smoke"):
            print("  [SMOKE] numbers discarded")
        return

    if args.analyze:
        report = analyze(out_dir, cells, eps_sensitivity=args.eps_sensitivity)
        for leg, r in report["legs"].items():
            g = r["contract_compliance"]
            print(f"  {leg}: parse {g['n_parsed']}/{g['n_elicited']} ({g['parse_rate']:.2%}); "
                  f"full-contract {g['n_contract_complete']}/{g['n_elicited']} "
                  f"({g['contract_complete_rate']:.2%}, gate>={PARSE_RATE_GATE:.0%}: "
                  f"{'PASS' if g['parse_gate_ok'] else 'FAIL'})")
        if "tau_v" in report:
            print(f"  tau_v = {report['tau_v']:.4f}"
                  + (" [SATURATED — peg, not a fit]" if report["tau_v_saturated"] else ""))
        if report.get("smoke"):
            print("  [SMOKE] machinery only — numbers discarded")
        return

    if not (args.leg and args.base_url and args.model):
        raise SystemExit("elicitation needs --leg, --base-url and --model (or use --analyze)")
    smoke = bool(args.limit or args.no_reason)
    if smoke:
        (out_dir / ".smoke").write_text("limit/no-reason set — machinery validation only\n")
        print("[SMOKE] numbers from this run are DISCARDED")
    run_cells = cells[: args.limit] if args.limit else cells
    k = args.fewshot_k
    from judex_calibration import fewshot as fs
    k_eff = k if k is not None else fs.default_k()
    blocks = ev.build_fewshot_by_criterion(cells, k=k_eff)
    ev.run_variant(args.base_url, args.model, run_cells, str(leg_path(out_dir, args.leg)),
                   fewshot=lambda c: blocks[c.criterion_id], reason=not args.no_reason,
                   budget=args.budget, fewshot_k=k_eff, workers=args.workers)
    analyze(out_dir, cells)


if __name__ == "__main__":
    main()
