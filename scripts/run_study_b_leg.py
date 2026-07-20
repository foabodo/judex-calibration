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
from judex_calibration import study_a

# Pre-registered B-Q1 gate (design doc §4): a leg is contract-compliant iff >= 90% of
# its elicited cells parse to valid compliance+confidence distributions.
PARSE_RATE_GATE = 0.90


def leg_path(out_dir: Path, leg: str) -> Path:
    return out_dir / f"{leg}_verbalized.json"


def analyze(out_dir: Path, cells) -> dict:
    report = {"channel": "verbalized", "epsilon": ev.EPSILON,
              "parse_rate_gate": PARSE_RATE_GATE, "legs": {}}
    views = {}
    for leg in ("pre", "post"):
        p = leg_path(out_dir, leg)
        if not p.exists():
            continue
        recs = json.loads(p.read_text())
        gate = ev.contract_compliance_summary(recs, n_cells=len(cells))
        gate["parse_gate_ok"] = gate["parse_rate"] >= PARSE_RATE_GATE
        # Epsilon-floor BEFORE any temperature machinery: stored vectors keep the
        # contract's exact grid zeros, but judex.calibration's inverse-softmax would
        # turn ln(1e-12 clip) into ~-27.6 logits — the pre-registered EPSILON (0.005,
        # half the grid step) is the channel's floor (design doc §5).
        comp = {k: ev.floor_and_renormalize(v) for k, v in ev.compliance_view(recs).items()}
        views[leg] = comp
        report["legs"][leg] = {"contract_compliance": gate,
                               "score": study_a.score_variant(comp, cells) if comp else {"n": 0}}
    if "pre" in views and "post" in views:
        tau_v = study_a.fit_tau_oc(views["post"], views["pre"], cells)
        report["tau_v"] = tau_v
        report["tau_v_saturated"] = study_a.saturated(tau_v)
        report["tau_v_note"] = ("verbalized-channel analog of tau_oc, measured FRESH; "
                                "never convert to/from Study A's tau_oc (channels are "
                                "incommensurable — grid-quantized, no deterministic bridge)")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run dir, e.g. runs/study_b_qwen (one per family)")
    ap.add_argument("--leg", choices=["pre", "post"], help="which twin this serving run is")
    ap.add_argument("--base-url", help="OpenAI-compatible /v1 server root")
    ap.add_argument("--model", help="model id as served")
    ap.add_argument("--fewshot-k", type=int, default=None, help="default: models.yaml fewshot_k")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, help="SMOKE: first N cells only (numbers discarded)")
    ap.add_argument("--no-reason", action="store_true", help="SMOKE: skip the reasoning stage")
    ap.add_argument("--analyze", action="store_true", help="score existing legs; no elicitation")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells()

    if args.analyze:
        report = analyze(out_dir, cells)
        for leg, r in report["legs"].items():
            g = r["contract_compliance"]
            print(f"  {leg}: parse {g['n_parsed']}/{g['n_elicited']} "
                  f"({g['parse_rate']:.2%}, gate>={PARSE_RATE_GATE:.0%}: "
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
