#!/usr/bin/env python3
"""Defined-answer control driver — one raw-completions leg per invocation, plus --analyze.

Phase 2c of the defined-answer control (design:
``docs/phase2_defined_answer_control_design.md``). Fork of ``run_study_b_leg.py``'s
conventions: one run dir per family, legs named pre/post, checkpoint + meta sidecar,
resume is crash-recovery only, ``--analyze`` is free and idempotent.

FILE NAMING IS A SAFETY DEVICE. Control legs are written as ``{pre,post}_control.json``,
never ``{pre,post}_verbalized.json`` — so a control run dir can never be picked up by the
AIReg driver (``run_study_b_leg.py --analyze`` finds no legs there) and a control leg can
never be scored by ``study_a.score_variant`` through a path that expects the 5-level
ordinal scale. The reverse holds too: this driver ignores ``*_verbalized.json``.

  run_control_leg.py --out runs/control_mmlu_qwen --leg pre  --base-url http://... --model Qwen/Qwen3.5-35B-A3B-Base
  run_control_leg.py --out runs/control_mmlu_qwen --leg post --base-url http://... --model Qwen/Qwen3.5-35B-A3B
  run_control_leg.py --out runs/control_mmlu_qwen --analyze

GATE (design D3): ``parse_ok >= 0.90``. The AIReg gate reads the six-field
``contract_complete`` tier; the control contract has five fields (findings is dropped and
has no defined-answer analogue), so the gate is re-derived onto ``parse_ok`` — the tier
that IS the same object across the two tasks. ``contract_complete`` is reported alongside.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from judex_calibration import control_fewshot as cfs
from judex_calibration import elicit_control as ec
from judex_calibration import mmlu, score_control

# Re-derived control gate (D3): parse_ok, not the six-field contract_complete tier.
PARSE_RATE_GATE = 0.90


def leg_path(out_dir: Path, leg: str) -> Path:
    """``{leg}_control.json`` — never ``_verbalized``; see the module docstring."""
    return out_dir / f"{leg}_control.json"


def analyze(out_dir: Path, items, *, bootstrap: bool = True) -> dict:
    recs_by_leg, metas = {}, {}
    for leg in ("pre", "post"):
        p = leg_path(out_dir, leg)
        if not p.exists():
            continue
        recs_by_leg[leg] = json.loads(p.read_text())
        mp = p.with_suffix(".meta.json")
        metas[leg] = json.loads(mp.read_text()) if mp.exists() else None
    report = score_control.analyze(recs_by_leg, bootstrap=bootstrap)
    report["parse_rate_gate"] = PARSE_RATE_GATE
    report["store_sha256"] = cfs.store_sha256()
    report["exemplar_selection"] = cfs.EXEMPLAR_SELECTION
    report["exemplar_span_band"] = list(cfs.EXEMPLAR_SPAN_BAND)
    for leg, recs in recs_by_leg.items():
        gate = ec.contract_compliance_summary(recs, n_cells=len(items))
        gate["parse_gate_ok"] = gate["parse_rate"] >= PARSE_RATE_GATE
        report["legs"][leg]["contract_compliance"] = gate
        report["legs"][leg]["leg_meta"] = metas.get(leg)
        report["legs"][leg]["scaffold_variant"] = (metas.get(leg) or {}).get(
            "scaffold_variant", "baseline")
    smoke = (out_dir / ".smoke").exists() or any(
        r.get("contract_compliance", {}).get("n_elicited", 0) < len(items)
        for r in report["legs"].values())
    report["smoke"] = smoke
    if smoke:
        report["smoke_note"] = "SMOKE run — machinery validation only; every number is DISCARDED"
    out = out_dir / "control_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return report


def print_report(report: dict) -> None:
    for leg, r in report["legs"].items():
        g = r.get("contract_compliance", {})
        print(f"  {leg} [{r.get('scaffold_variant')}]: parse {g.get('n_parsed')}/"
              f"{g.get('n_elicited')} ({g.get('parse_rate', 0):.2%}, gate>="
              f"{PARSE_RATE_GATE:.0%}: {'PASS' if g.get('parse_gate_ok') else 'FAIL'}); "
              f"5-field complete {g.get('contract_complete_rate', 0):.2%}")
        if r.get("n"):
            print(f"       acc {r['accuracy']:.3f} (chance {r['chance_floor']:.2f})  "
                  f"conf {r['mean_confidence']:.3f}  gap {r['overconfidence_gap']:+.3f}  "
                  f"ECE10 {r['ece10']:.3f}  AECE5 {r['aece5']:.3f}  MCE {r['mce10']:.3f}  "
                  f"AUROC {r['auroc_conf_vs_correct']:.3f}")
            b, bs = r.get("ece10_boot_item"), r.get("ece10_boot_subject")
            if b:
                tail = (f"  subject-clustered sensitivity [{bs['lo']:.3f}, {bs['hi']:.3f}] "
                        f"({bs['n_docs']} clusters)") if bs else "  (subject bootstrap: <3 clusters)"
                print(f"       ECE10 item-boot [{b['lo']:.3f}, {b['hi']:.3f}]"
                      f" ({b['n_docs']} clusters){tail}")
    if "e1_read" in report:
        e = report["e1_read"]
        print(f"  E1: base ECE10 {e['base_ece10']:.4f} -> {e['band']} "
              f"(inside AIReg v14 floor-clearing range: "
              f"{e['inside_aireg_v14_floor_clearing_range']})")
    if "e2_paired_ece_delta" in report:
        d = report["e2_paired_ece_delta"]
        bi = d.get("boot_item") or {}
        print(f"  E2: ECE10 pre {d['ece10_pre']:.4f} -> post {d['ece10_post']:.4f} "
              f"(delta {d['delta_post_minus_pre']:+.4f}, {d['direction']}); item-boot "
              f"[{bi.get('lo', float('nan')):+.4f}, {bi.get('hi', float('nan')):+.4f}] "
              f"excludes 0: {bi.get('excludes_zero')}")
    if "tau_tvd_descriptive" in report:
        t = report["tau_tvd_descriptive"]
        if t.get("tau_tvd") is not None:
            print(f"  tau (TVD, DESCRIPTIVE ONLY): {t['tau_tvd']:.4f}"
                  + (" [SATURATED — peg, not a fit]" if t.get("tau_tvd_saturated") else ""))
    if report.get("smoke"):
        print("  [SMOKE] machinery only — numbers discarded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run dir, e.g. runs/control_mmlu_qwen")
    ap.add_argument("--leg", choices=["pre", "post"], help="which twin this serving run is")
    ap.add_argument("--base-url", help="OpenAI-compatible /v1 server root")
    ap.add_argument("--model", help="model id as served")
    ap.add_argument("--fewshot-k", type=int, default=cfs.CONTROL_FEWSHOT_K,
                    help="control protocol default: 4 (one exemplar per option letter). "
                         "Must be a multiple of 4 — k/4 exemplars per letter; k=8 is the "
                         "doubled-density draw (E1 iteration (b))")
    ap.add_argument("--scaffold-variant", choices=list(cfs.SCAFFOLD_VARIANTS),
                    default="baseline",
                    help="coverage-preserving scaffold perturbation: baseline = the "
                         "shipped scaffold; alt_set = fully disjoint k/4-per-letter draw "
                         "(feasible on store v2 at k=4; only 3 of 6 subjects at k=8); "
                         "rev_order = same exemplars rendered D -> A. All three sit on "
                         "top of band-targeted selection. Pinned in the leg meta sidecar; "
                         "a resume under a different variant hard-errors")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, help="SMOKE: first N items only (numbers discarded)")
    ap.add_argument("--no-reason", action="store_true", help="SMOKE: skip the reasoning stage")
    ap.add_argument("--analyze", action="store_true", help="score existing legs; no elicitation")
    ap.add_argument("--no-bootstrap", action="store_true", help="--analyze without the B=2000 CIs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = mmlu.load_control_items()

    if args.analyze:
        print_report(analyze(out_dir, items, bootstrap=not args.no_bootstrap))
        return

    if not (args.leg and args.base_url and args.model):
        raise SystemExit("elicitation needs --leg, --base-url and --model (or use --analyze)")
    smoke = bool(args.limit or args.no_reason)
    if smoke:
        (out_dir / ".smoke").write_text("limit/no-reason set — machinery validation only\n")
        print("[SMOKE] numbers from this run are DISCARDED")
    run_items = items[: args.limit] if args.limit else items
    blocks = ec.build_fewshot_by_subject(items, k=args.fewshot_k,
                                         variant=args.scaffold_variant)
    spans = {s: [cfs.span_len(r) for r in cfs.order_rows(
        cfs.scaffold_rows(s, args.fewshot_k, args.scaffold_variant), args.scaffold_variant)]
        for s in blocks}
    print(f"[scaffold] contract={ec.CONTRACT} k={args.fewshot_k} "
          f"variant={args.scaffold_variant} selection={cfs.EXEMPLAR_SELECTION} "
          f"subjects={len(blocks)} "
          f"slice={mmlu.slice_sha256()[:12]} store={cfs.store_sha256()[:12]}")
    for s, v in spans.items():
        print(f"[scaffold]   {s:<24} spans {v}")
    ec.run_variant(args.base_url, args.model, run_items, str(leg_path(out_dir, args.leg)),
                   fewshot=lambda c: blocks[c.criterion_id], reason=not args.no_reason,
                   budget=args.budget, fewshot_k=args.fewshot_k, workers=args.workers,
                   scaffold_variant=args.scaffold_variant)
    print_report(analyze(out_dir, items))


if __name__ == "__main__":
    main()
