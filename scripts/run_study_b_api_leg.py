#!/usr/bin/env python3
"""Study B driver — one VERBALIZED chat-API leg per invocation (OpenRouter).

The API twin of ``run_study_b_leg.py``. Same run-dir layout, same leg names, same
meta-sidecar/resume discipline, same ``--analyze`` (imported from that driver, so the
report shape is identical) — the difference is the transport: one chat call per cell
through OpenRouter with 16-bit providers pinned (quantizations [bf16, fp16],
allow_fallbacks false), channel label ``verbalized_api_chat``.

Existed as a module (``judex_calibration.elicit_api_verbalized``) with no CLI until
2026-08-08; the gemma31/gemma26 API post legs were driven ad hoc. This driver makes the
Phase-1a scaffold-variant legs reproducible from the command line.

  run_study_b_api_leg.py --out runs/study_b_gemma31_api_k5v1 --leg post \
      --model google/gemma-4-31b-it --scaffold-variant alt_set --workers 8
  run_study_b_api_leg.py --out runs/study_b_gemma31_api_k5v1 --analyze

CROSS-MODE NOTE (carried from runs/study_b_gemma31_api/README.md): the gemma31 family's
baseline pre leg is a vast raw-completions leg, not an API leg — any tau_v computed in a
dir mixing the two is cross-mode and confound-labeled. Keep that pairing identical
between baseline and variant, or the variant delta absorbs the serving-mode difference.

The API key comes from the macOS Keychain (``security find-generic-password -s
openrouter-api-key -w``) and is never printed.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from judex_calibration.aireg import load_cells
from judex_calibration import elicit_verbalized as ev
from judex_calibration import elicit_api_verbalized as eva
from judex_calibration import fewshot as fs

# One source of truth for the leg layout + scoring: reuse the vast driver's helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_study_b_leg import analyze, leg_path, PARSE_RATE_GATE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run dir, e.g. runs/study_b_gemma31_api_k5v1")
    ap.add_argument("--leg", choices=["pre", "post"], help="which twin this API run is")
    ap.add_argument("--model", help="OpenRouter model id, e.g. google/gemma-4-31b-it "
                                    "(with --base-url: the id as served by vLLM)")
    ap.add_argument("--base-url", default=None,
                    help="self-hosted OpenAI-compatible endpoint root (e.g. "
                         "http://IP:PORT). Set => vLLM chat path: no OpenRouter key, no "
                         "provider pinning, channel verbalized_vllm_chat, dtype bf16 "
                         "pinned by the serving flags. Absent => OpenRouter, unchanged")
    ap.add_argument("--fewshot-k", type=int, default=None, help="default: models.yaml fewshot_k")
    ap.add_argument("--scaffold-variant", choices=list(fs.SCAFFOLD_VARIANTS), default="baseline",
                    help="k=5 coverage-preserving scaffold perturbation (Phase 1a): "
                         "baseline | alt_set | rev_order. Pinned in the leg meta sidecar; a "
                         "resume under a different variant hard-errors — use a fresh --out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="SMOKE: first N cells only (numbers discarded)")
    ap.add_argument("--analyze", action="store_true", help="score existing legs; no elicitation")
    ap.add_argument("--eps-sensitivity", action="store_true",
                    help="include the epsilon-floor sensitivity check in --analyze")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells()

    if args.analyze:
        report = analyze(out_dir, cells, eps_sensitivity=args.eps_sensitivity)
        for leg, r in report["legs"].items():
            g = r["contract_compliance"]
            print(f"  {leg} [{r.get('scaffold_variant')}]: parse {g['n_parsed']}/{g['n_elicited']} "
                  f"({g['parse_rate']:.2%}); full-contract {g['n_contract_complete']}/"
                  f"{g['n_elicited']} ({g['contract_complete_rate']:.2%}, "
                  f"gate>={PARSE_RATE_GATE:.0%}: {'PASS' if g['parse_gate_ok'] else 'FAIL'})")
        if "tau_v" in report:
            print(f"  tau_v = {report['tau_v']:.4f}"
                  + (" [SATURATED — peg, not a fit]" if report["tau_v_saturated"] else ""))
        if report.get("smoke"):
            print("  [SMOKE] machinery only — numbers discarded")
        return

    if not (args.leg and args.model):
        raise SystemExit("elicitation needs --leg and --model (or use --analyze)")
    if args.limit:
        (out_dir / ".smoke").write_text("limit set — machinery validation only\n")
        print("[SMOKE] numbers from this run are DISCARDED")
    run_cells = cells[: args.limit] if args.limit else cells
    k_eff = args.fewshot_k if args.fewshot_k is not None else fs.default_k()
    blocks = ev.build_fewshot_by_criterion(cells, k=k_eff, variant=args.scaffold_variant)
    if args.scaffold_variant != "baseline":
        print(f"[scaffold] variant={args.scaffold_variant} (k={k_eff}, coverage-preserving)")
    if args.base_url:
        print(f"[transport] self-hosted vLLM chat ({eva.SELF_HOSTED_CHANNEL}, "
              f"{eva.SELF_HOSTED_DTYPE}) — no OpenRouter key, no provider pinning")
    eva.run_variant_api(args.model, run_cells, str(leg_path(out_dir, args.leg)),
                        fewshot_by_crit=blocks, fewshot_k=k_eff, workers=args.workers,
                        scaffold_variant=args.scaffold_variant, base_url=args.base_url)
    analyze(out_dir, cells)


if __name__ == "__main__":
    main()
