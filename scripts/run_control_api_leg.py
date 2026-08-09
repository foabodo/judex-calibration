#!/usr/bin/env python3
"""Defined-answer control driver — one CHAT-transport leg per invocation.

The API twin of ``run_control_leg.py``, mirroring how ``run_study_b_api_leg.py`` twins
``run_study_b_leg.py``: same run-dir layout, same ``{leg}_control.json`` names, same
meta-sidecar/resume discipline, and ``analyze``/``leg_path``/``PARSE_RATE_GATE`` are
IMPORTED from the raw driver so the report shape is identical by construction.

D5 transport map for the 2c campaign: qwen, glm and maverick run raw-completions on BOTH
legs (``run_control_leg.py``); gemma31 runs raw pre + **self-hosted vLLM chat post** —
the matched-transport pairing established in Phase 1a, replicating that family's AIReg
transport exactly. Pass ``--base-url`` for the self-hosted bf16 chat path (no OpenRouter
key, no provider pinning, channel ``verbalized_vllm_chat``); omit it for OpenRouter.

  run_control_api_leg.py --out runs/control_mmlu_gemma31 --leg post \
      --base-url http://IP:PORT --model google/gemma-4-31b-it --workers 8
  run_control_api_leg.py --out runs/control_mmlu_gemma31 --analyze
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from judex_calibration import control_fewshot as cfs
from judex_calibration import elicit_control as ec
from judex_calibration import elicit_api_verbalized as eva
from judex_calibration import mmlu

# One source of truth for the leg layout + scoring: reuse the raw driver's helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_control_leg import PARSE_RATE_GATE, analyze, leg_path, print_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run dir, e.g. runs/control_mmlu_gemma31")
    ap.add_argument("--leg", choices=["pre", "post"], help="which twin this run is")
    ap.add_argument("--model", help="model id (OpenRouter id, or the id as served by vLLM)")
    ap.add_argument("--base-url", default=None,
                    help="self-hosted OpenAI-compatible endpoint root (e.g. http://IP:PORT). "
                         "Set => vLLM chat path: no OpenRouter key, no provider pinning, "
                         "channel verbalized_vllm_chat, bf16 pinned by the serving flags. "
                         "Absent => OpenRouter with 16-bit providers pinned")
    ap.add_argument("--fewshot-k", type=int, default=cfs.CONTROL_FEWSHOT_K,
                    help="control protocol default: 4 (one exemplar per option letter)")
    ap.add_argument("--scaffold-variant", choices=list(cfs.SCAFFOLD_VARIANTS),
                    default="baseline",
                    help="baseline | alt_set | rev_order; pinned in the leg meta sidecar, "
                         "a resume under a different variant hard-errors")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="SMOKE: first N items only (numbers discarded)")
    ap.add_argument("--analyze", action="store_true", help="score existing legs; no elicitation")
    ap.add_argument("--no-bootstrap", action="store_true", help="--analyze without the B=2000 CIs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = mmlu.load_control_items()

    if args.analyze:
        print_report(analyze(out_dir, items, bootstrap=not args.no_bootstrap))
        return

    if not (args.leg and args.model):
        raise SystemExit("elicitation needs --leg and --model (or use --analyze)")
    if args.limit:
        (out_dir / ".smoke").write_text("limit set — machinery validation only\n")
        print("[SMOKE] numbers from this run are DISCARDED")
    run_items = items[: args.limit] if args.limit else items
    blocks = ec.build_fewshot_by_subject(items, k=args.fewshot_k,
                                         variant=args.scaffold_variant)
    print(f"[scaffold] contract={ec.CONTRACT} k={args.fewshot_k} "
          f"variant={args.scaffold_variant} subjects={len(blocks)} "
          f"slice={mmlu.slice_sha256()[:12]} store={cfs.store_sha256()[:12]}")
    if args.base_url:
        print(f"[transport] self-hosted vLLM chat ({eva.SELF_HOSTED_CHANNEL}, "
              f"{eva.SELF_HOSTED_DTYPE}) — no OpenRouter key, no provider pinning")
    ec.run_variant_api(args.model, run_items, str(leg_path(out_dir, args.leg)),
                       fewshot_by_subject=blocks, fewshot_k=args.fewshot_k,
                       workers=args.workers, scaffold_variant=args.scaffold_variant,
                       base_url=args.base_url)
    print_report(analyze(out_dir, items))


if __name__ == "__main__":
    main()
