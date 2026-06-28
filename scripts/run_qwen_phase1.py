#!/usr/bin/env python3
"""Phase 1 driver: elicit base/post on a vLLM endpoint, then run Study A analysis.

Serve ONE model at a time (see serve_vllm_vastai.md); call this once per variant
with the matching --base-url/--post-url, then --analyze-only to compare.

Run with the evaluator venv python (study_a needs numpy/judex):
  .../judex-evaluator/.venv/bin/python scripts/run_qwen_phase1.py --help
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from judex_calibration.aireg import load_cells
from judex_calibration import elicit_base, study_a

DEFAULT_FEWSHOT = (
    "Example.\nEvidence:\nThe system card omits any data governance section.\n"
    "Criterion:\nArticle 10 - Data and data governance.\n"
    "Reasoning: No data governance is documented at all, so compliance evidence is essentially absent.\n"
    "Answer: A\n\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url"); ap.add_argument("--base-model")
    ap.add_argument("--post-url"); ap.add_argument("--post-model")
    ap.add_argument("--out", required=True, help="run dir, e.g. runs/phase1_qwen")
    ap.add_argument("--no-reason", action="store_true", help="skip CoT (smoke only)")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--closed-run", default="stage9-gemini-gpt-medium",
                    help="JUDEX run whose closed-evaluator (Gemini/GPT) AIReg preds get the Q4 check")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    reason = not args.no_reason

    if not args.analyze_only:
        for variant, url, model in (("pre", args.base_url, args.base_model),
                                    ("post", args.post_url, args.post_model)):
            if url and model:
                print(f"[{variant}] eliciting {len(cells)} cells from {model} @ {url} ...")
                elicit_base.run_variant(url, model, cells, str(out / f"{variant}.json"),
                                        fewshot=DEFAULT_FEWSHOT, reason=reason, budget=args.budget)
                print(f"[{variant}] wrote {out / f'{variant}.json'}")

    variants = {}
    for v in ("pre", "post"):
        p = out / f"{v}.json"
        if p.exists():
            variants[v] = study_a.load_predictions(p)
    if variants:
        report = study_a.cross_family({"qwen": variants}, cells)
        # Q4 — apply the open-derived constant (median tau_oc) to the closed evaluators on AIReg.
        T = report.get("tau_oc_summary", {}).get("tau_oc_median")
        if T is not None:
            mr = (Path(__file__).resolve().parents[2]
                  / "judex-evaluator" / "runs" / args.closed_run / "metrics_report.json")
            if mr.exists():
                closed = {it["item_label"]: it["prediction"]["probabilities"]
                          for it in json.loads(mr.read_text())["items"]}
                report["closed_side_check_Q4"] = study_a.closed_side_check(closed, cells, T)
        (out / "study_a_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
