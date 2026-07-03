#!/usr/bin/env python3
"""Study A driver — the SAME script runs both workflows; the flags/host/model pick which.

(S) MAC SMOKE — free plumbing test, numbers DISCARDED: one small int4/Q4 stand-in model
    (e.g. Qwen3-4B) on a local llama.cpp/Metal server, with --limit N --no-reason. Validates
    the machinery only; its tau_oc is MEANINGLESS — never report it or paste its
    pipeline_calibration_block.json anywhere. See docs/local_smoke_quickstart.md.
(L) VAST LIVE — the real, paid experiment, its tau_oc IS the study output: the seven panel
    base+post pairs on vLLM, bf16, ALL 120 cells, reasoning ON (omit --limit/--no-reason).
    See docs/vast_quickstart.md + docs/vast_claude_code_orchestration.md.

Serve ONE model at a time; call once per variant with --base-url/--post-url, then
--analyze-only to compare. Any run that sets --limit or --no-reason (or analyses <120 cells)
is auto-marked smoke (a .smoke sentinel + report["smoke"]=true), and its calibration block is
flagged do-not-integrate.

Run with the judex-arm conda python on the Mac (study_a needs numpy/judex); on a vast box use
the repo .venv:
  conda run -n judex-arm python scripts/run_qwen_phase1.py --help
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from judex_calibration.aireg import load_cells
from judex_calibration import elicit_base, study_a, fewshot as fewshot_mod

# Smoke-only fallback used with --no-reason; the live path builds a real per-Article
# k-shot block from the corpus-native exemplar store (see fewshot.build_fewshot_by_criterion).
SMOKE_FEWSHOT = (
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
    ap.add_argument("--no-reason", action="store_true",
                    help="SMOKE ONLY: skip CoT and use the static SMOKE_FEWSHOT instead of the real "
                         "per-Article corpus few-shot; tau_oc from a --no-reason run is MEANINGLESS. "
                         "Omit for the live experiment (reasoning ON).")
    ap.add_argument("--budget", type=int, default=2048,
                    help="generation/CoT token budget for the reasoning (live) path; ignored under --no-reason")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="SMOKE ONLY: elicit just N cells (stride-sampled across the 5 Articles) instead of "
                         "all 120. Any run with --limit>0 is a plumbing test — its report/tau_oc/"
                         "calibration_block are MEANINGLESS. The live experiment runs all 120 (--limit 0).")
    ap.add_argument("--closed-run", default="stage9-gemini-gpt-medium",
                    help="JUDEX run whose closed-evaluator AIReg preds get the Q4 check. Evaluators are "
                         "now Anthropic+GPT (Claude/GPT); the default gemini-gpt run is LEGACY — pass a "
                         "Claude+GPT AIReg run for a valid Q4.")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    smoke_marker = out / ".smoke"
    if args.no_reason or args.limit:
        smoke_marker.write_text(
            "PLUMBING SMOKE (--limit/--no-reason): tau_oc, study_a_report.json and "
            "pipeline_calibration_block.json in this dir are MEANINGLESS — discard them. The real "
            "experiment uses all 120 cells, bf16, reasoning ON.\n")
    cells = load_cells()
    if args.limit and args.limit < len(cells):
        step = max(1, len(cells) // args.limit)
        cells = cells[::step][:args.limit]
        print(f"[smoke] limited to {len(cells)} cells (stride-sampled across Articles: "
              f"{sorted({c.criterion_id for c in cells})})")
    reason = not args.no_reason

    if not args.analyze_only:
        # Real per-Article few-shot from the corpus-native exemplar store (fewshot_k from
        # models.yaml); one k-shot block per Article criterion, applied to that Article's cells.
        # --no-reason smoke runs keep the single static SMOKE_FEWSHOT.
        if reason:
            fewshot_by_crit = fewshot_mod.build_fewshot_by_criterion(cells)
            fewshot = lambda c: fewshot_by_crit.get(c.criterion_id, "")
            print(f"[fewshot] built {len(fewshot_by_crit)} per-Article blocks at k={fewshot_mod.default_k()} "
                  f"from {fewshot_mod.DIMENSION_STORE.name}")
        else:
            fewshot = SMOKE_FEWSHOT
        for variant, url, model in (("pre", args.base_url, args.base_model),
                                    ("post", args.post_url, args.post_model)):
            if url and model:
                print(f"[{variant}] eliciting {len(cells)} cells from {model} @ {url} ...")
                elicit_base.run_variant(url, model, cells, str(out / f"{variant}.json"),
                                        fewshot=fewshot, reason=reason, budget=args.budget)
                print(f"[{variant}] wrote {out / f'{variant}.json'}")

    variants = {}
    for v in ("pre", "post"):
        p = out / f"{v}.json"
        if p.exists():
            variants[v] = study_a.load_predictions(p)
    if variants:
        report = study_a.cross_family({"qwen": variants}, cells)
        # A --limit/--no-reason run, or analysing <120 cells, is a plumbing SMOKE: its numbers are
        # meaningless. Mark them so a smoke tau_oc / calibration block can never be mistaken for real.
        is_smoke = smoke_marker.exists() or any(len(p) < 120 for p in variants.values())
        report["smoke"] = is_smoke
        # Q4 — apply the open-derived constant (median tau_oc) to the closed evaluators on AIReg.
        T = report.get("tau_oc_summary", {}).get("tau_oc_median")
        if T is not None:
            mr = (Path(__file__).resolve().parents[2]
                  / "judex-evaluator" / "runs" / args.closed_run / "metrics_report.json")
            if mr.exists():
                closed = {it["item_label"]: it["prediction"]["probabilities"]
                          for it in json.loads(mr.read_text())["items"]}
                report["closed_side_check_Q4"] = study_a.closed_side_check(closed, cells, T)
            # Emit a drop-in judex-evaluator calibration block (mode: temperature). For the LIVE run
            # it integrates by copy-paste; for a SMOKE it is flagged do-not-integrate.
            block = study_a.calibration_block(report)
            if block is not None:
                block.setdefault("provenance", {})["smoke"] = is_smoke
                (out / "pipeline_calibration_block.json").write_text(json.dumps(block, indent=2))
                if is_smoke:
                    print("[SMOKE] wrote pipeline_calibration_block.json — tau_oc is MEANINGLESS "
                          "(--limit/--no-reason or <120 cells). Do NOT paste into judex-evaluator "
                          "pipeline.yaml; only a full 120-cell bf16 reasoning run is real.")
                else:
                    print(f"[integrate] wrote {out / 'pipeline_calibration_block.json'} "
                          f"(paste under judex-evaluator pipeline.yaml -> calibration; closed-evaluator-scoped)")
        (out / "study_a_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
