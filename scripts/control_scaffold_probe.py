#!/usr/bin/env python3
"""Offline feasibility + token-budget probe for the defined-answer control scaffold.

$0, no network, no elicitation — the pre-spend check that Phase 2b owes Phase 2c, in the
pattern of ``scripts/scaffold_variant_probe.py``:

  * per-subject store depth and the k=4 one-per-letter baseline draw;
  * ``alt_set`` feasibility under source-item exclusion (fail-loud, reported not raised);
  * ``rev_order`` = same exemplar SET, strictly reversed render order;
  * rendered few-shot block sizes per subject, in characters and (with ``--tokenize``)
    in family tokens;
  * worst-case live prompt tokens per family leg against the 32768 ``--max-model-len``
    pin, and against the AIReg scaffold's own worst case (the control's stems are short
    next to corpus-v2 dimension excerpts — ``professional_law`` is the long tail).

``--tokenize`` runs the family tokenizers strictly from the local HF cache
(HF_HUB_OFFLINE=1); a cache miss is REPORTED, never downloaded.

  control_scaffold_probe.py
  control_scaffold_probe.py --tokenize --json runs/control_scaffold_probe.json
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from judex_calibration import control_fewshot as cfs
from judex_calibration import elicit_control as ec
from judex_calibration import mmlu

# The four D5 families, base and post (tokenizers can differ within a pair). gemma31's
# post leg runs vLLM chat, but its RAW prompt budget is still the binding pre-provisioning
# number for the pre leg, so both sides are counted here.
TOKENIZE_REPOS = {
    "qwen/pre": "Qwen/Qwen3.5-35B-A3B-Base",
    "qwen/post": "Qwen/Qwen3.5-35B-A3B",
    "gemma31/pre": "google/gemma-4-31B",
    "glm/pre": "zai-org/GLM-4.5-Base",
    "glm/post": "zai-org/GLM-4.5",
    "maverick/pre": "meta-llama/Llama-4-Maverick-17B-128E",
    "maverick/post": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
}
CTX_PIN = 32768
# The AIReg scaffold's measured worst case under corpus-v2 + k=5 (CLAUDE.md): ~23.3k
# required, ~27k at the top of the scaffold-variant probe's range. Quoted so the control's
# number is read as a ratio, not in isolation.
AIREG_WORST_CASE_REQUIRED = 27000


def in_band(row: dict) -> bool:
    lo, hi = cfs.EXEMPLAR_SPAN_BAND
    return lo <= cfs.span_len(row) <= hi


def probe(k: int = cfs.CONTROL_FEWSHOT_K) -> dict:
    items = mmlu.load_control_items()
    store = cfs.load_store()
    rep = {"k": k, "contract": ec.CONTRACT, "n_items": len(items),
           "slice_sha256": mmlu.slice_sha256(), "store_sha256": cfs.store_sha256(),
           "exemplar_selection": cfs.EXEMPLAR_SELECTION,
           "exemplar_span_band": list(cfs.EXEMPLAR_SPAN_BAND),
           "subjects": sorted({i.subject for i in items}), "rows": {}}
    for subj in rep["subjects"]:
        rows = store.get(subj, [])
        depth = Counter(r["answer_letter"] for r in rows)
        src_depth = {L: len({r["source_item_label"] for r in rows if r["answer_letter"] == L})
                     for L in mmlu.OPTION_LABELS}
        base = cfs.order_rows(cfs.scaffold_rows(subj, k, "baseline"), "baseline")
        rev = cfs.order_rows(cfs.scaffold_rows(subj, k, "rev_order"), "rev_order")
        feas = cfs.alt_set_feasibility(subj, k)
        rep["rows"][subj] = {
            "store_rows": len(rows),
            "store_depth_by_letter": {L: depth.get(L, 0) for L in mmlu.OPTION_LABELS},
            "store_source_items_by_letter": src_depth,
            "baseline_ids": [r["id"] for r in base],
            "baseline_letters": [r["answer_letter"] for r in base],
            "baseline_raters": [r["rater_model"] for r in base],
            "baseline_source_items": sorted({r["source_item_label"] for r in base}),
            "baseline_coverage_complete": (
                len(base) == k and {r["answer_letter"] for r in base} == set(mmlu.OPTION_LABELS)),
            "rev_order_same_set": {r["id"] for r in rev} == {r["id"] for r in base},
            "rev_order_is_reversed": [r["id"] for r in rev] == [r["id"] for r in base][::-1],
            "alt_set_feasible": feas["alt_set_feasible"],
            "alt_set_starved_letters": feas["starved_letters"],
            "alt_set_error": feas["error"],
            "fewshot_chars": len(ec.build_fewshot(subj, k=k)),
            # band-targeted selection (E1 remediation): the spans the model is shown
            "baseline_span_lens": [cfs.span_len(r) for r in base],
            "baseline_out_of_band": [r["id"] for r in base if not in_band(r)],
            "alt_set_span_lens": feas["alt_set_span_lens"],
            "alt_set_out_of_band": feas["alt_set_out_of_band"],
            "store_in_band_by_letter": {
                L: sum(1 for r in rows if r["answer_letter"] == L and in_band(r))
                for L in mmlu.OPTION_LABELS},
        }
    rep["baseline_all_coverage_complete"] = all(
        r["baseline_coverage_complete"] and r["rev_order_same_set"] and r["rev_order_is_reversed"]
        for r in rep["rows"].values())
    rep["alt_set_available"] = all(r["alt_set_feasible"] for r in rep["rows"].values())
    all_spans = sorted(x for r in rep["rows"].values() for x in r["baseline_span_lens"])
    rep["baseline_spans"] = {
        "min": all_spans[0], "median": all_spans[len(all_spans) // 2], "max": all_spans[-1],
        "n_out_of_band": sum(len(r["baseline_out_of_band"]) for r in rep["rows"].values()),
        "n": len(all_spans)}
    rep["baseline_all_in_band"] = rep["baseline_spans"]["n_out_of_band"] == 0
    # longest live item (professional_law stems are the long tail)
    blocks = ec.build_fewshot_by_subject(items, k=k)
    longest = max(items, key=lambda i: len(ec.build_prompt(i.evidence_text, i.criterion_text,
                                                           blocks[i.criterion_id])))
    rep["longest_item"] = {"item_label": longest.item_label, "subject": longest.subject,
                           "prompt_chars": len(ec.build_prompt(longest.evidence_text,
                                                               longest.criterion_text,
                                                               blocks[longest.criterion_id]))}
    return rep


def tokenize_report(rep: dict, budget: int = 2048) -> dict:
    """Worst-case live prompt tokens per family leg, from the LOCAL HF cache."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    items = mmlu.load_control_items()
    blocks = ec.build_fewshot_by_subject(items, k=rep["k"])
    out = {}
    for tag, repo in TOKENIZE_REPOS.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo)
        except Exception as e:                       # cache miss: report, never download
            out[tag] = {"repo": repo, "error": f"{type(e).__name__}: {str(e)[:160]}"}
            continue
        fewshot_tok = {s: len(tok(b).input_ids) for s, b in blocks.items()}
        prompts = [len(tok(ec.build_prompt(i.evidence_text, i.criterion_text,
                                           blocks[i.criterion_id])).input_ids) for i in items]
        scaffold = len(tok(ec.JSON_SCAFFOLD).input_ids)
        required = max(prompts) + scaffold + budget + 1
        out[tag] = {
            "repo": repo,
            "fewshot_tokens_by_subject": fewshot_tok,
            "fewshot_tokens_max": max(fewshot_tok.values()),
            "fewshot_tokens_mean": round(sum(fewshot_tok.values()) / len(fewshot_tok), 1),
            "prompt_tokens_max": max(prompts),
            "prompt_tokens_mean": round(sum(prompts) / len(prompts), 1),
            "required_tokens": required,
            "ctx_pin": CTX_PIN, "fits_ctx_pin": required <= CTX_PIN,
            "headroom_tokens": CTX_PIN - required,
            "vs_aireg_worst_case_required": round(required / AIREG_WORST_CASE_REQUIRED, 3),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=cfs.CONTROL_FEWSHOT_K)
    ap.add_argument("--tokenize", action="store_true",
                    help="token counts from the LOCAL HF cache (offline; no downloads)")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args()

    rep = probe(args.k)
    print(f"contract={rep['contract']} k={rep['k']} items={rep['n_items']} "
          f"subjects={len(rep['subjects'])}")
    print(f"slice sha {rep['slice_sha256'][:16]}  store sha {rep['store_sha256'][:16]}\n")

    hdr = (f"{'subject':<24} {'rows':>5} {'min letter':>11} {'min src/letter':>15} "
           f"{'cover':>6} {'rev ok':>7} {'alt_set':>9} {'few-shot chars':>15}")
    print(hdr); print("-" * len(hdr))
    for s, r in rep["rows"].items():
        print(f"{s:<24} {r['store_rows']:>5} {min(r['store_depth_by_letter'].values()):>11} "
              f"{min(r['store_source_items_by_letter'].values()):>15} "
              f"{str(r['baseline_coverage_complete']):>6} "
              f"{str(r['rev_order_same_set'] and r['rev_order_is_reversed']):>7} "
              f"{('OK' if r['alt_set_feasible'] else 'starved:' + ','.join(r['alt_set_starved_letters'])):>9} "
              f"{r['fewshot_chars']:>15,}")
    print(f"\nbaseline coverage-complete everywhere: {rep['baseline_all_coverage_complete']}")
    print(f"alt_set available (optional robustness check): {rep['alt_set_available']}")
    if not rep["alt_set_available"]:
        print("  -> the v1 control store holds one source item for at least one letter per "
              "subject;\n     the fail-loud coverage guard blocks alt_set. rev_order is "
              "unaffected and available.")
    print(f"\nlongest live prompt: {rep['longest_item']['item_label']} "
          f"({rep['longest_item']['prompt_chars']:,} chars)")

    b = rep["baseline_spans"]
    print(f"\nexemplar selection: {rep['exemplar_selection']} "
          f"(band {rep['exemplar_span_band']}) — {b['n']} spans, "
          f"min {b['min']} / median {b['median']} / max {b['max']}, "
          f"out of band {b['n_out_of_band']}")

    print("\n=== baseline draw (letter ramp A->D, least-used-rater balancing) ===")
    for s, r in rep["rows"].items():
        print(f"{s}:  spans {r['baseline_span_lens']}")
        for lid, letter, rater, n in zip(r["baseline_ids"], r["baseline_letters"],
                                         r["baseline_raters"], r["baseline_span_lens"]):
            flag = "" if lid not in r["baseline_out_of_band"] else "  <- OUT OF BAND"
            print(f"    {letter}  {n:>5}  {rater:<48} {lid}{flag}")

    if args.tokenize:
        rep["tokens"] = tokenize_report(rep, budget=args.budget)
        print(f"\n=== token budget vs the {CTX_PIN} ctx pin (CoT budget {args.budget}; "
              f"local tokenizers, offline) ===")
        hdr = (f"{'family/leg':<16} {'few-shot max':>13} {'few-shot mean':>14} "
               f"{'prompt max':>11} {'required':>9} {'headroom':>9} {'fits':>6} {'vs AIReg':>9}")
        print(hdr); print("-" * len(hdr))
        for tag, t in rep["tokens"].items():
            if "error" in t:
                print(f"{tag:<16} TOKENIZER UNAVAILABLE OFFLINE — {t['error']}")
                continue
            print(f"{tag:<16} {t['fewshot_tokens_max']:>13,} {t['fewshot_tokens_mean']:>14,} "
                  f"{t['prompt_tokens_max']:>11,} {t['required_tokens']:>9,} "
                  f"{t['headroom_tokens']:>9,} {str(t['fits_ctx_pin']):>6} "
                  f"{t['vs_aireg_worst_case_required']:>9.3f}")

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {p}")
    return 0 if rep["baseline_all_coverage_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
