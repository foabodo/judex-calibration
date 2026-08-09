#!/usr/bin/env python3
"""Feasibility + dry-run probe for the k=5 coverage-preserving scaffold variants.

Phase 1a of `spec/plan_2026_08_08_verbalized_arm_gap_closure.md` (§1g) bounds the
scaffold confound by perturbing the instrument actually in use — k=5, one exemplar per
compliance level — instead of replaying the deprecated k=4 E-hole scaffold. This script
is the $0, no-network evidence that the two variants are well-formed:

  * store depth per (criterion, level) — the headroom an alternate draw needs;
  * V1 ``alt_set`` feasibility — after excluding the baseline draw's source excerpts,
    every level bucket must still be non-empty and the alternate draw must be
    coverage-complete AND row-disjoint from the baseline;
  * V1 exemplar ids vs the baseline ids, per criterion;
  * V2 ``rev_order`` — same exemplar SET as baseline, strictly reversed render order;
  * rendered few-shot block sizes per variant, in characters and (with
    ``--tokenize``) in family tokens, against the 32768 ``--max-model-len`` pin.

``--tokenize`` runs the family tokenizers strictly from the local HF cache
(HF_HUB_OFFLINE=1) — this probe never touches the network and never elicits.

  scaffold_variant_probe.py
  scaffold_variant_probe.py --tokenize --json runs/scaffold_variant_probe.json
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from judex_calibration.aireg import load_cells
from judex_calibration import fewshot as fs
from judex_calibration import elicit_verbalized as ev

# Every family the scaffold study elicits, base and post (tokenizers can differ within a
# pair). Phase 1a = qwen + gemma31; Phase 1b adds the remaining adoption-panel families,
# glm and maverick, whose legs are raw /v1/completions on BOTH sides — so the raw-prompt
# budget below is the binding pre-provisioning check for them (no chat-template path).
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


def criteria_texts(cells) -> dict:
    out = {}
    for c in cells:
        out.setdefault(c.criterion_id, c.criterion_text)
    return out


def probe(k: int = 5) -> dict:
    cells = load_cells()
    texts = criteria_texts(cells)
    store = fs._load_store()
    rep = {"k": k, "n_cells": len(cells), "criteria": sorted(texts), "rows": {}}
    for cid in sorted(texts):
        rows = store.get(cid, [])
        depth = Counter(r["compliance_level"] for r in rows)
        base = fs.scaffold_rows(cid, k, "baseline")
        base_src = sorted({r["source_item_label"] for r in base})
        left = [r for r in rows if r["source_item_label"] not in set(base_src)]
        left_depth = Counter(r["compliance_level"] for r in left)
        starved = [lv for lv in fs.LEVELS if left_depth.get(lv, 0) == 0]
        alt = fs.scaffold_rows(cid, k, "alt_set")
        rev = fs.scaffold_rows(cid, k, "rev_order")
        base_ord = fs.order_rows(base, "baseline")
        rev_ord = fs.order_rows(rev, "rev_order")
        rep["rows"][cid] = {
            "store_rows": len(rows),
            "store_depth": {lv: depth.get(lv, 0) for lv in fs.LEVELS},
            "baseline_ids": [r["id"] for r in base_ord],
            "baseline_source_items": base_src,
            "rows_after_source_exclusion": len(left),
            "depth_after_source_exclusion": {lv: left_depth.get(lv, 0) for lv in fs.LEVELS},
            "starved_levels": starved,
            "alt_set_ids": [r["id"] for r in fs.order_rows(alt, "baseline")],
            "alt_set_source_items": sorted({r["source_item_label"] for r in alt}),
            "alt_set_levels": [r["compliance_level"] for r in fs.order_rows(alt, "baseline")],
            "alt_set_coverage_complete": (
                len(alt) == k and {r["compliance_level"] for r in alt} == set(fs.LEVELS)),
            "alt_set_row_disjoint": not ({r["id"] for r in base} & {r["id"] for r in alt}),
            "alt_set_source_disjoint": not (
                set(base_src) & {r["source_item_label"] for r in alt}),
            "rev_order_same_set": {r["id"] for r in rev} == {r["id"] for r in base},
            "rev_order_is_reversed": [r["id"] for r in rev_ord] == [r["id"] for r in base_ord][::-1],
            "chars": {v: len(ev.build_fewshot(cid, texts[cid], k=k, variant=v))
                      for v in fs.SCAFFOLD_VARIANTS},
        }
    rep["all_feasible"] = all(
        not r["starved_levels"] and r["alt_set_coverage_complete"] and r["alt_set_row_disjoint"]
        and r["rev_order_same_set"] and r["rev_order_is_reversed"]
        for r in rep["rows"].values())
    return rep


def tokenize_report(rep: dict, budget: int = 2048) -> dict:
    """Worst-case live prompt tokens per (family-leg, variant), from the LOCAL HF cache."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    cells = load_cells()
    texts = criteria_texts(cells)
    out = {}
    for tag, repo in TOKENIZE_REPOS.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo)
        except Exception as e:                       # cache miss: report, never download
            out[tag] = {"repo": repo, "error": f"{type(e).__name__}: {str(e)[:160]}"}
            continue
        per_variant = {}
        for variant in fs.SCAFFOLD_VARIANTS:
            blocks = ev.build_fewshot_by_criterion(cells, k=rep["k"], variant=variant)
            fewshot_tok = {cid: len(tok(b).input_ids) for cid, b in blocks.items()}
            prompts = [len(tok(ev.build_prompt(c.evidence_text, c.criterion_text,
                                               blocks[c.criterion_id])).input_ids)
                       for c in cells]
            scaffold = len(tok(ev.JSON_SCAFFOLD).input_ids)
            per_variant[variant] = {
                "fewshot_tokens_by_criterion": fewshot_tok,
                "fewshot_tokens_max": max(fewshot_tok.values()),
                "fewshot_tokens_mean": round(sum(fewshot_tok.values()) / len(fewshot_tok), 1),
                "prompt_tokens_max": max(prompts),
                "required_tokens": max(prompts) + scaffold + budget + 1,
                "ctx_pin": CTX_PIN,
                "fits_ctx_pin": max(prompts) + scaffold + budget + 1 <= CTX_PIN,
            }
        out[tag] = {"repo": repo, "variants": per_variant}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--tokenize", action="store_true",
                    help="token counts from the LOCAL HF cache (offline; no downloads)")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args()

    rep = probe(args.k)
    print(f"k={rep['k']}  cells={rep['n_cells']}  criteria={len(rep['criteria'])}\n")
    print("=== V1 alt_set feasibility (exclusion by source_item_label) ===")
    hdr = f"{'criterion':<12} {'rows':>5} {'min lvl':>8} {'excl src':>9} {'left':>5} {'min left':>9} {'starved':>8} {'cover':>6} {'disjoint':>9}"
    print(hdr); print("-" * len(hdr))
    for cid, r in rep["rows"].items():
        print(f"{cid:<12} {r['store_rows']:>5} {min(r['store_depth'].values()):>8} "
              f"{len(r['baseline_source_items']):>9} {r['rows_after_source_exclusion']:>5} "
              f"{min(r['depth_after_source_exclusion'].values()):>9} "
              f"{(','.join(r['starved_levels']) or '-'):>8} "
              f"{str(r['alt_set_coverage_complete']):>6} {str(r['alt_set_row_disjoint']):>9}")
    print(f"\nALL FEASIBLE: {rep['all_feasible']}\n")

    print("=== baseline vs V1 exemplar ids ===")
    for cid, r in rep["rows"].items():
        print(f"{cid}:")
        for b, a in zip(r["baseline_ids"], r["alt_set_ids"]):
            print(f"    {b}\n  ->{a}")
    print()
    print("=== rendered few-shot block size (chars) ===")
    print(f"{'criterion':<12}" + "".join(f"{v:>12}" for v in fs.SCAFFOLD_VARIANTS))
    for cid, r in rep["rows"].items():
        print(f"{cid:<12}" + "".join(f"{r['chars'][v]:>12,}" for v in fs.SCAFFOLD_VARIANTS))

    if args.tokenize:
        rep["tokens"] = tokenize_report(rep, budget=args.budget)
        print("\n=== token budget vs the 32768 ctx pin (budget "
              f"{args.budget}; local tokenizers, offline) ===")
        for tag, t in rep["tokens"].items():
            if "error" in t:
                print(f"{tag} ({t['repo']}): TOKENIZER UNAVAILABLE OFFLINE — {t['error']}")
                continue
            print(f"{tag} ({t['repo']}):")
            for v, d in t["variants"].items():
                print(f"    {v:<9} few-shot max {d['fewshot_tokens_max']:>7,} "
                      f"mean {d['fewshot_tokens_mean']:>9,}  prompt max {d['prompt_tokens_max']:>7,}"
                      f"  required {d['required_tokens']:>7,}  fits={d['fits_ctx_pin']}")

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2))
        print(f"\nwrote {p}")
    return 0 if rep["all_feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
