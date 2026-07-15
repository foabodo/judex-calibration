#!/usr/bin/env python3
"""Measure the REAL Study A prompt budget per family x leg (corpus-v2 instrument).

Why this exists (2026-07-14): corpus v2 made the k=4 per-Article few-shot block ~14x
longer (dimension exemplar texts mean ~9.7k chars), so the pre-v2 `--max-model-len >=
16384` guidance is obsolete. This script renders the EXACT live prompts (real per-Article
k=4 few-shot from the v2 store + the AIReg evidence + criterion + CoT scaffold — the same
code path `run_qwen_phase1.py` uses), tokenizes them with each family's OWN tokenizer
(base and post separately: tokenizers can differ within a pair), reads each model's
context ceiling from its HF config, and prints a family x leg budget table with a
`--max-model-len` recommendation and bf16 KV-cache sizing.

Required context per cell = stage-2 prompt + 1 generated token, where
  stage-2 prompt = stage-1 prompt (few-shot + evidence + criterion) + reasoning (up to
  --budget tokens) + ANSWER_SCAFFOLD.
So: required = max_prompt_tokens + scaffold_tokens + budget + 1.

Run on the Mac with the judex-arm conda python; the HF token comes from the env
(HF_TOKEN, e.g. `export HF_TOKEN=$(security find-generic-password -s hf-token -w)`).
Only tokenizer/config files are downloaded — never weights.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from judex_calibration.aireg import load_cells
from judex_calibration import fewshot as fewshot_mod
from judex_calibration.elicit_base import build_cot_prompt, ANSWER_SCAFFOLD

MODELS_YAML = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"

# Smoke stand-ins (Mac llama.cpp int4) — measured for R4 planning only; NOT panel rows.
SMOKE_REPOS = {
    "smoke-qwen3-4b": ("Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B"),
    "smoke-gemma3-4b": ("google/gemma-3-4b-pt", "google/gemma-3-4b-it"),
}


def load_families() -> dict:
    import yaml
    cfg = yaml.safe_load(MODELS_YAML.read_text())
    fams = {}
    for name, f in cfg["families"].items():
        post = f.get("post_repo")
        if not post:
            raise SystemExit(
                f"family {name!r} has no post_repo in configs/models.yaml — the live plan "
                "serves BOTH legs from HF weights (vllm_hf), so the HF post twin must be pinned")
        fams[name] = (f["base_repo"], post)
    return fams


def get_tokenizer(repo: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(repo), False
    except Exception:
        return AutoTokenizer.from_pretrained(repo, trust_remote_code=True), True


def get_config(repo: str) -> dict:
    """config.json, falling back to Mistral-native params.json (normalized keys).

    A params.json-only repo (e.g. Mistral-Large-3) also means vLLM must serve it with
    --config-format mistral --load-format mistral --tokenizer-mode mistral.
    """
    from huggingface_hub import hf_hub_download
    try:
        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        if not context_ceiling(cfg):  # some configs omit the field; the class default supplies it
            from transformers import AutoConfig
            try:
                cfg = AutoConfig.from_pretrained(repo).to_dict()
            except Exception:
                cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True).to_dict()
        return cfg
    except Exception:
        p = json.loads(Path(hf_hub_download(repo, "params.json")).read_text())
        return {
            "max_position_embeddings": p.get("max_seq_len") or p.get("max_position_embeddings", 0),
            "num_hidden_layers": p.get("n_layers"),
            "num_key_value_heads": p.get("n_kv_heads"),
            "num_attention_heads": p.get("n_heads"),
            "head_dim": p.get("head_dim"),
            "hidden_size": p.get("dim"),
            # MLA (kv_lora_rank present, e.g. Mistral-Large-3): the compressed latent is cached
            "kv_lora_rank": p.get("kv_lora_rank"),
            "qk_rope_head_dim": p.get("qk_rope_head_dim"),
            "_mistral_native_format": True,
        }


def text_cfg(cfg: dict) -> dict:
    """Descend into text_config for multimodal wrappers (gemma/llama-4 style)."""
    return cfg.get("text_config", cfg)


def context_ceiling(cfg: dict) -> int:
    return int(text_cfg(cfg).get("max_position_embeddings", 0))


def kv_bytes_per_token(cfg: dict) -> int:
    """bf16 KV-cache bytes per token per sequence (upper bound; ignores sliding-window
    savings). MLA models (kv_lora_rank present) cache the compressed latent instead."""
    c = text_cfg(cfg)
    layers = int(c.get("num_hidden_layers", 0))
    if c.get("kv_lora_rank"):  # MLA (DeepSeek-V3 lineage, Kimi K2)
        per_layer = (int(c["kv_lora_rank"]) + int(c.get("qk_rope_head_dim", 0))) * 2
    else:
        heads = int(c.get("num_key_value_heads") or c.get("num_attention_heads", 0))
        head_dim = int(c.get("head_dim") or
                       (int(c["hidden_size"]) // int(c["num_attention_heads"])
                        if c.get("hidden_size") and c.get("num_attention_heads") else 0))
        per_layer = 2 * heads * head_dim * 2  # K+V, bf16
    return layers * per_layer


def tekken_counts(repo: str, prompts: list[str]):
    """Token counts under mistral-common's tekken — what vLLM --tokenizer-mode mistral
    actually uses to serve a params.json-only repo. The repo's HF tokenizer.json carries
    a known broken pretokenizer regex that INFLATES counts, so tekken is authoritative."""
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    tek = MistralTokenizer.from_hf_hub(repo).instruct_tokenizer.tokenizer
    return [len(tek.encode(p, bos=True, eos=False)) for p in prompts]


def measure_leg(repo: str, prompts: list[str]) -> dict:
    tok, remote_code = get_tokenizer(repo)
    counts = [len(tok(p, add_special_tokens=True).input_ids) for p in prompts]
    scaffold = len(tok(ANSWER_SCAFFOLD, add_special_tokens=False).input_ids)
    cfg = get_config(repo)
    if cfg.get("_mistral_native_format"):
        try:
            counts = tekken_counts(repo, prompts)
        except Exception as e:
            print(f"  [warn] tekken unavailable ({e}); HF tokenizer.json counts are an "
                  f"inflated upper bound for {repo}", flush=True)
    ceiling = context_ceiling(cfg)
    if not ceiling:  # last resort: the tokenizer's own limit, ignoring sentinel values
        mml = int(getattr(tok, "model_max_length", 0) or 0)
        ceiling = mml if 0 < mml < 10_000_000 else 0
    return {
        "repo": repo,
        "max_prompt_tokens": max(counts),
        "median_prompt_tokens": sorted(counts)[len(counts) // 2],
        "scaffold_tokens": scaffold,
        "ceiling": ceiling,
        "kv_bytes_per_token": kv_bytes_per_token(cfg),
        "trust_remote_code": remote_code,
        "mistral_native_format": bool(cfg.get("_mistral_native_format")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2048,
                    help="reasoning token budget (must match the driver's --budget)")
    ap.add_argument("--families", default="all",
                    help="comma list of families from models.yaml, or 'all'")
    ap.add_argument("--smoke", action="store_true",
                    help="also measure the Mac smoke stand-ins (R4 planning rows)")
    ap.add_argument("--json-out", help="write the full result table to this JSON path")
    args = ap.parse_args()

    fams = load_families()
    if args.families != "all":
        keep = set(args.families.split(","))
        missing = keep - fams.keys()
        if missing:
            raise SystemExit(f"unknown families: {sorted(missing)}")
        fams = {k: v for k, v in fams.items() if k in keep}
    if args.smoke:
        fams.update(SMOKE_REPOS)

    cells = load_cells()
    fs = fewshot_mod.build_fewshot_by_criterion(cells)
    prompts = [build_cot_prompt(c.evidence_text, c.criterion_text, fs[c.criterion_id])
               for c in cells]
    print(f"rendered {len(prompts)} live prompts "
          f"(k={fewshot_mod.default_k()} v2 few-shot; max {max(len(p) for p in prompts):,} chars)\n")

    rows, worst_required = [], 0
    for family, (base, post) in fams.items():
        for leg, repo in (("base", base), ("post", post)):
            print(f"[{family}/{leg}] {repo} ...", flush=True)
            try:
                m = measure_leg(repo, prompts)
            except Exception as e:
                rows.append({"family": family, "leg": leg, "repo": repo,
                             "error": f"{type(e).__name__}: {e}"})
                continue
            required = m["max_prompt_tokens"] + m["scaffold_tokens"] + args.budget + 1
            fits = required <= m["ceiling"] if m["ceiling"] else None
            rows.append({"family": family, "leg": leg, **m,
                         "required_ctx": required, "fits": fits})
            if family not in SMOKE_REPOS:
                worst_required = max(worst_required, required)

    print(f"\n| family | leg | repo | max prompt | median | required ctx (budget {args.budget}) "
          f"| model ceiling | fits | KV bf16 @required (GB/seq) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['family']} | {r['leg']} | {r['repo']} | ERROR: {r['error']} | | | | | |")
            continue
        kv_gb = r["kv_bytes_per_token"] * r["required_ctx"] / 2**30
        note = " (trust_remote_code)" if r["trust_remote_code"] else ""
        print(f"| {r['family']} | {r['leg']} | {r['repo']}{note} | {r['max_prompt_tokens']:,} "
              f"| {r['median_prompt_tokens']:,} | {r['required_ctx']:,} | {r['ceiling']:,} "
              f"| {'YES' if r['fits'] else 'NO' if r['fits'] is not None else '?'} | {kv_gb:.1f} |")

    if worst_required:
        # round the panel-wide worst case up to the next 4096 boundary for headroom
        rec = int(math.ceil(worst_required / 4096) * 4096)
        print(f"\npanel-wide worst-case required ctx: {worst_required:,} "
              f"=> recommended --max-model-len {rec} (next 4096 boundary)")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"budget": args.budget, "rows": rows, "worst_required": worst_required}, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
