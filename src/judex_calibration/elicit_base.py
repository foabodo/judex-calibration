"""vLLM token-slice elicitation with a reasoning-then-answer logit read.

Targets a vLLM OpenAI-compatible /v1/completions endpoint (one model per server;
see scripts/serve_vllm_vastai.md). Used for BOTH the base and post variants so
pre/post are compared in the SAME channel (token-sliced logits over the 5 level
letters), with the SAME few-shot CoT scaffold, isolating the weights.

Two stages per cell (Approach C):
  1) generate the reasoning, stopping before the answer scaffold;
  2) re-feed ``prompt + reasoning + 'Answer:'`` with max_tokens=1, logprobs=K and
     read the distribution over A..E at that single answer position -> softmax.
This guarantees the logit read is at a fixed position conditioned on the full
reasoning. A 5-call echo fallback (prompt-logprob of each appended letter) covers
the case where a letter misses top-K.

stdlib only (urllib). Pure HTTP/parse logic, unit-tested without a live server.
"""
from __future__ import annotations

import json, math, os, urllib.request
from pathlib import Path
from typing import Dict, List, Sequence

LABELS = ("very_low", "low", "moderate", "high", "very_high")
LETTERS = ("A", "B", "C", "D", "E")
ANSWER_SCAFFOLD = "\n\nFinal answer — the single most appropriate compliance level (A=very_low, B=low, C=moderate, D=high, E=very_high).\nAnswer:"


def _completions(base_url: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs); ex = [math.exp(x - m) for x in xs]; z = sum(ex)
    return [e / z for e in ex]


def generate_reasoning(base_url: str, model: str, prompt: str, budget: int = 2048) -> str:
    """Stage 1: produce the reasoning span, stopping before our answer scaffold."""
    out = _completions(base_url, {
        "model": model, "prompt": prompt, "max_tokens": budget, "temperature": 0,
        "stop": ["Answer:", "</think>"],
    })
    return out["choices"][0].get("text", "")


def _letters_from_topk(top: Dict[str, float]) -> Dict[str, float]:
    """Map a {token: logprob} top-K dict to {LETTER: logprob} (strip/upper)."""
    found: Dict[str, float] = {}
    for tok, lp in top.items():
        key = tok.strip().upper()
        if key in LETTERS and key not in found:
            found[key] = lp
    return found


def _objlist_to_map(items) -> Dict[str, float]:
    """[{token|tok_str, logprob|prob}, ...] -> {token_string: logprob}."""
    out: Dict[str, float] = {}
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        tok = item.get("token", item.get("tok_str"))
        if tok is None:
            continue
        if item.get("logprob") is not None:
            out[str(tok)] = float(item["logprob"])
        elif item.get("prob"):
            out[str(tok)] = math.log(max(float(item["prob"]), 1e-12))
    return out


def _first_position_token_logprobs(logprobs) -> Dict[str, float]:
    """{token_string: logprob} for the FIRST generated position, across server shapes.

    Accepts (a) vLLM / llama-cpp-python legacy completions where ``top_logprobs[0]`` is a
    ``{token: logprob}`` dict; (b) OpenAI chat-style where ``content[0].top_logprobs`` is a
    list of ``{token, logprob}``; (c) llama.cpp ``top_probs`` / object-list variants (incl.
    a ``prob`` field). Returns {} if none match (caller then tries the echo fallback). This
    keeps the token-slice identical whether served by vLLM (vast) or a Metal server (Mac).
    """
    if not isinstance(logprobs, dict):
        return {}
    content = logprobs.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        mapped = _objlist_to_map(content[0].get("top_logprobs"))
        if mapped:
            return mapped
    for key in ("top_logprobs", "top_probs"):
        seq = logprobs.get(key)
        if isinstance(seq, list) and seq:
            first = seq[0]
            if isinstance(first, dict) and not ("token" in first or "tok_str" in first):
                mapped = {str(k): float(v) for k, v in first.items() if v is not None}   # {token: logprob}
            elif isinstance(first, dict):
                mapped = _objlist_to_map([first])                                          # single object
            elif isinstance(first, list):
                mapped = _objlist_to_map(first)                                            # list of objects
            else:
                mapped = {}
            if mapped:
                return mapped
    return {}


def answer_logits_topk(base_url: str, model: str, answer_prompt: str, top_k: int = 20) -> Dict[str, float]:
    """Read the answer-position distribution over A..E, server-agnostically.

    Parses logprobs from vLLM, llama.cpp / llama-cpp-python, and chat-style responses
    (see ``_first_position_token_logprobs``), so the same token-slice works against a
    local Apple-Silicon Metal server (Mac smoke) and vLLM (vast).
    """
    out = _completions(base_url, {
        "model": model, "prompt": answer_prompt, "max_tokens": 1, "temperature": 0, "logprobs": top_k,
    })
    logprobs = out["choices"][0].get("logprobs") or {}
    return _letters_from_topk(_first_position_token_logprobs(logprobs))


def answer_logits_echo(base_url: str, model: str, answer_prompt: str) -> Dict[str, float]:
    """Fallback: prompt-logprob of each appended letter (echo). 5 calls.

    Uses the OpenAI-legacy ``echo`` + ``prompt``-logprob path (vLLM). Servers that do not
    support echo (e.g. llama.cpp) return an unexpected shape; each letter degrades to -50.0
    rather than raising, so the caller keeps its top-K result.
    """
    out: Dict[str, float] = {}
    for L in LETTERS:
        try:
            r = _completions(base_url, {
                "model": model, "prompt": answer_prompt + " " + L, "max_tokens": 0,
                "temperature": 0, "logprobs": 0, "echo": True,
            })
            lp = r["choices"][0]["logprobs"]["token_logprobs"]
            out[L] = float(lp[-1]) if lp and lp[-1] is not None else -50.0
        except Exception:
            out[L] = -50.0
    return out


def build_cot_prompt(evidence_text: str, criterion_text: str, fewshot: str = "") -> str:
    return (f"{fewshot}You are assessing EU AI Act compliance. Reason step by step, then answer.\n\n"
            f"Evidence:\n{evidence_text}\n\nCriterion:\n{criterion_text}\n\nReasoning:")


def elicit_cell(base_url: str, model: str, evidence_text: str, criterion_text: str,
                fewshot: str = "", reason: bool = True, budget: int = 2048) -> dict:
    """Return a token-sliced 5-way distribution for one cell (reasoning-then-answer)."""
    prompt = build_cot_prompt(evidence_text, criterion_text, fewshot)
    reasoning = generate_reasoning(base_url, model, prompt, budget) if reason else ""
    answer_prompt = prompt + reasoning + ANSWER_SCAFFOLD
    found = answer_logits_topk(base_url, model, answer_prompt)
    method = "topk"
    if len(found) < len(LETTERS):  # a letter missed top-K -> try exact via echo (server permitting)
        echo = answer_logits_echo(base_url, model, answer_prompt)
        # adopt echo only if it recovered strictly more real letters (servers w/o echo return all -50)
        if sum(1 for L in LETTERS if echo.get(L, -50.0) > -49.0) > len(found):
            found, method = echo, "echo"
    logits = [found.get(L, -50.0) for L in LETTERS]
    return {"labels": list(LABELS), "probabilities": _softmax(logits),
            "channel": "token_slice", "method": method, "covered": len(found),
            "reasoning_chars": len(reasoning)}


def run_variant(base_url: str, model: str, cells, out_path: str, *,
                fewshot="", reason: bool = True, budget: int = 2048,
                fewshot_k: "int | None" = None, workers: int = 1) -> dict:
    """Elicit every cell, write {item_label: [p...]} JSON for study_a; returns the map.

    ``fewshot`` may be a fixed string (same block for every cell) or a callable
    ``cell -> str`` (e.g. a per-Article block from ``fewshot.build_fewshot_by_criterion``).

    Checkpoints after EVERY cell (atomic tmp+rename) so a multi-hour leg survives a
    crash, and resumes by skipping labels already in ``out_path``. Resume is CRASH
    RECOVERY ONLY (the run-identity principle): a ``<leg>.meta.json`` sidecar records
    (model, reason, budget), and resuming against a different leg config hard-errors —
    use a fresh ``--out`` for a different experiment.
    """
    out = Path(out_path)
    meta_path = out.with_suffix(".meta.json")
    leg_meta = {"model": model, "reason": bool(reason), "budget": int(budget)}
    if fewshot_k is not None:
        # Pin the few-shot scaffold too (added 2026-07-19): k changes the prompt, so a
        # resume under a different k would silently mix scaffolds within one leg.
        leg_meta["fewshot_k"] = int(fewshot_k)
    preds: Dict[str, List[float]] = {}
    if out.exists():
        preds = json.loads(out.read_text())
        prior = json.loads(meta_path.read_text()) if meta_path.exists() else None
        # Compare only keys the stored sidecar has: legacy sidecars predate the
        # fewshot_k pin and must still resume; any key BOTH sides have must match.
        mismatched = (prior is not None
                      and any(prior[k] != leg_meta.get(k) for k in prior))
        if preds and (prior is None or mismatched):
            raise RuntimeError(
                f"{out} holds {len(preds)} cells from a different leg config "
                f"(sidecar {prior} != requested {leg_meta}); resume is crash-recovery "
                f"only — use a fresh --out for a different experiment")
        if preds:
            todo = sum(1 for c in cells if c.item_label not in preds)
            print(f"[{out.stem}] resuming: {len(preds)} cells cached, {todo} to go", flush=True)
    meta_path.write_text(json.dumps(leg_meta, indent=2))
    todo = [c for c in cells if c.item_label not in preds]

    def _one(c):
        fs = fewshot(c) if callable(fewshot) else fewshot
        return c, elicit_cell(base_url, model, c.evidence_text, c.criterion_text,
                              fewshot=fs, reason=reason, budget=budget)

    def _record(c, d):
        # Main-thread only: mutate + atomic checkpoint (same semantics as the
        # sequential path; order is irrelevant — preds is keyed by item_label).
        preds[c.item_label] = d["probabilities"]
        tmp = out.parent / (out.name + ".tmp")
        tmp.write_text(json.dumps(preds, indent=2))
        os.replace(tmp, out)
        print(f"[{out.stem}] {len(preds)}/{len(cells)} {c.item_label} "
              f"({d['method']}, covered {d['covered']})", flush=True)

    if workers <= 1:
        for c in todo:
            _record(*_one(c))
    else:
        # Bounded concurrency (2026-07-19): vLLM batches server-side, so parallel
        # cells reclaim idle GPU on big boxes. elicit_cell is stateless stdlib
        # urllib per call — thread-safe. All bookkeeping happens on this thread
        # via as_completed; a worker exception propagates after in-flight cells
        # checkpoint, so resume loses nothing.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_one, c) for c in todo]
            for f in as_completed(futs):
                _record(*f.result())
    return preds
