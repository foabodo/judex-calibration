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

import json, math, urllib.request
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


def answer_logits_topk(base_url: str, model: str, answer_prompt: str, top_k: int = 20) -> Dict[str, float]:
    out = _completions(base_url, {
        "model": model, "prompt": answer_prompt, "max_tokens": 1, "temperature": 0, "logprobs": top_k,
    })
    top = out["choices"][0]["logprobs"]["top_logprobs"][0]  # {token: logprob} at the answer position
    return _letters_from_topk(top)


def answer_logits_echo(base_url: str, model: str, answer_prompt: str) -> Dict[str, float]:
    """Fallback: prompt-logprob of each appended letter (echo). 5 calls."""
    out: Dict[str, float] = {}
    for L in LETTERS:
        r = _completions(base_url, {
            "model": model, "prompt": answer_prompt + " " + L, "max_tokens": 0,
            "temperature": 0, "logprobs": 0, "echo": True,
        })
        lp = r["choices"][0]["logprobs"]["token_logprobs"]
        out[L] = float(lp[-1]) if lp and lp[-1] is not None else -50.0
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
    if len(found) < len(LETTERS):  # a letter missed top-K -> exact via echo
        found = answer_logits_echo(base_url, model, answer_prompt)
        method = "echo"
    logits = [found.get(L, -50.0) for L in LETTERS]
    return {"labels": list(LABELS), "probabilities": _softmax(logits),
            "channel": "token_slice", "method": method, "covered": len(found),
            "reasoning_chars": len(reasoning)}


def run_variant(base_url: str, model: str, cells, out_path: str, *,
                fewshot="", reason: bool = True, budget: int = 2048) -> dict:
    """Elicit every cell, write {item_label: [p...]} JSON for study_a; returns the map.

    ``fewshot`` may be a fixed string (same block for every cell) or a callable
    ``cell -> str`` (e.g. a per-Article block from ``fewshot.build_fewshot_by_criterion``).
    """
    preds: Dict[str, List[float]] = {}
    for c in cells:
        fs = fewshot(c) if callable(fewshot) else fewshot
        d = elicit_cell(base_url, model, c.evidence_text, c.criterion_text,
                        fewshot=fs, reason=reason, budget=budget)
        preds[c.item_label] = d["probabilities"]
    with open(out_path, "w") as f:
        json.dump(preds, f, indent=2)
    return preds
