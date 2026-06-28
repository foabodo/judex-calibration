"""Post-trained elicitation via OpenAI-compatible API (OpenRouter for qwen).

Two channels per cell, both at temperature 0:
- verbalized: model writes a 5-way probability distribution as JSON.
- token_slice: read top_logprobs over the single-letter answer tokens (A..E) and
  softmax — the Type-A token-level distribution, comparable to the base leg's
  vLLM token-slicing (§4.3/4.4 of the guide).

stdlib only (urllib); the API key is read from the macOS Keychain and never logged.
"""
from __future__ import annotations

import json, math, subprocess, urllib.request
from typing import Sequence

LABELS = ("very_low", "low", "moderate", "high", "very_high")
LETTERS = ("A", "B", "C", "D", "E")


def keychain(secret_name: str) -> str:
    return subprocess.run(
        ["security", "find-generic-password", "-s", secret_name, "-w"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _post(base_url: str, key: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _softmax(xs: Sequence[float]) -> list[float]:
    m = max(xs); ex = [math.exp(x - m) for x in xs]; z = sum(ex)
    return [e / z for e in ex]


def elicit_token_slice(base_url: str, key: str, model: str, prompt: str) -> dict:
    """Single-letter answer; softmax the top_logprobs over A..E."""
    out = _post(base_url, key, {
        "model": model, "temperature": 0, "max_tokens": 1,
        "logprobs": True, "top_logprobs": 20,
        "messages": [{"role": "user", "content": prompt + "\nAnswer with exactly one letter (A-E). Answer:"}],
    })
    top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    lp = {t["token"].strip(): t["logprob"] for t in top}
    logits = [lp.get(L, -50.0) for L in LETTERS]
    return {"labels": list(LABELS), "probabilities": _softmax(logits),
            "channel": "token_slice", "covered": sum(L in lp for L in LETTERS)}


def elicit_verbalized(base_url: str, key: str, model: str, prompt: str,
                      reasoning_enabled: bool = True) -> dict:
    """Model writes the 5-way distribution as JSON; parsed and renormalized.

    For thinking models (e.g. qwen3.5-35b-a3b) pass ``reasoning_enabled=False`` —
    otherwise the model burns the whole budget on reasoning tokens and returns
    empty content. Disabling reasoning also makes the post pass a single
    forward-style judgement, closer to the (non-reasoning) base leg, and ~100x
    cheaper. Whether the *deployed* post model should reason is a study choice
    (Phase 1 defaults to OFF for cost + base-comparability; flip for live behavior).
    """
    instr = (prompt + "\nReturn ONLY a JSON object with keys "
             "very_low, low, moderate, high, very_high whose values are probabilities summing to 1.0.")
    body = {"model": model, "temperature": 0, "max_tokens": 300,
            "messages": [{"role": "user", "content": instr}]}
    if not reasoning_enabled:
        body["reasoning"] = {"enabled": False}
    out = _post(base_url, key, body)
    text = out["choices"][0]["message"]["content"]
    obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
    vals = [max(float(obj[L]), 0.0) for L in LABELS]
    z = sum(vals) or 1.0
    return {"labels": list(LABELS), "probabilities": [v / z for v in vals], "channel": "verbalized"}


def build_prompt(evidence_text: str, criterion_text: str, fewshot: str = "") -> str:
    return (f"{fewshot}You are assessing EU AI Act compliance.\n\n"
            f"Evidence:\n{evidence_text}\n\nCriterion:\n{criterion_text}\n\n"
            "Compliance level scale: A=very_low, B=low, C=moderate, D=high, E=very_high.")
