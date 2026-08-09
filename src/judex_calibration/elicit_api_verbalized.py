"""Study B addendum: verbalized FULL-contract elicitation via OpenRouter chat API.

USER DECISION 2026-07-21 (amending the all-self-hosted plan for ONE leg): the
Gemma-4-31B-it post leg collapsed under greedy raw-completions continuation on vast
(30.8% contract-complete — a decoding pathology, recorded and kept). Rather than settle
for a 2-family panel, the user accepted the serving-mode confound and directed an
API-based re-collection, preferring 16-bit precision. OpenRouter serves this model from
several bf16 providers (bf16 = the study's own vast precision), so we pin the request to
16-bit endpoints via OpenRouter's provider preferences: quantizations [bf16, fp16],
allow_fallbacks false. The responding provider + upstream id are recorded PER CELL.

Channel label: ``verbalized_api_chat`` — a DIFFERENT channel from the vast raw-completions
legs (chat template applied; provider stack differs). Every artifact from this module
carries that label; the vast post leg's recorded failure stands untouched in
``runs/study_b_gemma31/``. Cross-mode quantities (e.g. tau_v of an API post against a
vast pre) are confound-labeled by construction.

Instrument: one chat call per cell — the SAME k=5 full-contract few-shot + evidence +
criterion text as the vast legs, followed by an instruction to reason step by step and
then emit the six-field contract JSON. temperature 0, single pass, no resampling; a
parse failure is recorded, mirroring the vast discipline. Parsing scans ALL balanced
JSON objects in the reply and keeps the last one carrying a compliance_distribution
(the reply legitimately contains reasoning prose that may include braces).

SELF-HOSTED CHAT (USER DIRECTIVE 2026-08-09): the same chat instrument can be pointed at
an OpenAI-compatible endpoint we serve ourselves (vLLM on vast, bf16) by passing
``base_url``. That path takes NO OpenRouter key, sends NO provider-preference block, and
labels its channel ``verbalized_vllm_chat``. Its reason for existing is
quantization-matching: the base leg is bf16 raw-completions on our own box, so the post
leg must be bf16 too, and the chat *template* (not a different provider stack) is what
prevents the known greedy-repetition collapse of gemma-4 post on raw continuation. When
``base_url`` is absent every byte of the OpenRouter path is unchanged.

stdlib only (urllib); the API key comes from the macOS Keychain and is never logged.
"""
from __future__ import annotations

import json, os, subprocess, urllib.parse, urllib.request
from pathlib import Path
from typing import Dict, Optional

from .elicit_verbalized import (EPSILON, parse_contract_json, extract_balanced_object,
                                build_fewshot_by_criterion)
from .fewshot import SCAFFOLD_VARIANTS

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
QUANTIZATIONS = ["bf16", "fp16"]  # 16-bit only; allow_fallbacks false enforces it
# Self-hosted vLLM chat: a DIFFERENT channel again from both the raw-completions legs
# (``verbalized``) and the OpenRouter legs (``verbalized_api_chat``) — same chat template
# idea, our own bf16 weights, no third-party provider stack.
SELF_HOSTED_CHANNEL = "verbalized_vllm_chat"
SELF_HOSTED_DTYPE = "bfloat16"
# Sidecar keys that are PROVENANCE, not leg identity: recorded, printed, and never used
# by the resume guard. A rented box's IP is ephemeral — a crash-recovery resume against a
# re-provisioned endpoint serving the same model at the same dtype is still the same leg,
# and hard-erroring on the IP would defeat the checkpoint machinery it exists to protect.
UNGUARDED_META = ("endpoint_host",)

INSTRUCTION = (
    "\n\nReason step by step about the evidence against the criterion, then finish your "
    "reply with a single JSON object of exactly the shape shown in the examples — the six "
    "fields findings, compliance_level, compliance_distribution, compliance_justification, "
    "confidence_distribution, confidence_justification, with every probability a multiple "
    "of 0.05 and each distribution summing to 1.00. The JSON object must be the last thing "
    "in your reply."
)


def keychain(name: str) -> str:
    return subprocess.run(["security", "find-generic-password", "-s", name, "-w"],
                          capture_output=True, text=True, check=True).stdout.strip()


def chat_endpoint(base_url: str) -> str:
    """``http://host:port`` (with or without a trailing ``/v1``) -> the chat URL."""
    u = base_url.rstrip("/")
    return u + "/chat/completions" if u.endswith("/v1") else u + "/v1/chat/completions"


def _chat(key: Optional[str], model: str, content: str, max_tokens: int = 4096,
          timeout: int = 600, retries: int = 6, base_url: Optional[str] = None) -> dict:
    """One chat call with bounded backoff on transport errors (429/5xx/URLError).

    Transport retry is NOT resampling: no model output was produced. A response that
    arrives is parsed exactly once, per the no-resampling rule.

    ``base_url`` set => self-hosted OpenAI-compatible endpoint: no Authorization header,
    no OpenRouter provider-preference block (there is one provider — our own server, and
    its dtype is pinned by the serving flags). Absent => the OpenRouter path, unchanged.
    """
    import time
    body = {
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    if base_url:
        url = chat_endpoint(base_url)
        headers = {"Content-Type": "application/json"}
    else:
        url = OPENROUTER_URL
        body["provider"] = {"quantizations": QUANTIZATIONS, "allow_fallbacks": False}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    delay = 10.0
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            # OpenRouter can return HTTP 200 with an error body (no 'choices') on
            # provider hiccups — observed live 2026-07-21 (KeyError crash at cell 27
            # of the gemma26 leg). Treat as retryable transport noise, same as 5xx.
            if "choices" not in out:
                if attempt < retries - 1:
                    time.sleep(delay); delay = min(delay * 2, 120)
                    continue
                raise RuntimeError(f"no 'choices' after {retries} attempts: {str(out)[:300]}")
            return out
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay); delay = min(delay * 2, 120)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(delay); delay = min(delay * 2, 120)
                continue
            raise
    raise RuntimeError("unreachable")


def parse_last_contract(text: str) -> dict:
    """Last balanced JSON object carrying a compliance_distribution, through the
    strict contract parser. Falls back to the plain first-object parse."""
    best = None
    pos = 0
    while True:
        idx = text.find("{", pos)
        if idx < 0:
            break
        blob = extract_balanced_object(text[idx:])
        if blob is None:
            pos = idx + 1
            continue
        if '"compliance_distribution"' in blob:
            best = blob
        pos = idx + len(blob) if len(blob) > 1 else idx + 1
    return parse_contract_json(best if best is not None else text)


def elicit_cell_api(key: Optional[str], model: str, evidence_text: str, criterion_text: str,
                    fewshot: str, base_url: Optional[str] = None) -> dict:
    content = (f"{fewshot}You are assessing EU AI Act compliance.\n\n"
               f"Evidence:\n{evidence_text}\n\nCriterion:\n{criterion_text}"
               f"{INSTRUCTION}")
    out = _chat(key, model, content, base_url=base_url)
    choice = out["choices"][0]
    text = choice["message"].get("content") or ""
    rec = parse_last_contract(text)
    rec.update({
        "channel": SELF_HOSTED_CHANNEL if base_url else "verbalized_api_chat",
        "api_provider": out.get("provider"),
        "api_model": out.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "reply_chars": len(text),
    })
    if not rec.get("parse_ok"):
        rec["reply_tail"] = text[-400:]   # API legs keep failure text (vast-leg gap, fixed here)
    return rec


def run_variant_api(model: str, cells, out_path: str, *, fewshot_by_crit: Dict[str, str],
                    fewshot_k: int, workers: int = 4, key: Optional[str] = None,
                    scaffold_variant: str = "baseline",
                    base_url: Optional[str] = None) -> dict:
    """Same checkpoint/resume/meta-sidecar discipline as the vast legs, api-channel-pinned.

    ``scaffold_variant`` is provenance only here — the caller renders ``fewshot_by_crit``
    — but it is PINNED in the sidecar and guarded on resume exactly like ``fewshot_k``,
    so a variant leg can never silently continue into a baseline run dir.

    ``base_url`` set => self-hosted vLLM chat (bf16, our own box): no Keychain read, no
    provider pinning, channel ``verbalized_vllm_chat``, and the sidecar records the
    served dtype + the endpoint host instead of the OpenRouter quantization list. The
    host is recorded, not the full URL: a vast box's port is ephemeral provenance, and
    the resume guard must not hard-error just because the box was re-provisioned.
    """
    if scaffold_variant not in SCAFFOLD_VARIANTS:
        raise ValueError(f"unknown scaffold variant {scaffold_variant!r}; "
                         f"expected one of {SCAFFOLD_VARIANTS}")
    out = Path(out_path)
    meta_path = out.with_suffix(".meta.json")
    if base_url:
        key = None
        leg_meta = {"model": model, "channel": SELF_HOSTED_CHANNEL, "epsilon": EPSILON,
                    "fewshot_k": int(fewshot_k), "dtype": SELF_HOSTED_DTYPE,
                    "api": "vllm_chat", "scaffold_variant": scaffold_variant,
                    "endpoint_host": urllib.parse.urlparse(
                        base_url if "//" in base_url else "//" + base_url).hostname or ""}
    else:
        key = key or keychain("openrouter-api-key")
        leg_meta = {"model": model, "channel": "verbalized_api_chat", "epsilon": EPSILON,
                    "fewshot_k": int(fewshot_k), "quantizations": QUANTIZATIONS,
                    "api": "openrouter", "scaffold_variant": scaffold_variant}
    recs: Dict[str, dict] = {}
    if out.exists():
        recs = json.loads(out.read_text())
        prior = json.loads(meta_path.read_text()) if meta_path.exists() else None
        # A sidecar predating the scaffold-variant field describes a baseline leg.
        mismatched = prior is not None and (
            any(prior[k] != leg_meta.get(k) for k in prior if k not in UNGUARDED_META)
            or prior.get("scaffold_variant", "baseline") != scaffold_variant)
        if recs and (prior is None or mismatched):
            raise RuntimeError(f"{out} holds cells from a different leg config "
                               f"({prior} != {leg_meta}); use a fresh --out")
        if recs:
            print(f"[{out.stem}] resuming: {len(recs)} cached", flush=True)
    meta_path.write_text(json.dumps(leg_meta, indent=2))
    todo = [c for c in cells if c.item_label not in recs]

    def _one(c):
        return c, elicit_cell_api(key, model, c.evidence_text, c.criterion_text,
                                  fewshot_by_crit[c.criterion_id], base_url=base_url)

    def _record(c, d):
        recs[c.item_label] = d
        tmp = out.parent / (out.name + ".tmp")
        tmp.write_text(json.dumps(recs, indent=2))
        os.replace(tmp, out)
        status = "ok" if d.get("parse_ok") else f"PARSE-FAIL ({d.get('parse_error')})"
        print(f"[{out.stem}] {len(recs)}/{len(cells)} {c.item_label} "
              f"({status}; {d.get('api_provider')})", flush=True)

    if workers <= 1:
        for c in todo:
            _record(*_one(c))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(_one, c) for c in todo]):
                _record(*f.result())
    return recs
