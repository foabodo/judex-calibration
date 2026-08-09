"""Defined-answer control: verbalized elicitation of the K=4 MMLU contract.

A FORK-WITH-SHARED-PRIMITIVES of ``elicit_verbalized``. Everything that defines the
*harness* is imported from there and runs unchanged — the ``/v1/completions`` transport,
the two-stage reasoning -> JSON flow with its stop lists, balanced-object extraction,
distribution validation, the 0.05-grid check, and the registered EPSILON floor. What is
forked is only what the contract forces: the option labels, the prompt preamble, the
JSON shape, and the record assembly.

CONTRACT (design D3, ``mmlu_control_v1``):

    answer_letter             one of A-D
    answer_distribution       {A, B, C, D} on the 0.05 grid
    answer_justification      free text
    confidence_distribution   {low, medium, high}   (UNCHANGED from contract 0.2.0)
    confidence_justification  free text

``findings`` is DROPPED, not stubbed: requirement/status/evidence against a legal
criterion has no defined-answer analogue. The consequence, stated honestly in the paper:
the identical-harness claim is scoped to *scaffold structure + two-stage flow + decoding
params + the parse_ok-tier gate*, not to the six-field ``contract_complete`` tier, which
cannot be the same object without findings.

SCAFFOLD GRAMMAR — DELIBERATELY UNCHANGED. The blocks keep the AIReg scaffold's exact
``Example. / Evidence: / Criterion: / Reasoning: / JSON:`` skeleton, with the question
stem in the ``Evidence:`` slot and the A-D option list in the ``Criterion:`` slot.
Renaming them to ``Question:``/``Options:`` was considered and REJECTED: the D3 scope
sentence says the harness is identical in *scaffold structure*, and keeping the literal
section headers is what makes that sentence true byte-for-byte rather than by analogy.
It also keeps the imported stage-1/stage-2 stop lists (``JSON:``, ``Example.``,
``</think>``, ``\\n\\n\\n``) exactly the objects they were validated as. Four worked
examples establish the mapping for the model before the live item appears.

RECORD KEYS mirror the verbalized channel's names — the answer distribution is stored
under ``compliance`` — so ``elicit_verbalized.contract_compliance_summary`` and the
``*_view`` projections apply to control legs untouched. Each record additionally carries
``contract: "mmlu_control_v1"``, which is what marks it as a nominal-K=4 emission that
must never be fed to ordinal machinery (``score_control`` enforces this structurally).
"""
from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import control_fewshot as cfs
from . import elicit_api_verbalized as eva
from . import elicit_verbalized as ev
from . import mmlu
from .elicit_verbalized import (EPSILON, JSON_SCAFFOLD, _dist_json, _on_grid, _valid_dist,
                                extract_balanced_object, floor_and_renormalize,
                                generate_json, generate_reasoning)
from .mmlu import OPTION_LABELS

CONTRACT = "mmlu_control_v1"
CONF_LABELS = ev.CONF_LABELS          # ("low", "medium", "high") — unchanged from 0.2.0
CHANNEL = "verbalized"                # the SAME channel; only the contract differs


# ---------------------------------------------------------------------------
# Few-shot rendering (control contract shape)
# ---------------------------------------------------------------------------

def render_answer_json(row: dict) -> str:
    """The control-contract JSON a few-shot exemplar answers with (D3 field order)."""
    return ("{" + f'"answer_letter": "{row["answer_letter"]}", '
            f'"answer_distribution": {_dist_json(row["answer_distribution"], OPTION_LABELS)}, '
            f'"answer_justification": {json.dumps((row.get("answer_justification") or "").strip())}, '
            f'"confidence_distribution": {_dist_json(row["confidence_distribution"], CONF_LABELS)}, '
            f'"confidence_justification": {json.dumps((row.get("confidence_justification") or "").strip())}' + "}")


def render_block(row: dict) -> str:
    """One (question, options, reasoning, control-contract JSON) few-shot example.

    The exemplar's own options are rendered — the control's ``Criterion:`` body is
    item-specific by nature (it IS the option list), unlike the AIReg scaffold where one
    Article criterion is shared across a block.
    """
    justification = (row.get("answer_justification") or "").strip()
    return (f"Example.\nEvidence:\n{row['question'].strip()}\n\n"
            f"Criterion:\n{mmlu.render_options(row['choices'])}\n\n"
            f"Reasoning: {justification}\n"
            f"JSON: {render_answer_json(row)}\n\n")


def build_fewshot(subject: str, k: Optional[int] = None,
                  exclude: Optional[Sequence[str]] = None,
                  variant: str = "baseline") -> str:
    """k-example control few-shot prefix for one subject (k=4, one per letter)."""
    if k is None:
        k = cfs.default_k()
    rows = cfs.order_rows(cfs.scaffold_rows(subject, k, variant, exclude), variant)
    return "".join(render_block(r) for r in rows)


def build_fewshot_by_subject(items, k: Optional[int] = None,
                             variant: str = "baseline") -> Dict[str, str]:
    """{subject: control few-shot string} for every subject present in ``items``."""
    subjects: List[str] = []
    for it in items:
        if it.criterion_id not in subjects:
            subjects.append(it.criterion_id)
    return {s: build_fewshot(s, k=k, variant=variant) for s in subjects}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(evidence_text: str, criterion_text: str, fewshot: str = "") -> str:
    """The control prompt: ONE sentence differs from ``elicit_verbalized.build_prompt``.

    The EU-AI-Act framing sentence is replaced by the defined-answer frame; the
    instruction clause, the section headers, and the trailing ``Reasoning:`` cue are
    byte-identical to the AIReg harness.
    """
    return (f"{fewshot}You are answering a multiple-choice question that has exactly one "
            f"correct option. Reason step by step, then answer with a single JSON object "
            f"of the exact shape shown in the examples.\n\n"
            f"Evidence:\n{evidence_text}\n\nCriterion:\n{criterion_text}\n\nReasoning:")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_control_json(text: str) -> dict:
    """Parse a stage-2 emission into the control cell record. Never raises.

    Tiers (D3):
      ``parse_ok``          both distributions valid — the minimum for scoring;
      ``contract_complete`` parse_ok + a valid ``answer_letter`` + both justifications
                            non-empty. NO findings clause: the field is dropped, so a
                            complete control emission has FIVE fields, not six, and the
                            control gate is scoped to ``parse_ok`` accordingly.
    """
    blob = extract_balanced_object(text)
    if blob is None:
        return {"parse_ok": False, "parse_error": "no_json_object", "raw_chars": len(text),
                "contract": CONTRACT}
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as e:
        return {"parse_ok": False, "parse_error": f"json_decode: {e.msg}",
                "raw_chars": len(text), "contract": CONTRACT}
    ans_raw = obj.get("answer_distribution")
    conf_raw = obj.get("confidence_distribution")
    ans = _valid_dist(ans_raw, OPTION_LABELS)
    conf = _valid_dist(conf_raw, CONF_LABELS)
    if ans is None or conf is None:
        which = [n for n, v in (("answer_distribution", ans),
                                ("confidence_distribution", conf)) if v is None]
        return {"parse_ok": False, "parse_error": f"invalid: {'+'.join(which)}",
                "raw_chars": len(text), "contract": CONTRACT}
    ans_vals = [float(ans_raw[k]) for k in OPTION_LABELS]
    conf_vals = [float(conf_raw[k]) for k in CONF_LABELS]
    letter = obj.get("answer_letter")
    letter_ok = letter in OPTION_LABELS
    missing = [name for name, ok in (
        ("answer_letter", letter_ok),
        ("answer_justification", bool(str(obj.get("answer_justification") or "").strip())),
        ("confidence_justification", bool(str(obj.get("confidence_justification") or "").strip())),
    ) if not ok]
    return {
        "parse_ok": True,
        "contract": CONTRACT,
        # stored under the verbalized channel's key names so the shared gate summary and
        # the *_view projections apply untouched; ``compliance`` here IS the A-D answer
        # distribution (nominal, K=4 — never an ordinal compliance scale).
        "compliance": ans, "confidence": conf,
        "answer_letter": letter if letter_ok else None,
        "level_matches_argmax": (letter == OPTION_LABELS[ans.index(max(ans))]) if letter_ok else False,
        "contract_complete": not missing,
        "contract_missing": missing,
        "compliance_sum": round(sum(ans_vals), 4), "confidence_sum": round(sum(conf_vals), 4),
        "compliance_on_grid": _on_grid(ans_vals), "confidence_on_grid": _on_grid(conf_vals),
    }


def parse_last_control(text: str) -> dict:
    """Last balanced JSON object carrying an ``answer_distribution``.

    Mirrors ``elicit_api_verbalized.parse_last_contract``. Used on BOTH transports since
    the E1 remediation (2026-08-09): a stage-2 completion, like a chat reply, legitimately
    contains braces that are not the answer object. Phase 2c's Mode A had two distinct
    sub-causes, and only one of them was the store's:

      * illegal escapes the panel's own prose taught (``\\frac`` inside a JSON string) —
        fixed on the CONTENT side by store v2, which carries no backslash at all;
      * brace-hijack of the first-``{`` scan — MMLU's question text is full of TeX, so a
        model that restates ``\\frac{x}{12}`` before its JSON hands the first-object
        scanner ``{x}``, which is balanced, is not the answer, and fails ``json.loads``
        as ``Expecting property name enclosed in double quotes``. No store content can
        fix that; the parser has to skip non-answer objects, which is what this does.

    INERT on a well-formed single-object emission: the scan finds the same first object,
    it carries ``answer_distribution``, and it is what gets parsed — byte-identical to
    ``parse_control_json``. It is also inert on a no-object emission, which still falls
    through to ``no_json_object``.
    """
    best, pos = None, 0
    while True:
        idx = text.find("{", pos)
        if idx < 0:
            break
        blob = extract_balanced_object(text[idx:])
        if blob is None:
            pos = idx + 1
            continue
        if '"answer_distribution"' in blob:
            best = blob
        pos = idx + len(blob) if len(blob) > 1 else idx + 1
    return parse_control_json(best if best is not None else text)


# ---------------------------------------------------------------------------
# Elicitation
# ---------------------------------------------------------------------------

def elicit_cell(base_url: str, model: str, evidence_text: str, criterion_text: str,
                fewshot: str = "", reason: bool = True, budget: int = 2048) -> dict:
    """One control item through the verbalized channel (imported two-stage flow).

    Stage 2 is parsed with ``parse_last_control``, not the first-object scan — see that
    function's docstring for why (TeX braces in restated MMLU stems hijack the first-``{``
    scan). The RECORD SHAPE is unchanged: both paths return ``parse_control_json``'s dict.
    """
    prompt = build_prompt(evidence_text, criterion_text, fewshot)
    reasoning = generate_reasoning(base_url, model, prompt, budget) if reason else ""
    raw = generate_json(base_url, model, prompt + reasoning + JSON_SCAFFOLD)
    rec = parse_last_control(raw)
    rec.update({"channel": CHANNEL, "reasoning_chars": len(reasoning)})
    return rec


def _control_leg_meta(*, model: str, scaffold_variant: str, fewshot_k: int,
                      slice_sha: str, store_sha: str, **extra) -> dict:
    meta = {"model": model, "channel": CHANNEL, "epsilon": EPSILON,
            "contract": CONTRACT, "slice_sha256": slice_sha, "store_sha256": store_sha,
            "scaffold_variant": scaffold_variant, "fewshot_k": int(fewshot_k),
            # WHICH rows the scaffold drew, on top of WHICH store they came from. Two legs
            # can share a store_sha256 and still be different legs: the length-blind draw
            # and the band draw off store v2 differ by 31 parse-rate points on smoke.
            "exemplar_selection": cfs.EXEMPLAR_SELECTION}
    meta.update(extra)
    return meta


def _guard_resume(out: Path, meta_path: Path, leg_meta: dict, *, unguarded=()) -> Dict[str, dict]:
    """Load checkpointed records, hard-erroring on any leg-identity mismatch.

    The inherited guard iterates the PRIOR sidecar's keys, so an identity field the prior
    sidecar happens to lack is silently unguarded. Every control identity field therefore
    gets an EXPLICIT clause below (mirroring the ``scaffold_variant`` clause upstream) —
    a prior sidecar with no ``contract``/``slice_sha256``/``store_sha256``/
    ``exemplar_selection`` is a pre-control or pre-band artifact and must hard-error, not
    default to "compatible". ``exemplar_selection`` is the newest such field and its
    omission is exactly the Phase-2c case: those sidecars predate band selection, so
    resuming a Phase-2c leg under this harness hard-errors, which is the intent.
    """
    recs: Dict[str, dict] = {}
    if not out.exists():
        return recs
    recs = json.loads(out.read_text())
    prior = json.loads(meta_path.read_text()) if meta_path.exists() else None
    mismatched = prior is not None and (
        any(prior[k] != leg_meta.get(k) for k in prior if k not in unguarded)
        # explicit clauses — these must fire even when the PRIOR sidecar omits the key
        or prior.get("contract") != leg_meta["contract"]
        or prior.get("slice_sha256") != leg_meta["slice_sha256"]
        or prior.get("store_sha256") != leg_meta["store_sha256"]
        or prior.get("scaffold_variant") != leg_meta["scaffold_variant"]
        or prior.get("fewshot_k") != leg_meta["fewshot_k"]
        or prior.get("exemplar_selection") != leg_meta["exemplar_selection"]
        or prior.get("channel") != leg_meta["channel"])
    if recs and (prior is None or mismatched):
        raise RuntimeError(
            f"{out} holds {len(recs)} items from a different leg config "
            f"(sidecar {prior} != requested {leg_meta}); resume is crash-recovery only — "
            f"use a fresh --out for a different experiment")
    if recs:
        print(f"[{out.stem}] resuming: {len(recs)} items cached", flush=True)
    return recs


def _checkpointing_loop(items, todo, recs, out: Path, one, workers: int):
    def _record(c, d):
        recs[c.item_label] = d
        tmp = out.parent / (out.name + ".tmp")
        tmp.write_text(json.dumps(recs, indent=2))
        os.replace(tmp, out)
        status = "ok" if d.get("parse_ok") else f"PARSE-FAIL ({d.get('parse_error')})"
        print(f"[{out.stem}] {len(recs)}/{len(items)} {c.item_label} ({status})", flush=True)

    if workers <= 1:
        for c in todo:
            _record(*one(c))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in as_completed([pool.submit(one, c) for c in todo]):
                _record(*f.result())
    return recs


def run_variant(base_url: str, model: str, items, out_path: str, *,
                fewshot="", reason: bool = True, budget: int = 2048,
                fewshot_k: Optional[int] = None, workers: int = 1,
                scaffold_variant: str = "baseline",
                slice_sha: Optional[str] = None, store_sha: Optional[str] = None) -> dict:
    """Elicit every control item; checkpoint + sidecar, same discipline as Study B.

    The sidecar pins the control's SIX identity fields on top of the inherited ones:
    ``contract``, ``slice_sha256``, ``store_sha256``, ``scaffold_variant``, ``fewshot_k``,
    ``exemplar_selection`` — each with an explicit resume-guard clause (see
    ``_guard_resume``).
    """
    if scaffold_variant not in cfs.SCAFFOLD_VARIANTS:
        raise ValueError(f"unknown scaffold variant {scaffold_variant!r}; "
                         f"expected one of {cfs.SCAFFOLD_VARIANTS}")
    out = Path(out_path)
    meta_path = out.with_suffix(".meta.json")
    leg_meta = _control_leg_meta(
        model=model, scaffold_variant=scaffold_variant,
        fewshot_k=int(fewshot_k if fewshot_k is not None else cfs.default_k()),
        slice_sha=slice_sha or mmlu.slice_sha256(),
        store_sha=store_sha or cfs.store_sha256(),
        reason=bool(reason), budget=int(budget))
    recs = _guard_resume(out, meta_path, leg_meta)
    meta_path.write_text(json.dumps(leg_meta, indent=2))
    todo = [c for c in items if c.item_label not in recs]

    def _one(c):
        fs = fewshot(c) if callable(fewshot) else fewshot
        return c, elicit_cell(base_url, model, c.evidence_text, c.criterion_text,
                              fewshot=fs, reason=reason, budget=budget)

    return _checkpointing_loop(items, todo, recs, out, _one, workers)


# ---------------------------------------------------------------------------
# Chat-API twin (the gemma31 post transport, per D5)
# ---------------------------------------------------------------------------

INSTRUCTION = (
    "\n\nReason step by step about which option is correct, then finish your reply with a "
    "single JSON object of exactly the shape shown in the examples — the five fields "
    "answer_letter, answer_distribution, answer_justification, confidence_distribution, "
    "confidence_justification, with every probability a multiple of 0.05 and each "
    "distribution summing to 1.00. The JSON object must be the last thing in your reply."
)


def elicit_cell_api(key: Optional[str], model: str, evidence_text: str, criterion_text: str,
                    fewshot: str, base_url: Optional[str] = None) -> dict:
    content = (f"{fewshot}You are answering a multiple-choice question that has exactly "
               f"one correct option.\n\n"
               f"Evidence:\n{evidence_text}\n\nCriterion:\n{criterion_text}{INSTRUCTION}")
    out = eva._chat(key, model, content, base_url=base_url)
    choice = out["choices"][0]
    text = choice["message"].get("content") or ""
    rec = parse_last_control(text)
    rec.update({
        "channel": eva.SELF_HOSTED_CHANNEL if base_url else "verbalized_api_chat",
        "api_provider": out.get("provider"), "api_model": out.get("model"),
        "finish_reason": choice.get("finish_reason"), "reply_chars": len(text),
    })
    if not rec.get("parse_ok"):
        rec["reply_tail"] = text[-400:]
    return rec


def run_variant_api(model: str, items, out_path: str, *, fewshot_by_subject: Dict[str, str],
                    fewshot_k: int, workers: int = 4, key: Optional[str] = None,
                    scaffold_variant: str = "baseline", base_url: Optional[str] = None,
                    slice_sha: Optional[str] = None, store_sha: Optional[str] = None) -> dict:
    """Chat-transport control leg — same sidecar/resume discipline, api-channel-pinned.

    ``base_url`` set => self-hosted vLLM chat (bf16, our own box): no Keychain read, no
    provider pinning, channel ``verbalized_vllm_chat``. This is the D5 transport for the
    gemma31 POST leg, matching the Phase-1a pairing (raw pre x vllm-chat post).
    """
    if scaffold_variant not in cfs.SCAFFOLD_VARIANTS:
        raise ValueError(f"unknown scaffold variant {scaffold_variant!r}; "
                         f"expected one of {cfs.SCAFFOLD_VARIANTS}")
    out = Path(out_path)
    meta_path = out.with_suffix(".meta.json")
    if base_url:
        key = None
        extra = {"dtype": eva.SELF_HOSTED_DTYPE, "api": "vllm_chat",
                 "endpoint_host": urllib.parse.urlparse(
                     base_url if "//" in base_url else "//" + base_url).hostname or ""}
        channel = eva.SELF_HOSTED_CHANNEL
    else:
        key = key or eva.keychain("openrouter-api-key")
        extra = {"quantizations": eva.QUANTIZATIONS, "api": "openrouter"}
        channel = "verbalized_api_chat"
    leg_meta = _control_leg_meta(
        model=model, scaffold_variant=scaffold_variant, fewshot_k=int(fewshot_k),
        slice_sha=slice_sha or mmlu.slice_sha256(),
        store_sha=store_sha or cfs.store_sha256(), **extra)
    leg_meta["channel"] = channel
    recs = _guard_resume(out, meta_path, leg_meta, unguarded=eva.UNGUARDED_META)
    meta_path.write_text(json.dumps(leg_meta, indent=2))
    todo = [c for c in items if c.item_label not in recs]

    def _one(c):
        return c, elicit_cell_api(key, model, c.evidence_text, c.criterion_text,
                                  fewshot_by_subject[c.criterion_id], base_url=base_url)

    return _checkpointing_loop(items, todo, recs, out, _one, workers)


# ---------------------------------------------------------------------------
# Views + gate summary (shared machinery, control-labelled)
# ---------------------------------------------------------------------------

def answer_view(recs: Dict[str, dict]) -> Dict[str, List[float]]:
    """{item_label: [p_A, p_B, p_C, p_D]} over parsed items."""
    return ev.compliance_view(recs)


def confidence_view(recs: Dict[str, dict]) -> Dict[str, List[float]]:
    """{item_label: [low, medium, high]} over parsed items."""
    return ev.confidence_view(recs)


def contract_compliance_summary(recs: Dict[str, dict], n_cells: int) -> dict:
    """The control gate numbers — ``elicit_verbalized``'s summary, control-labelled.

    Scoped to ``parse_ok`` per D3: ``contract_complete`` is a FIVE-field tier here and is
    reported, not gated, because it is not the same object as the AIReg six-field tier.
    """
    out = ev.contract_compliance_summary(recs, n_cells=n_cells)
    out["contract"] = CONTRACT
    out["mean_n_findings"] = None      # findings dropped by D3 — never emitted
    parsed = [r for r in recs.values() if r.get("parse_ok")]
    out["answer_letter_valid_rate"] = (
        sum(1 for r in parsed if r.get("answer_letter") in OPTION_LABELS) / len(parsed)
    ) if parsed else None
    return out
