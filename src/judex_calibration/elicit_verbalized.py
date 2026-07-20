"""Study B: verbalized-channel elicitation over /v1/completions (base AND post).

The production JUDEX evaluators emit the contract-0.2.0 verbalized output — a 5-level
``compliance_distribution`` and a 3-level ``confidence_distribution`` as 0.05-grid JSON —
not logits. Study B measures the pre/post phenomenon in THAT channel: both legs of a
family emit a reduced-contract JSON via k-shot few-shot continuation on the same
``/v1/completions`` endpoint Study A used, so pre/post are compared in the SAME channel
with the SAME scaffold, isolating the weights (Study A's design, ported to the
verbalized channel).

Two stages per cell (Approach C, verbalized):
  1) generate the reasoning span, stopping before the JSON scaffold;
  2) re-feed ``prompt + reasoning + '\\nJSON:'`` and greedy-generate the JSON object.
Parsing is strict-but-repairing: balanced-brace extraction, then validation of both
distributions (all keys present, non-negative, positive sum) with renormalization.
A cell that fails to parse is RECORDED as a failure — never resampled (temperature 0 is
deterministic) and never silently rerouted to the logit channel (B-Q1 is the question).

The few-shot exemplars come from the corpus-v2 dimension store, whose rows carry the
full contract shape (findings, both distributions, both justifications) — rendered here
as (Evidence, Criterion, Reasoning, FULL-contract JSON) so the base sees exactly the
shape it must continue. FULL contract (USER DECISION 2026-07-20, superseding the B0
reduced-contract draft): all six top-level fields of contract 0.2.0, in contract order —
findings, compliance_level, compliance_distribution, compliance_justification,
confidence_distribution, confidence_justification. The B-Q1 gate is two-tier:
``parse_ok`` (both distributions valid — the minimum for tau_v scoring) and
``contract_complete`` (all six fields well-formed — the gated rate).

stdlib only (urllib). Pure HTTP/parse logic, unit-tested without a live server.
"""
from __future__ import annotations

import json, os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .elicit_base import _completions  # same transport; monkeypatched in tests
from . import fewshot as fewshot_mod

LABELS = ("very_low", "low", "moderate", "high", "very_high")
CONF_LABELS = ("low", "medium", "high")
JSON_SCAFFOLD = "\nJSON:"

# The verbalized channel's zero-mass trap: the 0.2.0 contract REQUIRES zero-mass levels
# written as 0.0, and ln 0 = -inf, so every inverse-softmax operation on this channel
# needs an epsilon floor. PRE-REGISTERED at half the 0.05 elicitation-grid step; applied
# at ANALYSIS time (floor_and_renormalize below), never at storage — stored vectors are
# the raw parsed+renormalized emissions. Recorded in the leg meta sidecar so a resume
# under a different epsilon convention hard-errors.
EPSILON = 0.005


def floor_and_renormalize(probs: Sequence[float], eps: float = EPSILON) -> List[float]:
    """Epsilon-floor grid zeros, then renormalize — the mandatory pre-inverse-softmax step.

    After this, judex.calibration's ops (whose internal 1e-12 clip would otherwise
    manufacture ~-27.6 logits out of grid zeros) apply unchanged.
    """
    floored = [max(float(p), eps) for p in probs]
    z = sum(floored)
    return [p / z for p in floored]


# ---------------------------------------------------------------------------
# Few-shot: contract-shaped rendering from the corpus-v2 dimension store
# ---------------------------------------------------------------------------

def _dist_json(d: Dict[str, float], keys: Sequence[str]) -> str:
    """Canonical-order JSON for a distribution ({} preserved as 0.0, two-decimal)."""
    return "{" + ", ".join(f'"{k}": {float(d.get(k, 0.0)):.2f}' for k in keys) + "}"


FINDING_KEYS = ("requirement", "status", "evidence")
FINDING_STATUSES = ("met", "partially_met", "unmet", "indeterminate")


def render_answer_json(row: dict) -> str:
    """The FULL-contract JSON a few-shot exemplar answers with (contract 0.2.0 field order)."""
    findings = [{k: f.get(k, "") for k in FINDING_KEYS} for f in row.get("findings", [])]
    return ("{" + f'"findings": {json.dumps(findings)}, '
            f'"compliance_level": "{row["compliance_level"]}", '
            f'"compliance_distribution": {_dist_json(row["compliance_distribution"], LABELS)}, '
            f'"compliance_justification": {json.dumps((row.get("compliance_justification") or "").strip())}, '
            f'"confidence_distribution": {_dist_json(row["confidence_distribution"], CONF_LABELS)}, '
            f'"confidence_justification": {json.dumps((row.get("confidence_justification") or "").strip())}' + "}")


def render_block(row: dict, criterion_text: str) -> str:
    """One (excerpt, criterion, reasoning, reduced-contract JSON) few-shot example."""
    justification = (row.get("compliance_justification") or "").strip()
    return (f"Example.\nEvidence:\n{row['text']}\n\n"
            f"Criterion:\n{criterion_text}\n\n"
            f"Reasoning: {justification}\n"
            f"JSON: {render_answer_json(row)}\n\n")


def build_fewshot(criterion_id: str, criterion_text: str, k: Optional[int] = None,
                  exclude: Optional[Sequence[str]] = None) -> str:
    """k-example contract-shaped few-shot prefix, same selection as the token-slice leg.

    Reuses ``fewshot.select_rows`` (stratified fixed_set over levels x raters) so Study B
    picks the SAME exemplar rows Study A did — only the rendering differs.
    """
    if k is None:
        k = fewshot_mod.default_k()
    rows = fewshot_mod.select_rows(criterion_id, k, exclude=exclude)
    rows.sort(key=lambda r: int(r["compliance_1to5"]))
    return "".join(render_block(r, criterion_text) for r in rows)


def build_fewshot_by_criterion(cells, k: Optional[int] = None) -> Dict[str, str]:
    """{criterion_id: contract-shaped few-shot string} for every criterion in ``cells``."""
    text_by_crit: Dict[str, str] = {}
    for c in cells:
        text_by_crit.setdefault(c.criterion_id, c.criterion_text)
    return {cid: build_fewshot(cid, txt, k=k) for cid, txt in text_by_crit.items()}


# ---------------------------------------------------------------------------
# Elicitation
# ---------------------------------------------------------------------------

def build_prompt(evidence_text: str, criterion_text: str, fewshot: str = "") -> str:
    return (f"{fewshot}You are assessing EU AI Act compliance. Reason step by step, then "
            f"answer with a single JSON object of the exact shape shown in the examples.\n\n"
            f"Evidence:\n{evidence_text}\n\nCriterion:\n{criterion_text}\n\nReasoning:")


def generate_reasoning(base_url: str, model: str, prompt: str, budget: int = 2048) -> str:
    """Stage 1: reasoning span, stopping before the JSON scaffold (or a runaway Example)."""
    out = _completions(base_url, {
        "model": model, "prompt": prompt, "max_tokens": budget, "temperature": 0,
        "stop": ["JSON:", "Example.", "</think>"],
    })
    return out["choices"][0].get("text", "")


def generate_json(base_url: str, model: str, answer_prompt: str, max_tokens: int = 1600) -> str:
    """Stage 2: greedy JSON continuation after the scaffold."""
    out = _completions(base_url, {
        "model": model, "prompt": answer_prompt, "max_tokens": max_tokens, "temperature": 0,
        "stop": ["Example.", "\n\n\n"],
    })
    return out["choices"][0].get("text", "")


def extract_balanced_object(text: str) -> Optional[str]:
    """First balanced {...} object in ``text`` (string-literal aware), else None."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _valid_dist(obj, keys: Sequence[str]) -> Optional[List[float]]:
    """Parse one distribution: all keys present, numeric, non-negative, positive sum.

    Returns the RAW (renormalized) vector or None. Renormalization tolerance is
    unbounded by design — the raw sum is reported separately (``*_sum``) so gross
    non-distributions are visible in QA rather than silently laundered.
    """
    if not isinstance(obj, dict):
        return None
    try:
        vals = [float(obj[k]) for k in keys]
    except (KeyError, TypeError, ValueError):
        return None
    if any(v < 0 for v in vals) or sum(vals) <= 0:
        return None
    z = sum(vals)
    return [v / z for v in vals]


def _on_grid(vals: Sequence[float], step: float = 0.05, tol: float = 1e-6) -> bool:
    return all(abs(v / step - round(v / step)) <= tol for v in vals)


def parse_contract_json(text: str) -> dict:
    """Parse a stage-2 emission into the Study B cell record.

    Never raises: parse failure returns ``{"parse_ok": False, "parse_error": ...}``
    plus the raw text length, so the failure is a recorded observation (the B-Q1
    contract-compliance gate counts these), not an exception.
    """
    blob = extract_balanced_object(text)
    if blob is None:
        return {"parse_ok": False, "parse_error": "no_json_object", "raw_chars": len(text)}
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as e:
        return {"parse_ok": False, "parse_error": f"json_decode: {e.msg}", "raw_chars": len(text)}
    comp_raw = obj.get("compliance_distribution")
    conf_raw = obj.get("confidence_distribution")
    comp = _valid_dist(comp_raw, LABELS)
    conf = _valid_dist(conf_raw, CONF_LABELS)
    if comp is None or conf is None:
        which = [n for n, v in (("compliance_distribution", comp), ("confidence_distribution", conf)) if v is None]
        return {"parse_ok": False, "parse_error": f"invalid: {'+'.join(which)}", "raw_chars": len(text)}
    comp_vals = [float(comp_raw[k]) for k in LABELS]
    conf_vals = [float(conf_raw[k]) for k in CONF_LABELS]
    level = obj.get("compliance_level")
    # contract_complete: the FULL-contract tier (the B-Q1 gated rate) — all six 0.2.0
    # fields well-formed, not just the two distributions parse_ok requires.
    findings = obj.get("findings")
    findings_ok = (isinstance(findings, list) and len(findings) > 0
                   and all(isinstance(f, dict) and all(k in f for k in FINDING_KEYS)
                           and f.get("status") in FINDING_STATUSES for f in findings))
    missing = [name for name, ok in (
        ("findings", findings_ok),
        ("compliance_level", level in LABELS),
        ("compliance_justification", bool(str(obj.get("compliance_justification") or "").strip())),
        ("confidence_justification", bool(str(obj.get("confidence_justification") or "").strip())),
    ) if not ok]
    return {
        "parse_ok": True,
        "compliance": comp, "confidence": conf,
        "compliance_level": level if level in LABELS else None,
        "level_matches_argmax": (level == LABELS[comp.index(max(comp))]) if level in LABELS else False,
        "contract_complete": not missing,
        "contract_missing": missing,
        "n_findings": len(findings) if isinstance(findings, list) else 0,
        # QA diagnostics (reported, not gated): raw sums pre-renormalization + 0.05-grid conformance
        "compliance_sum": round(sum(comp_vals), 4), "confidence_sum": round(sum(conf_vals), 4),
        "compliance_on_grid": _on_grid(comp_vals), "confidence_on_grid": _on_grid(conf_vals),
    }


def elicit_cell(base_url: str, model: str, evidence_text: str, criterion_text: str,
                fewshot: str = "", reason: bool = True, budget: int = 2048) -> dict:
    """One cell through the verbalized channel; returns the parsed record + provenance."""
    prompt = build_prompt(evidence_text, criterion_text, fewshot)
    reasoning = generate_reasoning(base_url, model, prompt, budget) if reason else ""
    raw = generate_json(base_url, model, prompt + reasoning + JSON_SCAFFOLD)
    rec = parse_contract_json(raw)
    rec.update({"channel": "verbalized", "reasoning_chars": len(reasoning)})
    return rec


def run_variant(base_url: str, model: str, cells, out_path: str, *,
                fewshot="", reason: bool = True, budget: int = 2048,
                fewshot_k: "int | None" = None, workers: int = 1) -> dict:
    """Elicit every cell; write the full per-cell record map; checkpoint + sidecar.

    Output layout (one file, two derivable views):
      ``<out>.json``            {item_label: full cell record} — the artifact of record.
      ``compliance_view()`` / ``confidence_view()`` project it to the {label: [p..]}
      shape ``study_a`` machinery consumes (parse-failed cells are absent — study_a
      then scores the parsed subset, and the B-Q1 gate reports the failure count).

    Same checkpoint/resume/meta-sidecar discipline as elicit_base.run_variant, with the
    channel + epsilon convention pinned: resume is CRASH RECOVERY ONLY, and a resume
    against a different (model, reason, budget, fewshot_k, channel, epsilon) hard-errors.
    """
    out = Path(out_path)
    meta_path = out.with_suffix(".meta.json")
    leg_meta = {"model": model, "reason": bool(reason), "budget": int(budget),
                "channel": "verbalized", "epsilon": EPSILON}
    if fewshot_k is not None:
        leg_meta["fewshot_k"] = int(fewshot_k)
    recs: Dict[str, dict] = {}
    if out.exists():
        recs = json.loads(out.read_text())
        prior = json.loads(meta_path.read_text()) if meta_path.exists() else None
        mismatched = (prior is not None
                      and any(prior[k] != leg_meta.get(k) for k in prior))
        if recs and (prior is None or mismatched):
            raise RuntimeError(
                f"{out} holds {len(recs)} cells from a different leg config "
                f"(sidecar {prior} != requested {leg_meta}); resume is crash-recovery "
                f"only — use a fresh --out for a different experiment")
        if recs:
            todo = sum(1 for c in cells if c.item_label not in recs)
            print(f"[{out.stem}] resuming: {len(recs)} cells cached, {todo} to go", flush=True)
    meta_path.write_text(json.dumps(leg_meta, indent=2))
    todo = [c for c in cells if c.item_label not in recs]

    def _one(c):
        fs = fewshot(c) if callable(fewshot) else fewshot
        return c, elicit_cell(base_url, model, c.evidence_text, c.criterion_text,
                              fewshot=fs, reason=reason, budget=budget)

    def _record(c, d):
        recs[c.item_label] = d
        tmp = out.parent / (out.name + ".tmp")
        tmp.write_text(json.dumps(recs, indent=2))
        os.replace(tmp, out)
        status = "ok" if d.get("parse_ok") else f"PARSE-FAIL ({d.get('parse_error')})"
        print(f"[{out.stem}] {len(recs)}/{len(cells)} {c.item_label} ({status})", flush=True)

    if workers <= 1:
        for c in todo:
            _record(*_one(c))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_one, c) for c in todo]
            for f in as_completed(futs):
                _record(*f.result())
    return recs


# ---------------------------------------------------------------------------
# Views + the B-Q1 gate summary
# ---------------------------------------------------------------------------

def compliance_view(recs: Dict[str, dict]) -> Dict[str, List[float]]:
    """{item_label: [5 probs]} over the parsed cells — study_a-compatible."""
    return {k: r["compliance"] for k, r in recs.items() if r.get("parse_ok")}


def confidence_view(recs: Dict[str, dict]) -> Dict[str, List[float]]:
    """{item_label: [low, medium, high]} over the parsed cells."""
    return {k: r["confidence"] for k, r in recs.items() if r.get("parse_ok")}


def contract_compliance_summary(recs: Dict[str, dict], n_cells: int) -> dict:
    """The B-Q1 gate numbers: parse rate + shape QA over one leg."""
    parsed = [r for r in recs.values() if r.get("parse_ok")]
    n = len(recs)
    fails: Dict[str, int] = {}
    for r in recs.values():
        if not r.get("parse_ok"):
            fails[r.get("parse_error", "?")] = fails.get(r.get("parse_error", "?"), 0) + 1
    complete = [r for r in parsed if r.get("contract_complete")]
    miss_counts: Dict[str, int] = {}
    for r in parsed:
        for m in r.get("contract_missing", []):
            miss_counts[m] = miss_counts.get(m, 0) + 1
    return {
        "n_cells": n_cells, "n_elicited": n, "n_parsed": len(parsed),
        "parse_rate": (len(parsed) / n) if n else 0.0,
        "parse_failures": fails,
        # FULL-contract tier (user decision 2026-07-20): the B-Q1 gated rate
        "n_contract_complete": len(complete),
        "contract_complete_rate": (len(complete) / n) if n else 0.0,
        "contract_missing_counts": miss_counts,
        "mean_n_findings": (sum(r.get("n_findings", 0) for r in parsed) / len(parsed)) if parsed else None,
        "level_matches_argmax_rate": (sum(r.get("level_matches_argmax", False) for r in parsed) / len(parsed)) if parsed else None,
        "compliance_on_grid_rate": (sum(r.get("compliance_on_grid", False) for r in parsed) / len(parsed)) if parsed else None,
        "confidence_on_grid_rate": (sum(r.get("confidence_on_grid", False) for r in parsed) / len(parsed)) if parsed else None,
        "epsilon": EPSILON,
    }
