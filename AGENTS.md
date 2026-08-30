# AGENTS.md — judex-calibration

## Scope and required reading

This repository contains the experimental calibration studies for JUDEX. Before
changing scientific logic, protocols, model rosters, result interpretation, or paper
claims, read this file, `README.md`, and `CLAUDE.md`. The latter is the detailed legacy
state record; its “Current state,” integrity-invariant, and adopted-protocol sections
remain required reading, but dated handoffs and superseded sections are historical.
Verify any result against the current configs, tracked protocol documents, and artifacts.

When this checkout is nested under the JUDEX umbrella, the parent `AGENTS.md` also
applies. In a standalone checkout, retain these shared rules: use
`~/.claude/bin/judex-python`, use `develop` as the integration branch, keep secrets out
of files and logs, and obtain explicit approval before any paid or live action.

## Scientific invariants

- Keep the logit-channel Study A record distinct from the verbalized/production-channel
  Study B program. Do not transfer constants, bands, or conclusions between channels
  unless an adopted protocol explicitly permits it.
- The adopted r3 ratio estimand and the frozen rA1 absolute-mechanism estimand are
  different questions. Neither supersedes the other; never merge their constants or
  describe one as a validation of the other.
- Do not edit frozen protocol constants, acceptance gates, registered decision ladders,
  or results of record in place. A correction needs a new amendment or version with
  explicit provenance.
- Fit temperatures on this repository's explicit `T_BOUNDS = (0.25, 20.0)` wherever
  the study protocol requires it. A boundary value is a peg, not an interior fit.
- Never mix fitting objectives as though W1, RPS, Murphy reliability, entropy, and Brier
  temperatures were directly commensurable.
- Ground truth must come from the canonical tracked ground-truth bundle, not a
  `runs/` metrics report. Re-derive values after a ground-truth vintage change.
- Closed-pair work uses the evaluator's current production configuration bundle. Verify
  the config directory and model-family identifiers from current protocol/config files;
  do not infer them from an old run name.
- Quantized or smoke-model runs validate plumbing only. Never promote their scientific
  outputs into a live-study result.

## Execution and cost controls

- Run `~/.claude/bin/judex-keys-preload` before any key-using job, then export required
  keys only in the same command as the job. Never expose values or lengths.
- Live studies, provider calls, rented compute, top-ups, and teardown require explicit
  user authorization. Build and validate all free harnesses before renting hardware.
- A run ID is immutable experiment identity. Resume only an exact matching experiment;
  use a fresh run directory when any pinned input changes.
- Preserve raw responses, sidecars, provenance, and result reports needed for audit.
  `runs/` is often ignored and machine-local, so do not assume another checkout has it.
- On rented inference hardware, follow the relevant adopted runbook exactly; do not
  lower numerical precision merely to fit a cheaper host when logits are the estimand.

## Verification

Use the JUDEX wrapper for all commands. Run the narrow tests for edited modules first,
then the repository suite when practical:

```bash
~/.claude/bin/judex-python -m unittest discover -s tests -p 'test_*.py'
```

Protocol drivers with a documented `--selftest` gate must pass it before analysis or
live execution. Check diffs for stale constants, mismatched config bundles, and claims
that omit known limitations.
