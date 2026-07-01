# judex-calibration

Study A — **open pre/post-pair calibration** for the JUDEX closed evaluators.

Measures the *clean post-training overconfidence temperature* by running the **pre-trained (base)** and **post-trained** variants of six open models on the AIReg-Bench validation cells (independent human GT), to decide whether a **transferred constant temperature** can correct closed-evaluator (Gemini/GPT) overconfidence on out-of-sample documents — where DACA, internal-dispersion, and accuracy-contaminated supervised fits all failed.

- **Base leg:** vLLM on rented GPU, token-sliced logits over the 5 compliance levels.
- **Post leg:** API from local macOS (Claude Code orchestrates).
- **Analysis:** reuses `judex-evaluator` tooling verbatim — `judex.calibration` (`fit_temperature`, `apply_temperature`, `fit_dispersion_temperature`) and `judex.experiments` (`murphy_decomposition`, `dispersion_calibration_recovery`) — scored against the **canonical, manifest-verified** AIReg GT (`aireg.py` synthesizes it from the git-tracked `judex-ground-truth` bundle; no run dependency). See [`docs/integration_remediation_2026_07_01.md`](docs/integration_remediation_2026_07_01.md).

**Start here:** [`docs/study_a_implementation_guide.md`](docs/study_a_implementation_guide.md). **Do nothing that costs money until Phase 0 (free accuracy pre-check) passes.** To validate the whole pipeline on your own GPU before any spend, see [`docs/local_smoke_quickstart.md`](docs/local_smoke_quickstart.md) (plumbing smoke — a small model on a local CUDA box; not a scientific run).

Setup: `pip install -e ../judex-evaluator` then `pip install -e .`. Integration branch is `develop` (never `integration`).
