# judex-calibration

Study A — **open pre/post-pair calibration** for the JUDEX closed evaluators.

Measures the *clean post-training overconfidence temperature* by running the **pre-trained (base)** and **post-trained** variants of six open models on the AIReg-Bench validation cells (independent human GT), to decide whether a **transferred constant temperature** can correct closed-evaluator (Claude/GPT) overconfidence on out-of-sample documents — where DACA, internal-dispersion, and accuracy-contaminated supervised fits all failed.

- **Base leg:** vLLM on rented GPU, token-sliced logits over the 5 compliance levels.
- **Post leg:** API from local macOS (Claude Code orchestrates).
- **Analysis:** reuses `judex-evaluator` calibration tooling (`fit_temperature`, `fit_dispersion_temperature`, `dispersion_calibration_recovery`, `murphy_decomposition`).

**Start here:** [`docs/study_a_implementation_guide.md`](docs/study_a_implementation_guide.md). **Do nothing that costs money until Phase 0 (free accuracy pre-check) passes.**

Setup: `pip install -e ../judex-evaluator` then `pip install -e .`. Integration branch is `develop` (never `integration`).
