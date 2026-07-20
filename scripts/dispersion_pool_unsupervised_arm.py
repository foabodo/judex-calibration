#!/usr/bin/env python3
"""Pre-registered GT-free dispersion-pool estimator — the E6 unsupervised arm.

This is the FROZEN instrument for the held on-pair sweep (Sonnet 4.6 + GPT 5.4,
24 docs x 5 Articles = 120 cells). It computes the decorrelated-dispersion-pool
temperature for a closed-pair run with NO supervision on the closed models:

  pool      = the three gate-passing Study A base (pre) distributions
              {qwen_k5, gemma31_k5, llama31}, raw (NOT one-hot), equal weight,
              AUGMENTING the run's own cross-family Phase-1 replicates;
  statistic = judex.calibration.fit_dispersion_temperature (entropy-consistency,
              pooled per-item squared gap), bounds (0.25, 20.0) explicit;
  estimate  = log-space shrinkage of the raw fit toward the pre-registered
              sensitivity band's geometric midpoint:
                T_hat = exp(0.65*ln T_raw + 0.35*ln sqrt(1.6008*4.8772)).

Provenance: promotion analysis 2026-07-19 (umbrella
spec/analysis_2026_07_19_dispersion_pool_promotion.md), as AMENDED 2026-07-20
by the band amendment (umbrella spec/amendment_2026_07_20_band_glm_excision.md:
GLM's gate-failed tau_oc excised; band = the gate-passing range [1.60, 4.88],
anchor = its geometric midpoint 2.794; lambda re-selected by the same frozen
rule — see the promotion doc's amendment block). The shrinkage repairs
the two characterized defects of the raw fit — closed-run overshoot from
objective flatness under heterogeneous item widths, and the sharpening flip
when the pool loses its wide member — at <=3pp mean RPS-gap cost on the six
validation targets. Do NOT retune LAMBDA, the pool membership, or the anchor
against the on-pair run; that would unfreeze the pre-registration.

GT rule: judex_calibration.aireg.load_cells() is used ONLY as the
(document_id, article) -> item_label join index. Ground-truth probabilities
are never read by this script; the supervised arm lives elsewhere.

Run from the calibration checkout/worktree:
  JUDEX_UMBRELLA=/path/to/judex PYTHONPATH=src python \
      scripts/dispersion_pool_unsupervised_arm.py \
      --closed-run-dir /path/to/judex-evaluator/runs/<run-id> \
      --bases-runs-dir /path/to/main-checkout/judex-calibration/runs \
      --out <dir>
  --selftest reproduces the two 2026-07-19 legacy-run numbers (INDICATIVE,
  off-pair: Gemini+GPT, not the current closed pair).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from judex.calibration import fit_dispersion_temperature, realized_dispersion  # noqa: E402
from judex.core import ComplianceDistribution  # noqa: E402
from judex.experiments import replicates_by_level_from_cross_family  # noqa: E402

from judex_calibration import aireg  # noqa: E402  (join index only — GT never read)
from judex_calibration.study_a import T_BOUNDS, saturated  # noqa: E402

# ----------------------------------------------------------------- FROZEN SPEC
POOL_BASES = {  # family -> base (pre) leg file name under --bases-runs-dir
    "qwen": "qwen_k5/pre.json",
    "gemma31": "gemma31_k5/pre.json",
    "llama31": "llama31/pre.json",
}
REPLICATE_SOURCE = "cross_family_phase1"  # augment construction: cfp1 + bases
# Band + anchor amended 2026-07-20 (GLM's gate-failed tau_oc excised —
# spec/amendment_2026_07_20_band_glm_excision.md): band = gate-passing tau_oc
# range; anchor = geometric midpoint of the exact shipped values.
BAND = (1.60, 4.88)                       # pre-registered E6 sensitivity band (amended)
ANCHOR = math.sqrt(1.6007889248849985 * 4.877199362107526)  # 2.7942 (band geometric midpoint)
LAMBDA = 0.35                             # log-space shrinkage weight (re-selected 2026-07-20
                                          # by the frozen rule under the amended anchor:
                                          # plateau {0.25,0.30,0.35} tied, best mean -> 0.35)
MIN_POOL_ENTROPY = 1.00                   # nats; vetting floor (flip at ~0.96)
# Concurrence criterion for the sweep analysis (advisory here, SECONDARY —
# RPS-gap closure is the primary acceptance): the arms concur when
# |ln(T_hat / T_star)| <= ln(CONCURRENCE_RATIO). Frozen 2026-07-19 at the
# doc-clustered bootstrap envelope of |ln(T_hat/T_star)| on the legacy closed
# validation runs (then q95 = 1.18 on phase23 => e^1.18 ~= 3.25). Kept frozen
# by the 2026-07-20 amendment: the amended-estimator envelope tightens
# (q95 = 1.04 => 2.83), so 3.25 remains conservative. Wide because BOTH
# arms' temperatures are within-basin weakly identified on closed runs.
# NOTE (amendment knock-on): under the amended band the legacy closed runs'
# shrunk T_hat (5.24 / 5.46) sit ABOVE the band's upper edge — A3 is
# materially stricter for closed-run-like targets than under the r1 freeze.
CONCURRENCE_RATIO = 3.25

LABELS = ("very_low", "low", "moderate", "high", "very_high")
FULL_TO_SHORT = {
    "Very low probability of compliance": "very_low",
    "Low probability of compliance": "low",
    "Moderate probability of compliance": "moderate",
    "High probability of compliance": "high",
    "Very high probability of compliance": "very_high",
}

# 2026-07-19 promotion-analysis raw fits (legacy off-pair runs) for --selftest.
# T_raw is anchor-independent (unchanged by the 2026-07-20 amendment); the
# selftest additionally re-pins T_hat against the amended anchor via shrink().
SELFTEST_EXPECT = {
    "stage9-gemini-gpt-medium": 7.352,
    "phase23-deference-fix-native": 7.822,
}


def default_umbrella() -> Path:
    env = os.environ.get("JUDEX_UMBRELLA")
    return Path(env) if env else REPO.parent


def canon_payload(dist_payload: dict) -> ComplianceDistribution:
    by_label = dist_payload["by_label"]
    return ComplianceDistribution.from_values(
        [float(by_label[full]) for full in FULL_TO_SHORT], LABELS)


def recanon(dist: ComplianceDistribution) -> ComplianceDistribution:
    by = dict(zip(dist.labels, dist.probabilities))
    return ComplianceDistribution.from_values([by[full] for full in FULL_TO_SHORT], LABELS)


def load_bases(bases_runs_dir: Path) -> dict[str, dict[str, ComplianceDistribution]]:
    pools = {}
    for fam, rel in POOL_BASES.items():
        raw = json.loads((bases_runs_dir / rel).read_text())
        pools[fam] = {label: ComplianceDistribution.from_values([float(p) for p in probs], LABELS)
                      for label, probs in raw.items()}
        if len(pools[fam]) != 120:
            raise SystemExit(f"base leg {rel}: expected 120 cells, got {len(pools[fam])}")
    return pools


def load_closed_run(run_dir: Path, by_doc_article) -> tuple[dict, dict]:
    """{item_label: reported final dist}, {item_label: [cfp1 replicates]}."""
    finals: dict[str, ComplianceDistribution] = {}
    reps: dict[str, list[ComplianceDistribution]] = {}
    for doc_dir in sorted((run_dir / "documents").iterdir()):
        payload_path = doc_dir / "cross_family_evaluation.json"
        if not payload_path.exists():
            continue
        payload = json.loads(payload_path.read_text())
        doc_id = payload["document_id"]
        per_level = replicates_by_level_from_cross_family(payload, source=REPLICATE_SOURCE)
        for level in payload["phase4"]["reconciled_levels"]:
            if level.get("hierarchy_level") != "dimension":
                continue
            article = level["level_id"].split(":")[0].replace("article_", "")
            cell = by_doc_article.get((doc_id, article))
            if cell is None:
                raise SystemExit(f"no AIReg cell for ({doc_id}, {article})")
            finals[cell.item_label] = canon_payload(level["final_distribution"])
            reps[cell.item_label] = [recanon(r) for r in per_level.get(level["level_id"], [])]
    return finals, reps


def estimate(finals, cfp1_reps, bases) -> dict:
    labels = sorted(finals)
    pool = {lb: list(cfp1_reps.get(lb, [])) + [bases[f][lb] for f in POOL_BASES] for lb in labels}
    fit = fit_dispersion_temperature([(finals[lb], pool[lb]) for lb in labels], bounds=T_BOUNDS)
    t_raw = fit.temperature
    t_hat = math.exp((1 - LAMBDA) * math.log(t_raw) + LAMBDA * math.log(ANCHOR))

    # vetting diagnostics (bases-only realized entropy + wide-member dependence)
    def pool_entropy(fams):
        return sum(realized_dispersion([bases[f][lb] for f in fams]).mixture_entropy
                   for lb in labels) / len(labels)

    h_bases = pool_entropy(list(POOL_BASES))
    lobo_h = {f: pool_entropy([g for g in POOL_BASES if g != f]) for f in POOL_BASES}
    return {
        "estimator": "dispersion_pool_unsupervised_arm",
        "frozen": {"pool_bases": POOL_BASES, "replicate_source": REPLICATE_SOURCE,
                   "band": list(BAND), "anchor": ANCHOR, "lambda": LAMBDA,
                   "t_bounds": list(T_BOUNDS), "min_pool_entropy": MIN_POOL_ENTROPY,
                   "concurrence_ratio": CONCURRENCE_RATIO},
        "item_count": len(labels),
        "T_raw": t_raw,
        "T_raw_saturated": saturated(t_raw),
        "T_hat": t_hat,
        "claimed_entropy_at_unit": fit.claimed_entropy_at_unit,
        "realized_entropy_mean": fit.realized_entropy_mean,
        "bases_pool_entropy": h_bases,
        "bases_pool_entropy_ok": h_bases >= MIN_POOL_ENTROPY,
        "lobo_pool_entropy": lobo_h,
        "lobo_entropy_warning": [f for f, h in lobo_h.items() if h < MIN_POOL_ENTROPY],
        "in_band": BAND[0] <= t_hat <= BAND[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--closed-run-dir", type=Path)
    parser.add_argument("--bases-runs-dir", type=Path,
                        default=default_umbrella() / "judex-calibration" / "runs")
    parser.add_argument("--supervised-t", type=float, default=None,
                        help="externally supplied supervised-arm T* for the advisory concurrence check")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--selftest", action="store_true",
                        help="reproduce the 2026-07-19 legacy-run raw fits (indicative, off-pair)")
    args = parser.parse_args()

    cells = aireg.load_cells()  # join index only — GT probabilities never read
    by_doc_article = {(c.document_id, str(c.article)): c for c in cells}
    bases = load_bases(args.bases_runs_dir)

    if args.selftest:
        eval_runs = default_umbrella() / "judex-evaluator" / "runs"
        ok = True
        for run_id, expected in SELFTEST_EXPECT.items():
            finals, reps = load_closed_run(eval_runs / run_id, by_doc_article)
            res = estimate(finals, reps, bases)
            expected_hat = math.exp((1 - LAMBDA) * math.log(expected) + LAMBDA * math.log(ANCHOR))
            good = (abs(math.log(res["T_raw"] / expected)) < 5e-3
                    and abs(math.log(res["T_hat"] / expected_hat)) < 5e-3)
            ok &= good
            print(f"[selftest] {run_id}: T_raw={res['T_raw']:.3f} expected~{expected} "
                  f"T_hat={res['T_hat']:.3f} expected~{expected_hat:.3f} "
                  f"{'OK' if good else 'MISMATCH'} (INDICATIVE off-pair)")
        raise SystemExit(0 if ok else 1)

    if not args.closed_run_dir:
        parser.error("--closed-run-dir is required (or use --selftest)")
    finals, reps = load_closed_run(args.closed_run_dir, by_doc_article)
    res = estimate(finals, reps, bases)
    res["closed_run_dir"] = str(args.closed_run_dir)
    if args.supervised_t is not None:
        ratio = res["T_hat"] / args.supervised_t
        res["supervised_t"] = args.supervised_t
        res["concurs_with_supervised"] = abs(math.log(ratio)) <= math.log(CONCURRENCE_RATIO)
    print(json.dumps(res, indent=1, default=float))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "dispersion_pool_unsupervised_arm.json").write_text(
            json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
