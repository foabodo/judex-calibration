"""Tests for the tau_DACA triangulation leg (fit_tau_daca / daca_triangulation).

Synthetic cells only — no live server, no run artifacts, no AIReg bundle read —
so these run on a fresh clone and on a vast box. The scenarios pin the contract:
GT never enters the fit (alignment is to the reference), the agreement filter
actually filters, an overconfident closed leg aligned to a well-spread reference
yields tau > 1, saturation and below-chance-reference guards fire, and the
triangulation summary compares correctly against the other two estimators.
"""
import math
import unittest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from judex_calibration.aireg import Cell
from judex_calibration.study_a import (
    T_BOUNDS,
    daca_triangulation,
    fit_tau_daca,
)

LABELS = ("very_low", "low", "moderate", "high", "very_high")


def _cell(i, gt_probs):
    gt = tuple(gt_probs)
    return Cell(
        item_id=f"item{i}", item_label=f"Art 9 / cell {i}", document_id=f"doc{i % 4}",
        article="9", criterion_id="article_9_rms", criterion_text="", evidence_text="",
        gt_probs=gt, gt_labels=LABELS, gt_argmax=max(range(5), key=lambda k: gt[k]),
    )


# Ten cells, GT centered on moderate/high — GT is deliberately IRRELEVANT to the
# fit (only the gate metrics see it); tests assert that via a GT-swap invariance.
CELLS = [_cell(i, [0.05, 0.15, 0.45, 0.25, 0.10] if i % 2 else [0.05, 0.10, 0.30, 0.40, 0.15])
         for i in range(10)]

SHARP_MODERATE = [0.02, 0.03, 0.90, 0.03, 0.02]   # overconfident closed prediction
WIDE_MODERATE = [0.10, 0.20, 0.40, 0.20, 0.10]    # well-spread reference, same argmax
WIDE_HIGH = [0.05, 0.10, 0.25, 0.45, 0.15]        # reference disagreeing on argmax


def _preds(vec, labels=None):
    return {c.item_label: list(vec) for c in (CELLS if labels is None else labels)}


class FitTauDacaTests(unittest.TestCase):
    def test_overconfident_closed_vs_spread_reference_softens(self):
        out = fit_tau_daca(_preds(SHARP_MODERATE), _preds(WIDE_MODERATE), CELLS)
        self.assertEqual(out["n_overlap"], 10)
        self.assertEqual(out["n_agreement"], 10)      # same argmax everywhere
        self.assertGreater(out["tau_daca"], 1.0)      # softening required
        self.assertFalse(out["tau_daca_saturated"])
        self.assertEqual(out["objective"], "rps_alignment_to_reference")
        self.assertEqual(out["T_bounds"], list(T_BOUNDS))

    def test_gt_never_enters_the_fit(self):
        # Same closed/reference, GT argmax flipped to very_low everywhere: the
        # gate metrics move, tau_daca must not.
        flipped = [_cell(i, [0.60, 0.20, 0.10, 0.07, 0.03]) for i in range(10)]
        a = fit_tau_daca(_preds(SHARP_MODERATE), _preds(WIDE_MODERATE), CELLS)
        b = fit_tau_daca(_preds(SHARP_MODERATE), _preds(WIDE_MODERATE), flipped)
        self.assertAlmostEqual(a["tau_daca"], b["tau_daca"], places=9)
        self.assertNotEqual(a["reference_argmax_acc_gate_only"],
                            b["reference_argmax_acc_gate_only"])

    def test_agreement_filter_drops_disagreeing_cells(self):
        # Reference argmax = high on half the cells -> those cells are filtered out.
        reference = {}
        for i, c in enumerate(CELLS):
            reference[c.item_label] = list(WIDE_MODERATE if i % 2 else WIDE_HIGH)
        out = fit_tau_daca(_preds(SHARP_MODERATE), reference, CELLS)
        self.assertEqual(out["n_overlap"], 10)
        self.assertEqual(out["n_agreement"], 5)
        self.assertAlmostEqual(out["agreement_rate"], 0.5)

    def test_no_agreement_returns_nan_saturated(self):
        out = fit_tau_daca(_preds(SHARP_MODERATE), _preds(WIDE_HIGH), CELLS)
        self.assertEqual(out["n_agreement"], 0)
        self.assertTrue(math.isnan(out["tau_daca"]))
        self.assertTrue(out["tau_daca_saturated"])

    def test_below_chance_reference_is_flagged(self):
        # Reference argmax (moderate) never matches a very_low GT: accuracy 0 < 0.2.
        flipped = [_cell(i, [0.60, 0.20, 0.10, 0.07, 0.03]) for i in range(10)]
        out = fit_tau_daca(_preds(SHARP_MODERATE), _preds(WIDE_MODERATE), flipped)
        self.assertTrue(out["reference_below_chance"])

    def test_identical_distributions_fit_near_unit(self):
        out = fit_tau_daca(_preds(WIDE_MODERATE), _preds(WIDE_MODERATE), CELLS)
        self.assertAlmostEqual(out["tau_daca"], 1.0, places=2)


class DacaTriangulationTests(unittest.TestCase):
    def _references(self):
        return {"qwen": _preds(WIDE_MODERATE),
                "gemma": _preds([0.08, 0.17, 0.45, 0.20, 0.10]),
                "disagreeing": _preds(WIDE_HIGH)}  # zero agreement -> excluded

    def test_summary_excludes_invalid_references_and_compares(self):
        out = daca_triangulation(_preds(SHARP_MODERATE), self._references(), CELLS,
                                 tau_transfer=1.5, T_supervised=1.4)
        self.assertEqual(out["summary"]["n_references"], 3)
        self.assertEqual(out["summary"]["n_valid"], 2)
        self.assertIn("disagreeing", out["summary"]["excluded_references"])
        self.assertGreater(out["summary"]["tau_daca_median"], 1.0)
        vs = out["summary"]["vs_tau_transfer"]
        self.assertEqual(vs["value"], 1.5)
        self.assertAlmostEqual(vs["abs_diff"], abs(out["summary"]["tau_daca_median"] - 1.5))
        self.assertAlmostEqual(
            vs["log_ratio"], math.log(out["summary"]["tau_daca_median"] / 1.5))
        self.assertIn("vs_T_supervised", out["summary"])

    def test_comparators_optional_and_nan_ignored(self):
        out = daca_triangulation(_preds(SHARP_MODERATE), {"qwen": _preds(WIDE_MODERATE)},
                                 CELLS, tau_transfer=float("nan"))
        self.assertNotIn("vs_tau_transfer", out["summary"])
        self.assertNotIn("vs_T_supervised", out["summary"])

    def test_all_invalid_yields_no_median(self):
        out = daca_triangulation(_preds(SHARP_MODERATE), {"x": _preds(WIDE_HIGH)}, CELLS)
        self.assertEqual(out["summary"]["n_valid"], 0)
        self.assertNotIn("tau_daca_median", out["summary"])


if __name__ == "__main__":
    unittest.main()
