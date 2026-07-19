"""Tests for the tau_oc clustering guard (cluster_fields / calibration_block provenance).

The adopted Q3 stopping rule (user directive, 2026-07-19): median(tau_oc) is adoptable
only if max/min <= 2 across the gate-passing families. These tests pin the guard's
contract: the ratio spans CLEAN fits only (saturated and degenerate-reference families
excluded), rule_ok has three states (True / False / None-for-<2-clean-fits), the fields
ride into calibration_block provenance, and an end-to-end cross_family() on synthetic
pre/post pairs carries them. Synthetic cells only — no live server, no AIReg bundle.
"""
import math
import unittest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from judex_calibration.aireg import Cell
from judex_calibration.study_a import (
    CLUSTER_RULE_MAX_RATIO,
    calibration_block,
    cluster_fields,
    cross_family,
)

LABELS = ("very_low", "low", "moderate", "high", "very_high")


def _row(tau, saturated=False, degenerate=False):
    return {"tau_oc": tau, "tau_oc_saturated": saturated,
            "tau_oc_reference_degenerate": degenerate}


class ClusterFieldsTests(unittest.TestCase):
    def test_clustered_panel_passes(self):
        out = cluster_fields({"a": _row(1.5), "b": _row(2.0), "c": _row(2.9)})
        self.assertEqual(out["tau_oc_clean_families_n"], 3)
        self.assertAlmostEqual(out["tau_oc_cluster_ratio"], 2.9 / 1.5)
        self.assertIs(out["tau_oc_cluster_rule_ok"], True)

    def test_measured_study_a_panel_fails(self):
        # The actual 2026-07-19 outcome: {1.60, 1.86, 4.88} + GLM 8.84 -> ratio 5.52 > 2.
        out = cluster_fields({"qwen": _row(1.6007889248849985),
                              "llama31": _row(1.8571440700827275),
                              "gemma31": _row(4.877199362107526),
                              "glm": _row(8.835204542762193)})
        self.assertEqual(out["tau_oc_clean_families_n"], 4)
        self.assertGreater(out["tau_oc_cluster_ratio"], CLUSTER_RULE_MAX_RATIO)
        self.assertIs(out["tau_oc_cluster_rule_ok"], False)

    def test_boundary_ratio_exactly_two_passes(self):
        out = cluster_fields({"a": _row(1.5), "b": _row(3.0)})
        self.assertIs(out["tau_oc_cluster_rule_ok"], True)

    def test_saturated_and_degenerate_families_are_excluded(self):
        # Without exclusions the ratio would be 20/1.5 = 13.3; the clean pair is 1.5/1.7.
        out = cluster_fields({"a": _row(1.5), "b": _row(1.7),
                              "pegged": _row(20.0, saturated=True),
                              "bad_ref": _row(9.0, degenerate=True)})
        self.assertEqual(out["tau_oc_clean_families_n"], 2)
        self.assertAlmostEqual(out["tau_oc_cluster_ratio"], 1.7 / 1.5)
        self.assertIs(out["tau_oc_cluster_rule_ok"], True)

    def test_single_clean_family_cannot_establish_clustering(self):
        out = cluster_fields({"a": _row(1.6), "pegged": _row(20.0, saturated=True)})
        self.assertEqual(out["tau_oc_clean_families_n"], 1)
        self.assertIsNone(out["tau_oc_cluster_ratio"])
        self.assertIsNone(out["tau_oc_cluster_rule_ok"])

    def test_nan_tau_is_not_a_clean_fit(self):
        out = cluster_fields({"a": _row(float("nan")), "b": _row(1.6)})
        self.assertEqual(out["tau_oc_clean_families_n"], 1)
        self.assertIsNone(out["tau_oc_cluster_rule_ok"])


class CalibrationBlockProvenanceTests(unittest.TestCase):
    def test_cluster_fields_ride_into_provenance(self):
        report = {"tau_oc_summary": {"tau_oc_median": 3.2,
                                     **cluster_fields({"a": _row(1.6), "b": _row(4.9)})}}
        block = calibration_block(report)
        prov = block["provenance"]
        self.assertAlmostEqual(prov["tau_oc_cluster_ratio"], 4.9 / 1.6)
        self.assertIs(prov["tau_oc_cluster_rule_ok"], False)
        self.assertEqual(prov["tau_oc_clean_families_n"], 2)
        self.assertIn("gate-passing", prov["tau_oc_cluster_rule"])


def _cell(i, gt):
    gt = tuple(gt)
    return Cell(item_id=f"item{i}", item_label=f"Art 9 / cell {i}", document_id=f"doc{i % 4}",
                article="9", criterion_id="article_9_rms", criterion_text="", evidence_text="",
                gt_probs=gt, gt_labels=LABELS,
                gt_argmax=max(range(5), key=lambda k: gt[k]))


def _sharpen(probs, tau):
    """q_i \\propto p_i^tau — logit-space sharpening by factor tau, so the tau_oc that
    maps the post back onto the pre (softening) is ~tau."""
    powered = [p ** tau for p in probs]
    total = sum(powered)
    return [p / total for p in powered]


class CrossFamilyEndToEndTests(unittest.TestCase):
    def test_scattered_families_fail_the_rule_end_to_end(self):
        # Two families whose posts are the pre sharpened by ~1.5x and ~6x: the fitted
        # tau_oc pair should straddle the <=2 rule (ratio ~4) and the summary must say so.
        # GT equals each cell's pre so the pre legs are calibrated (T* ~ 1, interior) and
        # neither family trips the degenerate-reference exclusion.
        pre_vecs = [[0.10, 0.20, 0.40, 0.20, 0.10], [0.05, 0.15, 0.35, 0.30, 0.15]]
        cells = [_cell(i, pre_vecs[i % 2]) for i in range(10)]
        families = {}
        for fam, tau in (("mild", 1.5), ("wild", 6.0)):
            pre = {c.item_label: pre_vecs[i % 2] for i, c in enumerate(cells)}
            post = {c.item_label: _sharpen(pre_vecs[i % 2], tau) for i, c in enumerate(cells)}
            families[fam] = {"pre": pre, "post": post}
        summary = cross_family(families, cells)["tau_oc_summary"]
        self.assertEqual(summary["tau_oc_clean_families_n"], 2)
        self.assertGreater(summary["tau_oc_cluster_ratio"], CLUSTER_RULE_MAX_RATIO)
        self.assertIs(summary["tau_oc_cluster_rule_ok"], False)
        # sanity: the grid fits actually recovered the engineered scales (within grid step)
        self.assertLess(abs(math.log(summary["tau_oc_min"] / 1.5)), 0.25)
        self.assertLess(abs(math.log(summary["tau_oc_max"] / 6.0)), 0.25)


if __name__ == "__main__":
    unittest.main()
