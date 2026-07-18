"""Integration-seam tests against the git-tracked sibling repos (no live server).

These pin the contracts re-verified in docs/integration_remediation_2026_07_01.md
(2026-07-03 addendum): the canonical AIReg GT load, the 7-rater exemplar store
shape, the stratified few-shot selection, and the corpus<->AIReg firewall. They
read only tracked sibling artifacts, so they run on a fresh clone and on a vast box.
"""
import csv
import json
import unittest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from judex_calibration import fewshot
from judex_calibration.aireg import load_cells, GT_LABELS_DIR

EXPECTED_RATERS = {
    "deepseek-v4-pro",
    "gemma-4-26b-a4b-it",
    "glm-4.5",
    "kimi-k2-thinking",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "mistral-large-2512",
    "qwen3.5-35b-a3b",
}
# The 7-field distributional contract every store row must carry (subset checked).
CONTRACT_FIELDS = {
    "compliance_distribution", "compliance_level", "compliance_1to5",
    "compliance_justification", "confidence_distribution", "findings",
    "probabilities", "rater_model", "source_item_label", "text",
}


class AiregGtSeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = load_cells()

    def test_loads_120_manifest_verified_cells(self):
        self.assertEqual(len(self.cells), 120)
        for c in self.cells:
            self.assertAlmostEqual(sum(c.gt_probs), 1.0, places=9)
            self.assertEqual(len(c.gt_probs), 5)
            self.assertTrue(c.evidence_text)

    def test_gt_matches_canonical_barycenter_csv(self):
        path = GT_LABELS_DIR / "airegbench_labels_mgmfrm_anchored_projection_barycenter.csv"
        bary = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                key = row.get("item_label") or row.get("item_id")
                bary[key] = [float(v) for k, v in row.items() if k.startswith("p_")]
        matched = 0
        for c in self.cells:
            b = bary.get(c.item_label) or bary.get(c.item_id)
            if b is None:
                continue
            matched += 1
            for p, q in zip(c.gt_probs, b):
                self.assertLess(abs(p - q), 1e-12)
        self.assertEqual(matched, 120)


class GtVintageGuardTests(unittest.TestCase):
    """Pin the FORMULATION of the canonical AIReg GT, not just its shape.

    Why this exists: judex-calibration reads the siblings by relative path with no SHA pin, so it
    always gets whatever GT is checked out. That is the right design (§ aireg.py) — but it means a
    sibling-side re-materialization lands here silently. It has now happened twice, and both times
    argmax was preserved, so every accuracy-shaped check stayed green while the probability vectors
    moved (2026-07-03: W1 ~0.0008; 2026-07-09: W1 mean 0.194, mean H 1.065 -> 1.252).

    A distribution-fitting calibration study cannot absorb that silently. These assertions fail
    LOUDLY the next time the bundle's formulation moves. A failure here is NOT a bug to patch out:
    it means the GT changed, and the fix is to re-derive any affected T*/tau_oc numbers and update
    the provenance block in ``aireg.py`` + ``CLAUDE.md`` to the new formulation.
    """

    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            (GT_LABELS_DIR / "airegbench_labels_mgmfrm_anchored_projection_manifest.json").read_text())
        cls.diag = json.loads((GT_LABELS_DIR / "validation_diagnostics_v4.json").read_text())

    def test_bundle_is_the_canonical_production_object(self):
        self.assertEqual(self.manifest["status"], "canonical_production_ground_truth")
        self.assertEqual(self.manifest["methodology"]["name"], "mgmfrm_anchored_projection")
        self.assertEqual(self.diag["model"], "cumulative_consistency_fixed_sigma_item_mgmfrm.v1")
        self.assertEqual(self.manifest["counts"], {"items": 120, "observations": 360, "raters": 3})

    def test_readout_formulation_is_freethresh_tau_0675_continuous(self):
        # The 2026-07-09 re-materialization (89f40b7 + 3c2ebdb). Study A's estimands are
        # temperatures fit against this object, so a change to any of the three is material.
        self.assertEqual(self.diag["sampling"]["thresholds"], "free")
        self.assertAlmostEqual(self.diag["sampling"]["readout_tau"], 0.675, places=6)
        self.assertFalse(self.manifest["methodology"]["barycenter_grid_discretization"]["applied"])

    def test_labels_are_continuous_not_grid_snapped(self):
        # Belt-and-braces on the flag above, measured on the labels we actually load.
        on_grid = sum(1 for c in load_cells() for p in c.gt_probs if abs(p * 20 - round(p * 20)) < 1e-9)
        self.assertEqual(on_grid, 0, "GT looks grid-snapped again — the de-snap was reverted upstream")

    def test_validation_caveat_still_holds_as_documented(self):
        # CLAUDE.md / aireg.py describe this bundle as usable-but-not-a-validated-benchmark, on
        # TWO hard failures. If upstream ever clears them, the docs overstate the caveat.
        self.assertEqual(self.manifest["validation"]["response_model_validation_status"], "unavailable")
        self.assertEqual(sorted(self.manifest["validation"]["hard_failure_families"]),
                         ["convergence_rhat", "prior_predictive"])


class FewshotStoreSeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = fewshot._load_store()
        cls.rows = [r for rows in cls.store.values() for r in rows]

    def test_dimension_store_is_the_7_rater_build(self):
        self.assertEqual(len(self.rows), 308)  # 44 excerpts x 7 raters (corpus v2, 2026-07-13)
        self.assertEqual({r["rater_model"] for r in self.rows}, EXPECTED_RATERS)

    def test_leaf_plus_dimension_totals_1757(self):
        # The leaf store sibling of the adopted dimension store (same corpus tree),
        # so this seam tracks whichever corpus vintage fewshot.DIMENSION_STORE adopts.
        leaf_path = fewshot.DIMENSION_STORE.parent / "exemplar_store.json"
        leaf = json.loads(leaf_path.read_text())
        leaf_store = leaf.get("store", leaf)
        n_leaf = sum(len(v) for v in leaf_store.values())
        self.assertEqual(n_leaf, 1449)
        self.assertEqual(n_leaf + len(self.rows), 1757)

    def test_rows_carry_the_distributional_contract(self):
        for r in self.rows:
            self.assertTrue(CONTRACT_FIELDS <= set(r), f"missing fields in row {r.get('id')}")

    def test_letter_matches_argmax(self):
        for r in self.rows:
            argmax = max(range(5), key=lambda i: r["probabilities"][i])
            self.assertEqual(int(r["compliance_1to5"]) - 1, argmax, r.get("id"))

    def test_k4_selection_spans_distinct_raters(self):
        for cid in self.store:
            picked = fewshot.select_rows(cid, 4)
            self.assertEqual(len(picked), 4, cid)
            self.assertEqual(len({p["rater_model"] for p in picked}), 4, cid)

    def test_firewall_store_disjoint_from_aireg(self):
        aireg_labels = {c.item_label for c in load_cells()}
        src = {r["source_item_label"] for r in self.rows}
        self.assertFalse(src & aireg_labels)


if __name__ == "__main__":
    unittest.main()
