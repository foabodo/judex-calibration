"""Unit tests for the B-Q4 confidence-calibration machinery (no live server, no GT files
beyond the canonical loader used elsewhere in the suite)."""
import math, unittest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from judex_calibration import study_b as sb


class TemperLinkTests(unittest.TestCase):
    def test_T1_identity_after_floor(self):
        p = sb.temper([0.1, 0.3, 0.6], 1.0)
        self.assertAlmostEqual(sum(p), 1.0, places=12)
        for a, b in zip(p, [0.1, 0.3, 0.6]):
            self.assertAlmostEqual(a, b, places=9)

    def test_high_T_flattens_low_T_sharpens(self):
        base = [0.1, 0.3, 0.6]
        flat = sb.temper(base, 10.0)
        sharp = sb.temper(base, 0.3)
        self.assertLess(max(flat), max(base))
        self.assertGreater(max(sharp), max(base))

    def test_zeros_survive_via_floor(self):
        p = sb.temper([0.0, 0.2, 0.8], 2.0)
        self.assertTrue(all(x > 0 for x in p))
        self.assertAlmostEqual(sum(p), 1.0, places=12)

    def test_link_ordering(self):
        self.assertGreater(sb.link([0.0, 0.2, 0.8]), sb.link([0.8, 0.2, 0.0]))
        self.assertAlmostEqual(sb.link([0.0, 1.0, 0.0]), 0.5)


def _rows(confs, corrects, w1s, docs=None):
    docs = docs or [f"doc{i%4:02d}" for i in range(len(confs))]
    return [{"label": f"c{i}", "doc": docs[i], "conf": confs[i],
             "correct": corrects[i], "w1": w1s[i]} for i in range(len(confs))]


class FitTests(unittest.TestCase):
    def test_overconfident_confidence_fits_T_above_1(self):
        # states high confidence everywhere but is right only half the time ->
        # flattening (T>1) reduces Brier
        rows = _rows([[0.0, 0.1, 0.9]] * 20, [i % 2 == 0 for i in range(20)], [0.2] * 20)
        T = sb.fit_Tc(rows)
        self.assertGreater(T, 1.0)
        self.assertLess(sb.brier(rows, T), sb.brier(rows, 1.0))

    def test_well_calibrated_stays_near_1(self):
        # 80% correct with p_hat ~= 0.8: T=1 should already be near-optimal
        rows = _rows([[0.05, 0.2, 0.75]] * 10, [True] * 8 + [False] * 2, [0.1] * 10)
        T = sb.fit_Tc(rows)
        self.assertLess(sb.brier(rows, T), sb.brier(rows, 1.0) + 1e-9)
        self.assertLess(abs(math.log(T)), abs(math.log(5.0)))

    def test_kendall_perfect_orders(self):
        self.assertAlmostEqual(sb.kendall_tau_b([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
        self.assertAlmostEqual(sb.kendall_tau_b([1, 2, 3, 4], [8, 6, 4, 2]), -1.0)

    def test_aurc_rewards_informative_ranking(self):
        # confident cells have low w1 -> AURC beats the anti-informative ranking
        confs = [[0.0, 0.1, 0.9]] * 5 + [[0.9, 0.1, 0.0]] * 5
        w1s = [0.05] * 5 + [0.8] * 5
        good = sb.aurc(_rows(confs, [True] * 10, w1s), None)
        bad = sb.aurc(_rows(confs[::-1], [True] * 10, w1s), None)
        self.assertLess(good, bad)

    def test_lodo_deterministic_and_clustered(self):
        rows = _rows([[0.0, 0.1, 0.9]] * 24, [i % 2 == 0 for i in range(24)], [0.2] * 24,
                     docs=[f"d{i%6}" for i in range(24)])
        a = sb.lodo(rows, n_boot=200)
        b = sb.lodo(rows, n_boot=200)
        self.assertEqual(a["mean_oos_delta_brier"], b["mean_oos_delta_brier"])
        self.assertEqual(a["ci95"], b["ci95"])
        self.assertEqual(a["n_docs"], 6)
        self.assertIn("calibratable", a)


class AnalyzeTests(unittest.TestCase):
    class Cell:
        def __init__(self, label, doc, gt):
            self.item_label = label
            self.document_id = doc
            self.gt_labels = ("very_low", "low", "moderate", "high", "very_high")
            self.gt_probs = gt

    def test_analyze_confidence_end_to_end(self):
        cells = [self.Cell(f"c{i}", f"d{i%3}", [0.6, 0.2, 0.1, 0.1, 0.0]) for i in range(12)]
        recs = {}
        for i in range(12):
            correct = i % 2 == 0
            comp = [0.7, 0.2, 0.1, 0.0, 0.0] if correct else [0.0, 0.1, 0.2, 0.7, 0.0]
            recs[f"c{i}"] = {"parse_ok": True, "compliance": comp,
                             "confidence": [0.0, 0.1, 0.9]}
        out = sb.analyze_confidence(recs, cells)
        self.assertEqual(out["n"], 12)
        self.assertAlmostEqual(out["accuracy"], 0.5)
        self.assertGreater(out["T_c_full_sample"], 1.0)   # overconfident by construction
        self.assertLessEqual(out["brier_at_Tc"], out["brier_T1"] + 1e-12)
        self.assertEqual(out["lodo"]["n_docs"], 3)

    def test_parse_failures_excluded(self):
        cells = [self.Cell("c0", "d0", [0.6, 0.2, 0.1, 0.1, 0.0])]
        out = sb.analyze_confidence({"c0": {"parse_ok": False}}, cells)
        self.assertEqual(out["n"], 0)


if __name__ == "__main__":
    unittest.main()
