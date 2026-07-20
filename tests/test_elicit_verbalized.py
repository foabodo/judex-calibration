"""Unit tests for the Study B verbalized-channel elicitation — no live server."""
import json, tempfile, unittest
from pathlib import Path

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from judex_calibration import elicit_verbalized as ev

GOOD_JSON = ('{"compliance_level": "low", '
             '"compliance_distribution": {"very_low": 0.20, "low": 0.45, "moderate": 0.25, '
             '"high": 0.10, "very_high": 0.00}, '
             '"confidence_distribution": {"low": 0.15, "medium": 0.60, "high": 0.25}}')


def _fake_completions(json_text, reasoning=" The evidence shows gaps in data governance."):
    def fake(base_url, body, timeout=600):
        if body["prompt"].rstrip().endswith("JSON:"):
            return {"choices": [{"text": " " + json_text}]}
        return {"choices": [{"text": reasoning}]}
    return fake


class ParseTests(unittest.TestCase):
    def test_good_emission_parses(self):
        rec = ev.parse_contract_json(GOOD_JSON)
        self.assertTrue(rec["parse_ok"])
        self.assertAlmostEqual(sum(rec["compliance"]), 1.0, places=9)
        self.assertAlmostEqual(sum(rec["confidence"]), 1.0, places=9)
        self.assertEqual(rec["compliance_level"], "low")
        self.assertTrue(rec["level_matches_argmax"])
        self.assertTrue(rec["compliance_on_grid"])
        self.assertTrue(rec["confidence_on_grid"])
        self.assertAlmostEqual(rec["compliance_sum"], 1.0)

    def test_prose_wrapped_json_extracted(self):
        rec = ev.parse_contract_json("Sure, here is the assessment:\n" + GOOD_JSON + "\nDone.")
        self.assertTrue(rec["parse_ok"])

    def test_nested_braces_and_strings_balanced(self):
        wrapped = '{"note": "a {brace} in \\"string\\"", ' + GOOD_JSON[1:]
        rec = ev.parse_contract_json(wrapped)
        self.assertTrue(rec["parse_ok"])

    def test_missing_confidence_fails_with_named_error(self):
        broken = GOOD_JSON.replace("confidence_distribution", "conf")
        rec = ev.parse_contract_json(broken)
        self.assertFalse(rec["parse_ok"])
        self.assertIn("confidence_distribution", rec["parse_error"])

    def test_no_object_fails(self):
        rec = ev.parse_contract_json("I cannot answer.")
        self.assertFalse(rec["parse_ok"])
        self.assertEqual(rec["parse_error"], "no_json_object")

    def test_negative_mass_rejected(self):
        bad = GOOD_JSON.replace('"very_low": 0.20', '"very_low": -0.20')
        self.assertFalse(ev.parse_contract_json(bad)["parse_ok"])

    def test_off_sum_renormalized_and_reported(self):
        off = GOOD_JSON.replace('"low": 0.45', '"low": 0.55')  # sums to 1.10
        rec = ev.parse_contract_json(off)
        self.assertTrue(rec["parse_ok"])
        self.assertAlmostEqual(sum(rec["compliance"]), 1.0, places=9)
        self.assertAlmostEqual(rec["compliance_sum"], 1.10)

    def test_off_grid_flagged_not_failed(self):
        off = GOOD_JSON.replace("0.45", "0.47").replace("0.25, ", "0.23, ", 1)
        rec = ev.parse_contract_json(off)
        self.assertTrue(rec["parse_ok"])
        self.assertFalse(rec["compliance_on_grid"])

    def test_mismatched_level_flagged(self):
        rec = ev.parse_contract_json(GOOD_JSON.replace('"low"', '"high"', 1))
        self.assertTrue(rec["parse_ok"])
        self.assertFalse(rec["level_matches_argmax"])


class FloorTests(unittest.TestCase):
    def test_zeros_floored_and_renormalized(self):
        out = ev.floor_and_renormalize([0.65, 0.25, 0.10, 0.0, 0.0])
        self.assertAlmostEqual(sum(out), 1.0, places=12)
        self.assertTrue(all(p >= ev.EPSILON / 2 for p in out))
        self.assertGreater(min(out), 0.0)

    def test_identity_ordering_preserved(self):
        out = ev.floor_and_renormalize([0.5, 0.3, 0.2, 0.0, 0.0])
        self.assertEqual(sorted(range(5), key=lambda i: -out[i])[:3], [0, 1, 2])


class RenderTests(unittest.TestCase):
    ROW = {"compliance_level": "very_low", "compliance_1to5": 1,
           "compliance_distribution": {"very_low": 0.65, "low": 0.25, "moderate": 0.1,
                                       "high": 0.0, "very_high": 0.0},
           "confidence_distribution": {"low": 0.6, "medium": 0.3, "high": 0.1},
           "compliance_justification": "Almost no governance evidence.",
           "text": "The system card omits any data governance section."}

    def test_answer_json_round_trips_through_parser(self):
        rec = ev.parse_contract_json(ev.render_answer_json(self.ROW))
        self.assertTrue(rec["parse_ok"])
        self.assertEqual(rec["compliance_level"], "very_low")
        self.assertTrue(rec["level_matches_argmax"])
        self.assertTrue(rec["compliance_on_grid"] and rec["confidence_on_grid"])

    def test_block_ends_with_json_line(self):
        block = ev.render_block(self.ROW, "Article 10 - Data and data governance.")
        self.assertIn("Reasoning: Almost no governance evidence.", block)
        self.assertIn("JSON: {", block)
        self.assertTrue(block.endswith("\n\n"))


class Cell:
    def __init__(self, label):
        self.item_label = label
        self.evidence_text = "ev"
        self.criterion_text = "crit"


class RunVariantTests(unittest.TestCase):
    def test_elicit_cell_two_stage(self):
        ev._completions_orig = None
        orig = ev._completions
        try:
            ev.__dict__["_completions"] = _fake_completions(GOOD_JSON)
            rec = ev.elicit_cell("http://x", "m", "ev", "crit")
            self.assertTrue(rec["parse_ok"])
            self.assertEqual(rec["channel"], "verbalized")
            self.assertGreater(rec["reasoning_chars"], 0)
        finally:
            ev.__dict__["_completions"] = orig

    def test_run_variant_checkpoints_views_and_gate(self):
        orig = ev._completions
        try:
            ev.__dict__["_completions"] = _fake_completions(GOOD_JSON)
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "pre_verbalized.json"
                cells = [Cell("c1"), Cell("c2")]
                recs = ev.run_variant("http://x", "m", cells, str(out), fewshot_k=5)
                self.assertEqual(len(recs), 2)
                meta = json.loads(out.with_suffix(".meta.json").read_text())
                self.assertEqual(meta["channel"], "verbalized")
                self.assertEqual(meta["epsilon"], ev.EPSILON)
                self.assertEqual(meta["fewshot_k"], 5)
                comp = ev.compliance_view(recs)
                conf = ev.confidence_view(recs)
                self.assertEqual(set(comp) | set(conf), {"c1", "c2"})
                self.assertEqual(len(comp["c1"]), 5)
                self.assertEqual(len(conf["c1"]), 3)
                gate = ev.contract_compliance_summary(recs, n_cells=2)
                self.assertEqual(gate["parse_rate"], 1.0)
                self.assertEqual(gate["n_parsed"], 2)
        finally:
            ev.__dict__["_completions"] = orig

    def test_resume_config_mismatch_hard_errors(self):
        orig = ev._completions
        try:
            ev.__dict__["_completions"] = _fake_completions(GOOD_JSON)
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "pre_verbalized.json"
                ev.run_variant("http://x", "m", [Cell("c1")], str(out), fewshot_k=5)
                with self.assertRaises(RuntimeError):
                    ev.run_variant("http://x", "OTHER", [Cell("c1"), Cell("c2")], str(out), fewshot_k=5)
        finally:
            ev.__dict__["_completions"] = orig

    def test_parse_failures_recorded_not_raised(self):
        orig = ev._completions
        try:
            ev.__dict__["_completions"] = _fake_completions("no json here")
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "pre_verbalized.json"
                recs = ev.run_variant("http://x", "m", [Cell("c1")], str(out))
                self.assertFalse(recs["c1"]["parse_ok"])
                self.assertEqual(ev.compliance_view(recs), {})
                gate = ev.contract_compliance_summary(recs, n_cells=1)
                self.assertEqual(gate["parse_rate"], 0.0)
                self.assertIn("no_json_object", gate["parse_failures"])
        finally:
            ev.__dict__["_completions"] = orig


if __name__ == "__main__":
    unittest.main()
