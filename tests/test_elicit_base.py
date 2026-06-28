"""Unit tests for the vLLM token-slice elicitation — no live server (monkeypatched)."""
import math, unittest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from judex_calibration import elicit_base as eb


def _reasoning_or(top=None, *, echo_vals=None):
    """Build a fake _completions that routes stage-1 reasoning / stage-2 topk / echo."""
    def fake(base_url, body, timeout=600):
        if body.get("echo"):
            L = body["prompt"].strip()[-1].upper()
            return {"choices": [{"logprobs": {"token_logprobs": [None, -0.1, echo_vals[L]]}}]}
        if body.get("max_tokens") == 1 and "logprobs" in body:
            return {"choices": [{"logprobs": {"top_logprobs": [top]}}]}
        return {"choices": [{"text": " The evidence shows partial compliance, gaps in reassessment."}]}
    return fake


class ElicitBaseTests(unittest.TestCase):
    def test_topk_path_softmaxes_five_letters(self):
        eb._completions = _reasoning_or({" A": -3.0, " B": -1.0, "C": -0.5, "D": -2.0, "E": -4.0})
        out = eb.elicit_cell("http://x", "m", "evidence", "criterion")
        self.assertEqual(out["method"], "topk")
        self.assertEqual(out["covered"], 5)
        self.assertEqual(out["labels"], list(eb.LABELS))
        self.assertAlmostEqual(sum(out["probabilities"]), 1.0, places=6)
        # C had the highest logit -> moderate should be the argmax
        self.assertEqual(out["probabilities"].index(max(out["probabilities"])), 2)
        self.assertGreater(out["reasoning_chars"], 0)

    def test_falls_back_to_echo_when_letter_missing(self):
        # top-K only surfaces C and D -> must fall back to echo for all five
        eb._completions = _reasoning_or({"C": -0.5, "D": -2.0},
                                        echo_vals={"A": -3, "B": -1, "C": -0.5, "D": -2, "E": -4})
        out = eb.elicit_cell("http://x", "m", "evidence", "criterion")
        self.assertEqual(out["method"], "echo")
        self.assertEqual(out["covered"], 5)
        self.assertAlmostEqual(sum(out["probabilities"]), 1.0, places=6)

    def test_no_reason_skips_stage1(self):
        eb._completions = _reasoning_or({"A": -1, "B": -1, "C": -1, "D": -1, "E": -1})
        out = eb.elicit_cell("http://x", "m", "evidence", "criterion", reason=False)
        self.assertEqual(out["reasoning_chars"], 0)
        for p in out["probabilities"]:
            self.assertAlmostEqual(p, 0.2, places=6)  # uniform logits -> uniform dist


if __name__ == "__main__":
    unittest.main()
