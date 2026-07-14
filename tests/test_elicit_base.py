"""Unit tests for the vLLM token-slice elicitation — no live server (monkeypatched)."""
import json, math, tempfile, unittest
from collections import namedtuple
from pathlib import Path

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


class ServerShapeTests(unittest.TestCase):
    """The answer-position logprob read must parse vLLM, llama.cpp, and chat-style shapes
    identically (the Mac smoke serves via a Metal server, not vLLM)."""

    LP = {" A": -3.0, " B": -1.0, "C": -0.5, "D": -2.0, "E": -4.0}  # C is the argmax (moderate)

    def _elicit(self, logprobs_payload):
        def fake(base_url, body, timeout=600):
            return {"choices": [{"logprobs": logprobs_payload}]}
        eb._completions = fake
        return eb.elicit_cell("http://x", "m", "evidence", "criterion", reason=False)

    def _assert_moderate(self, out):
        self.assertEqual(out["covered"], 5)
        self.assertAlmostEqual(sum(out["probabilities"]), 1.0, places=6)
        self.assertEqual(out["probabilities"].index(max(out["probabilities"])), 2)  # C -> moderate

    def test_vllm_dict_top_logprobs(self):
        self._assert_moderate(self._elicit({"top_logprobs": [self.LP]}))

    def test_chat_style_content(self):
        tl = [{"token": t.strip(), "logprob": lp} for t, lp in self.LP.items()]
        self._assert_moderate(self._elicit({"content": [{"top_logprobs": tl}]}))

    def test_llamacpp_top_probs_objlist(self):
        objs = [{"token": t.strip(), "logprob": lp} for t, lp in self.LP.items()]
        self._assert_moderate(self._elicit({"top_probs": [objs]}))

    def test_prob_field_converted_to_logprob(self):
        objs = [{"token": t.strip(), "prob": math.exp(lp)} for t, lp in self.LP.items()]
        self._assert_moderate(self._elicit({"content": [{"top_logprobs": objs}]}))

    def test_unparseable_logprobs_degrades_uniform(self):
        # a shape we don't recognise -> no letters -> echo returns all -50 -> uniform (no crash)
        out = self._elicit({"unexpected": True})
        self.assertAlmostEqual(sum(out["probabilities"]), 1.0, places=6)


FakeCell = namedtuple("FakeCell", "item_label evidence_text criterion_text")
_CELLS = [FakeCell(f"Art 9 / Cell {i}", f"evidence {i}", "criterion") for i in range(3)]


class CheckpointTests(unittest.TestCase):
    """run_variant checkpoints per cell and resumes (crash recovery only)."""

    def _counting_fake(self):
        """Fake _completions that counts stage-2 (answer-logit) calls per elicitation."""
        counter = {"answers": 0}
        top = {"A": -1.0, "B": -2.0, "C": -0.5, "D": -3.0, "E": -4.0}
        def fake(base_url, body, timeout=600):
            if body.get("max_tokens") == 1 and "logprobs" in body:
                counter["answers"] += 1
                return {"choices": [{"logprobs": {"top_logprobs": [top]}}]}
            return {"choices": [{"text": " reasoning."}]}
        return fake, counter

    def test_checkpoints_every_cell_and_writes_meta(self):
        eb._completions, counter = self._counting_fake()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pre.json"
            preds = eb.run_variant("http://x", "m", _CELLS, str(out))
            self.assertEqual(len(preds), 3)
            self.assertEqual(counter["answers"], 3)
            on_disk = json.loads(out.read_text())
            self.assertEqual(set(on_disk), {c.item_label for c in _CELLS})
            meta = json.loads((Path(d) / "pre.meta.json").read_text())
            self.assertEqual(meta, {"model": "m", "reason": True, "budget": 2048})
            self.assertFalse((Path(d) / "pre.json.tmp").exists())  # tmp always renamed away

    def test_resume_skips_cached_cells(self):
        eb._completions, counter = self._counting_fake()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pre.json"
            # simulate a leg that crashed after 2 of 3 cells
            out.write_text(json.dumps({c.item_label: [0.2] * 5 for c in _CELLS[:2]}))
            (Path(d) / "pre.meta.json").write_text(
                json.dumps({"model": "m", "reason": True, "budget": 2048}))
            preds = eb.run_variant("http://x", "m", _CELLS, str(out))
            self.assertEqual(len(preds), 3)
            self.assertEqual(counter["answers"], 1)  # only the missing cell was elicited
            self.assertEqual(preds[_CELLS[0].item_label], [0.2] * 5)  # cached rows kept verbatim

    def test_resume_with_different_leg_config_raises(self):
        eb._completions, _ = self._counting_fake()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pre.json"
            out.write_text(json.dumps({_CELLS[0].item_label: [0.2] * 5}))
            (Path(d) / "pre.meta.json").write_text(
                json.dumps({"model": "m", "reason": False, "budget": 2048}))  # a --no-reason leg
            with self.assertRaises(RuntimeError):
                eb.run_variant("http://x", "m", _CELLS, str(out))  # reasoning-ON resume must refuse
            # missing sidecar with existing preds is equally untrusted
            (Path(d) / "pre.meta.json").unlink()
            with self.assertRaises(RuntimeError):
                eb.run_variant("http://x", "m", _CELLS, str(out))


if __name__ == "__main__":
    unittest.main()
