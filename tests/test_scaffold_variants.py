"""Phase 1a: k=5 coverage-preserving scaffold-sensitivity variants (no live server).

Guards the three properties the variants must have to be a valid scaffold-sensitivity
bound for the instrument in use (plan_2026_08_08_verbalized_arm_gap_closure §1g):

  * V1 ``alt_set`` — deterministic, coverage-complete (one exemplar per compliance
    level), and fully row-disjoint from the shipped baseline draw, for EVERY criterion
    the 120 AIReg cells use;
  * V2 ``rev_order`` — identical exemplar SET, render order strictly reversed;
  * defaulting to ``baseline`` reproduces the pre-2026-08-08 code path byte for byte
    (the variants must not perturb the shipped legs), and the leg meta sidecar pins the
    variant so a cross-variant resume hard-errors.
"""
import hashlib, json, tempfile, unittest
from pathlib import Path

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from judex_calibration import fewshot as fs
from judex_calibration import elicit_verbalized as ev
from judex_calibration import elicit_api_verbalized as eva
from judex_calibration.aireg import load_cells

K = 5

# sha256 of the rendered per-criterion few-shot block at k=5 under the DEFAULT
# (baseline) path, captured from the code as it stood BEFORE the scaffold-variant
# change (2026-08-08). Both channels: verbalized (contract-shaped) and token-slice
# (A-E letter). A diff here means either the variant plumbing perturbed the shipped
# scaffold or the corpus-v2 store pin moved — both are load-bearing.
PRE_CHANGE_DIGESTS = {
    "article_10": ("422db13c46508b74014b5ae20f1651e44c3f6468b8b0416029f74aedd8c446dc",
                   "948e8a4214b392f3e75d2644e53cfb96c0b47711eff1622238af1fff5cabe301"),
    "article_12": ("9490706786b4187bc6865e8a223086601756fa798a79ef14502b06aac061e509",
                   "5de419219657e56fa2caf1e90179893a517fa05698427f0199777a7b3b6137ba"),
    "article_14": ("89c19c9b1a4366d19174274662513d0c35e0b1ce707fb0a8c1626503e7aa7603",
                   "d4f16bd724e4a416db8ecafd2aa3a18d627eaeeb3db82168843d23cdb1562db0"),
    "article_15": ("140d772cb1149eac083679c49224f19a05f96ed2d36967b0b4cc75c02764c423",
                   "eb19986c312e23972a3a3202c52a28db810b49df58d7b81cd895e429b34b9d1d"),
    "article_9": ("0d50ce11e6d079eed533d097f1669e3d8d974a4a669eb6a9167374c9761d3c4a",
                  "255b0f2af071586bc75e1eede95392487f85d62d7b1f86338902bd9836ddb6f4"),
}

_CELLS = None
_TEXTS = None


def cells_and_texts():
    global _CELLS, _TEXTS
    if _CELLS is None:
        _CELLS = load_cells()
        _TEXTS = {}
        for c in _CELLS:
            _TEXTS.setdefault(c.criterion_id, c.criterion_text)
    return _CELLS, _TEXTS


def _legacy_build_fewshot(criterion_id, criterion_text, k):
    """The pre-change build_fewshot body, verbatim — the regression reference."""
    rows = fs.select_rows(criterion_id, k)
    rows.sort(key=lambda r: int(r["compliance_1to5"]))
    return "".join(ev.render_block(r, criterion_text) for r in rows)


class DefaultKTests(unittest.TestCase):
    def test_default_k_reads_config(self):
        self.assertEqual(fs.default_k(), 5)

    def test_fallback_is_five_not_four(self):
        """A yaml read failure must not regress a leg to the deprecated k=4 scaffold."""
        orig = fs.MODELS_YAML
        try:
            fs.MODELS_YAML = Path("/nonexistent/models.yaml")
            self.assertEqual(fs.default_k(), 5)
        finally:
            fs.MODELS_YAML = orig


class AltSetTests(unittest.TestCase):
    def test_every_criterion_is_feasible(self):
        _, texts = cells_and_texts()
        self.assertTrue(texts, "no criteria loaded")
        for cid in sorted(texts):
            with self.subTest(criterion=cid):
                base = fs.scaffold_rows(cid, K, "baseline")
                alt = fs.scaffold_rows(cid, K, "alt_set")
                self.assertEqual(len(alt), K)
                # coverage: exactly one exemplar per compliance level
                levels = [r["compliance_level"] for r in alt]
                self.assertEqual(sorted(levels), sorted(fs.LEVELS))
                self.assertEqual(len(set(levels)), K)
                # row-disjoint from baseline
                self.assertFalse({r["id"] for r in base} & {r["id"] for r in alt})
                # and (stronger) source-excerpt-disjoint, which is the mechanism
                self.assertFalse({r["source_item_label"] for r in base}
                                 & {r["source_item_label"] for r in alt})

    def test_baseline_is_also_coverage_complete(self):
        _, texts = cells_and_texts()
        for cid in sorted(texts):
            levels = [r["compliance_level"] for r in fs.scaffold_rows(cid, K, "baseline")]
            self.assertEqual(sorted(levels), sorted(fs.LEVELS), cid)

    def test_deterministic_across_calls(self):
        _, texts = cells_and_texts()
        for cid in sorted(texts):
            a = [r["id"] for r in fs.scaffold_rows(cid, K, "alt_set")]
            b = [r["id"] for r in fs.scaffold_rows(cid, K, "alt_set")]
            self.assertEqual(a, b, cid)
            self.assertEqual(ev.build_fewshot(cid, texts[cid], k=K, variant="alt_set"),
                             ev.build_fewshot(cid, texts[cid], k=K, variant="alt_set"))

    def test_rendered_block_differs_from_baseline_and_keeps_the_ramp(self):
        _, texts = cells_and_texts()
        for cid in sorted(texts):
            base = ev.build_fewshot(cid, texts[cid], k=K)
            alt = ev.build_fewshot(cid, texts[cid], k=K, variant="alt_set")
            self.assertNotEqual(base, alt, cid)
            levels = [r["compliance_1to5"] for r in
                      fs.order_rows(fs.scaffold_rows(cid, K, "alt_set"), "alt_set")]
            self.assertEqual(levels, sorted(levels), cid)

    def test_token_slice_channel_supports_the_variant_too(self):
        _, texts = cells_and_texts()
        cid = sorted(texts)[0]
        self.assertNotEqual(fs.build_fewshot(cid, texts[cid], k=K),
                            fs.build_fewshot(cid, texts[cid], k=K, variant="alt_set"))

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            fs.scaffold_rows("article_10", K, "shuffled")


class RevOrderTests(unittest.TestCase):
    def test_same_set_as_baseline(self):
        _, texts = cells_and_texts()
        for cid in sorted(texts):
            base = {r["id"] for r in fs.scaffold_rows(cid, K, "baseline")}
            rev = {r["id"] for r in fs.scaffold_rows(cid, K, "rev_order")}
            self.assertEqual(base, rev, cid)

    def test_order_strictly_reversed_in_row_list(self):
        _, texts = cells_and_texts()
        for cid in sorted(texts):
            base = [r["id"] for r in fs.order_rows(fs.scaffold_rows(cid, K, "baseline"), "baseline")]
            rev = [r["id"] for r in fs.order_rows(fs.scaffold_rows(cid, K, "rev_order"), "rev_order")]
            self.assertEqual(rev, base[::-1], cid)

    def test_order_strictly_reversed_in_rendered_prompt_text(self):
        _, texts = cells_and_texts()
        for cid in sorted(texts):
            base_txt = ev.build_fewshot(cid, texts[cid], k=K)
            rev_txt = ev.build_fewshot(cid, texts[cid], k=K, variant="rev_order")
            self.assertNotEqual(base_txt, rev_txt, cid)
            base_levels = [ln.split('"compliance_level": "')[1].split('"')[0]
                           for ln in base_txt.splitlines() if '"compliance_level": "' in ln]
            rev_levels = [ln.split('"compliance_level": "')[1].split('"')[0]
                          for ln in rev_txt.splitlines() if '"compliance_level": "' in ln]
            self.assertEqual(base_levels, list(fs.LEVELS), cid)
            self.assertEqual(rev_levels, list(reversed(fs.LEVELS)), cid)
            # the SET of rendered examples is unchanged — a pure permutation
            self.assertEqual(sorted(base_txt.split("Example.")), sorted(rev_txt.split("Example.")))

    def test_token_slice_channel_reverses_the_letter_ramp(self):
        _, texts = cells_and_texts()
        cid = sorted(texts)[0]
        letters = lambda t: [ln.split("Answer:")[1].strip()
                             for ln in t.splitlines() if ln.startswith("Answer:")]
        self.assertEqual(letters(fs.build_fewshot(cid, texts[cid], k=K)), list("ABCDE"))
        self.assertEqual(letters(fs.build_fewshot(cid, texts[cid], k=K, variant="rev_order")),
                         list("EDCBA"))


class BaselineRegressionTests(unittest.TestCase):
    """--scaffold-variant absent must reproduce the pre-change bytes exactly."""

    def test_default_matches_legacy_code_path(self):
        _, texts = cells_and_texts()
        for cid in sorted(texts):
            self.assertEqual(ev.build_fewshot(cid, texts[cid], k=K),
                             _legacy_build_fewshot(cid, texts[cid], K), cid)

    def test_default_equals_explicit_baseline(self):
        _, texts = cells_and_texts()
        for cid in sorted(texts):
            self.assertEqual(ev.build_fewshot(cid, texts[cid], k=K),
                             ev.build_fewshot(cid, texts[cid], k=K, variant="baseline"), cid)

    def test_pre_change_digests(self):
        _, texts = cells_and_texts()
        self.assertEqual(sorted(texts), sorted(PRE_CHANGE_DIGESTS))
        for cid, (verb_sha, tok_sha) in PRE_CHANGE_DIGESTS.items():
            v = hashlib.sha256(ev.build_fewshot(cid, texts[cid], k=K).encode()).hexdigest()
            t = hashlib.sha256(fs.build_fewshot(cid, texts[cid], k=K).encode()).hexdigest()
            self.assertEqual(v, verb_sha, f"{cid} verbalized block drifted")
            self.assertEqual(t, tok_sha, f"{cid} token-slice block drifted")

    def test_by_criterion_default_unchanged(self):
        cells, texts = cells_and_texts()
        blocks = ev.build_fewshot_by_criterion(cells, k=K)
        self.assertEqual(set(blocks), set(texts))
        for cid, b in blocks.items():
            self.assertEqual(b, _legacy_build_fewshot(cid, texts[cid], K))


class Cell:
    def __init__(self, label):
        self.item_label = label
        self.evidence_text = "ev"
        self.criterion_text = "crit"
        self.criterion_id = "article_10"


def _fake_completions(json_text):
    def fake(base_url, body, timeout=600):
        if body["prompt"].rstrip().endswith("JSON:"):
            return {"choices": [{"text": " " + json_text}]}
        return {"choices": [{"text": " reasoning"}]}
    return fake


GOOD_JSON = ('{"findings": [{"requirement": "r", "status": "unmet", "evidence": "e"}], '
             '"compliance_level": "low", '
             '"compliance_distribution": {"very_low": 0.20, "low": 0.45, "moderate": 0.25, '
             '"high": 0.10, "very_high": 0.00}, '
             '"compliance_justification": "j", '
             '"confidence_distribution": {"low": 0.15, "medium": 0.60, "high": 0.25}, '
             '"confidence_justification": "j"}')


class VllmSidecarTests(unittest.TestCase):
    def setUp(self):
        self._orig = ev._completions
        ev.__dict__["_completions"] = _fake_completions(GOOD_JSON)

    def tearDown(self):
        ev.__dict__["_completions"] = self._orig

    def test_variant_pinned_in_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pre_verbalized.json"
            ev.run_variant("http://x", "m", [Cell("c1")], str(out), fewshot_k=5,
                           scaffold_variant="alt_set")
            meta = json.loads(out.with_suffix(".meta.json").read_text())
            self.assertEqual(meta["scaffold_variant"], "alt_set")
            self.assertEqual(meta["fewshot_k"], 5)

    def test_baseline_default_pinned_explicitly(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pre_verbalized.json"
            ev.run_variant("http://x", "m", [Cell("c1")], str(out), fewshot_k=5)
            meta = json.loads(out.with_suffix(".meta.json").read_text())
            self.assertEqual(meta["scaffold_variant"], "baseline")

    def test_resume_across_variants_hard_errors(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pre_verbalized.json"
            ev.run_variant("http://x", "m", [Cell("c1")], str(out), fewshot_k=5,
                           scaffold_variant="alt_set")
            for other in ("baseline", "rev_order"):
                with self.assertRaises(RuntimeError, msg=other):
                    ev.run_variant("http://x", "m", [Cell("c1"), Cell("c2")], str(out),
                                   fewshot_k=5, scaffold_variant=other)

    def test_legacy_sidecar_without_the_field_resumes_as_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pre_verbalized.json"
            ev.run_variant("http://x", "m", [Cell("c1")], str(out), fewshot_k=5)
            meta_p = out.with_suffix(".meta.json")
            meta = json.loads(meta_p.read_text())
            meta.pop("scaffold_variant")                      # a pre-2026-08-08 sidecar
            meta_p.write_text(json.dumps(meta))
            ev.run_variant("http://x", "m", [Cell("c1")], str(out), fewshot_k=5)   # resumes
            with self.assertRaises(RuntimeError):
                ev.run_variant("http://x", "m", [Cell("c1")], str(out), fewshot_k=5,
                               scaffold_variant="rev_order")

    def test_unknown_variant_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                ev.run_variant("http://x", "m", [Cell("c1")],
                               str(Path(d) / "pre_verbalized.json"), scaffold_variant="nope")


class ApiSidecarTests(unittest.TestCase):
    """Same pinning + guard on the OpenRouter chat path (gemma31 legs)."""

    def setUp(self):
        self._orig = eva.elicit_cell_api
        eva.__dict__["elicit_cell_api"] = lambda key, model, ev_txt, cr_txt, fewshot: {
            "parse_ok": True, "compliance": [0.2, 0.45, 0.25, 0.1, 0.0],
            "confidence": [0.15, 0.6, 0.25], "channel": "verbalized_api_chat",
            "api_provider": "fake"}

    def tearDown(self):
        eva.__dict__["elicit_cell_api"] = self._orig

    def _run(self, out, variant, cells=(Cell("c1"),)):
        return eva.run_variant_api("google/gemma-4-31b-it", list(cells), str(out),
                                   fewshot_by_crit={"article_10": "fs"}, fewshot_k=5,
                                   workers=1, key="dummy", scaffold_variant=variant)

    def test_variant_pinned_in_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "post_verbalized.json"
            self._run(out, "rev_order")
            meta = json.loads(out.with_suffix(".meta.json").read_text())
            self.assertEqual(meta["scaffold_variant"], "rev_order")
            self.assertEqual(meta["channel"], "verbalized_api_chat")
            self.assertEqual(meta["quantizations"], ["bf16", "fp16"])

    def test_resume_across_variants_hard_errors(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "post_verbalized.json"
            self._run(out, "alt_set")
            with self.assertRaises(RuntimeError):
                self._run(out, "baseline", cells=(Cell("c1"), Cell("c2")))

    def test_legacy_sidecar_without_the_field_resumes_as_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "post_verbalized.json"
            self._run(out, "baseline")
            meta_p = out.with_suffix(".meta.json")
            meta = json.loads(meta_p.read_text())
            meta.pop("scaffold_variant")
            meta_p.write_text(json.dumps(meta))
            self._run(out, "baseline")
            with self.assertRaises(RuntimeError):
                self._run(out, "alt_set")

    def test_unknown_variant_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                self._run(Path(d) / "post_verbalized.json", "nope")


class DriverWiringTests(unittest.TestCase):
    """The CLI seams: both drivers expose --scaffold-variant, defaulting to baseline."""

    SCRIPTS = ("run_study_b_leg.py", "run_study_b_api_leg.py")

    def _run(self, script, *args):
        import subprocess
        p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / script
        return subprocess.run([sys.executable, str(p), *args], capture_output=True, text=True)

    def test_flag_documented_in_help(self):
        for script in self.SCRIPTS:
            out = self._run(script, "--help")
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("--scaffold-variant", out.stdout, script)
            for v in fs.SCAFFOLD_VARIANTS:
                self.assertIn(v, out.stdout, f"{script}: {v}")

    def test_unknown_variant_rejected_by_argparse(self):
        with tempfile.TemporaryDirectory() as d:
            for script in self.SCRIPTS:
                out = self._run(script, "--out", d, "--analyze", "--scaffold-variant", "shuffled")
                self.assertNotEqual(out.returncode, 0, script)
                self.assertIn("invalid choice", out.stderr, script)

    def test_scaffold_variants_tuple(self):
        self.assertEqual(fs.SCAFFOLD_VARIANTS, ("baseline", "alt_set", "rev_order"))


if __name__ == "__main__":
    unittest.main()
