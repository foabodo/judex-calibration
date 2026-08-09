"""Defined-answer control (Phase 2b): slice adapter, k=4 scaffold, K=4 contract parser.

Guards the properties the control leg's validity rests on:

  * the vendored slice and the control exemplar store are sha-pinned, and a moved
    artifact hard-errors rather than silently re-scoping every leg;
  * the item adapter is duck-type-compatible with the elicitation harness AND with the
    scoring machinery (one-hot key, D7's two document_id modes);
  * the k=4 selector is deterministic, coverage-complete (one exemplar per option
    letter) for every subject, and its alt_set variant fails LOUD when the store cannot
    supply a disjoint draw;
  * the store is firewall-disjoint from the scored slice (split AND text);
  * the K=4 contract parser's two tiers behave exactly like the verbalized battery's,
    minus the findings clause;
  * ``run_variant``'s seam: two-stage call, checkpointing, and an explicit resume guard
    on EVERY control identity field — including the fields a prior sidecar might omit.
"""
import json, tempfile, unittest
from pathlib import Path

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from judex_calibration import control_fewshot as cfs
from judex_calibration import elicit_control as ec
from judex_calibration import elicit_verbalized as ev
from judex_calibration import mmlu

GOOD_JSON = ('{"answer_letter": "B", '
             '"answer_distribution": {"A": 0.10, "B": 0.65, "C": 0.20, "D": 0.05}, '
             '"answer_justification": "B follows from the modus tollens step.", '
             '"confidence_distribution": {"low": 0.10, "medium": 0.60, "high": 0.30}, '
             '"confidence_justification": "The inference is standard but the wording is loose."}')


def _fake_completions(json_text, reasoning=" The second option matches the premise."):
    def fake(base_url, body, timeout=600):
        if body["prompt"].rstrip().endswith("JSON:"):
            return {"choices": [{"text": " " + json_text}]}
        return {"choices": [{"text": reasoning}]}
    return fake


_ITEMS = None


def items():
    global _ITEMS
    if _ITEMS is None:
        _ITEMS = mmlu.load_control_items()
    return _ITEMS


# ---------------------------------------------------------------------------
# Pins + adapter
# ---------------------------------------------------------------------------

class SlicePinTests(unittest.TestCase):
    def test_slice_hashes_to_the_pin(self):
        self.assertEqual(mmlu.slice_sha256(), mmlu.SLICE_SHA256)
        self.assertEqual(mmlu.verify_slice_sha256(), mmlu.SLICE_SHA256)

    def test_store_hashes_to_the_pin(self):
        self.assertEqual(cfs.store_sha256(), cfs.STORE_SHA256)
        self.assertEqual(cfs.verify_store_sha256(), cfs.STORE_SHA256)

    def test_slice_sha_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "slice.json"
            p.write_text(json.dumps({"items": [], "meta": {}}))
            with self.assertRaises(RuntimeError) as cm:
                mmlu.verify_slice_sha256(p)
            self.assertIn("sha256", str(cm.exception))
            with self.assertRaises(RuntimeError):
                mmlu.load_control_items(p)

    def test_store_sha_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "store.json"
            p.write_text("{}")
            with self.assertRaises(RuntimeError):
                cfs.verify_store_sha256(p)


class AdapterTests(unittest.TestCase):
    def test_shape_and_counts(self):
        its = items()
        self.assertEqual(len(its), 120)
        self.assertEqual(len({i.subject for i in its}), 6)
        for s in mmlu.SUBJECTS:
            self.assertEqual(sum(1 for i in its if i.subject == s), 20, s)

    def test_duck_typed_for_elicitation(self):
        it = items()[0]
        self.assertEqual(it.item_label, it.item_id)
        self.assertTrue(it.item_label.startswith("mmlu:"))
        self.assertEqual(len(it.item_label.split(":")), 4)
        self.assertEqual(it.evidence_text, it.question)
        self.assertEqual(it.criterion_id, it.subject)
        self.assertEqual(it.criterion_text, mmlu.render_options(it.choices))
        self.assertIn("A. ", it.criterion_text)
        self.assertIn("D. ", it.criterion_text)

    def test_scoring_fields_are_one_hot_on_the_key(self):
        for it in items():
            self.assertEqual(it.gt_labels, ("A", "B", "C", "D"))
            self.assertEqual(sum(it.gt_probs), 1.0)
            self.assertEqual(it.gt_probs[it.gt_argmax], 1.0)
            self.assertEqual(mmlu.OPTION_LABELS[it.gt_argmax], it.answer_letter)

    def test_document_id_modes(self):
        by_item = mmlu.load_control_items(document_id="item")
        by_subject = mmlu.load_control_items(document_id="subject")
        self.assertEqual(len({i.document_id for i in by_item}), 120)
        self.assertEqual(len({i.document_id for i in by_subject}), 6)
        self.assertEqual(by_item[0].document_id, by_item[0].item_label)
        self.assertEqual(by_subject[0].document_id, by_subject[0].subject)

    def test_unknown_document_id_mode_raises(self):
        with self.assertRaises(ValueError):
            mmlu.load_control_items(document_id="paragraph")

    def test_items_are_frozen(self):
        with self.assertRaises(Exception):
            items()[0].question = "tampered"

    def test_render_item_text_matches_every_store_row(self):
        """The store's ``text`` and a live item's (evidence, criterion) are one object."""
        n = 0
        for rows in cfs.load_store().values():
            for r in rows:
                self.assertEqual(mmlu.render_item_text(r["question"], r["choices"]), r["text"])
                n += 1
        self.assertEqual(n, 237)

    def test_render_options_rejects_wrong_arity(self):
        with self.assertRaises(ValueError):
            mmlu.render_options(["a", "b", "c"])


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class SelectorTests(unittest.TestCase):
    def test_k_is_four_and_not_read_from_models_yaml(self):
        self.assertEqual(cfs.default_k(), 4)
        self.assertEqual(cfs.CONTROL_FEWSHOT_K, 4)

    def test_one_exemplar_per_letter_for_every_subject(self):
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                rows = cfs.scaffold_rows(s, 4)
                self.assertEqual(len(rows), 4)
                self.assertEqual({r["answer_letter"] for r in rows}, set(mmlu.OPTION_LABELS))

    def test_determinism(self):
        for s in mmlu.SUBJECTS:
            a = [r["id"] for r in cfs.scaffold_rows(s, 4)]
            b = [r["id"] for r in cfs.scaffold_rows(s, 4)]
            self.assertEqual(a, b)

    def test_least_used_rater_balancing_spreads_seats(self):
        for s in mmlu.SUBJECTS:
            raters = [r["rater_model"] for r in cfs.scaffold_rows(s, 4)]
            self.assertEqual(len(set(raters)), 4, f"{s}: {raters}")

    def test_render_order_is_the_letter_ramp(self):
        rows = cfs.scaffold_rows("philosophy", 4)
        self.assertEqual([r["answer_letter"] for r in cfs.order_rows(rows)], list(mmlu.OPTION_LABELS))
        self.assertEqual([r["answer_letter"] for r in cfs.order_rows(rows, "rev_order")],
                         list(reversed(mmlu.OPTION_LABELS)))

    def test_rev_order_is_the_same_set_reversed(self):
        for s in mmlu.SUBJECTS:
            base = cfs.order_rows(cfs.scaffold_rows(s, 4, "baseline"), "baseline")
            rev = cfs.order_rows(cfs.scaffold_rows(s, 4, "rev_order"), "rev_order")
            self.assertEqual({r["id"] for r in rev}, {r["id"] for r in base})
            self.assertEqual([r["id"] for r in rev], [r["id"] for r in base][::-1])

    def test_alt_set_fails_loud_on_the_v1_store(self):
        """The v1 store has ONE source item for at least one letter per subject, so a
        source-exclusion re-walk cannot be coverage-preserving. The guard must raise —
        silently dropping a letter would confound the sensitivity read with a coverage
        hole, which is exactly the k=4 E-hole failure the AIReg scaffold was fixed for."""
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                with self.assertRaises(RuntimeError):
                    cfs.scaffold_rows(s, 4, "alt_set")
                feas = cfs.alt_set_feasibility(s, 4)
                self.assertFalse(feas["alt_set_feasible"])
                self.assertTrue(feas["starved_letters"])

    def test_alt_set_succeeds_when_the_store_has_depth(self):
        """The guard is a coverage check, not a blanket refusal: give it a store with two
        source items per letter and the disjoint draw goes through."""
        rows = []
        for src in ("srcA", "srcB"):
            for i, L in enumerate(mmlu.OPTION_LABELS):
                rows.append({"id": f"{src}:{L}", "answer_letter": L, "answer_1to4": i + 1,
                             "source_item_label": f"{src}:{L}", "rater_model": f"r{i}"})
        orig = cfs._STORE_CACHE
        try:
            cfs._STORE_CACHE = {"toy": rows}
            base = cfs.scaffold_rows("toy", 4, "baseline")
            alt = cfs.scaffold_rows("toy", 4, "alt_set")
            self.assertEqual({r["answer_letter"] for r in alt}, set(mmlu.OPTION_LABELS))
            self.assertFalse({r["id"] for r in base} & {r["id"] for r in alt})
        finally:
            cfs._STORE_CACHE = orig

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            cfs.scaffold_rows("philosophy", 4, "shuffle")


class FirewallTests(unittest.TestCase):
    def test_store_sources_are_disjoint_from_the_scored_slice(self):
        slice_labels = {i.item_label for i in items()}
        store_srcs = {r["source_item_label"] for rows in cfs.load_store().values() for r in rows}
        self.assertTrue(store_srcs)
        self.assertFalse(slice_labels & store_srcs)

    def test_store_questions_are_disjoint_from_slice_questions(self):
        slice_q = {i.question.strip() for i in items()}
        store_q = {r["question"].strip() for rows in cfs.load_store().values() for r in rows}
        self.assertFalse(slice_q & store_q)

    def test_store_is_dev_validation_only_slice_is_test(self):
        self.assertEqual({i.split for i in items()}, {"test"})
        splits = {r["source_item_label"].split(":")[2]
                  for rows in cfs.load_store().values() for r in rows}
        self.assertEqual(splits, {"dev", "validation"})


# ---------------------------------------------------------------------------
# Parser tiers (K=4 battery, mirroring the verbalized 16-case battery)
# ---------------------------------------------------------------------------

class ParseTests(unittest.TestCase):
    def test_good_emission_parses(self):
        rec = ec.parse_control_json(GOOD_JSON)
        self.assertTrue(rec["parse_ok"])
        self.assertEqual(rec["contract"], "mmlu_control_v1")
        self.assertEqual(len(rec["compliance"]), 4)
        self.assertAlmostEqual(sum(rec["compliance"]), 1.0, places=9)
        self.assertAlmostEqual(sum(rec["confidence"]), 1.0, places=9)
        self.assertEqual(rec["answer_letter"], "B")
        self.assertTrue(rec["level_matches_argmax"])
        self.assertTrue(rec["compliance_on_grid"] and rec["confidence_on_grid"])
        self.assertAlmostEqual(rec["compliance_sum"], 1.0)

    def test_prose_wrapped_json_extracted(self):
        self.assertTrue(ec.parse_control_json("Here you go:\n" + GOOD_JSON + "\nDone.")["parse_ok"])

    def test_nested_braces_and_strings_balanced(self):
        wrapped = '{"note": "a {brace} in \\"string\\"", ' + GOOD_JSON[1:]
        self.assertTrue(ec.parse_control_json(wrapped)["parse_ok"])

    def test_no_object_fails(self):
        rec = ec.parse_control_json("I cannot answer.")
        self.assertFalse(rec["parse_ok"])
        self.assertEqual(rec["parse_error"], "no_json_object")

    def test_malformed_json_fails_with_named_error(self):
        rec = ec.parse_control_json('{"answer_letter": "B",}')   # balanced but invalid
        self.assertFalse(rec["parse_ok"])
        self.assertIn("json_decode", rec["parse_error"])
        # unbalanced never yields a blob at all
        self.assertEqual(ec.parse_control_json('{"answer_letter": "B", ')["parse_error"],
                         "no_json_object")

    def test_missing_answer_distribution_fails_with_named_error(self):
        rec = ec.parse_control_json(GOOD_JSON.replace("answer_distribution", "answers"))
        self.assertFalse(rec["parse_ok"])
        self.assertIn("answer_distribution", rec["parse_error"])

    def test_missing_confidence_distribution_fails_with_named_error(self):
        rec = ec.parse_control_json(GOOD_JSON.replace("confidence_distribution", "conf"))
        self.assertFalse(rec["parse_ok"])
        self.assertIn("confidence_distribution", rec["parse_error"])

    def test_partial_option_keys_fail(self):
        rec = ec.parse_control_json(GOOD_JSON.replace('"D": 0.05', '"E": 0.05'))
        self.assertFalse(rec["parse_ok"])
        self.assertIn("answer_distribution", rec["parse_error"])

    def test_negative_mass_rejected(self):
        self.assertFalse(ec.parse_control_json(
            GOOD_JSON.replace('"A": 0.10', '"A": -0.10'))["parse_ok"])

    def test_zero_sum_rejected(self):
        rec = ec.parse_control_json(GOOD_JSON.replace(
            '{"A": 0.10, "B": 0.65, "C": 0.20, "D": 0.05}',
            '{"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}'))
        self.assertFalse(rec["parse_ok"])

    def test_off_sum_renormalized_and_reported(self):
        rec = ec.parse_control_json(GOOD_JSON.replace('"B": 0.65', '"B": 0.75'))
        self.assertTrue(rec["parse_ok"])
        self.assertAlmostEqual(sum(rec["compliance"]), 1.0, places=9)
        self.assertAlmostEqual(rec["compliance_sum"], 1.10)

    def test_off_grid_flagged_not_failed(self):
        rec = ec.parse_control_json(GOOD_JSON.replace('"B": 0.65', '"B": 0.63'))
        self.assertTrue(rec["parse_ok"])
        self.assertFalse(rec["compliance_on_grid"])
        self.assertTrue(rec["confidence_on_grid"])

    def test_mismatched_letter_flagged(self):
        rec = ec.parse_control_json(GOOD_JSON.replace('"answer_letter": "B"',
                                                      '"answer_letter": "C"'))
        self.assertTrue(rec["parse_ok"])
        self.assertFalse(rec["level_matches_argmax"])

    def test_contract_complete_on_good_emission(self):
        rec = ec.parse_control_json(GOOD_JSON)
        self.assertTrue(rec["contract_complete"])
        self.assertEqual(rec["contract_missing"], [])

    def test_missing_answer_letter_parses_but_incomplete(self):
        rec = ec.parse_control_json(GOOD_JSON.replace('"answer_letter": "B", ', ""))
        self.assertTrue(rec["parse_ok"])
        self.assertFalse(rec["contract_complete"])
        self.assertIn("answer_letter", rec["contract_missing"])
        self.assertIsNone(rec["answer_letter"])
        self.assertFalse(rec["level_matches_argmax"])

    def test_out_of_alphabet_letter_incomplete(self):
        rec = ec.parse_control_json(GOOD_JSON.replace('"answer_letter": "B"',
                                                      '"answer_letter": "E"'))
        self.assertTrue(rec["parse_ok"])
        self.assertIn("answer_letter", rec["contract_missing"])

    def test_empty_answer_justification_incomplete(self):
        rec = ec.parse_control_json(GOOD_JSON.replace(
            '"answer_justification": "B follows from the modus tollens step."',
            '"answer_justification": "   "'))
        self.assertTrue(rec["parse_ok"])
        self.assertIn("answer_justification", rec["contract_missing"])

    def test_empty_confidence_justification_incomplete(self):
        rec = ec.parse_control_json(GOOD_JSON.replace(
            '"confidence_justification": "The inference is standard but the wording is loose."',
            '"confidence_justification": ""'))
        self.assertTrue(rec["parse_ok"])
        self.assertIn("confidence_justification", rec["contract_missing"])

    def test_findings_are_not_part_of_the_control_tier(self):
        """D3: findings is DROPPED. Its absence must not make a record incomplete, and its
        presence must not be graded — the control tier is five fields, not six."""
        rec = ec.parse_control_json(GOOD_JSON)
        self.assertTrue(rec["contract_complete"])
        self.assertNotIn("findings", rec["contract_missing"])
        self.assertNotIn("n_findings", rec)
        with_findings = ec.parse_control_json('{"findings": [], ' + GOOD_JSON[1:])
        self.assertTrue(with_findings["contract_complete"])

    def test_parse_last_control_picks_the_final_object(self):
        first = GOOD_JSON.replace('"answer_letter": "B"', '"answer_letter": "A"')
        rec = ec.parse_last_control("draft " + first + " ... final answer " + GOOD_JSON)
        self.assertTrue(rec["parse_ok"])
        self.assertEqual(rec["answer_letter"], "B")


class FloorTests(unittest.TestCase):
    def test_zeros_floored_and_renormalized_on_4_vectors(self):
        out = ev.floor_and_renormalize([0.65, 0.35, 0.0, 0.0])
        self.assertEqual(len(out), 4)
        self.assertAlmostEqual(sum(out), 1.0, places=12)
        self.assertGreater(min(out), 0.0)

    def test_one_hot_4_vector_floored(self):
        out = ev.floor_and_renormalize([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(out), 1.0, places=12)
        self.assertAlmostEqual(out[0], 1.0 / (1.0 + 3 * ev.EPSILON), places=12)
        self.assertEqual(out.index(max(out)), 0)

    def test_ordering_preserved(self):
        out = ev.floor_and_renormalize([0.5, 0.3, 0.2, 0.0])
        self.assertEqual(sorted(range(4), key=lambda i: -out[i]), [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Render -> parse round trip on REAL store rows
# ---------------------------------------------------------------------------

class RenderRoundTripTests(unittest.TestCase):
    def test_every_store_row_round_trips(self):
        n = 0
        for subject, rows in cfs.load_store().items():
            for r in rows:
                rec = ec.parse_control_json(ec.render_answer_json(r))
                self.assertTrue(rec["parse_ok"], r["id"])
                self.assertTrue(rec["contract_complete"], (r["id"], rec.get("contract_missing")))
                self.assertEqual(rec["answer_letter"], r["answer_letter"], r["id"])
                self.assertTrue(rec["level_matches_argmax"], r["id"])
                self.assertTrue(rec["compliance_on_grid"], r["id"])
                self.assertTrue(rec["confidence_on_grid"], r["id"])
                n += 1
        self.assertEqual(n, 237)

    def test_rendered_blocks_carry_four_parsable_exemplars(self):
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                block = ec.build_fewshot(s)
                objs = [ln.split("JSON: ", 1)[1] for ln in block.splitlines()
                        if ln.startswith("JSON: ")]
                self.assertEqual(len(objs), 4)
                letters = [ec.parse_control_json(o)["answer_letter"] for o in objs]
                self.assertEqual(letters, list(mmlu.OPTION_LABELS))
                self.assertTrue(all(ec.parse_control_json(o)["contract_complete"] for o in objs))
                self.assertEqual(block.count("Example.\nEvidence:\n"), 4)
                self.assertEqual(block.count("\n\nCriterion:\n"), 4)
                self.assertTrue(block.endswith("\n\n"))

    def test_prompt_keeps_the_aireg_scaffold_grammar(self):
        it = items()[0]
        p = ec.build_prompt(it.evidence_text, it.criterion_text, ec.build_fewshot(it.subject))
        self.assertIn("multiple-choice question", p)
        self.assertNotIn("EU AI Act", p)
        # the section headers and the trailing cue are the AIReg harness's, verbatim
        self.assertIn("\n\nEvidence:\n", p)
        self.assertIn("\n\nCriterion:\n", p)
        self.assertTrue(p.endswith("\n\nReasoning:"))
        self.assertIn("Reason step by step, then answer with a single JSON object "
                      "of the exact shape shown in the examples.", p)

    def test_fewshot_by_subject_covers_every_subject(self):
        blocks = ec.build_fewshot_by_subject(items())
        self.assertEqual(set(blocks), set(mmlu.SUBJECTS))


# ---------------------------------------------------------------------------
# run_variant seam
# ---------------------------------------------------------------------------

class RunVariantTests(unittest.TestCase):
    def setUp(self):
        self._orig = ev._completions
        ev.__dict__["_completions"] = _fake_completions(GOOD_JSON)
        self.items = items()[:2]

    def tearDown(self):
        ev.__dict__["_completions"] = self._orig

    def _run(self, out, **kw):
        kw.setdefault("fewshot_k", 4)
        kw.setdefault("slice_sha", "SLICE")
        kw.setdefault("store_sha", "STORE")
        return ec.run_variant("http://x", kw.pop("model", "m"), self.items, str(out), **kw)

    def test_elicit_cell_two_stage(self):
        rec = ec.elicit_cell("http://x", "m", "ev", "crit")
        self.assertTrue(rec["parse_ok"])
        self.assertEqual(rec["channel"], "verbalized")
        self.assertEqual(rec["contract"], "mmlu_control_v1")
        self.assertGreater(rec["reasoning_chars"], 0)

    def test_checkpoint_sidecar_and_views(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pre_control.json"
            recs = self._run(out)
            self.assertEqual(len(recs), 2)
            self.assertTrue(out.exists())
            meta = json.loads(out.with_suffix(".meta.json").read_text())
            self.assertEqual(meta["contract"], "mmlu_control_v1")
            self.assertEqual(meta["slice_sha256"], "SLICE")
            self.assertEqual(meta["store_sha256"], "STORE")
            self.assertEqual(meta["scaffold_variant"], "baseline")
            self.assertEqual(meta["fewshot_k"], 4)
            self.assertEqual(meta["epsilon"], ev.EPSILON)
            av, cv = ec.answer_view(recs), ec.confidence_view(recs)
            self.assertEqual(set(av), {i.item_label for i in self.items})
            self.assertEqual(len(av[self.items[0].item_label]), 4)
            self.assertEqual(len(cv[self.items[0].item_label]), 3)

    def test_gate_summary_is_control_labelled(self):
        with tempfile.TemporaryDirectory() as d:
            recs = self._run(Path(d) / "pre_control.json")
            g = ec.contract_compliance_summary(recs, n_cells=2)
            self.assertEqual(g["parse_rate"], 1.0)
            self.assertEqual(g["contract_complete_rate"], 1.0)
            self.assertEqual(g["contract"], "mmlu_control_v1")
            self.assertIsNone(g["mean_n_findings"])
            self.assertEqual(g["answer_letter_valid_rate"], 1.0)

    def test_parse_failures_recorded_not_raised(self):
        ev.__dict__["_completions"] = _fake_completions("no json here")
        with tempfile.TemporaryDirectory() as d:
            recs = self._run(Path(d) / "pre_control.json")
            self.assertFalse(any(r["parse_ok"] for r in recs.values()))
            self.assertEqual(ec.answer_view(recs), {})
            g = ec.contract_compliance_summary(recs, n_cells=2)
            self.assertEqual(g["parse_rate"], 0.0)
            self.assertIn("no_json_object", g["parse_failures"])

    def test_resume_same_config_is_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pre_control.json"
            self._run(out)
            recs = self._run(out)          # crash-recovery resume: no error, nothing to do
            self.assertEqual(len(recs), 2)

    def test_resume_mismatch_hard_errors_on_every_identity_field(self):
        mismatches = {
            "model": {"model": "OTHER"},
            "slice_sha256": {"slice_sha": "MOVED"},
            "store_sha256": {"store_sha": "MOVED"},
            "scaffold_variant": {"scaffold_variant": "rev_order"},
            "fewshot_k": {"fewshot_k": 5},
            "reason": {"reason": False},
            "budget": {"budget": 1024},
        }
        for field, kw in mismatches.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                out = Path(d) / "pre_control.json"
                self._run(out)
                with self.assertRaises(RuntimeError):
                    self._run(out, **kw)

    def test_prior_sidecar_missing_an_identity_field_hard_errors(self):
        """The inherited guard only iterates the PRIOR sidecar's keys, so a sidecar that
        predates a field would be silently unguarded. The explicit clauses must fire."""
        for dropped in ("contract", "slice_sha256", "store_sha256", "scaffold_variant",
                        "fewshot_k", "channel"):
            with self.subTest(dropped=dropped), tempfile.TemporaryDirectory() as d:
                out = Path(d) / "pre_control.json"
                self._run(out)
                mp = out.with_suffix(".meta.json")
                meta = json.loads(mp.read_text())
                meta.pop(dropped)
                mp.write_text(json.dumps(meta))
                with self.assertRaises(RuntimeError):
                    self._run(out)

    def test_wrong_contract_in_prior_sidecar_hard_errors(self):
        """A run dir holding AIReg-contract records can never be continued as a control
        leg (and vice versa) — the contract is leg identity, not metadata."""
        for wrong in ("contract_0_2_0", "mmlu_control_v2"):
            with self.subTest(contract=wrong), tempfile.TemporaryDirectory() as d:
                out = Path(d) / "pre_control.json"
                self._run(out)
                mp = out.with_suffix(".meta.json")
                meta = json.loads(mp.read_text())
                meta["contract"] = wrong
                mp.write_text(json.dumps(meta))
                with self.assertRaises(RuntimeError):
                    self._run(out)

    def test_missing_sidecar_hard_errors(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pre_control.json"
            self._run(out)
            out.with_suffix(".meta.json").unlink()
            with self.assertRaises(RuntimeError):
                self._run(out)

    def test_unknown_scaffold_variant_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                self._run(Path(d) / "pre_control.json", scaffold_variant="shuffle")

    def test_default_shas_come_from_the_pinned_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pre_control.json"
            ec.run_variant("http://x", "m", self.items, str(out), fewshot_k=4)
            meta = json.loads(out.with_suffix(".meta.json").read_text())
            self.assertEqual(meta["slice_sha256"], mmlu.SLICE_SHA256)
            self.assertEqual(meta["store_sha256"], cfs.STORE_SHA256)


class DriverLayoutTests(unittest.TestCase):
    def test_control_legs_cannot_be_read_by_the_aireg_driver(self):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
        import run_control_leg, run_study_b_leg
        d = Path("/tmp/never-created")
        self.assertEqual(run_control_leg.leg_path(d, "pre").name, "pre_control.json")
        self.assertEqual(run_study_b_leg.leg_path(d, "pre").name, "pre_verbalized.json")
        self.assertNotEqual(run_control_leg.leg_path(d, "pre"), run_study_b_leg.leg_path(d, "pre"))
        self.assertEqual(run_control_leg.PARSE_RATE_GATE, 0.90)


if __name__ == "__main__":
    unittest.main()
