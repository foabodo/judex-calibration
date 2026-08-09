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
import collections, json, tempfile, unittest
from pathlib import Path

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from judex_calibration import control_fewshot as cfs
from judex_calibration import elicit_control as ec
from judex_calibration import elicit_verbalized as ev
from judex_calibration import mmlu

# Row count of the pinned control exemplar store. Pinned as a constant so a store that
# silently shrinks (a filter regression) fails here rather than in a live leg. v1 = 237;
# v2 (E1 remediation) is the current pin, see cfs.STORE_SHA256.
STORE_ROWS = 333  # v2 (E1 remediation); v1 was 237

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
        self.assertEqual(n, STORE_ROWS)

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

    def test_every_selected_exemplar_is_in_the_span_band(self):
        """CHANGE 1's headline property, on the real store, for every variant.

        The band is a preference with a documented nearest-the-band fallback, so the
        assertion is scoped to what the store can actually deliver: every (subject,
        letter) bucket on store v2 holds at least one in-band row (asserted separately
        below), so the BASELINE and rev_order draws must be fully in band. ``alt_set``
        re-walks under source exclusion and may legitimately fall back — it is checked
        under the fallback test instead.
        """
        lo, hi = cfs.EXEMPLAR_SPAN_BAND
        self.assertEqual((lo, hi), (400, 800))
        self.assertEqual(cfs.EXEMPLAR_SELECTION, "band_400_800")
        for s in mmlu.SUBJECTS:
            for variant in ("baseline", "rev_order"):
                with self.subTest(subject=s, variant=variant):
                    for r in cfs.scaffold_rows(s, 4, variant):
                        n = cfs.span_len(r)
                        self.assertTrue(lo <= n <= hi, f"{s}/{variant} {r['id']}: {n} chars")

    def test_every_bucket_has_an_in_band_row(self):
        """Why the baseline draw can be asserted fully in band: the store supports it.

        Pinned on the store rather than inferred from the selector, so a future store
        that loses in-band depth in some bucket fails HERE, naming the bucket, instead of
        silently exercising the fallback inside a live leg.
        """
        lo, hi = cfs.EXEMPLAR_SPAN_BAND
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                for L in mmlu.OPTION_LABELS:
                    n = sum(1 for r in cfs.load_store()[s]
                            if r["answer_letter"] == L and lo <= cfs.span_len(r) <= hi)
                    self.assertGreaterEqual(n, 1, f"{s}/{L} has no in-band exemplar")

    def test_band_selection_actually_moved_the_draw(self):
        """The change is not a no-op: a length-blind walk over the same store draws spans
        outside the band, which is the configuration the Mac smoke ladder measured at
        0.538 parse vs 0.846 for the band draw."""
        lo, hi = cfs.EXEMPLAR_SPAN_BAND
        blind = []
        for s in mmlu.SUBJECTS:
            by_letter = {}
            for r in cfs.load_store()[s]:
                by_letter.setdefault(r["answer_letter"], []).append(r)
            blind += [by_letter[L][0] for L in mmlu.OPTION_LABELS]   # stored order, no band
        self.assertTrue([r for r in blind if not (lo <= cfs.span_len(r) <= hi)],
                        "the length-blind draw is already all in band — test is vacuous")

    def test_band_restrict_falls_back_to_nearest_and_never_empties(self):
        """The fallback contract: no in-band row => nearest-the-band row(s), never [] —
        and an empty pool is a coverage failure that must be LOUD, not a band miss."""
        rows = [{"id": "far", "answer_justification": "x" * 2000},
                {"id": "near", "answer_justification": "x" * 900},
                {"id": "near2", "answer_justification": "y" * 900}]
        got = cfs.band_restrict(rows)
        self.assertEqual([r["id"] for r in got], ["near", "near2"])   # ties both kept
        below = [{"id": "short", "answer_justification": "x" * 100},
                 {"id": "shorter", "answer_justification": "x" * 10}]
        self.assertEqual([r["id"] for r in cfs.band_restrict(below)], ["short"])
        mixed = [{"id": "out", "answer_justification": "x" * 2000},
                 {"id": "in", "answer_justification": "x" * 500}]
        self.assertEqual([r["id"] for r in cfs.band_restrict(mixed)], ["in"])
        with self.assertRaises(ValueError):
            cfs.band_restrict([])

    def test_band_selection_is_deterministic_and_still_coverage_complete(self):
        """Determinism and coverage are the two properties the band must not cost."""
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                a = [r["id"] for r in cfs.scaffold_rows(s, 4)]
                b = [r["id"] for r in cfs.scaffold_rows(s, 4)]
                self.assertEqual(a, b)
                self.assertEqual(len(a), 4)
                self.assertEqual({r["answer_letter"] for r in cfs.scaffold_rows(s, 4)},
                                 set(mmlu.OPTION_LABELS))

    def test_band_survives_a_bucket_with_no_in_band_row(self):
        """A toy store whose C bucket is entirely out of band: the letter is still
        covered (fallback), and the other letters still draw in band."""
        lo, hi = cfs.EXEMPLAR_SPAN_BAND
        rows = []
        for i, L in enumerate(mmlu.OPTION_LABELS):
            for j, n in enumerate((2000, 550) if L != "C" else (2000, 1500)):
                rows.append({"id": f"{L}{j}", "answer_letter": L, "answer_1to4": i + 1,
                             "source_item_label": f"src:{L}{j}", "rater_model": f"r{j}",
                             "answer_justification": "x" * n})
        orig = cfs._STORE_CACHE
        try:
            cfs._STORE_CACHE = {"toy": rows}
            picked = cfs.scaffold_rows("toy", 4, "baseline")
            self.assertEqual({r["answer_letter"] for r in picked}, set(mmlu.OPTION_LABELS))
            by_letter = {r["answer_letter"]: r for r in picked}
            for L in ("A", "B", "D"):
                self.assertTrue(lo <= cfs.span_len(by_letter[L]) <= hi, L)
            self.assertEqual(cfs.span_len(by_letter["C"]), 1500)   # nearest the band
        finally:
            cfs._STORE_CACHE = orig

    def test_alt_set_is_feasible_on_the_v2_store(self):
        """FLIPPED at the v2 re-pin (was ``test_alt_set_fails_loud_on_the_v1_store``).

        The v1 candidate pool used min_candidates_per_letter=1, so at least one letter per
        subject traced to a SINGLE source item and the source-exclusion re-walk could not
        be coverage-preserving: the guard raised for all six subjects and the assertion
        was that it did. The v2 pool is depth 2 by construction and survives the
        correct-answer + backslash filters at depth 2 in every bucket, so the disjoint
        draw must now GO THROUGH — on the real store, not a toy one.

        The guard itself is unchanged, and the toy-store test below still pins its
        raising behaviour."""
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                base = cfs.scaffold_rows(s, 4, "baseline")
                alt = cfs.scaffold_rows(s, 4, "alt_set")
                self.assertEqual(len(alt), 4)
                self.assertEqual({r["answer_letter"] for r in alt}, set(mmlu.OPTION_LABELS))
                self.assertFalse({r["id"] for r in base} & {r["id"] for r in alt})
                self.assertFalse({r["source_item_label"] for r in base}
                                 & {r["source_item_label"] for r in alt})
                feas = cfs.alt_set_feasibility(s, 4)
                self.assertTrue(feas["alt_set_feasible"], feas["error"])
                self.assertEqual(feas["starved_letters"], [])
                self.assertIsNone(feas["error"])

    def test_alt_set_guard_still_raises_on_a_shallow_store(self):
        """The v2 store no longer trips the guard, so the guard's raising path needs its
        own coverage: a one-source-per-letter store must still fail LOUD rather than
        silently drop a letter."""
        rows = [{"id": f"only:{L}", "answer_letter": L, "answer_1to4": i + 1,
                 "source_item_label": f"only:{L}", "rater_model": f"r{i}"}
                for i, L in enumerate(mmlu.OPTION_LABELS)]
        orig = cfs._STORE_CACHE
        try:
            cfs._STORE_CACHE = {"shallow": rows}
            with self.assertRaises(RuntimeError):
                cfs.scaffold_rows("shallow", 4, "alt_set")
            feas = cfs.alt_set_feasibility("shallow", 4)
            self.assertFalse(feas["alt_set_feasible"])
            self.assertTrue(feas["starved_letters"])
        finally:
            cfs._STORE_CACHE = orig

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

    def test_alt_set_band_selects_among_the_remainder(self):
        """Variant interplay: alt_set excludes the BAND-selected primary's source items,
        then band-selects among what is left.

        alt_set is where the fallback earns its keep — four (subject, letter) buckets on
        store v2 hold a source item with no in-band row, so a hard in-band requirement
        would make the disjoint draw infeasible. The assertion is therefore the honest
        one: alt_set stays feasible and coverage-preserving everywhere, its draw is
        source-disjoint from the primary, and it is in band wherever the remainder
        allowed it — with any fallback row reported rather than hidden.
        """
        lo, hi = cfs.EXEMPLAR_SPAN_BAND
        fallbacks = 0
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                base = cfs.scaffold_rows(s, 4, "baseline")
                alt = cfs.scaffold_rows(s, 4, "alt_set")
                self.assertEqual({r["answer_letter"] for r in alt}, set(mmlu.OPTION_LABELS))
                self.assertFalse({r["source_item_label"] for r in base}
                                 & {r["source_item_label"] for r in alt})
                feas = cfs.alt_set_feasibility(s, 4)
                self.assertEqual(feas["exemplar_selection"], cfs.EXEMPLAR_SELECTION)
                self.assertEqual(feas["baseline_out_of_band"], [])
                self.assertEqual(feas["baseline_span_lens"], [cfs.span_len(r) for r in base])
                for r in alt:
                    if not (lo <= cfs.span_len(r) <= hi):
                        fallbacks += 1
                        # a fallback is only legitimate if the remainder had no in-band row
                        rest = [x for x in cfs.load_store()[s]
                                if x["answer_letter"] == r["answer_letter"]
                                and x["source_item_label"] not in
                                {b["source_item_label"] for b in base}]
                        self.assertFalse([x for x in rest if lo <= cfs.span_len(x) <= hi],
                                         f"{s}: alt_set fell back with in-band rows left")
        self.assertLessEqual(fallbacks, 24)

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            cfs.scaffold_rows("philosophy", 4, "shuffle")


class DensityTests(unittest.TestCase):
    """k=8 — the doubled-density draw (E1 iteration (b), 2026-08-09).

    The coverage rule generalizes from "one exemplar per letter" to "k/4 exemplars per
    letter, every letter covered". These tests state it at BOTH densities so the property
    is the rule and not a k=4 coincidence, and they pin the two things a density change
    could quietly break: letter balance and the rendered ramp.
    """
    KS = (4, 8)

    def test_default_k_is_unchanged_by_the_density_iteration(self):
        """k=8 is a per-leg --fewshot-k, not a new protocol constant. The default moves
        only if the orchestrator adopts it."""
        self.assertEqual(cfs.default_k(), 4)
        self.assertEqual(cfs.CONTROL_FEWSHOT_K, 4)

    def test_every_letter_covered_k_over_four_times(self):
        for k in self.KS:
            per = k // len(mmlu.OPTION_LABELS)
            for s in mmlu.SUBJECTS:
                with self.subTest(k=k, subject=s):
                    rows = cfs.scaffold_rows(s, k)
                    self.assertEqual(len(rows), k)
                    self.assertEqual(len({r["id"] for r in rows}), k)     # no repeats
                    counts = collections.Counter(r["answer_letter"] for r in rows)
                    self.assertEqual(dict(counts), {L: per for L in mmlu.OPTION_LABELS})

    def test_k8_draw_is_a_superset_of_the_k4_draw(self):
        """The walk is deterministic and pass 1 is unchanged, so raising the density ADDS
        exemplars rather than substituting them — which is what makes the k=4 leg and the
        k=8 leg a clean density contrast."""
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                four = [r["id"] for r in cfs.select_rows(s, 4)]
                eight = [r["id"] for r in cfs.select_rows(s, 8)]
                self.assertEqual(eight[:4], four)

    def test_render_order_is_the_repeated_letter_ramp(self):
        for k in self.KS:
            per = k // len(mmlu.OPTION_LABELS)
            expect = [L for L in mmlu.OPTION_LABELS for _ in range(per)]
            for s in mmlu.SUBJECTS:
                with self.subTest(k=k, subject=s):
                    rows = cfs.scaffold_rows(s, k)
                    self.assertEqual([r["answer_letter"] for r in cfs.order_rows(rows)], expect)
                    self.assertEqual(
                        [r["answer_letter"] for r in cfs.order_rows(rows, "rev_order")],
                        list(reversed(expect)))

    def test_determinism_and_rater_spread_at_both_densities(self):
        for k in self.KS:
            for s in mmlu.SUBJECTS:
                with self.subTest(k=k, subject=s):
                    a = [r["id"] for r in cfs.scaffold_rows(s, k)]
                    self.assertEqual(a, [r["id"] for r in cfs.scaffold_rows(s, k)])
                    c = collections.Counter(r["rater_model"] for r in cfs.scaffold_rows(s, k))
                    # the least-used walk must never collapse onto one seat: all four
                    # letters draw distinct raters, and no seat exceeds k/4 + 1 picks
                    self.assertGreaterEqual(len(c), len(mmlu.OPTION_LABELS))
                    self.assertLessEqual(max(c.values()), k // len(mmlu.OPTION_LABELS) + 1)

    def test_band_holds_at_k8_and_every_miss_is_a_documented_fallback(self):
        """At k=8 the draw needs TWO in-band rows per bucket, and one bucket on store v2
        has only one (econometrics/B). The assertion is therefore the honest one: any
        out-of-band pick must be a bucket whose in-band depth was genuinely exhausted."""
        lo, hi = cfs.EXEMPLAR_SPAN_BAND
        per = 2
        fallbacks = 0
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                for r in cfs.scaffold_rows(s, 8):
                    if lo <= cfs.span_len(r) <= hi:
                        continue
                    fallbacks += 1
                    depth = sum(1 for x in cfs.load_store()[s]
                                if x["answer_letter"] == r["answer_letter"]
                                and lo <= cfs.span_len(x) <= hi)
                    self.assertLess(depth, per,
                                    f"{s}/{r['answer_letter']} fell back with in-band rows left")
        self.assertEqual(fallbacks, 1)   # pinned: econometrics/B, the only depth-1 bucket

    def test_every_bucket_has_two_in_band_rows_except_the_pinned_one(self):
        """Pinned on the STORE, so a future store that loses in-band depth fails here,
        naming the bucket, rather than silently exercising the fallback in a live leg."""
        lo, hi = cfs.EXEMPLAR_SPAN_BAND
        thin = []
        for s in mmlu.SUBJECTS:
            for L in mmlu.OPTION_LABELS:
                n = sum(1 for r in cfs.load_store()[s]
                        if r["answer_letter"] == L and lo <= cfs.span_len(r) <= hi)
                self.assertGreaterEqual(n, 1, f"{s}/{L} has no in-band exemplar")
                if n < 2:
                    thin.append(f"{s}/{L}")
        self.assertEqual(thin, ["econometrics/B"])

    def test_non_multiple_of_four_k_is_a_loud_caller_error(self):
        """k=5 on the control would cover one letter twice and three once — the exact
        imbalance that made k=4 defective on the 5-level AIReg instrument."""
        for bad_k in (1, 2, 3, 5, 6, 7, 0):
            with self.subTest(k=bad_k), self.assertRaises(ValueError):
                cfs.scaffold_rows("philosophy", bad_k)

    def test_balance_guard_fires_when_a_bucket_starves_at_density(self):
        """A store with depth 2 everywhere except one letter: fine at k=4, LOUD at k=8 —
        the guard must be stated at the density, not at coverage-of-the-letter-set."""
        rows = []
        for i, L in enumerate(mmlu.OPTION_LABELS):
            for j in range(1 if L == "C" else 2):
                rows.append({"id": f"{L}{j}", "answer_letter": L, "answer_1to4": i + 1,
                             "source_item_label": f"src:{L}{j}", "rater_model": f"r{j}",
                             "answer_justification": "x" * 600})
        orig = cfs._STORE_CACHE
        try:
            cfs._STORE_CACHE = {"thin": rows}
            self.assertEqual(len(cfs.scaffold_rows("thin", 4)), 4)
            with self.assertRaises(RuntimeError):
                cfs.scaffold_rows("thin", 8)
        finally:
            cfs._STORE_CACHE = orig

    def test_alt_set_feasibility_at_k8_is_reported_not_required(self):
        """alt_set is default-OFF (D4). At k=8 the primary can spend both source items of
        a letter, leaving the disjoint re-walk starved — the guard must raise there and
        the draw must still be clean where the depth allows it. Pinned as measured."""
        feasible, infeasible = [], []
        for s in mmlu.SUBJECTS:
            f = cfs.alt_set_feasibility(s, 8)
            (feasible if f["alt_set_feasible"] else infeasible).append(s)
            if f["alt_set_feasible"]:
                self.assertEqual(f["starved_letters"], [])
                base = cfs.scaffold_rows(s, 8, "baseline")
                alt = cfs.scaffold_rows(s, 8, "alt_set")
                self.assertEqual(len(alt), 8)
                self.assertFalse({r["id"] for r in base} & {r["id"] for r in alt})
                self.assertFalse({r["source_item_label"] for r in base}
                                 & {r["source_item_label"] for r in alt})
            else:
                self.assertTrue(f["starved_letters"])
                with self.assertRaises(RuntimeError):
                    cfs.scaffold_rows(s, 8, "alt_set")
        self.assertEqual(sorted(feasible), ["clinical_knowledge",
                                            "high_school_mathematics", "professional_law"])
        self.assertEqual(sorted(infeasible), ["econometrics", "formal_logic", "philosophy"])

    def test_rendered_k8_block_carries_eight_parsable_exemplars(self):
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                block = ec.build_fewshot(s, k=8)
                objs = [ln.split("JSON: ", 1)[1] for ln in block.splitlines()
                        if ln.startswith("JSON: ")]
                self.assertEqual(len(objs), 8)
                self.assertEqual([ec.parse_control_json(o)["answer_letter"] for o in objs],
                                 [L for L in mmlu.OPTION_LABELS for _ in range(2)])
                self.assertTrue(all(ec.parse_control_json(o)["contract_complete"] for o in objs))
                self.assertEqual(block.count("Example.\nEvidence:\n"), 8)
                self.assertTrue(block.endswith("\n\n"))
                # the k=4 block is a strict PREFIX of the k=8 block only in exemplar SET,
                # not in text (the ramp interleaves pass 2), so assert the set relation
                self.assertTrue(set(ln for ln in ec.build_fewshot(s, k=4).splitlines()
                                    if ln.startswith("JSON: ")) <= set(
                    ln for ln in block.splitlines() if ln.startswith("JSON: ")))

    def test_k8_exemplar_prose_carries_no_backslash_or_control_char(self):
        """Store v2's content invariant (Mode A), restated at the doubled density: the
        four ADDED exemplars per subject must be as backslash-free as the first four.

        Asserted on the store PROSE, not the rendered block — ``json.dumps`` legitimately
        emits ``\\uXXXX`` for the non-ASCII logic/maths glyphs the panel wrote, and that
        escaping is valid JSON, not the illegal-escape material Mode A is about."""
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                for r in cfs.scaffold_rows(s, 8):
                    for field in ("answer_justification", "confidence_justification"):
                        text = r.get(field) or ""
                        self.assertNotIn("\\", text, f"{r['id']}.{field}")
                        self.assertFalse([c for c in text if ord(c) < 32],
                                         f"{r['id']}.{field}")


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


class StoreV2ScaffoldContentTests(unittest.TestCase):
    """The two properties store v2 exists to guarantee (E1 remediation).

    Phase 2c's four base legs all failed the parse gate through two content-driven modes
    traced to the v1 store, not to the harness: LaTeX backslashes inside JSON string
    fields (`Invalid \\escape`), and a reasoning span roughly 3x shorter than the AIReg
    scaffold's, which taught base checkpoints that reasoning is optional. Both are
    properties of the exemplar CONTENT, so both are pinned here — the harness, the
    contract and the organic credences are unchanged and are covered by the tests above.
    """

    def _rendered_strings(self, row):
        return [row.get("reasoning") or "", row.get("answer_justification") or "",
                row.get("confidence_justification") or ""]

    def test_no_backslash_in_any_rendered_string_field(self):
        offenders = [(r["id"], s) for rows in cfs.load_store().values() for r in rows
                     for s in self._rendered_strings(r) if "\\" in s]
        self.assertEqual(offenders, [], f"{len(offenders)} store strings carry a backslash")

    def test_every_rendered_exemplar_json_is_valid_and_round_trips(self):
        """The end-to-end property, stated precisely.

        ``render_answer_json`` puts the panel's prose through ``json.dumps``, which emits
        its own legal escapes: ``\\"`` for a quote, and ``\\uXXXX`` for any non-ASCII
        character (the panel does use Unicode logic and math symbols such as the
        disjunction sign in formal_logic and econometrics). Those are backslashes in the
        rendered block, but they are JSON-VALID ones — a base checkpoint that imitates
        them emits parseable JSON, which is not the failure mode.

        What breaks stage 2 is a backslash the panel WROTE, i.e. a TeX command like
        ``\\frac`` sitting raw inside a string, because that is an ILLEGAL escape
        (``Invalid \\escape``). That property is pinned by
        ``test_no_backslash_in_any_rendered_string_field`` above, on the source strings.

        Here we pin the consequence that actually matters: every exemplar JSON the model
        is shown parses, and parses back to exactly the prose the panel wrote.

        The MMLU question and option text is benchmark quotation and may carry TeX; it is
        deliberately out of scope — the scaffold teaches by the prose it shows, and the
        prose is the panel's.
        """
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                for r in cfs.scaffold_rows(s, 4):
                    rendered = ec.render_answer_json(r)
                    obj = json.loads(rendered)           # must not raise
                    self.assertEqual(obj["answer_justification"],
                                     r["answer_justification"].strip(), r["id"])
                    self.assertEqual(obj["confidence_justification"],
                                     r["confidence_justification"].strip(), r["id"])
                    self.assertTrue(ec.parse_control_json(rendered)["parse_ok"], r["id"])

    def test_rendered_json_lines_contain_no_illegal_escape(self):
        """Mode A at the level the base checkpoint actually imitates it.

        Scoped to the ``JSON:`` lines of the rendered block, deliberately. The
        ``Evidence:``/``Criterion:`` sections carry the MMLU question and options
        verbatim, and MMLU's own text contains TeX (``$x\\%$`` in
        high_school_mathematics, for one) — that is benchmark quotation sitting in PROSE,
        not inside a JSON string, and it is not ours to rewrite.

        This is the honest boundary of the fix and it is worth stating plainly: the
        scaffold cannot stop showing the model LaTeX, because the question text has
        LaTeX in it. What v2 changes is what the model is shown to DO with it — every
        exemplar answers a TeX-bearing question in plain ASCII prose. So the property
        pinned here is that the JSON the model is taught to emit never contains an
        illegal escape, which is what made stage 2 unparseable.
        """
        legal = set('"\\/bfnrtu')
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                json_lines = [ln for ln in ec.build_fewshot(s).splitlines()
                              if ln.startswith("JSON: ")]
                self.assertEqual(len(json_lines), 4)
                for ln in json_lines:
                    for i, ch in enumerate(ln):
                        if ch == "\\":
                            self.assertIn(ln[i + 1], legal,
                                          f"{s}: illegal escape at {ln[max(0, i - 60):i + 20]!r}")

    def test_justification_spans_are_in_the_aireg_band(self):
        """AIReg dimension exemplars: min 246, median 610 characters. v1 control exemplars
        were median 188 with 56% under 200. v2 must clear the floor everywhere and sit in
        the target band on the median."""
        lens = sorted(len(r["answer_justification"])
                      for rows in cfs.load_store().values() for r in rows)
        self.assertGreaterEqual(lens[0], 246, "a stored span is shorter than AIReg's minimum")
        median = lens[len(lens) // 2]
        self.assertGreaterEqual(median, 400)
        self.assertLessEqual(median, 900)
        self.assertEqual([x for x in lens if x < 200], [])

    def test_every_subject_letter_bucket_has_two_source_items(self):
        """The depth alt_set needs, asserted on the store rather than inferred from the
        selector: this is the v1 limitation the v2 candidate top-up was built to remove."""
        for s in mmlu.SUBJECTS:
            with self.subTest(subject=s):
                by_letter = {}
                for r in cfs.load_store()[s]:
                    by_letter.setdefault(r["answer_letter"], set()).add(r["source_item_label"])
                for L in mmlu.OPTION_LABELS:
                    self.assertGreaterEqual(len(by_letter.get(L, ())), 2, f"{s}/{L}")


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

    def test_latex_braces_before_the_json_parse_under_last_object(self):
        """CHANGE 2's regression case, taken from Phase 2c's Mode A.

        MMLU stems carry TeX, and a model that restates one before answering emits
        ``\\frac{x}{12}``. The first-``{`` scan grabs ``{x}`` — balanced, not the answer,
        and a ``json.loads`` failure. The last-object scan skips it.
        """
        text = ("The rate is given by \\frac{x}{12} so the share is \\frac{1}{4}.\n"
                "JSON: " + GOOD_JSON)
        first = ec.parse_control_json(text)
        self.assertFalse(first["parse_ok"])           # the mode, reproduced
        self.assertIn("json_decode", first["parse_error"])
        rec = ec.parse_last_control(text)             # the fix
        self.assertTrue(rec["parse_ok"], rec.get("parse_error"))
        self.assertEqual(rec["answer_letter"], "B")
        self.assertTrue(rec["contract_complete"])

    def test_last_object_is_inert_on_a_plain_single_object(self):
        """Inertness: on emissions with no decoy brace the two parsers agree exactly,
        including on the failure paths, so switching the raw path changes nothing but
        the brace-hijack case."""
        for text in (GOOD_JSON,
                     "Here you go:\n" + GOOD_JSON + "\nDone.",
                     " " + GOOD_JSON,
                     "I cannot answer.",
                     '{"answer_letter": "B",}',
                     GOOD_JSON.replace("answer_distribution", "answers"),
                     GOOD_JSON.replace('"B": 0.65', '"B": 0.63')):
            with self.subTest(text=text[:40]):
                self.assertEqual(ec.parse_control_json(text), ec.parse_last_control(text))


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
        self.assertEqual(n, STORE_ROWS)

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
            self.assertEqual(meta["exemplar_selection"], "band_400_800")
            self.assertEqual(meta["exemplar_selection"], cfs.EXEMPLAR_SELECTION)
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
            "fewshot_k": {"fewshot_k": 8},
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
                        "fewshot_k", "exemplar_selection", "channel"):
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

    def test_wrong_exemplar_selection_in_prior_sidecar_hard_errors(self):
        """Band selection is leg identity, not metadata: a Phase-2c leg (length-blind, no
        such key) and a legacy leg claiming a different policy must both refuse to
        continue under the band draw — same store_sha256 notwithstanding."""
        for wrong in ("length_blind", "band_300_900"):
            with self.subTest(selection=wrong), tempfile.TemporaryDirectory() as d:
                out = Path(d) / "pre_control.json"
                self._run(out)
                mp = out.with_suffix(".meta.json")
                meta = json.loads(mp.read_text())
                meta["exemplar_selection"] = wrong
                mp.write_text(json.dumps(meta))
                with self.assertRaises(RuntimeError):
                    self._run(out)

    def test_elicit_cell_uses_last_object_parsing(self):
        """CHANGE 2 at the seam, not just at the parser: a stage-2 completion whose TeX
        precedes the JSON must produce a parse_ok record through ``elicit_cell``."""
        ev.__dict__["_completions"] = _fake_completions(
            "recall \\frac{x}{12} and \\frac{1}{4}, so: " + GOOD_JSON)
        rec = ec.elicit_cell("http://x", "m", "ev", "crit")
        self.assertTrue(rec["parse_ok"], rec.get("parse_error"))
        self.assertEqual(rec["answer_letter"], "B")
        self.assertEqual(rec["contract"], "mmlu_control_v1")
        self.assertEqual(rec["channel"], "verbalized")

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
