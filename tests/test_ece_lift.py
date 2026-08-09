"""The ECE lift out of base_calibration_premise_probe.py, and control scoring.

Three jobs:

  1. REGRESSION — ``judex_calibration.ece`` must compute exactly what the probe computed
     before the lift. Frozen verbatim copies of the pre-lift bodies live here (the
     ``_legacy_*`` functions, same pattern as ``test_scaffold_variants._legacy_build_fewshot``)
     and are asserted equal on synthetic rows AND on rows reconstructed from the shipped
     probe artifact. The grid + bounds re-declared in ``ece`` are asserted identical to
     ``study_a``'s, element for element — that identity is what keeps the lifted
     ``fit_T_ece`` on the same axis as everything else.
     (The whole-artifact proof is stronger and was run out-of-band: re-running the probe
     after the lift reproduced the pre-lift artifact byte for byte, sha256
     e77e1368483751a46b92d3828b49946cb756d3b8e09273474b4090880707e3c1.)

  2. TOY CASES — hand-computable ECE/MCE, AUROC at ties and at perfect separation,
     bootstrap seed determinism, and the <3-cluster guard.

  3. THE ORDINAL-METRIC GUARD — ``score_control`` must be structurally incapable of
     emitting RPS / W1 / Murphy on the nominal K=4 control (design D7). Checked at
     source level AND in a fresh interpreter's import graph.
"""
import json, math, random, subprocess, sys, unittest
from pathlib import Path

import pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from judex_calibration import ece, score_control, study_a  # noqa: E402
from judex_calibration import mmlu  # noqa: E402

ARTIFACT = REPO / "runs" / "base_calibration_premise" / "base_calibration_premise.json"


# ---------------------------------------------------------------------------
# Frozen pre-lift bodies (verbatim from scripts/base_calibration_premise_probe.py
# as it stood on 2026-08-09, before the lift). DO NOT "improve" these.
# ---------------------------------------------------------------------------

def _legacy_ece_equal_width(rows, bins=10):
    n = len(rows)
    if n == 0:
        return {"ece": float("nan"), "mce": float("nan"), "bins": []}
    buckets = [[] for _ in range(bins)]
    for _, conf, corr, _ in rows:
        idx = min(bins - 1, int(conf * bins))
        buckets[idx].append((conf, corr))
    ece_ = 0.0
    mce = 0.0
    table = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        acc = sum(c for _, c in b) / len(b)
        conf = sum(cf for cf, _ in b) / len(b)
        gap = abs(conf - acc)
        ece_ += (len(b) / n) * gap
        mce = max(mce, gap)
        table.append({"bin": f"[{i / bins:.1f},{(i + 1) / bins:.1f})", "n": len(b),
                      "mean_conf": round(conf, 4), "acc": round(acc, 4),
                      "signed_gap": round(conf - acc, 4)})
    return {"ece": ece_, "mce": mce, "bins": table}


def _legacy_aece_equal_mass(rows, bins=5):
    n = len(rows)
    if n == 0:
        return {"aece": float("nan"), "bins": []}
    order = sorted(rows, key=lambda r: r[1])
    edges = [round(i * n / bins) for i in range(bins + 1)]
    aece = 0.0
    table = []
    for i in range(bins):
        b = order[edges[i]:edges[i + 1]]
        if not b:
            continue
        acc = sum(r[2] for r in b) / len(b)
        conf = sum(r[1] for r in b) / len(b)
        aece += (len(b) / n) * abs(conf - acc)
        table.append({"n": len(b), "conf_lo": round(b[0][1], 4), "conf_hi": round(b[-1][1], 4),
                      "mean_conf": round(conf, 4), "acc": round(acc, 4),
                      "signed_gap": round(conf - acc, 4)})
    return {"aece": aece, "bins": table}


def _legacy_auroc(rows):
    pos = [r[1] for r in rows if r[2] == 1]
    neg = [r[1] for r in rows if r[2] == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Row fixtures
# ---------------------------------------------------------------------------

def synthetic_rows(n=240, seed=7):
    """Rows spanning the whole confidence range, including exact bin edges and ties."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        conf = [0.25, 0.5, 0.7, 0.9, 1.0][i % 5] if i % 11 == 0 else rng.uniform(0.25, 1.0)
        rows.append((f"doc{i % 12}", conf, int(rng.random() < conf), 0))
    return rows


def artifact_rows():
    """Rows reconstructed from the SHIPPED probe artifact's per-leg AECE5 bin tables.

    Each equal-mass bin contributes its ``n`` rows at the bin's mean confidence with
    ``round(acc * n)`` of them correct, so the confidence/accuracy profile driving the
    comparison is a real leg's, not an invented one. Returns {leg_key: rows}.
    """
    if not ARTIFACT.exists():
        return {}
    rep = json.loads(ARTIFACT.read_text())
    out = {}
    for key, fam in rep.get("families", {}).items():
        for ch, cv in fam.get("channels", {}).items():
            for leg in ("pre", "post"):
                bins = cv.get(leg, {}).get("correctness", {}).get("aece5_bins") or []
                rows = []
                for bi, b in enumerate(bins):
                    n, acc, conf = b["n"], b["acc"], b["mean_conf"]
                    ncorr = round(acc * n)
                    for j in range(n):
                        rows.append((f"doc{j % 24}", conf, int(j < ncorr), 0))
                if rows:
                    out[f"{key}|{ch}|{leg}"] = rows
    return out


class LiftRegressionTests(unittest.TestCase):
    def test_grid_and_bounds_identical_to_study_a(self):
        self.assertEqual(tuple(ece.T_BOUNDS), tuple(study_a.T_BOUNDS))
        self.assertEqual(len(ece.GRID), len(study_a.GRID))
        for a, b in zip(ece.GRID, study_a.GRID):
            self.assertEqual(a, b)

    def test_constants_unchanged(self):
        self.assertEqual((ece.EPSILON, ece.BOOT_N, ece.BOOT_SEED, ece.ECE_BINS, ece.AECE_BINS),
                         (0.005, 2000, 20260721, 10, 5))

    def test_saturated_matches_study_a(self):
        for T in [0.25, 0.2500001, 1.0, 19.99999, 20.0, float("nan")]:
            self.assertEqual(ece.saturated(T), study_a.saturated(T), T)

    def test_estimators_match_legacy_on_synthetic_rows(self):
        rows = synthetic_rows()
        new_ew, old_ew = ece.ece_equal_width(rows), _legacy_ece_equal_width(rows)
        self.assertEqual(new_ew["ece"], old_ew["ece"])
        self.assertEqual(new_ew["mce"], old_ew["mce"])
        self.assertEqual(new_ew["bins"], old_ew["bins"])
        self.assertEqual(ece.mce(rows), old_ew["mce"])
        self.assertEqual(ece.aece_equal_mass(rows), _legacy_aece_equal_mass(rows))
        self.assertEqual(ece.auroc(rows), _legacy_auroc(rows))

    def test_estimators_match_legacy_on_artifact_rows(self):
        rows_by_leg = artifact_rows()
        if not rows_by_leg:
            self.skipTest("shipped probe artifact absent (runs/ is gitignored)")
        self.assertGreaterEqual(len(rows_by_leg), 8)
        for key, rows in rows_by_leg.items():
            with self.subTest(leg=key):
                self.assertEqual(ece.ece_equal_width(rows), _legacy_ece_equal_width(rows))
                self.assertEqual(ece.aece_equal_mass(rows), _legacy_aece_equal_mass(rows))
                self.assertEqual(ece.auroc(rows), _legacy_auroc(rows))

    def test_estimators_agree_on_empty_and_degenerate_input(self):
        self.assertTrue(math.isnan(ece.ece_equal_width([])["ece"]))
        self.assertTrue(math.isnan(ece.aece_equal_mass([])["aece"]))
        allright = [("d", 0.9, 1, 0)] * 5
        self.assertTrue(math.isnan(ece.auroc(allright)))
        self.assertEqual(ece.ece_equal_width(allright)["ece"], _legacy_ece_equal_width(allright)["ece"])

    def test_shipped_artifact_ece_reproduces_from_its_own_bin_table(self):
        """The stored ECE10 must equal the weighted mean |conf - acc| of its own bins."""
        if not ARTIFACT.exists():
            self.skipTest("shipped probe artifact absent (runs/ is gitignored)")
        rep = json.loads(ARTIFACT.read_text())
        checked = 0
        for fam in rep.get("families", {}).values():
            for cv in fam.get("channels", {}).values():
                for leg in ("pre", "post"):
                    c = cv.get(leg, {}).get("correctness")
                    if not c or not c.get("ece10_bins"):
                        continue
                    n = sum(b["n"] for b in c["ece10_bins"])
                    recomputed = sum(b["n"] / n * abs(b["mean_conf"] - b["acc"])
                                     for b in c["ece10_bins"])
                    # the stored bin table rounds mean_conf/acc to 4dp, so the
                    # reconstruction carries ~1e-4 of rounding noise by construction
                    self.assertAlmostEqual(recomputed, c["ece10"], delta=5e-4,
                                           msg=f"{fam['label']} {leg}")
                    checked += 1
        self.assertGreaterEqual(checked, 8)


class ToyScoringTests(unittest.TestCase):
    def test_hand_computed_two_bin_ece_and_mce(self):
        # bin [0.9,1.0): 4 rows at conf .95, 2 correct -> gap .45, weight .4
        # bin [0.3,0.4): 6 rows at conf .35, 3 correct -> gap .15, weight .6
        rows = ([("d", 0.95, 1, 0)] * 2 + [("d", 0.95, 0, 0)] * 2
                + [("d", 0.35, 1, 0)] * 3 + [("d", 0.35, 0, 0)] * 3)
        out = ece.ece_equal_width(rows)
        self.assertAlmostEqual(out["ece"], 0.4 * 0.45 + 0.6 * 0.15, places=12)
        self.assertAlmostEqual(out["ece"], 0.27, places=12)
        self.assertAlmostEqual(out["mce"], 0.45, places=12)
        self.assertEqual(len(out["bins"]), 2)

    def test_perfectly_calibrated_rows_have_zero_ece(self):
        rows = [("d", 0.5, 1, 0)] * 5 + [("d", 0.5, 0, 0)] * 5
        self.assertAlmostEqual(ece.ece_equal_width(rows)["ece"], 0.0, places=12)

    def test_auroc_all_ties_is_one_half(self):
        rows = [("d", 0.7, 1, 0)] * 4 + [("d", 0.7, 0, 0)] * 4
        self.assertEqual(ece.auroc(rows), 0.5)

    def test_auroc_perfect_separation_is_one(self):
        rows = [("d", 0.9, 1, 0)] * 3 + [("d", 0.3, 0, 0)] * 3
        self.assertEqual(ece.auroc(rows), 1.0)
        flipped = [("d", 0.3, 1, 0)] * 3 + [("d", 0.9, 0, 0)] * 3
        self.assertEqual(ece.auroc(flipped), 0.0)

    def test_auroc_undefined_without_both_classes(self):
        self.assertTrue(math.isnan(ece.auroc([("d", 0.9, 1, 0)])))


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.items = mmlu.load_control_items()
        rng = random.Random(3)
        self.preds = {}
        for it in self.items:
            v = [rng.uniform(0.05, 1.0) for _ in range(4)]
            if rng.random() < 0.7:          # informative but overconfident, ~70% correct
                v[it.gt_argmax] += 1.2
            z = sum(v)
            self.preds[it.item_label] = [x / z for x in v]

    def test_seed_determinism(self):
        a = ece.boot_ece(self.preds, self.items, n=50)
        b = ece.boot_ece(self.preds, self.items, n=50)
        self.assertEqual(a, b)
        c = ece.boot_ece(self.preds, self.items, n=50, seed=ece.BOOT_SEED + 1)
        self.assertNotEqual(a["median"], c["median"])

    def test_paired_delta_seed_determinism_and_pairing(self):
        post = {k: [x ** 2 for x in v] for k, v in self.preds.items()}
        post = {k: [x / sum(v) for x in v] for k, v in post.items()}
        a = ece.boot_paired_ece_delta(self.preds, post, self.items, n=50)
        b = ece.boot_paired_ece_delta(self.preds, post, self.items, n=50)
        self.assertEqual(a, b)
        self.assertEqual(a["n_docs"], 120)
        same = ece.boot_paired_ece_delta(self.preds, self.preds, self.items, n=50)
        self.assertAlmostEqual(same["median"], 0.0, places=12)

    def test_item_and_subject_clustering_differ(self):
        subj_items = mmlu.load_control_items(document_id="subject")
        item_boot = ece.boot_ece(self.preds, self.items, n=50)
        subj_boot = ece.boot_ece(self.preds, subj_items, n=50)
        self.assertEqual(item_boot["n_docs"], 120)   # D7 primary: i.i.d. items
        self.assertEqual(subj_boot["n_docs"], 6)     # sensitivity: 6 subject clusters
        for b in (item_boot, subj_boot):
            self.assertLessEqual(b["lo"], b["median"])
            self.assertLessEqual(b["median"], b["hi"])
        self.assertNotEqual(item_boot["median"], subj_boot["median"])

    def test_fewer_than_three_clusters_returns_none(self):
        two = self.items[:2]
        preds = {i.item_label: self.preds[i.item_label] for i in two}
        self.assertIsNone(ece.boot_ece(preds, two, n=10))
        self.assertIsNone(ece.boot_paired_ece_delta(preds, preds, two, n=10))


class ScoreControlTests(unittest.TestCase):
    def setUp(self):
        self.items = mmlu.load_control_items()
        self.subj = mmlu.load_control_items(document_id="subject")
        rng = random.Random(11)
        self.pre, self.post = {}, {}
        for it in self.items:
            v = [rng.uniform(0.05, 1.0) for _ in range(4)]
            if rng.random() < 0.65:         # ~65% accurate, so AUROC is defined
                v[it.gt_argmax] += 1.0
            z = sum(v)
            p = [x / z for x in v]
            self.pre[it.item_label] = p
            sharp = [x ** 3 for x in p]
            self.post[it.item_label] = [x / sum(sharp) for x in sharp]

    def _recs(self, preds):
        return {k: {"parse_ok": True, "compliance": v, "confidence": [0.2, 0.5, 0.3],
                    "contract": "mmlu_control_v1", "contract_complete": True,
                    "contract_missing": [], "answer_letter": "A",
                    "level_matches_argmax": True,
                    "compliance_on_grid": False, "confidence_on_grid": True}
                for k, v in preds.items()}

    def test_answer_view_applies_the_epsilon_floor(self):
        recs = {"x": {"parse_ok": True, "compliance": [1.0, 0.0, 0.0, 0.0]},
                "y": {"parse_ok": False}}
        v = score_control.answer_view(recs)
        self.assertEqual(set(v), {"x"})
        self.assertGreater(min(v["x"]), 0.0)
        self.assertAlmostEqual(sum(v["x"]), 1.0, places=12)

    def test_score_leg_shape_and_chance_floor(self):
        out = score_control.score_leg(self.pre, self.items, self.subj, bootstrap=False)
        self.assertEqual(out["n"], 120)
        self.assertEqual(out["chance_floor"], 0.25)
        self.assertAlmostEqual(out["accuracy_above_chance"], out["accuracy"] - 0.25)
        self.assertAlmostEqual(out["overconfidence_gap"],
                               out["mean_confidence"] - out["accuracy"], places=12)
        for k in ("ece10", "aece5", "mce10", "auroc_conf_vs_correct"):
            self.assertIsInstance(out[k], float)
        self.assertEqual(out["temperature"]["T_ece10"] in ece.GRID, True)
        self.assertEqual(list(map(float, [0.25, 20.0])), list(map(float, ece.T_BOUNDS)))

    def test_score_leg_empty_predictions(self):
        self.assertEqual(score_control.score_leg({}, self.items, self.subj), {"n": 0})

    def test_full_analyze_report(self):
        rep = score_control.analyze({"pre": self._recs(self.pre), "post": self._recs(self.post)},
                                    bootstrap=True, boot_n=40)
        self.assertEqual(rep["contract"], "mmlu_control_v1")
        self.assertEqual(rep["slice_sha256"], mmlu.SLICE_SHA256)
        self.assertEqual(rep["n_items"], 120)
        self.assertEqual(set(rep["legs"]), {"pre", "post"})
        self.assertEqual(rep["legs"]["pre"]["ece10_boot_item"]["n_docs"], 120)
        self.assertEqual(rep["legs"]["pre"]["ece10_boot_subject"]["n_docs"], 6)
        d = rep["e2_paired_ece_delta"]
        self.assertAlmostEqual(d["delta_post_minus_pre"], d["ece10_post"] - d["ece10_pre"])
        self.assertIn(d["direction"], ("post_improves", "post_degrades", "tie"))
        self.assertIn(rep["e1_read"]["band"],
                      ("daca_like_below_0.10", "intermediate_0.10_to_0.20",
                       "aireg_like_above_0.20"))
        self.assertFalse(math.isnan(rep["legs"]["pre"]["auroc_conf_vs_correct"]))
        # nothing ordinal is EMITTED: no report key names an ordinal quantity (the prose
        # values deliberately do — that is where the guarantee is written down)
        def keys(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield k
                    yield from keys(v)
            elif isinstance(node, list):
                for v in node:
                    yield from keys(v)
        emitted = {k.lower() for k in keys(rep)}
        for banned in ("rps", "mean_rps", "w1", "mean_w1", "murphy", "wasserstein",
                       "resolution", "reliability", "t_rps", "t_rel", "tau_oc"):
            self.assertNotIn(banned, emitted, banned)

    def test_tau_tvd_is_descriptive_and_permutation_invariant(self):
        t = score_control.fit_tau_tvd(self.post, self.pre, self.items)
        self.assertIn(t["tau_tvd"], ece.GRID)
        self.assertIn("DESCRIPTIVE ONLY", t["status"])
        self.assertLessEqual(t["mean_tvd_at_tau"], t["mean_tvd_at_T1"])
        # permuting the option alphabet consistently must not move the fit; a W1
        # objective would move, which is exactly why D6 mandates TVD.
        perm = [2, 0, 3, 1]
        pre_p = {k: [v[i] for i in perm] for k, v in self.pre.items()}
        post_p = {k: [v[i] for i in perm] for k, v in self.post.items()}
        items_p = [type(it)(**{**it.__dict__, "gt_argmax": perm.index(it.gt_argmax)})
                   for it in self.items]
        t2 = score_control.fit_tau_tvd(post_p, pre_p, items_p)
        self.assertEqual(t["tau_tvd"], t2["tau_tvd"])
        self.assertAlmostEqual(t["mean_tvd_at_tau"], t2["mean_tvd_at_tau"], places=12)

    def test_tau_tvd_identity_when_post_equals_pre(self):
        t = score_control.fit_tau_tvd(self.pre, self.pre, self.items)
        self.assertAlmostEqual(t["mean_tvd_at_T1"], 0.0, places=12)
        self.assertAlmostEqual(t["tau_tvd"], 1.0, delta=0.15)


class OrdinalMetricGuardTests(unittest.TestCase):
    """D7: the control path must be structurally incapable of emitting ordinal metrics."""
    BANNED = ("score_variant", "murphy_decomposition", "wasserstein_1",
              "ranked_probability_score")

    def test_source_of_score_control_and_ece_is_clean(self):
        for mod in ("score_control.py", "ece.py"):
            src = (REPO / "src" / "judex_calibration" / mod).read_text()
            body = "\n".join(l for l in src.splitlines()
                             if not l.lstrip().startswith("#"))
            # the module docstring names them to explain the guard; strip it before grepping
            body = body.split('"""', 2)[-1]
            for name in self.BANNED:
                self.assertNotIn(name, body, f"{mod} mentions {name}")
            self.assertNotIn("study_a", body, mod)
            self.assertNotIn("judex.experiments", body, mod)

    def test_no_banned_name_is_bound_in_either_namespace(self):
        for mod in (score_control, ece):
            for name in self.BANNED:
                self.assertFalse(hasattr(mod, name), f"{mod.__name__}.{name}")

    def test_fresh_interpreter_import_graph(self):
        """study_a / judex.experiments must not even be IMPORTED by score_control.

        (``judex.core.metrics`` does arrive transitively via ``judex.calibration`` — that
        is upstream's package layout, not a call site; the namespace assertions above
        prove no name from it is bound here.)
        """
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "from judex_calibration import score_control\n"
            "bad = [m for m in ('judex_calibration.study_a', 'judex.experiments',\n"
            "                   'judex_calibration.aireg') if m in sys.modules]\n"
            "print(','.join(bad))\n" % str(REPO / "src")
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "", f"score_control imported: {r.stdout.strip()}")


if __name__ == "__main__":
    unittest.main()
