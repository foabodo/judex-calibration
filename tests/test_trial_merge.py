"""Unit tests for the driver's per-family run dirs + cross-family --merge helpers.

Pure-helper tests (no live server, no elicitation): FAMILY=DIR spec parsing, run-dir
loading, and the smoke-inheritance rule — a merge must flag smoke if ANY source dir
carries a .smoke sentinel or ANY leg has <120 cells.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_DRIVER = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen_phase1.py"
_spec = importlib.util.spec_from_file_location("run_qwen_phase1", _DRIVER)
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)


def _preds(n):
    return {f"cell_{i}": [0.2] * 5 for i in range(n)}


def _write_run_dir(root, name, pre=None, post=None, smoke=False):
    d = Path(root) / name
    d.mkdir()
    if pre is not None:
        (d / "pre.json").write_text(json.dumps(pre))
    if post is not None:
        (d / "post.json").write_text(json.dumps(post))
    if smoke:
        (d / ".smoke").write_text("smoke\n")
    return d


class ParseMergeSpecsTests(unittest.TestCase):
    def test_parses_family_dir_pairs(self):
        specs = driver.parse_merge_specs(["qwen=runs/qwen", "gemma=runs/gemma"])
        self.assertEqual(list(specs), ["qwen", "gemma"])
        self.assertEqual(specs["gemma"], Path("runs/gemma"))

    def test_rejects_malformed_spec(self):
        for bad in ("qwenruns/qwen", "=runs/qwen", "qwen="):
            with self.assertRaises(SystemExit):
                driver.parse_merge_specs([bad])

    def test_rejects_duplicate_family(self):
        with self.assertRaises(SystemExit):
            driver.parse_merge_specs(["qwen=runs/a", "qwen=runs/b"])


class MergeFamilyRunsTests(unittest.TestCase):
    def test_two_clean_families_not_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = _write_run_dir(tmp, "qwen", pre=_preds(120), post=_preds(120))
            g = _write_run_dir(tmp, "gemma", pre=_preds(120), post=_preds(120))
            families, smoke = driver.merge_family_runs({"qwen": q, "gemma": g})
            self.assertEqual(set(families), {"qwen", "gemma"})
            self.assertEqual(set(families["gemma"]), {"pre", "post"})
            self.assertFalse(smoke)

    def test_short_leg_in_one_family_taints_the_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = _write_run_dir(tmp, "qwen", pre=_preds(120), post=_preds(120))
            g = _write_run_dir(tmp, "gemma", pre=_preds(6), post=_preds(120))
            _, smoke = driver.merge_family_runs({"qwen": q, "gemma": g})
            self.assertTrue(smoke)

    def test_smoke_sentinel_in_one_family_taints_the_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = _write_run_dir(tmp, "qwen", pre=_preds(120), post=_preds(120), smoke=True)
            g = _write_run_dir(tmp, "gemma", pre=_preds(120), post=_preds(120))
            _, smoke = driver.merge_family_runs({"qwen": q, "gemma": g})
            self.assertTrue(smoke)

    def test_empty_dir_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = _write_run_dir(tmp, "qwen")  # no legs at all
            with self.assertRaises(SystemExit):
                driver.merge_family_runs({"qwen": q})

    def test_single_leg_family_is_allowed(self):
        # a pre-only dir merges fine (tau_oc just won't be fit for it)
        with tempfile.TemporaryDirectory() as tmp:
            q = _write_run_dir(tmp, "qwen", pre=_preds(120))
            families, smoke = driver.merge_family_runs({"qwen": q})
            self.assertEqual(set(families["qwen"]), {"pre"})
            self.assertFalse(smoke)


if __name__ == "__main__":
    unittest.main()
