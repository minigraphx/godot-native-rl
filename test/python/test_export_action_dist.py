"""Tests for scripts/export_action_dist.py (std extraction from SB3 PPO log_std)."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import export_action_dist as ead  # noqa: E402

# Guarded heavy import (#141): missing torch -> skip, not error, under bare python.
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class TestStdFromLogStd(unittest.TestCase):
    def test_exp_of_log_std(self):
        stats = ead.std_from_log_std([0.0, math.log(2.0), math.log(0.5)])
        self.assertEqual(stats["action_dim"], 3)
        self.assertAlmostEqual(stats["std"][0], 1.0, places=6)
        self.assertAlmostEqual(stats["std"][1], 2.0, places=6)
        self.assertAlmostEqual(stats["std"][2], 0.5, places=6)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ead.std_from_log_std([])


class TestStdFromModel(unittest.TestCase):
    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_extracts_from_policy_log_std(self):
        import torch
        model = SimpleNamespace(policy=SimpleNamespace(log_std=torch.tensor([0.0, math.log(3.0)])))
        stats = ead.std_from_model(model)
        self.assertEqual(stats["action_dim"], 2)
        self.assertAlmostEqual(stats["std"][1], 3.0, places=5)

    def test_no_log_std_raises(self):
        model = SimpleNamespace(policy=SimpleNamespace())
        with self.assertRaises(ValueError):
            ead.std_from_model(model)


class TestWriteJson(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "action_dist.json"
            ead.write_action_dist_json({"std": [1.0, 2.0], "action_dim": 2}, out)
            loaded = json.loads(out.read_text())
            self.assertEqual(loaded["std"], [1.0, 2.0])
            self.assertEqual(loaded["action_dim"], 2)


if __name__ == "__main__":
    unittest.main()
