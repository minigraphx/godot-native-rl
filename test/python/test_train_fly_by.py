"""Pure-helper tests for scripts/train_fly_by.py (arg parsing; no SB3/torch import)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import train_fly_by as tfb  # noqa: E402


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        a = tfb.parse_args([])
        self.assertGreater(a.timesteps, 0)
        self.assertTrue(a.save_model_path.endswith(".zip"))
        self.assertTrue(a.pt_export_path.endswith(".pt"))

    def test_overrides(self):
        a = tfb.parse_args(["--timesteps", "1234", "--speedup", "4"])
        self.assertEqual(a.timesteps, 1234)
        self.assertEqual(a.speedup, 4)


if __name__ == "__main__":
    unittest.main()
