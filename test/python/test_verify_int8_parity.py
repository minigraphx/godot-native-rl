import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_int8_parity as v  # noqa: E402


class TestInt8ParitySummary(unittest.TestCase):
    def test_pass(self):
        ok, summary = v.int8_parity_summary(0.96, 3, 50, 0.9)
        self.assertTrue(ok)
        self.assertIn("0.96", summary)

    def test_below_threshold_fails(self):
        ok, summary = v.int8_parity_summary(0.80, 4, 50, 0.9)
        self.assertFalse(ok)
        self.assertIn("threshold", summary)

    def test_degenerate_distinct_fails(self):
        ok, summary = v.int8_parity_summary(1.0, 1, 50, 0.9)
        self.assertFalse(ok)
        self.assertIn("distinct", summary)

    def test_threshold_boundary_inclusive(self):
        ok, _ = v.int8_parity_summary(0.9, 2, 50, 0.9)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
