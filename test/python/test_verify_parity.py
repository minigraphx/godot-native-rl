import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_ncnn_parity as vp  # noqa: E402


class TestParitySummary(unittest.TestCase):
    def test_all_pass(self):
        ok, summary = vp.parity_summary(0, 0, 3, 50)
        self.assertTrue(ok)
        self.assertIn("50/50 argmax match", summary)
        self.assertIn("3 distinct actions", summary)

    def test_argmax_mismatch_fails_first(self):
        ok, summary = vp.parity_summary(4, 7, 1, 50)
        self.assertFalse(ok)
        self.assertEqual(summary, "4/50 argmax mismatches")

    def test_value_mismatch_message(self):
        ok, summary = vp.parity_summary(0, 2, 3, 50)
        self.assertFalse(ok)
        self.assertIn("2/50", summary)
        self.assertIn("atol=", summary)

    def test_value_mismatch_message_custom_atol(self):
        ok, summary = vp.parity_summary(0, 2, 3, 50, atol=0.05)
        self.assertFalse(ok)
        self.assertIn("atol=0.05", summary)

    def test_degenerate_single_action(self):
        ok, summary = vp.parity_summary(0, 0, 1, 50)
        self.assertFalse(ok)
        self.assertIn("1 distinct action", summary)

    def test_two_distinct_actions_passes(self):
        ok, summary = vp.parity_summary(0, 0, 2, 50)
        self.assertTrue(ok)

    def test_verify_result_is_frozen(self):
        r = vp.VerifyResult(True, 0, 0, 3, 50, "ok")
        with self.assertRaises(Exception):
            r.ok = False  # frozen dataclass


if __name__ == "__main__":
    unittest.main()
