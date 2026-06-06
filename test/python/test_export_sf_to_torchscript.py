import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_sf_to_torchscript as ex  # noqa: E402


class TestActorLogitLayout(unittest.TestCase):
    def test_single_discrete(self):
        total, nvec = ex.actor_logit_layout([5])
        self.assertEqual(total, 5)
        self.assertEqual(nvec, [5])

    def test_multi_discrete(self):
        total, nvec = ex.actor_logit_layout([3, 2, 4])
        self.assertEqual(total, 9)
        self.assertEqual(nvec, [3, 2, 4])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ex.actor_logit_layout([])

    def test_non_positive_raises(self):
        with self.assertRaises(ValueError):
            ex.actor_logit_layout([5, 0])


if __name__ == "__main__":
    unittest.main()
