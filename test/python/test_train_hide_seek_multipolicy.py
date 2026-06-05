import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_hide_seek_multipolicy as mp  # noqa: E402


class TestPolicyIndexMap(unittest.TestCase):
    def test_single_world(self):
        self.assertEqual(mp.policy_index_map(["seeker", "hider"]),
                         {"seeker": [0], "hider": [1]})

    def test_parallel_interleaved(self):
        names = ["seeker", "hider", "seeker", "hider"]
        self.assertEqual(mp.policy_index_map(names),
                         {"seeker": [0, 2], "hider": [1, 3]})

    def test_first_seen_key_order(self):
        self.assertEqual(list(mp.policy_index_map(["hider", "seeker"]).keys()),
                         ["hider", "seeker"])


if __name__ == "__main__":
    unittest.main()
