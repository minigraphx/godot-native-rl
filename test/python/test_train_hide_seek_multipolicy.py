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


class TestSplitStitch(unittest.TestCase):
    def test_split_by_policy(self):
        index_map = {"seeker": [0, 2], "hider": [1, 3]}
        batched = np.array([[10.0], [11.0], [12.0], [13.0]])
        out = mp.split_by_policy(batched, index_map)
        np.testing.assert_array_equal(out["seeker"], np.array([[10.0], [12.0]]))
        np.testing.assert_array_equal(out["hider"], np.array([[11.0], [13.0]]))

    def test_stitch_is_inverse_of_split(self):
        index_map = {"seeker": [0, 2], "hider": [1, 3]}
        actions = np.array([[1], [2], [3], [4]], dtype=np.int64)  # (n_agents, action_dim)
        per_policy = mp.split_by_policy(actions, index_map)
        stitched = mp.stitch_actions(per_policy, index_map, n_agents=4)
        np.testing.assert_array_equal(stitched, actions)


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        cfg = mp.parse_args([])
        self.assertEqual(cfg.timesteps, 800_000)
        self.assertEqual(cfg.export_dir, "models")
        self.assertEqual(cfg.policy_names, ("seeker", "hider"))

    def test_overrides(self):
        cfg = mp.parse_args(["--timesteps", "1234", "--speedup", "4"])
        self.assertEqual(cfg.timesteps, 1234)
        self.assertEqual(cfg.speedup, 4)


if __name__ == "__main__":
    unittest.main()
