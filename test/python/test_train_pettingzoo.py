import sys
import unittest
from pathlib import Path

import numpy as np
from gymnasium import spaces

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_pettingzoo as tp  # noqa: E402


class TestStackByAgent(unittest.TestCase):
    def test_orders_by_agent_list(self):
        per_agent = {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}
        out = tp.stack_by_agent(per_agent, [0, 1])
        self.assertEqual(out.shape, (2, 2))
        np.testing.assert_array_equal(out[1], np.array([3.0, 4.0]))

    def test_respects_agent_order(self):
        per_agent = {0: np.array([1.0]), 1: np.array([2.0])}
        out = tp.stack_by_agent(per_agent, [1, 0])
        np.testing.assert_array_equal(out[0], np.array([2.0]))


class TestToActionDict(unittest.TestCase):
    def test_scatters_rows_to_agents(self):
        full = np.array([[1], [2]], dtype=np.int64)
        d = tp.to_action_dict(full, [0, 1])
        np.testing.assert_array_equal(d[0], np.array([1]))
        np.testing.assert_array_equal(d[1], np.array([2]))

    def test_respects_agent_order(self):
        full = np.array([[1], [2]], dtype=np.int64)
        d = tp.to_action_dict(full, [1, 0])
        np.testing.assert_array_equal(d[1], np.array([1]))
        np.testing.assert_array_equal(d[0], np.array([2]))


class TestActionNvec(unittest.TestCase):
    def test_reads_tuple_of_discrete(self):
        space = spaces.Tuple((spaces.Discrete(5), spaces.Discrete(3)))
        self.assertEqual(tp.action_nvec(space), [5, 3])

    def test_rejects_non_tuple_space(self):
        with self.assertRaises(ValueError):
            tp.action_nvec(spaces.MultiDiscrete([5, 3]))


class TestUnwrapObs(unittest.TestCase):
    def test_extracts_inner_obs_key(self):
        # godot_rl GodotEnv returns per-agent Dict obs {'obs': vec}; unwrap to {agent: vec}.
        obs_dict = {0: {"obs": np.array([1.0, 2.0])}, 1: {"obs": np.array([3.0, 4.0])}}
        out = tp.unwrap_obs(obs_dict)
        np.testing.assert_array_equal(out[0], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(out[1], np.array([3.0, 4.0]))

    def test_custom_key(self):
        obs_dict = {0: {"camera": np.array([5.0])}}
        out = tp.unwrap_obs(obs_dict, key="camera")
        np.testing.assert_array_equal(out[0], np.array([5.0]))


if __name__ == "__main__":
    unittest.main()
