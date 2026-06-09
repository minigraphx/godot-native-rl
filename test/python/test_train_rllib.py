import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_rllib as tr  # noqa: E402


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        cfg = tr.parse_args([])
        self.assertEqual(cfg.base_port, 11008)
        self.assertEqual(cfg.experiment, "chase_rllib")
        self.assertEqual(cfg.train_dir, "logs/rllib")
        self.assertEqual(cfg.speedup, 8)
        self.assertEqual(cfg.action_repeat, 8)

    def test_overrides(self):
        cfg = tr.parse_args(["--timesteps", "2000", "--base_port", "12000"])
        self.assertEqual(cfg.timesteps, 2000)
        self.assertEqual(cfg.base_port, 12000)


class TestPpoConfigOverrides(unittest.TestCase):
    def test_single_socket_and_parity_safe(self):
        cfg = tr.parse_args(["--timesteps", "2000"])
        o = tr.ppo_config_overrides(cfg)
        # num_env_runners=0 => rollouts on the driver: exactly one env, one socket.
        self.assertEqual(o["num_env_runners"], 0)
        # No obs normalization anywhere (ncnn parity: the exported actor must be a plain MLP).
        self.assertFalse(o.get("normalize_obs", False))
        self.assertEqual(o["framework"], "torch")


class TestNestAction(unittest.TestCase):
    def test_scalar_to_godot_structure(self):
        # GodotEnv.step wants one list per agent, one entry per action key: Discrete scalar -> [[a]].
        self.assertEqual(tr.nest_action(3), [[3]])


if __name__ == "__main__":
    unittest.main()
