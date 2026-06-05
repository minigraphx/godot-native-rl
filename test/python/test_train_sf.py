import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_sf as ts  # noqa: E402


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        cfg = ts.parse_args([])
        self.assertEqual(cfg.timesteps, 1_000_000)
        self.assertEqual(cfg.base_port, 11008)
        self.assertEqual(cfg.env_agents, 1)
        self.assertEqual(cfg.experiment, "chase_sf")
        self.assertEqual(cfg.train_dir, "logs/sf")

    def test_overrides(self):
        cfg = ts.parse_args(["--timesteps", "2000", "--base_port", "12000", "--env_agents", "4"])
        self.assertEqual(cfg.timesteps, 2000)
        self.assertEqual(cfg.base_port, 12000)
        self.assertEqual(cfg.env_agents, 4)


class TestClientPort(unittest.TestCase):
    def test_single_worker_offset(self):
        # godot_rl's make_godot_env_func uses base_port + 1 + env_id; the single serial
        # worker is env_id=0, so the Godot client connects on base_port + 1.
        self.assertEqual(ts.client_port(11008), 11009)


class TestBuildSfArgv(unittest.TestCase):
    def test_contains_macos_and_parity_safe_overrides(self):
        cfg = ts.parse_args(["--timesteps", "2000", "--base_port", "11008", "--env_agents", "1"])
        argv = ts.build_sf_argv(cfg)
        self.assertIn("--serial_mode=True", argv)
        self.assertIn("--async_rl=False", argv)
        self.assertIn("--num_workers=1", argv)
        self.assertIn("--num_envs_per_worker=1", argv)
        self.assertIn("--normalize_input=False", argv)
        self.assertIn("--normalize_returns=False", argv)
        self.assertIn("--use_rnn=False", argv)
        self.assertIn("--device=cpu", argv)
        self.assertIn("--train_for_env_steps=2000", argv)
        self.assertIn("--base_port=11008", argv)
        self.assertIn("--env_agents=1", argv)
        self.assertIn("--experiment=chase_sf", argv)


if __name__ == "__main__":
    unittest.main()
