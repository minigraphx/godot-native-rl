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
        self.assertEqual(cfg.batch_size, 512)

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
        self.assertIn("--env=gdrl", argv)
        self.assertIn("--serial_mode=True", argv)
        self.assertIn("--async_rl=False", argv)
        self.assertIn("--num_workers=1", argv)
        self.assertIn("--num_envs_per_worker=1", argv)
        self.assertIn("--normalize_input=False", argv)
        self.assertIn("--normalize_returns=False", argv)
        self.assertIn("--use_rnn=False", argv)
        self.assertIn("--device=cpu", argv)
        # Checkpoint-reliability knobs for tiny budgets (small batch + 1 batch/epoch + frequent save).
        self.assertIn("--batch_size=512", argv)
        self.assertIn("--num_batches_per_epoch=1", argv)
        self.assertIn("--save_every_sec=5", argv)
        self.assertIn("--keep_checkpoints=1", argv)
        self.assertIn("--train_for_env_steps=2000", argv)
        self.assertIn("--base_port=11008", argv)
        self.assertIn("--env_agents=1", argv)
        self.assertIn("--experiment=chase_sf", argv)


class TestNestScalarActions(unittest.TestCase):
    def test_bare_scalar_single_agent(self):
        # SF's NonBatchedMultiAgentWrapper unwraps the 1-agent list -> our step gets a bare int.
        # from_numpy needs [agent][key], so 3 -> [[3]].
        self.assertEqual(ts.nest_scalar_actions(3), [[3]])

    def test_per_agent_scalars_are_wrapped(self):
        # Multi-agent single-key: one bare int per agent -> [[a],[b],...].
        self.assertEqual(ts.nest_scalar_actions([3]), [[3]])
        self.assertEqual(ts.nest_scalar_actions([3, 1, 4]), [[3], [1], [4]])

    def test_existing_sequences_pass_through(self):
        # Multi-key action spaces already arrive as per-key sequences; leave them alone.
        self.assertEqual(ts.nest_scalar_actions([[1, 2], [3, 4]]), [[1, 2], [3, 4]])

    def test_numpy_bare_zero_d_scalar(self):
        import numpy as np

        self.assertEqual(ts.nest_scalar_actions(np.int64(7)), [[7]])

    def test_numpy_zero_d_scalars_are_wrapped(self):
        import numpy as np

        out = ts.nest_scalar_actions([np.int32(3), np.int64(1)])
        self.assertEqual(out, [[3], [1]])

    def test_numpy_1d_per_agent_passes_through(self):
        import numpy as np

        out = ts.nest_scalar_actions([np.array([2]), np.array([5])])
        self.assertEqual([list(x) for x in out], [[2], [5]])


if __name__ == "__main__":
    unittest.main()
