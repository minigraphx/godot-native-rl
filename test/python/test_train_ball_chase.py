import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_ball_chase as t  # noqa: E402

# Checkpoint selection now lives in the shared `checkpoints` module (see
# test_checkpoints.py); the trainer calls select_checkpoint(..., policy="resume").


class TestRemainingTimesteps(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(t.remaining_timesteps(100, 30), 70)

    def test_never_negative(self):
        self.assertEqual(t.remaining_timesteps(100, 250), 0)


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        a = t.parse_args([])
        self.assertEqual(a.timesteps, 200_000)
        self.assertEqual(a.speedup, 8)
        self.assertEqual(a.action_repeat, 8)
        self.assertEqual(a.seed, 0)
        self.assertEqual(a.save_model_path, "models/ball_chase_sac.zip")
        self.assertEqual(a.pt_export_path, "models/ball_chase_sac.pt")
        self.assertEqual(a.checkpoint_freq, 25_000)
        self.assertEqual(a.checkpoint_dir, "models/ball_chase_checkpoints")
        self.assertFalse(a.fresh)
        self.assertFalse(a.best_checkpoint)

    def test_overrides(self):
        a = t.parse_args(["--timesteps", "1234", "--speedup", "4", "--fresh",
                          "--best_checkpoint"])
        self.assertEqual(a.timesteps, 1234)
        self.assertEqual(a.speedup, 4)
        self.assertTrue(a.fresh)
        self.assertTrue(a.best_checkpoint)


if __name__ == "__main__":
    unittest.main()
