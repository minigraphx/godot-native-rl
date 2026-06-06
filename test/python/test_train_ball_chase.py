import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_ball_chase as t  # noqa: E402


class TestLatestCheckpoint(unittest.TestCase):
    def test_missing_dir_returns_none(self):
        self.assertIsNone(t.latest_checkpoint("/nonexistent/dir/xyz"))

    def test_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(t.latest_checkpoint(d))

    def test_picks_highest_step(self):
        with tempfile.TemporaryDirectory() as d:
            for n in (5000, 25000, 10000):
                (Path(d) / f"ball_chase_ckpt_{n}_steps.zip").touch()
            self.assertEqual(
                t.latest_checkpoint(d),
                str(Path(d) / "ball_chase_ckpt_25000_steps.zip"),
            )

    def test_ignores_non_matching_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "ignore_me.txt").touch()
            (Path(d) / "ball_chase_ckpt_notanumber_steps.zip").touch()
            (Path(d) / "rover_ckpt_99000_steps.zip").touch()  # wrong prefix
            self.assertIsNone(t.latest_checkpoint(d))


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

    def test_overrides(self):
        a = t.parse_args(["--timesteps", "1234", "--speedup", "4", "--fresh"])
        self.assertEqual(a.timesteps, 1234)
        self.assertEqual(a.speedup, 4)
        self.assertTrue(a.fresh)


if __name__ == "__main__":
    unittest.main()
