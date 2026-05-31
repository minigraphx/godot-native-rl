import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_rover as tr  # noqa: E402


class TestLatestCheckpoint(unittest.TestCase):
    def test_missing_dir_returns_none(self):
        self.assertIsNone(tr.latest_checkpoint("/no/such/dir/anywhere"))

    def test_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(tr.latest_checkpoint(d))

    def test_picks_highest_step_count(self):
        with tempfile.TemporaryDirectory() as d:
            for name in (
                "rover_ckpt_5000_steps.zip",
                "rover_ckpt_50000_steps.zip",
                "rover_ckpt_25000_steps.zip",
                "unrelated.txt",
            ):
                (Path(d) / name).touch()
            self.assertEqual(
                tr.latest_checkpoint(d),
                str(Path(d) / "rover_ckpt_50000_steps.zip"),
            )

    def test_ignores_non_matching_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "model.zip").touch()
            (Path(d) / "rover_ckpt_notanumber_steps.zip").touch()
            self.assertIsNone(tr.latest_checkpoint(d))


class TestRemainingTimesteps(unittest.TestCase):
    def test_difference(self):
        self.assertEqual(tr.remaining_timesteps(400_000, 125_000), 275_000)

    def test_done_equals_total(self):
        self.assertEqual(tr.remaining_timesteps(400_000, 400_000), 0)

    def test_overshoot_clamps_to_zero(self):
        self.assertEqual(tr.remaining_timesteps(400_000, 450_000), 0)


if __name__ == "__main__":
    unittest.main()
