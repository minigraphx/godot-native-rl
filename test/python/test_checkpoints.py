import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import checkpoints as cp  # noqa: E402


def _touch(d, name, mtime=None):
    p = Path(d) / name
    p.touch()
    if mtime is not None:
        import os
        os.utime(p, (mtime, mtime))
    return p


class TestHighestStep(unittest.TestCase):
    def test_missing_dir_returns_none(self):
        self.assertIsNone(cp.highest_step_checkpoint("/no/such/dir/anywhere"))

    def test_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(cp.highest_step_checkpoint(d))

    def test_picks_highest_step_not_mtime(self):
        # Write the highest-step file FIRST (oldest mtime) so an mtime picker would miss it.
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "rover_ckpt_50000_steps.zip", mtime=1000)
            _touch(d, "rover_ckpt_5000_steps.zip", mtime=2000)
            _touch(d, "rover_ckpt_25000_steps.zip", mtime=3000)
            _touch(d, "unrelated.txt")
            self.assertEqual(
                cp.highest_step_checkpoint(d),
                str(Path(d) / "rover_ckpt_50000_steps.zip"),
            )

    def test_prefix_agnostic(self):
        # Any CheckpointCallback name_prefix works (not just "rover_ckpt").
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "ball_chase_ckpt_25000_steps.zip")
            self.assertEqual(
                cp.highest_step_checkpoint(d),
                str(Path(d) / "ball_chase_ckpt_25000_steps.zip"),
            )

    def test_ignores_non_matching_files(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "model.zip")
            _touch(d, "rover_ckpt_notanumber_steps.zip")
            self.assertIsNone(cp.highest_step_checkpoint(d))


class TestBestReward(unittest.TestCase):
    def test_none_when_absent(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "rover_ckpt_50000_steps.zip")
            self.assertIsNone(cp.best_reward_checkpoint(d))

    def test_picks_best_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "rover_ckpt_50000_steps.zip")
            _touch(d, "rover_best.zip")
            self.assertEqual(
                cp.best_reward_checkpoint(d),
                str(Path(d) / "rover_best.zip"),
            )

    def test_newest_best_when_several(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "a_best.zip", mtime=1000)
            _touch(d, "b_best.zip", mtime=2000)
            self.assertEqual(
                cp.best_reward_checkpoint(d),
                str(Path(d) / "b_best.zip"),
            )


class TestNewestByMtime(unittest.TestCase):
    def test_missing_dir_returns_none(self):
        self.assertIsNone(cp.newest_by_mtime("/no/such/dir/anywhere"))

    def test_picks_newest(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "old.zip", mtime=1000)
            new = _touch(d, "new.zip", mtime=2000)
            self.assertEqual(cp.newest_by_mtime(d), str(new))


class TestSelectCheckpoint(unittest.TestCase):
    def test_resume_uses_highest_step_over_mtime(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "rover_ckpt_50000_steps.zip", mtime=1000)  # highest step, oldest
            _touch(d, "rover_ckpt_5000_steps.zip", mtime=9000)   # newest mtime, low step
            self.assertEqual(
                cp.select_checkpoint(d, policy="resume"),
                str(Path(d) / "rover_ckpt_50000_steps.zip"),
            )

    def test_resume_ignores_best(self):
        # resume never prefers *_best.zip (deploy artifact, not a resumable training state).
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "rover_ckpt_50000_steps.zip")
            _touch(d, "rover_best.zip")
            self.assertEqual(
                cp.select_checkpoint(d, policy="resume"),
                str(Path(d) / "rover_ckpt_50000_steps.zip"),
            )

    def test_resume_best_only_dir_starts_fresh(self):
        # A dir containing ONLY a best zip must not be resumed from via the mtime
        # fallback (#139 review carry-over): best is a deploy artifact.
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "rover_ckpt_best.zip")
            self.assertIsNone(cp.select_checkpoint(d, policy="resume"))
            # ...while deploy still finds it.
            self.assertEqual(
                cp.select_checkpoint(d, policy="deploy"),
                str(Path(d) / "rover_ckpt_best.zip"),
            )

    def test_resume_mtime_fallback_skips_best(self):
        # Non-step zips fall back to mtime for resume, but a newer best zip never wins.
        with tempfile.TemporaryDirectory() as d:
            plain = _touch(d, "model.zip", mtime=1000)
            _touch(d, "rover_ckpt_best.zip", mtime=2000)
            self.assertEqual(cp.select_checkpoint(d, policy="resume"), str(plain))

    def test_deploy_prefers_best(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "rover_ckpt_50000_steps.zip")
            _touch(d, "rover_best.zip")
            self.assertEqual(
                cp.select_checkpoint(d, policy="deploy"),
                str(Path(d) / "rover_best.zip"),
            )

    def test_deploy_falls_through_to_highest_step(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "rover_ckpt_25000_steps.zip")
            _touch(d, "rover_ckpt_50000_steps.zip")
            self.assertEqual(
                cp.select_checkpoint(d, policy="deploy"),
                str(Path(d) / "rover_ckpt_50000_steps.zip"),
            )

    def test_falls_through_to_mtime_for_nonstandard_names(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(d, "model_a.zip", mtime=1000)
            newest = _touch(d, "model_b.zip", mtime=2000)
            self.assertEqual(cp.select_checkpoint(d, policy="deploy"), str(newest))

    def test_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(cp.select_checkpoint(d, policy="deploy"))
            self.assertIsNone(cp.select_checkpoint(d, policy="resume"))

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            cp.select_checkpoint("/tmp", policy="bogus")


class TestBestZipPath(unittest.TestCase):
    def test_naming_matches_best_suffix(self):
        # The writer's path must be discoverable by the deploy picker.
        with tempfile.TemporaryDirectory() as d:
            best = cp.best_zip_path(d, "rover_ckpt")
            self.assertEqual(best, Path(d) / "rover_ckpt_best.zip")
            best.touch()
            self.assertEqual(cp.best_reward_checkpoint(d), str(best))


if __name__ == "__main__":
    unittest.main()
