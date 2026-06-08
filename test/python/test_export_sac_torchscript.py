import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_sac_torchscript as m  # noqa: E402


class TestLatestCheckpoint(unittest.TestCase):
    def test_missing_dir_returns_empty(self):
        self.assertEqual(m.latest_checkpoint("/nonexistent/dir/xyz"), "")

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(m.latest_checkpoint(d), "")

    def test_picks_newest_by_mtime(self):
        import os
        import time
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "ball_chase_ckpt_5000_steps.zip"
            new = Path(d) / "ball_chase_ckpt_25000_steps.zip"
            old.touch()
            time.sleep(0.01)
            new.touch()
            # Force distinct mtimes regardless of fs granularity.
            os.utime(old, (1, 1))
            os.utime(new, (2, 2))
            self.assertEqual(m.latest_checkpoint(d), str(new))


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        a = m.parse_args([])
        self.assertEqual(a.checkpoint, "")
        self.assertEqual(a.checkpoint_dir, "models/ball_chase_checkpoints")
        self.assertEqual(a.pt_export_path, "models/ball_chase_sac.pt")

    def test_overrides(self):
        a = m.parse_args(["--checkpoint", "x.zip", "--pt_export_path", "out.pt"])
        self.assertEqual(a.checkpoint, "x.zip")
        self.assertEqual(a.pt_export_path, "out.pt")


if __name__ == "__main__":
    unittest.main()
