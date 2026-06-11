import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_rover as tr  # noqa: E402

# Checkpoint selection now lives in the shared `checkpoints` module (see
# test_checkpoints.py); the trainer calls select_checkpoint(..., policy="resume").


class TestRemainingTimesteps(unittest.TestCase):
    def test_difference(self):
        self.assertEqual(tr.remaining_timesteps(400_000, 125_000), 275_000)

    def test_done_equals_total(self):
        self.assertEqual(tr.remaining_timesteps(400_000, 400_000), 0)

    def test_overshoot_clamps_to_zero(self):
        self.assertEqual(tr.remaining_timesteps(400_000, 450_000), 0)


if __name__ == "__main__":
    unittest.main()
