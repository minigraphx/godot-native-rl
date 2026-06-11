import json
import pathlib
import sys
import tempfile
import unittest

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import reward_checkpoint as rc  # noqa: E402

try:
    import stable_baselines3  # noqa: F401
    HAVE_SB3 = True
except ImportError:
    HAVE_SB3 = False


class TestRollingMeanReward(unittest.TestCase):
    def test_none_below_min_episodes(self):
        self.assertIsNone(rc.rolling_mean_reward([{"r": 1.0}], min_episodes=2))

    def test_none_when_empty(self):
        self.assertIsNone(rc.rolling_mean_reward([], min_episodes=1))

    def test_zero_min_episodes_still_needs_one(self):
        # max(min_episodes, 1): an empty buffer never produces a mean.
        self.assertIsNone(rc.rolling_mean_reward([], min_episodes=0))

    def test_mean(self):
        infos = [{"r": 1.0}, {"r": 2.0}, {"r": 3.0}]
        self.assertAlmostEqual(rc.rolling_mean_reward(infos, min_episodes=3), 2.0)


class TestIsNewBest(unittest.TestCase):
    def test_beats_prior(self):
        self.assertTrue(rc.is_new_best(2.0, 1.0))

    def test_equal_is_not_new(self):
        self.assertFalse(rc.is_new_best(1.0, 1.0))

    def test_worse_is_not_new(self):
        self.assertFalse(rc.is_new_best(0.5, 1.0))

    def test_anything_beats_neg_inf(self):
        self.assertTrue(rc.is_new_best(-1e9, float("-inf")))

    def test_nan_never_wins(self):
        self.assertFalse(rc.is_new_best(float("nan"), float("-inf")))


class TestSidecarPersistence(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            side = pathlib.Path(d) / "rover_ckpt_best.zip.json"
            rc.save_best_meta(side, 3.25, 125_000)
            self.assertEqual(rc.load_best_reward(side), 3.25)
            data = json.loads(side.read_text())
            self.assertEqual(data["num_timesteps"], 125_000)
            self.assertIn("saved_at", data)

    def test_missing_is_neg_inf(self):
        self.assertEqual(
            rc.load_best_reward(pathlib.Path("/no/such/sidecar.json")), float("-inf")
        )

    def test_malformed_is_neg_inf(self):
        with tempfile.TemporaryDirectory() as d:
            side = pathlib.Path(d) / "bad.json"
            side.write_text("not json{")
            self.assertEqual(rc.load_best_reward(side), float("-inf"))
            side.write_text('{"wrong_key": 1}')
            self.assertEqual(rc.load_best_reward(side), float("-inf"))

    def test_sidecar_path_naming(self):
        best = pathlib.Path("/m/rover_ckpt_best.zip")
        self.assertEqual(
            rc.sidecar_path(best), pathlib.Path("/m/rover_ckpt_best.zip.json")
        )


class _StubModel:
    """Just enough model surface for the callback: ep_info_buffer + save()."""

    def __init__(self):
        self.ep_info_buffer = []
        self.num_timesteps = 0
        self.saved_to = []

    def save(self, path):
        self.saved_to.append(pathlib.Path(path))
        pathlib.Path(path).write_text("fake-zip")


@unittest.skipUnless(HAVE_SB3, "stable_baselines3 not installed")
class TestRewardGatedCheckpoint(unittest.TestCase):
    def _make(self, d, min_episodes=2):
        cb = rc.make_reward_gated_checkpoint(d, "rover_ckpt", min_episodes=min_episodes)
        cb.model = _StubModel()
        return cb

    def test_saves_on_improvement_and_skips_otherwise(self):
        with tempfile.TemporaryDirectory() as d:
            cb = self._make(d)
            cb.model.ep_info_buffer = [{"r": 1.0}, {"r": 2.0}]
            cb.model.num_timesteps = 512
            cb.on_rollout_end()
            self.assertEqual(len(cb.model.saved_to), 1)
            self.assertEqual(cb.model.saved_to[0].name, "rover_ckpt_best.zip")
            self.assertEqual(cb.best, 1.5)
            # Same mean again -> no second save.
            cb.on_rollout_end()
            self.assertEqual(len(cb.model.saved_to), 1)
            # Worse mean -> still no save.
            cb.model.ep_info_buffer = [{"r": 0.0}, {"r": 1.0}]
            cb.on_rollout_end()
            self.assertEqual(len(cb.model.saved_to), 1)
            # Better mean -> saves again and updates the sidecar.
            cb.model.ep_info_buffer = [{"r": 3.0}, {"r": 3.0}]
            cb.model.num_timesteps = 1024
            cb.on_rollout_end()
            self.assertEqual(len(cb.model.saved_to), 2)
            self.assertEqual(rc.load_best_reward(cb.sidecar), 3.0)

    def test_respects_min_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            cb = self._make(d, min_episodes=5)
            cb.model.ep_info_buffer = [{"r": 10.0}]  # high but too few episodes
            cb.on_rollout_end()
            self.assertEqual(cb.model.saved_to, [])

    def test_resume_reloads_best_from_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            first = self._make(d)
            first.model.ep_info_buffer = [{"r": 5.0}, {"r": 5.0}]
            first.on_rollout_end()
            self.assertEqual(len(first.model.saved_to), 1)
            # A "resumed run" constructs a fresh callback: best must come back as 5.0,
            # so a worse post-restart mean cannot overwrite the better best zip.
            resumed = self._make(d)
            self.assertEqual(resumed.best, 5.0)
            resumed.model.ep_info_buffer = [{"r": 2.0}, {"r": 2.0}]
            resumed.on_rollout_end()
            self.assertEqual(resumed.model.saved_to, [])


if __name__ == "__main__":
    unittest.main()
