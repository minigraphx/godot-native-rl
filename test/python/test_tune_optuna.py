"""Unit tests for the pure helpers in scripts/tune_optuna.py (issue #113).

tune_optuna is stdlib-only at module load (torch/SB3/optuna/godot_rl are imported lazily inside the
trial-running functions), so these tests run with no ML stack installed — they cover the HP sampling
space, the PPO-kwarg mapping (incl. the minibatch-divisibility fix), the ep_rew_mean extraction, and
the best-trial summary."""
import sys
import unittest
from collections import deque
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import tune_optuna as to  # noqa: E402  (stdlib-only at import; no dep probe needed)


class RecordingTrial:
    """Minimal stand-in for an optuna Trial: records each suggested param and returns a deterministic
    in-range value (the low end for floats/ints, the first choice for categoricals) so the sampled
    dict is predictable and the suggest_* call surface is asserted."""

    def __init__(self):
        self.calls = {}

    def suggest_float(self, name, low, high, log=False):
        self.calls[name] = ("float", low, high, log)
        return low

    def suggest_int(self, name, low, high):
        self.calls[name] = ("int", low, high)
        return low

    def suggest_categorical(self, name, choices):
        self.calls[name] = ("categorical", list(choices))
        return choices[0]


class TestSampleHyperparams(unittest.TestCase):
    def test_samples_all_expected_keys(self):
        hp = to.sample_ppo_hyperparams(RecordingTrial())
        self.assertEqual(
            set(hp),
            {"learning_rate", "n_steps", "batch_size", "n_epochs",
             "gamma", "gae_lambda", "ent_coef", "clip_range"},
        )

    def test_records_suggest_calls_and_ranges(self):
        t = RecordingTrial()
        to.sample_ppo_hyperparams(t)
        # learning_rate + ent_coef + gamma are log-scale floats.
        self.assertEqual(t.calls["learning_rate"][0], "float")
        self.assertTrue(t.calls["learning_rate"][3])  # log=True
        self.assertTrue(t.calls["ent_coef"][3])
        self.assertTrue(t.calls["gamma"][3])
        # n_steps / batch_size are categorical over the documented choice lists.
        self.assertEqual(t.calls["n_steps"], ("categorical", to.N_STEPS_CHOICES))
        self.assertEqual(t.calls["batch_size"], ("categorical", to.BATCH_SIZE_CHOICES))
        self.assertEqual(t.calls["n_epochs"][0], "int")


class TestValidBatchSize(unittest.TestCase):
    def test_exact_divisor_kept(self):
        self.assertEqual(to._valid_batch_size(2048, 256), 256)
        self.assertEqual(to._valid_batch_size(512, 64), 64)

    def test_batch_larger_than_n_steps_clamped(self):
        # batch_size 256 > n_steps 256 stays 256; > n_steps drops to n_steps then to a divisor.
        self.assertEqual(to._valid_batch_size(256, 256), 256)
        self.assertEqual(to._valid_batch_size(256, 512), 256)  # clamp to n_steps (256 % 256 == 0)

    def test_non_divisor_rounds_down_to_divisor(self):
        # 1000 isn't a power of two; 256 doesn't divide it -> step down to the largest divisor <= 256.
        b = to._valid_batch_size(1000, 256)
        self.assertLessEqual(b, 256)
        self.assertEqual(1000 % b, 0)

    def test_always_divides_for_documented_choices(self):
        for n in to.N_STEPS_CHOICES:
            for bs in to.BATCH_SIZE_CHOICES:
                b = to._valid_batch_size(n, bs)
                self.assertGreaterEqual(b, 1)
                self.assertEqual(n % b, 0, f"{n} % {b} != 0 (from batch={bs})")
                self.assertLessEqual(b, n)


class TestMakePpoKwargs(unittest.TestCase):
    def test_maps_and_fixes_batch_size(self):
        hp = {
            "learning_rate": 3e-4, "n_steps": 1000, "batch_size": 256, "n_epochs": 10,
            "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0, "clip_range": 0.2,
        }
        kw = to.make_ppo_kwargs(hp)
        self.assertEqual(kw["n_steps"], 1000)
        self.assertEqual(1000 % kw["batch_size"], 0)  # corrected to a divisor
        self.assertIsInstance(kw["n_epochs"], int)
        self.assertAlmostEqual(kw["learning_rate"], 3e-4)


class TestMeanEpisodeReward(unittest.TestCase):
    def test_mean_of_rewards(self):
        buf = deque([{"r": 1.0, "l": 5}, {"r": 3.0, "l": 7}])
        self.assertAlmostEqual(to.mean_episode_reward(buf), 2.0)

    def test_empty_is_negative_infinity(self):
        self.assertEqual(to.mean_episode_reward(deque()), float("-inf"))
        self.assertEqual(to.mean_episode_reward(None), float("-inf"))

    def test_ignores_entries_without_reward(self):
        buf = [{"l": 5}, {"r": 4.0}]
        self.assertAlmostEqual(to.mean_episode_reward(buf), 4.0)


class TestBestResult(unittest.TestCase):
    def test_summary_shape(self):
        class FakeStudy:
            best_value = 12.5
            best_params = {"learning_rate": 3e-4, "n_steps": 512}
            trials = [object(), object(), object()]
        s = to.best_result(FakeStudy())
        self.assertEqual(s["best_value"], 12.5)
        self.assertEqual(s["best_params"]["n_steps"], 512)
        self.assertEqual(s["n_trials"], 3)


if __name__ == "__main__":
    unittest.main()
