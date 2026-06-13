"""Tests for scripts/coop_mapoca.py (issue #30 M2/M3).

Pure helpers (team layout, GAE, counterfactual advantage, posthumous masking) — numpy-only, run
everywhere. The torch trainer pieces (attention critic) are exercised by the guarded e2e smoke in
run_tests.sh, not here."""
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import coop_mapoca as mp  # noqa: E402


class TestTeamLayout(unittest.TestCase):
    def test_team_count(self):
        self.assertEqual(mp.validate_team_layout(2, 2), 1)
        self.assertEqual(mp.validate_team_layout(16, 2), 8)

    def test_indivisible_raises(self):
        with self.assertRaises(ValueError):
            mp.validate_team_layout(5, 2)

    def test_nonpositive_raises(self):
        with self.assertRaises(ValueError):
            mp.validate_team_layout(0, 2)
        with self.assertRaises(ValueError):
            mp.validate_team_layout(4, 0)

    def test_slices(self):
        self.assertEqual(mp.team_slices(4, 2), [slice(0, 2), slice(2, 4)])
        self.assertEqual(mp.team_slices(6, 3), [slice(0, 3), slice(3, 6)])


class TestGAE(unittest.TestCase):
    def test_zero_reward_zero_value_zero_adv(self):
        adv, ret = mp.compute_gae(
            np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)),
            np.zeros(1), np.zeros(1), gamma=0.99, gae_lambda=0.95)
        self.assertTrue(np.allclose(adv, 0.0))
        self.assertTrue(np.allclose(ret, 0.0))

    def test_single_step_advantage(self):
        # one step, reward 1, value 0, no bootstrap -> advantage == reward.
        adv, ret = mp.compute_gae(
            np.array([[1.0]]), np.array([[0.0]]), np.array([[0.0]]),
            np.array([0.0]), np.array([1.0]), gamma=0.99, gae_lambda=0.95)
        self.assertAlmostEqual(float(adv[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(ret[0, 0]), 1.0, places=5)

    def test_matches_reference_two_step(self):
        rewards = np.array([[1.0], [2.0]])
        values = np.array([[0.5], [0.5]])
        dones = np.array([[0.0], [0.0]])
        adv, ret = mp.compute_gae(rewards, values, dones, np.array([0.0]), np.array([1.0]),
                                  gamma=0.9, gae_lambda=1.0)
        # gae_lambda=1 -> advantages are plain (discounted return - value).
        # t=1: delta = 2 + 0 - 0.5 = 1.5 ; t=0: delta = 1 + 0.9*0.5 - 0.5 = 0.95, adv = 0.95+0.9*1.5
        self.assertAlmostEqual(float(adv[1, 0]), 1.5, places=5)
        self.assertAlmostEqual(float(adv[0, 0]), 0.95 + 0.9 * 1.5, places=5)


class TestCounterfactualAdvantage(unittest.TestCase):
    def test_broadcast_and_subtract(self):
        team_returns = np.array([[10.0], [20.0]])          # (steps=2, teams=1)
        baselines = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])  # (2, 1, team_size=2)
        adv = mp.counterfactual_advantage(team_returns, baselines)
        self.assertEqual(adv.shape, (2, 1, 2))
        self.assertTrue(np.allclose(adv[0, 0], [9.0, 8.0]))
        self.assertTrue(np.allclose(adv[1, 0], [17.0, 16.0]))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mp.counterfactual_advantage(np.zeros((2, 1)), np.zeros((2, 2)))  # baselines not 3-D
        with self.assertRaises(ValueError):
            mp.counterfactual_advantage(np.zeros((2, 1)), np.zeros((3, 1, 2)))  # steps disagree


class TestNormalize(unittest.TestCase):
    def test_zero_mean_unit_std(self):
        out = mp.normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(float(out.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(out.std()), 1.0, places=3)

    def test_constant_input_no_nan(self):
        out = mp.normalize(np.array([5.0, 5.0, 5.0]))
        self.assertFalse(np.any(np.isnan(out)))


class TestPosthumousMasking(unittest.TestCase):
    def test_alive_until_finish(self):
        # agent 0 finishes at step 1, agent 1 never finishes. (steps=3, team_size=2)
        dones = np.array([[0, 0], [1, 0], [0, 0]])
        m = mp.alive_mask(dones)
        self.assertTrue(np.allclose(m[:, 0], [1.0, 0.0, 0.0]))  # 0 leaves at step 1, gone after
        self.assertTrue(np.allclose(m[:, 1], [1.0, 1.0, 1.0]))  # 1 present throughout

    def test_masked_mean_ignores_absent(self):
        vals = np.array([1.0, 2.0, 100.0])
        mask = np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(mp.masked_mean(vals, mask), 1.5, places=5)

    def test_masked_mean_all_absent_is_zero(self):
        self.assertEqual(mp.masked_mean(np.array([1.0, 2.0]), np.array([0.0, 0.0])), 0.0)


if __name__ == "__main__":
    unittest.main()
