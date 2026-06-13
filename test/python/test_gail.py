"""Tests for scripts/gail.py (#61 GAIL imitation).

Pure helpers (reward transform, expert sampler) run everywhere; the torch discriminator tests are
guarded by a dep-probe (they run in CI's .venv-train)."""
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gail  # noqa: E402  (stdlib-only at import; torch is lazy)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class TestRewardTransform(unittest.TestCase):
    def test_nonnegative(self):
        for r in gail.gail_reward_from_logits([-10.0, -1.0, 0.0, 1.0, 10.0]):
            self.assertGreaterEqual(r, 0.0)

    def test_monotone_increasing(self):
        rs = gail.gail_reward_from_logits([-5.0, -1.0, 0.0, 1.0, 5.0])
        self.assertEqual(rs, sorted(rs))

    def test_zero_logit_is_log2(self):
        import math
        self.assertAlmostEqual(gail.gail_reward_from_logits([0.0])[0], math.log(2.0), places=6)

    def test_stable_at_extremes(self):
        # softplus must not overflow/underflow.
        rs = gail.gail_reward_from_logits([-1000.0, 1000.0])
        self.assertAlmostEqual(rs[0], 0.0, places=6)
        self.assertAlmostEqual(rs[1], 1000.0, places=3)


class TestSampler(unittest.TestCase):
    def test_in_range_and_sized(self):
        idx = gail.sample_indices(10, 32, seed=1)
        self.assertEqual(len(idx), 32)
        self.assertTrue(all(0 <= i < 10 for i in idx))

    def test_deterministic(self):
        self.assertEqual(gail.sample_indices(7, 16, seed=3), gail.sample_indices(7, 16, seed=3))

    def test_empty_pool_raises(self):
        with self.assertRaises(ValueError):
            gail.sample_indices(0, 8)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestDiscriminator(unittest.TestCase):
    def test_reward_shape_and_nonneg(self):
        d, _ = gail.make_discriminator(obs_dim=5, n_actions=4)
        obs = torch.randn(8, 5)
        act = torch.randint(0, 4, (8,))
        r = d.reward(obs, act)
        self.assertEqual(tuple(r.shape), (8,))
        self.assertTrue(bool((r >= 0).all()))

    def test_update_separates_expert_from_policy(self):
        torch.manual_seed(0)
        d, opt = gail.make_discriminator(obs_dim=4, n_actions=3, lr=5e-3)
        # Expert pairs cluster at one obs region + action 0; policy at another + action 2.
        e_obs = torch.randn(64, 4) + 3.0
        e_act = torch.zeros(64, dtype=torch.long)
        p_obs = torch.randn(64, 4) - 3.0
        p_act = torch.full((64,), 2, dtype=torch.long)
        for _ in range(150):
            d.update(p_obs, p_act, e_obs, e_act, opt)
        # After training, expert pairs should score higher (more expert-like) than policy pairs.
        self.assertGreater(float(d.reward(e_obs, e_act).mean()),
                           float(d.reward(p_obs, p_act).mean()))


if __name__ == "__main__":
    unittest.main()
