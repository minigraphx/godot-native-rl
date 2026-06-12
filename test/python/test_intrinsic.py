"""Tests for scripts/intrinsic.py (issue #27).

The pure stdlib helpers (RunningMeanStd, combine_rewards, normalize_intrinsic) run everywhere — no
ML stack. The RND network tests are guarded by a torch dep-probe (they run in CI's .venv-train)."""
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import intrinsic as ic  # noqa: E402  (stdlib-only at import; torch is lazy)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class TestRunningMeanStd(unittest.TestCase):
    def test_batch_mean_and_var(self):
        rms = ic.RunningMeanStd()
        rms.update([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(rms.mean, 2.5, places=3)
        self.assertAlmostEqual(rms.var, 1.25, places=2)   # population var of [1,2,3,4]
        self.assertAlmostEqual(rms.std, 1.25 ** 0.5, places=2)

    def test_batched_matches_incremental(self):
        batch = ic.RunningMeanStd()
        batch.update([1.0, 2.0, 3.0, 4.0, 5.0])
        incr = ic.RunningMeanStd()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            incr.update([v])
        self.assertAlmostEqual(batch.mean, incr.mean, places=6)
        self.assertAlmostEqual(batch.var, incr.var, places=6)

    def test_empty_update_is_noop(self):
        rms = ic.RunningMeanStd()
        rms.update([7.0, 7.0])
        m, v, c = rms.mean, rms.var, rms.count
        rms.update([])
        self.assertEqual((rms.mean, rms.var, rms.count), (m, v, c))

    def test_constant_stream_zero_variance(self):
        rms = ic.RunningMeanStd()
        rms.update([3.0] * 10)
        self.assertAlmostEqual(rms.mean, 3.0, places=3)
        self.assertAlmostEqual(rms.var, 0.0, places=4)


class TestCombineRewards(unittest.TestCase):
    def test_elementwise_mix(self):
        out = ic.combine_rewards([1.0, 2.0], [10.0, 20.0], coef=0.5)
        self.assertEqual(out, [6.0, 12.0])

    def test_zero_coef_passes_extrinsic_through(self):
        self.assertEqual(ic.combine_rewards([1.0, -2.0], [99.0, 99.0], 0.0), [1.0, -2.0])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ic.combine_rewards([1.0], [1.0, 2.0], 0.5)


class TestNormalizeIntrinsic(unittest.TestCase):
    def test_divides_by_running_std(self):
        rms = ic.RunningMeanStd()
        # Prime the running std with a spread so the denominator is well-defined.
        rms.update([0.0, 2.0, 4.0, 6.0])
        std = rms.std
        out = ic.normalize_intrinsic([std], rms)  # updates rms again, then divides
        # The returned value is the input divided by the (post-update) running std — finite, > 0.
        self.assertGreater(out[0], 0.0)
        self.assertTrue(all(x == x for x in out))  # not NaN

    def test_near_zero_std_falls_back_to_unit_denominator(self):
        rms = ic.RunningMeanStd()
        # A constant batch keeps std ~0 -> denominator falls back to 1.0 (pass-through).
        out = ic.normalize_intrinsic([5.0, 5.0, 5.0], rms)
        for x in out:
            self.assertAlmostEqual(x, 5.0, places=3)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestRNDModel(unittest.TestCase):
    def test_intrinsic_reward_shape(self):
        model, _ = ic.make_rnd(obs_dim=5)
        obs = torch.randn(8, 5)
        r = model.intrinsic_reward(obs)
        self.assertEqual(tuple(r.shape), (8,))
        self.assertTrue(bool((r >= 0).all()))  # squared error is non-negative

    def test_target_is_frozen(self):
        model, _ = ic.make_rnd(obs_dim=4)
        self.assertTrue(all(not p.requires_grad for p in model.target.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.predictor.parameters()))

    def test_update_reduces_novelty_on_repeated_obs(self):
        torch.manual_seed(0)
        model, opt = ic.make_rnd(obs_dim=6, lr=1e-3)
        obs = torch.randn(16, 6)
        before = float(model.intrinsic_reward(obs).mean())
        for _ in range(50):
            model.update(obs, opt)
        after = float(model.intrinsic_reward(obs).mean())
        # The predictor learns the target on these states -> novelty drops markedly.
        self.assertLess(after, before)

    def test_update_returns_decreasing_loss(self):
        torch.manual_seed(0)
        model, opt = ic.make_rnd(obs_dim=4, lr=1e-3)
        obs = torch.randn(32, 4)
        first = model.update(obs, opt)
        for _ in range(30):
            last = model.update(obs, opt)
        self.assertLess(last, first)


if __name__ == "__main__":
    unittest.main()
