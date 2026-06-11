import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Dep probe (#141/#148): only the third-party dep lives in the try; the script under
# test is stdlib-only at module load, so its own import failures stay loud.
try:
    import numpy as np  # noqa: E402
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False
import train_cleanrl as tc  # noqa: E402


@unittest.skipUnless(HAVE_DEPS, "numpy not installed")
class TestComputeGae(unittest.TestCase):
    """GAE advantages + returns against hand-computed values (gamma=0.99, lam=0.95)."""

    def test_two_steps_no_dones(self):
        rewards = np.array([[1.0], [2.0]], dtype=np.float32)
        values = np.array([[0.5], [1.5]], dtype=np.float32)
        dones = np.array([[0.0], [0.0]], dtype=np.float32)
        adv, ret = tc.compute_gae(
            rewards, values, dones,
            next_value=np.array([3.0], dtype=np.float32),
            next_done=np.array([0.0], dtype=np.float32),
            gamma=0.99, gae_lambda=0.95,
        )
        # t=1: delta = 2 + 0.99*3 - 1.5 = 3.47 ; adv[1] = 3.47
        # t=0: delta = 1 + 0.99*1.5 - 0.5 = 1.985 ; adv[0] = 1.985 + 0.99*0.95*3.47
        self.assertAlmostEqual(float(adv[1, 0]), 3.47, places=4)
        self.assertAlmostEqual(float(adv[0, 0]), 5.248535, places=4)
        # returns = adv + values
        self.assertAlmostEqual(float(ret[1, 0]), 4.97, places=4)
        self.assertAlmostEqual(float(ret[0, 0]), 5.748535, places=4)

    def test_terminal_cuts_bootstrap(self):
        # dones[1]=1 => at t=0 the next-step bootstrap is zeroed (terminal cut).
        rewards = np.array([[1.0], [2.0]], dtype=np.float32)
        values = np.array([[0.5], [1.5]], dtype=np.float32)
        dones = np.array([[0.0], [1.0]], dtype=np.float32)
        adv, ret = tc.compute_gae(
            rewards, values, dones,
            next_value=np.array([3.0], dtype=np.float32),
            next_done=np.array([0.0], dtype=np.float32),
            gamma=0.99, gae_lambda=0.95,
        )
        # t=0: nextnonterminal = 1 - dones[1] = 0 -> delta = 1 - 0.5 = 0.5, adv[0] = 0.5
        self.assertAlmostEqual(float(adv[0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(ret[0, 0]), 1.0, places=5)

    def test_preserves_shape(self):
        rewards = np.zeros((3, 4), dtype=np.float32)
        values = np.zeros((3, 4), dtype=np.float32)
        dones = np.zeros((3, 4), dtype=np.float32)
        adv, ret = tc.compute_gae(
            rewards, values, dones,
            next_value=np.zeros(4, dtype=np.float32),
            next_done=np.zeros(4, dtype=np.float32),
            gamma=0.99, gae_lambda=0.95,
        )
        self.assertEqual(adv.shape, (3, 4))
        self.assertEqual(ret.shape, (3, 4))


@unittest.skipUnless(HAVE_DEPS, "numpy not installed")
class TestDiscreteActionDims(unittest.TestCase):
    def test_single_head(self):
        total, nvec = tc.discrete_action_dims([5])
        self.assertEqual(total, 5)
        self.assertEqual(nvec, [5])

    def test_multi_head(self):
        total, nvec = tc.discrete_action_dims([2, 3])
        self.assertEqual(total, 5)
        self.assertEqual(nvec, [2, 3])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            tc.discrete_action_dims([])

    def test_non_positive_raises(self):
        with self.assertRaises(ValueError):
            tc.discrete_action_dims([3, 0])


@unittest.skipUnless(HAVE_DEPS, "numpy not installed")
class TestNumUpdates(unittest.TestCase):
    def test_chase_default(self):
        self.assertEqual(tc.num_updates(300_000, 256, 1), 1171)

    def test_multi_env_floor(self):
        self.assertEqual(tc.num_updates(2048, 256, 4), 2)

    def test_clamps_to_zero(self):
        self.assertEqual(tc.num_updates(10, 256, 1), 0)


@unittest.skipUnless(HAVE_DEPS, "numpy not installed")
class TestObsAndActLayout(unittest.TestCase):
    class _FakeBox:
        def __init__(self, shape):
            self.shape = shape

    class _FakeMultiDiscrete:
        def __init__(self, nvec):
            self.nvec = np.array(nvec)

    def test_obs_dim(self):
        self.assertEqual(tc.obs_dim(self._FakeBox((5,))), 5)

    def test_act_layout_from_multidiscrete(self):
        total, nvec = tc.act_layout(self._FakeMultiDiscrete([5]))
        self.assertEqual(total, 5)
        self.assertEqual(nvec, [5])


@unittest.skipUnless(HAVE_DEPS, "numpy not installed")
class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        cfg = tc.parse_args([])
        self.assertEqual(cfg.timesteps, 300_000)
        self.assertEqual(cfg.num_steps, 256)
        self.assertEqual(cfg.seed, 0)
        self.assertEqual(cfg.onnx_export_path, "models/chase_cleanrl_policy.onnx")
        self.assertAlmostEqual(cfg.gamma, 0.99)
        self.assertAlmostEqual(cfg.gae_lambda, 0.95)

    def test_override(self):
        cfg = tc.parse_args(["--gamma", "0.9", "--timesteps", "1000"])
        self.assertAlmostEqual(cfg.gamma, 0.9)
        self.assertEqual(cfg.timesteps, 1000)

    def test_config_is_immutable(self):
        cfg = tc.parse_args([])
        with self.assertRaises(AttributeError):
            cfg.gamma = 0.1  # NamedTuple is immutable

    def test_unknown_arg_exits(self):
        with self.assertRaises(SystemExit):
            tc.parse_args(["--not-a-real-arg", "1"])


@unittest.skipUnless(HAVE_DEPS, "numpy not installed")
class TestLayerInit(unittest.TestCase):
    def test_orthogonal_and_bias(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        layer = torch.nn.Linear(4, 4)
        out = tc.layer_init(layer, std=1.0, bias_const=0.5)
        self.assertIs(out, layer)
        self.assertTrue(torch.allclose(layer.bias, torch.full((4,), 0.5)))
        # orthogonal init => W @ W.T ~= I for a square layer
        w = layer.weight.detach()
        self.assertTrue(torch.allclose(w @ w.t(), torch.eye(4), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
