"""Tests for scripts/export_vecnormalize.py (stats extraction from SB3 VecNormalize)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import export_vecnormalize as ev  # noqa: E402

# Guarded heavy imports (#141): missing deps -> skips, not errors, under bare python.
try:
    import numpy  # noqa: F401
    import gymnasium  # noqa: F401
    import stable_baselines3  # noqa: F401
    HAVE_SB3 = True
except ImportError:
    HAVE_SB3 = False

needs_sb3 = unittest.skipUnless(HAVE_SB3, "numpy/gymnasium/SB3 not installed")


def _make_vecnormalize(seed: int = 0, obs_dim: int = 4):
    import gymnasium as gym
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class _Dummy(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.zeros(obs_dim, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(obs_dim, dtype=np.float32), 0.0, False, False, {}

    venv = DummyVecEnv([lambda: _Dummy()])
    vn = VecNormalize(venv, norm_obs=True, norm_reward=False)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        vn.obs_rms.update(rng.normal(size=(8, obs_dim)).astype(np.float32))
    return vn


class TestStatsFromVecNormalize(unittest.TestCase):
    @needs_sb3
    def test_extracts_mean_var_epsilon_clip(self):
        vn = _make_vecnormalize()
        stats = ev.stats_from_vecnormalize(vn)
        self.assertEqual(stats["obs_size"], 4)
        self.assertEqual(len(stats["mean"]), 4)
        self.assertEqual(len(stats["var"]), 4)
        for got, exp in zip(stats["mean"], vn.obs_rms.mean):
            self.assertAlmostEqual(got, float(exp), places=6)
        for got, exp in zip(stats["var"], vn.obs_rms.var):
            self.assertAlmostEqual(got, float(exp), places=6)
        self.assertAlmostEqual(stats["epsilon"], float(vn.epsilon))
        self.assertAlmostEqual(stats["clip_obs"], float(vn.clip_obs))

    @needs_sb3
    def test_rejects_norm_obs_disabled(self):
        vn = _make_vecnormalize()
        vn.norm_obs = False
        with self.assertRaises(ValueError):
            ev.stats_from_vecnormalize(vn)

    def test_rejects_non_vecnormalize(self):
        with self.assertRaises(ValueError):
            ev.stats_from_vecnormalize(object())

    @needs_sb3
    def test_rejects_dict_obs_rms(self):
        vn = _make_vecnormalize()
        vn.obs_rms = {"a": vn.obs_rms}
        with self.assertRaises(ValueError):
            ev.stats_from_vecnormalize(vn)

    @needs_sb3
    def test_write_and_reread(self):
        vn = _make_vecnormalize()
        stats = ev.stats_from_vecnormalize(vn)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "stats.json"
            ev.write_stats_json(stats, path)
            reread = json.loads(path.read_text())
        self.assertEqual(reread, stats)


if __name__ == "__main__":
    unittest.main()
