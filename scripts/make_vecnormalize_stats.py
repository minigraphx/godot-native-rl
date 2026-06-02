#!/usr/bin/env python3
"""Generate seeded VecNormalize stats + a golden fixture for the obs-normalization replay test.

Builds a real SB3 VecNormalize over a dummy Box-obs env, updates it with seeded random
observations, then writes:
  models/synthetic_vecnormalize.json         - exported stats (via export_vecnormalize)
  models/synthetic_vecnormalize_golden.json  - {stats_path, cases:[{raw, normalized}]} where
                                               `normalized` is SB3's own vn.normalize_obs(raw).

So test/unit/test_obs_normalize.gd asserts the GDScript ObsNormalize.normalize reproduces SB3's
actual output (atol 1e-6), not a re-implementation of it. The .pkl is not committed (this script
recreates it deterministically); only the derived JSON fixtures are committed.

Run under .venv-train.  Regenerate:  .venv-train/bin/python scripts/make_vecnormalize_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "scripts"))

import export_vecnormalize as ev  # noqa: E402

OBS_DIM = 6
SEED = 11


def build_vecnormalize():
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class _Dummy(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.zeros(OBS_DIM, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}

    venv = DummyVecEnv([lambda: _Dummy()])
    vn = VecNormalize(venv, norm_obs=True, norm_reward=False)
    rng = np.random.default_rng(SEED)
    for _ in range(100):
        vn.obs_rms.update(rng.normal(loc=0.5, scale=2.0, size=(16, OBS_DIM)).astype(np.float32))
    return vn


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    vn = build_vecnormalize()

    stats = ev.stats_from_vecnormalize(vn)
    stats_path = MODELS / "synthetic_vecnormalize.json"
    ev.write_stats_json(stats, stats_path)

    rng = np.random.default_rng(SEED + 1)
    cases = []
    for _ in range(5):
        raw = rng.normal(loc=0.5, scale=2.0, size=(OBS_DIM,)).astype(np.float32)
        normalized = np.asarray(vn.normalize_obs(raw), dtype=np.float32).ravel()
        cases.append({
            "raw": [float(x) for x in raw],
            "normalized": [float(x) for x in normalized],
        })
    golden = {"stats_path": "res://models/synthetic_vecnormalize.json", "cases": cases}
    golden_path = MODELS / "synthetic_vecnormalize_golden.json"
    golden_path.write_text(json.dumps(golden, indent=2) + "\n")

    print(f"OK: {stats_path}")
    print(f"OK: {golden_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
