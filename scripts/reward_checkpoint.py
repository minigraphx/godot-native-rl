#!/usr/bin/env python3
"""Reward-gated "best model" checkpointing for the SB3 trainers (#138).

The periodic `CheckpointCallback` keeps the *latest* policy; this keeps the *best one
seen*: whenever the rolling mean of recent training-episode rewards beats the prior
best, the model is saved to `<checkpoint_dir>/<name_prefix>_best.zip` -- the artifact
the deploy-side picker (`checkpoints.select_checkpoint(policy="deploy")`) prefers.

Design (decided on #138):
  - Signal: `model.ep_info_buffer` (the rolling episode-reward window VecMonitor
    already fills). No eval rollouts -- in-editor training is a single-socket bridge,
    so SB3's EvalCallback (which wants a parallel eval env) is a poor fit.
  - Cadence: `_on_rollout_end` (once per rollout), NOT `_on_step` -- early in training
    the mean rises fast and a per-step gate would hammer the disk.
  - Resume correctness: the best mean reward is persisted to a JSON sidecar next to
    the best zip and reloaded on construction, so resuming a run can't overwrite a
    better `*_best.zip` with a worse post-restart one.
  - Opt-in and additive: wired behind a trainer flag, alongside (not replacing) the
    periodic checkpoints. Stable PPO usually ends near its peak; the gate earns its
    keep on noisy SAC / self-play runs.

Pure helpers live at module level (stdlib-only, unit-testable); the SB3 import is
deferred into `make_reward_gated_checkpoint()` per the repo convention.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from checkpoints import best_zip_path  # noqa: E402


def rolling_mean_reward(ep_infos, min_episodes: int) -> float | None:
    """Mean of the `"r"` entries in an ep_info_buffer-style iterable, or None.

    None when fewer than `min_episodes` episodes have completed -- too little signal
    to bless a "best" yet (a lucky first episode shouldn't pin the gate).
    """
    rewards = [float(info["r"]) for info in ep_infos]
    if len(rewards) < max(min_episodes, 1):
        return None
    return sum(rewards) / len(rewards)


def is_new_best(mean_reward: float, best: float) -> bool:
    """True when mean_reward strictly beats the prior best (NaN never wins)."""
    if math.isnan(mean_reward):
        return False
    return mean_reward > best


def sidecar_path(best_zip: pathlib.Path) -> pathlib.Path:
    """The JSON sidecar persisting the best mean reward across resumes."""
    return best_zip.with_name(best_zip.name + ".json")


def load_best_reward(sidecar: pathlib.Path) -> float:
    """Best mean reward recorded in `sidecar`, or -inf when missing/malformed.

    -inf (not an error) so a fresh run and a corrupt sidecar both just mean "no best
    yet"; the next improvement rewrites both files.
    """
    try:
        data = json.loads(sidecar.read_text())
        return float(data["best_mean_reward"])
    except (OSError, ValueError, TypeError, KeyError):
        return float("-inf")


def save_best_meta(sidecar: pathlib.Path, mean_reward: float, num_timesteps: int) -> None:
    """Record the newly blessed best (reward + step count + wall-clock) to `sidecar`."""
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(
            {
                "best_mean_reward": mean_reward,
                "num_timesteps": int(num_timesteps),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
    )


def make_reward_gated_checkpoint(
    checkpoint_dir: str,
    name_prefix: str,
    min_episodes: int = 10,
    verbose: int = 0,
):
    """Build the RewardGatedCheckpoint callback (lazy SB3 import).

    Saves `<checkpoint_dir>/<name_prefix>_best.zip` (+ `.json` sidecar) whenever the
    rolling mean episode reward improves; combine with the periodic CheckpointCallback
    by passing both in the `callback=[...]` list to `model.learn`.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class RewardGatedCheckpoint(BaseCallback):
        def __init__(self):
            super().__init__(verbose)
            self.zip_path = best_zip_path(checkpoint_dir, name_prefix)
            self.sidecar = sidecar_path(self.zip_path)
            # Reload the prior best so a resumed run can't bless a worse policy.
            self.best = load_best_reward(self.sidecar)

        def _on_step(self) -> bool:  # required abstract; the gate runs per rollout
            return True

        def _on_rollout_end(self) -> None:
            mean = rolling_mean_reward(self.model.ep_info_buffer, min_episodes)
            if mean is None or not is_new_best(mean, self.best):
                return
            self.best = mean
            self.zip_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.zip_path)
            save_best_meta(self.sidecar, mean, self.model.num_timesteps)
            if self.verbose > 0:
                print(
                    "New best mean reward %.3f at %d steps -> %s"
                    % (mean, self.model.num_timesteps, self.zip_path)
                )

    return RewardGatedCheckpoint()
