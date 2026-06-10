#!/usr/bin/env python3
"""Train the Chase The Target agent with stock Ray/RLlib PPO (new API stack) over the godot-rl wire.

Fourth training backend alongside SB3 (train_chase.py), CleanRL (train_cleanrl.py) and
SampleFactory (train_sf.py). Ecosystem interop, not a replacement: it proves an unmodified
RLlib release trains against our env over godot_rl's wire protocol. Runs in the isolated
.venv-rllib (ray pins gymnasium==1.2.2, which godot-rl's declared deps conflict with — see
requirements-rllib.txt; godot-rl itself is installed --no-deps and is runtime-compatible).

The stock godot_rl `RayVectorGodotEnv` targets RLlib's OLD API stack, so a thin custom
gymnasium adapter (`GodotRLlibEnv`, built by make_godot_env_cls below) wraps `GodotEnv`
directly: Dict({'obs': Box}) -> Box, Tuple(Discrete) -> Discrete, batch-of-one squeezed.
Spaces confirmed against the live handshake (plan Task 2): obs Box(-1, 1, (5,), float32),
action Discrete(5), action nesting [[a]].

Run this FIRST (the env opens the server on --base_port and waits), THEN launch the Godot
training scene which connects as the client. See scripts/train_rllib.sh for orchestration.

Design: docs/superpowers/specs/2026-06-09-rllib-backend-design.md (GitHub #110)

Convention: heavy imports (ray / torch / numpy / gymnasium / godot_rl) are LAZY (inside
main() or the factory) so the pure helpers stay unit-testable without those deps installed.
"""
from __future__ import annotations

import argparse
from typing import NamedTuple, Sequence


class RLlibConfig(NamedTuple):
    """Immutable run configuration (built from argv by parse_args)."""

    timesteps: int
    base_port: int
    speedup: int
    action_repeat: int
    seed: int
    experiment: str
    train_dir: str


def parse_args(argv: Sequence[str] | None = None) -> RLlibConfig:
    """Parse argv into an immutable RLlibConfig. Raises SystemExit on unknown args (argparse)."""
    p = argparse.ArgumentParser(allow_abbrev=False, description="Ray/RLlib PPO (new API stack) for chase.")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--base_port", type=int, default=11008)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiment", type=str, default="chase_rllib")
    p.add_argument("--train_dir", type=str, default="logs/rllib")
    a = p.parse_args(argv)
    return RLlibConfig(
        timesteps=a.timesteps,
        base_port=a.base_port,
        speedup=a.speedup,
        action_repeat=a.action_repeat,
        seed=a.seed,
        experiment=a.experiment,
        train_dir=a.train_dir,
    )


def nest_action(a) -> list[list[int]]:
    """Re-nest a scalar Discrete action into GodotEnv.step's [agent][action_key] structure.

    The chase env is single-agent with one discrete action key, so a -> [[a]]
    (confirmed against the live wire in plan Task 2).
    """
    return [[int(a)]]


def ppo_config_overrides(cfg: RLlibConfig) -> dict:
    """New-API-stack PPO knobs as a plain dict (pure: no ray import).

    num_env_runners=0 keeps rollouts on the driver — exactly one env, one socket, so the
    single headless Godot client orchestration stays as simple as the CleanRL backend's.
    No obs/return normalization anywhere: the exported actor must be a plain MLP for the
    ncnn parity check. Hyperparameters are modest (interop proof, not a leaderboard) and
    borrow their magnitudes from train_cleanrl.py.
    """
    return {
        "framework": "torch",
        "num_env_runners": 0,
        "normalize_obs": False,
        "seed": cfg.seed,
        "train_batch_size": 512,
        "minibatch_size": 128,
        "num_epochs": 4,
        "lr": 2.5e-4,
        "gamma": 0.99,
        "entropy_coeff": 0.01,
    }


def make_godot_env_cls():
    """Build the GodotRLlibEnv class (lazy: imports gymnasium/numpy/godot_rl on first call).

    Defined inside a factory so importing this module stays dependency-light for the unit
    tests, which run in .venv-train (no ray) — see test/python/test_train_rllib.py.
    """
    import gymnasium as gym
    import numpy as np
    from godot_rl.core.godot_env import GodotEnv

    class GodotRLlibEnv(gym.Env):
        """Single-agent gymnasium adapter over godot_rl's GodotEnv for RLlib's new API stack.

        GodotEnv speaks in batches of agents (per-agent obs dicts / reward lists); RLlib's
        SingleAgentEnvRunner wants a plain gymnasium.Env. This adapter squeezes the
        batch-of-one and unwraps the Dict obs / Tuple action spaces from the handshake.
        """

        def __init__(self, config=None):
            env_config = dict(config or {})
            self._env = GodotEnv(
                env_path=None,
                port=int(env_config.get("base_port", 11008)),
                show_window=False,
                seed=int(env_config.get("seed", 0)),
                action_repeat=int(env_config.get("action_repeat", 8)),
                speedup=int(env_config.get("speedup", 8)),
            )
            # Handshake spaces (Task 2): Dict('obs': Box(-1,1,(5,),f32)) / Tuple(Discrete(5)).
            obs_space = self._env.observation_space["obs"]
            act_space = self._env.action_space
            if isinstance(act_space, gym.spaces.Tuple):
                if len(act_space.spaces) != 1:
                    raise ValueError(
                        f"GodotRLlibEnv supports exactly one action key, got {act_space}"
                    )
                act_space = act_space.spaces[0]
            if not isinstance(act_space, gym.spaces.Discrete):
                raise ValueError(f"GodotRLlibEnv supports a single Discrete action, got {act_space}")
            self.observation_space = obs_space
            self.action_space = act_space

        def reset(self, *, seed=None, options=None):
            # GodotEnv seeds via its constructor only; gymnasium's seed kwarg is accepted
            # for API compliance but cannot re-seed a live Godot episode stream.
            super().reset(seed=seed)
            obs, _info = self._env.reset()
            return self._squeeze_obs(obs), {}

        def step(self, action):
            obs, reward, term, trunc, _info = self._env.step(nest_action(action), order_ij=True)
            return (
                self._squeeze_obs(obs),
                float(reward[0]),
                bool(term[0]),
                bool(trunc[0]),
                {},
            )

        def close(self):
            self._env.close()

        @staticmethod
        def _squeeze_obs(obs):
            # Batch-of-one per-agent dicts -> the flat float32 obs vector.
            return np.asarray(obs[0]["obs"], dtype=np.float32)

    return GodotRLlibEnv


def _find_rl_module_dirs(checkpoint_dir: str) -> list[str]:
    """All `rl_module` subdirectories under a saved checkpoint (lazy os import not needed: stdlib).

    New-stack checkpoints nest the RLModule under learner_group/learner/rl_module/<module_id>;
    the exporter (export_rllib_to_torchscript.py) consumes this layout. Walked rather than
    hardcoded so a layout change in a future ray fails loud here, at save time.
    """
    import os

    found = []
    for root, dirs, _files in os.walk(checkpoint_dir):
        for d in dirs:
            if d == "rl_module":
                found.append(os.path.join(root, d))
    return found


def main(argv: Sequence[str] | None = None) -> int:
    import os

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    cfg = parse_args(argv)
    overrides = ppo_config_overrides(cfg)
    env_cls = make_godot_env_cls()

    ray.init(include_dashboard=False, ignore_reinit_error=True, num_cpus=2)
    config = (
        PPOConfig()
        # New stack is the 2.55 default; explicit so a future default flip can't silently
        # move us back to the old stack this backend deliberately targets.
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env_cls,
            env_config={
                "base_port": cfg.base_port,
                "speedup": cfg.speedup,
                "action_repeat": cfg.action_repeat,
                "seed": cfg.seed,
            },
        )
        .env_runners(num_env_runners=overrides["num_env_runners"])
        .framework(overrides["framework"])
        .debugging(seed=overrides["seed"])
        .training(
            lr=overrides["lr"],
            gamma=overrides["gamma"],
            train_batch_size_per_learner=overrides["train_batch_size"],
            minibatch_size=overrides["minibatch_size"],
            num_epochs=overrides["num_epochs"],
            entropy_coeff=overrides["entropy_coeff"],
        )
    )
    algo = config.build_algo()

    sampled = 0
    iteration = 0
    while sampled < cfg.timesteps:
        result = algo.train()
        iteration = int(result.get("training_iteration", iteration + 1))
        env_runner_results = result.get("env_runners", {})
        if "num_env_steps_sampled_lifetime" not in env_runner_results:
            raise RuntimeError(
                "env_runners/num_env_steps_sampled_lifetime missing from RLlib result "
                f"(keys: {sorted(env_runner_results)}); the step-counter key moved — "
                "update train_rllib.py for this ray version."
            )
        sampled = int(env_runner_results["num_env_steps_sampled_lifetime"])
        episode_return = env_runner_results.get("episode_return_mean", float("nan"))
        print(f"iter {iteration} steps={sampled}/{cfg.timesteps} episode_return_mean={episode_return}")

    ckpt_dir = os.path.abspath(
        os.path.join(cfg.train_dir, cfg.experiment, f"checkpoint_{iteration:06d}")
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    algo.save_to_path(ckpt_dir)
    print("checkpoint:", ckpt_dir)
    for rl_module_dir in _find_rl_module_dirs(ckpt_dir):
        print("rl_module dir:", rl_module_dir)

    algo.stop()  # closes the env -> the Godot client exits
    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
