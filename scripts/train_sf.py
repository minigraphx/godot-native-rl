#!/usr/bin/env python3
"""Train the Chase The Target agent with SampleFactory (async PPO) over the godot-rl bridge.

Third training backend, alongside scripts/train_chase.py (SB3) and scripts/train_cleanrl.py
(CleanRL). It drives godot_rl's supported SampleFactory entry point
(`godot_rl.wrappers.sample_factory_wrapper.sample_factory_training`) with macOS-safe and
parity-safe overrides, then scripts/export_sf_to_onnx.py turns the SF checkpoint into ONNX that
flows unchanged into scripts/export_to_ncnn.py -> native ncnn deploy.

Runs in the isolated .venv-sf (SF pins gymnasium<1.0). Heavy imports (sample_factory / godot_rl /
torch) are LAZY inside main() so the pure helpers below stay unit-testable without those deps.

Design: docs/superpowers/specs/2026-06-05-sample-factory-backend-design.md
"""
from __future__ import annotations

import argparse
from typing import NamedTuple, Sequence


class SFConfig(NamedTuple):
    """Immutable SampleFactory run config (built from argv by parse_args)."""

    timesteps: int
    base_port: int
    env_agents: int
    speedup: int
    seed: int
    experiment: str
    train_dir: str


def parse_args(argv: Sequence[str] | None = None) -> SFConfig:
    """Parse argv into an immutable SFConfig. Raises SystemExit on unknown args (argparse)."""
    p = argparse.ArgumentParser(allow_abbrev=False, description="SampleFactory PPO for chase.")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--base_port", type=int, default=11008)
    p.add_argument("--env_agents", type=int, default=1)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiment", type=str, default="chase_sf")
    p.add_argument("--train_dir", type=str, default="logs/sf")
    a = p.parse_args(argv)
    return SFConfig(
        timesteps=a.timesteps,
        base_port=a.base_port,
        env_agents=a.env_agents,
        speedup=a.speedup,
        seed=a.seed,
        experiment=a.experiment,
        train_dir=a.train_dir,
    )


def client_port(base_port: int) -> int:
    """Port the Godot client must connect on.

    godot_rl's make_godot_env_func computes `port = base_port; if env_config: port += 1 + env_id`.
    Our single serial worker is env_id=0, so the client listens on base_port + 1.
    """
    return base_port + 1


def build_sf_argv(cfg: SFConfig) -> list[str]:
    """Translate an SFConfig into the SampleFactory CLI argv list (the `extras` for parse_gdrl_args).

    Bakes in the overrides that make the run macOS-safe (serial/sync, single worker) and
    ncnn-parity-safe (input/return normalization OFF, so the exported actor is a plain MLP).
    """
    return [
        f"--experiment={cfg.experiment}",
        f"--train_for_env_steps={cfg.timesteps}",
        f"--base_port={cfg.base_port}",
        f"--env_agents={cfg.env_agents}",
        f"--seed={cfg.seed}",
        "--serial_mode=True",
        "--async_rl=False",
        "--num_workers=1",
        "--num_envs_per_worker=1",
        "--worker_num_splits=1",
        "--normalize_input=False",
        "--normalize_returns=False",
        "--use_rnn=False",
        "--device=cpu",
    ]


def _build_args_namespace(cfg: SFConfig) -> argparse.Namespace:
    """Build the `args` namespace godot_rl's sample_factory_training expects.

    register_gdrl_env reads args.env_path / args.speedup / args.seed / args.viz;
    parse_gdrl_args reads args.experiment_dir / args.experiment_name / args.eval.
    env_path=None => in-editor training: SF opens the server and waits for the Godot client.
    """
    return argparse.Namespace(
        env_path=None,
        experiment_dir=cfg.train_dir,
        experiment_name=cfg.experiment,
        speedup=cfg.speedup,
        seed=cfg.seed,
        viz=False,
        eval=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    # Heavy import is lazy: only when actually training (keeps the pure helpers import-light).
    from godot_rl.wrappers.sample_factory_wrapper import sample_factory_training

    cfg = parse_args(argv)
    args = _build_args_namespace(cfg)
    extras = build_sf_argv(cfg)
    print(f"SampleFactory training: experiment={cfg.experiment} timesteps={cfg.timesteps} "
          f"base_port={cfg.base_port} client_port={client_port(cfg.base_port)} env_agents={cfg.env_agents}")
    status = sample_factory_training(args, extras)
    print(f"SampleFactory finished with status={status}")
    # SF status is an enum; treat the normal-termination value as success (0).
    return int(getattr(status, "value", status) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
