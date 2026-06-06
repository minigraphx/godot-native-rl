#!/usr/bin/env python3
"""Train the Chase The Target agent with SampleFactory (async PPO) over the godot-rl bridge.

Third training backend, alongside scripts/train_chase.py (SB3) and scripts/train_cleanrl.py
(CleanRL). It bypasses godot_rl's `sample_factory_training()` and instead registers a
module-level, picklable env factory via SF's `register_env`, then calls `run_rl` directly
(this works around godot_rl-0.8.2 / SF-2.1.1 incompatibilities: the single-key discrete-action
scalar crash and the is_multiagent double-wrap — see below) with macOS-safe and parity-safe
overrides. The trained checkpoint is exported to TorchScript via scripts/export_sf_to_torchscript.py
(the .venv-sf venv can't onnx-export), which flows unchanged into scripts/export_to_ncnn.py ->
native ncnn deploy.

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
    batch_size: int


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
    # Small default so the learner trains (and checkpoints) within a few thousand frames; see
    # build_sf_argv. Chase's policy is a tiny MLP, so a modest batch is fine for real runs too.
    p.add_argument("--batch_size", type=int, default=512)
    a = p.parse_args(argv)
    return SFConfig(
        timesteps=a.timesteps,
        base_port=a.base_port,
        env_agents=a.env_agents,
        speedup=a.speedup,
        seed=a.seed,
        experiment=a.experiment,
        train_dir=a.train_dir,
        batch_size=a.batch_size,
    )


def client_port(base_port: int) -> int:
    """Port the real SampleFactory SAMPLER env listens on (env_config.env_id=0 -> base_port + 1).

    godot_rl's make_godot_env_func computes `port = base_port; if env_config: port += 1 + env_id`.
    NOTE: SF actually opens TWO sockets per run: the env-info probe (env_config=None -> base_port)
    and then the real sampler (env_config.env_id=0 -> base_port + 1). The orchestrator therefore
    does NOT hard-code this offset; it watches the trainer log and launches a Godot client on
    whatever port each "waiting for remote GODOT connection on port N" announces. This helper
    documents the sampler port (the one training data actually flows over).
    """
    return base_port + 1


def build_sf_argv(cfg: SFConfig) -> list[str]:
    """Translate an SFConfig into the SampleFactory CLI argv list (the `extras` for parse_gdrl_args).

    Bakes in the overrides that make the run:
      - macOS-safe: serial/sync, single worker (no spawn-based async multiprocessing);
      - ncnn-parity-safe: input/return normalization OFF and no RNN, so the exported actor is a
        plain MLP that converts cleanly to ncnn;
      - checkpoint-reliable for tiny budgets: SampleFactory only advances train_for_env_steps and
        writes a checkpoint once the LEARNER processes a batch. The godot_rl defaults inherit
        Atari-tuned sizes (batch_size=2048, num_batches_per_epoch=4 -> ~8192 frames before the first
        learner step), so a 3k-step smoke would collect samples forever without ever checkpointing.
        We shrink batch_size and force one batch per epoch so the learner trains (and saves) well
        within a few thousand frames, and save_every_sec/keep_checkpoints guarantee a .pth lands.
        These are training-throughput knobs only; they do NOT change the exported MLP architecture.
    """
    return [
        "--env=gdrl",
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
        f"--batch_size={cfg.batch_size}",
        "--num_batches_per_epoch=1",
        "--save_every_sec=5",
        "--keep_checkpoints=1",
        "--save_best_after=0",
    ]


def _is_scalar(x) -> bool:
    """True for a Python int/float or a 0-d numpy value (no usable __len__)."""
    if getattr(x, "ndim", None) == 0:
        return True
    return not (hasattr(x, "__len__") and not isinstance(x, (str, bytes)))


def _as_int(x):
    """Coerce a Python/numpy scalar to a plain int."""
    return x.item() if hasattr(x, "item") else int(x)


def nest_scalar_actions(actions):
    """Reshape SampleFactory's action output into godot_rl's expected [agent][action_key] layout.

    godot_rl's GodotEnv.from_numpy indexes the per-step action as action[agent_idx][key_idx] (one
    entry per AIController action KEY). SampleFactory collapses a single-key all-discrete action
    space (chase: one Discrete(5)) to a bare scalar per agent (the `all_discrete` fast path in
    sample_factory's curr_actions / preprocess_actions), so from_numpy's `action[agent_idx][0]`
    raises `'int' object is not subscriptable`. Worse, SF's NonBatchedMultiAgentWrapper unwraps the
    single-agent list (`action = action[0]`), so our env.step can even receive a *bare* scalar.

    This adapter restores the [agent][key] structure for both shapes:
      - a bare scalar `a`                  -> `[[a]]`        (single agent, single key)
      - a per-agent list of scalars `[a,b]`-> `[[a],[b]]`   (multi-agent, single key)
      - a per-agent list of sequences      -> passed through (multi-key spaces; no collapse to fix)
    """
    # Top-level bare scalar: single agent whose single-key action SF already unwrapped.
    if _is_scalar(actions):
        return [[_as_int(actions)]]

    nested = []
    for a in actions:
        if _is_scalar(a):
            nested.append([_as_int(a)])  # per-agent scalar -> 1-element [key] list
        else:
            nested.append(a)  # already a per-key sequence (multi-key action space)
    return nested


def _make_nested_godot_env(env_path, full_env_name, cfg=None, env_config=None, render_mode=None,
                           seed=0, speedup=1, viz=False):
    """SF env factory that returns a godot_rl env whose step() nests scalar discrete actions.

    Module-level (NOT a closure) so it is picklable: SampleFactory's env-info probe spawns a
    process (multiprocessing 'spawn') and pickles the registered factory. Mirrors godot_rl's
    make_godot_env_func port/seed math (port = base_port [+ 1 + env_id when env_config is set]),
    then builds a thin GodotEnv subclass that applies nest_scalar_actions before delegating to
    step() — fixing the single-key-discrete scalar-vs-[agent][key] crash (see nest_scalar_actions).
    """
    from godot_rl.wrappers import sample_factory_wrapper as sfw

    class _NestedActionGodotEnv(sfw.SampleFactoryEnvWrapperNonBatched):
        # godot_rl's wrapper already returns per-agent LISTS for obs/reward/done/info but does not
        # declare is_multiagent, so SF (seeing num_agents==1) wraps it again in
        # NonBatchedMultiAgentWrapper and double-lists everything -> `infos[0]` becomes a list and
        # `info.get(...)` crashes. Declaring is_multiagent=True makes SF skip that re-wrap and treat
        # godot_rl's lists as the authoritative per-agent layout.
        is_multiagent = True

        def step(self, action):
            return super().step(nest_scalar_actions(action))

    port = cfg.base_port
    _seed = seed
    if env_config:
        port += 1 + env_config.env_id
        _seed += 1 + env_config.env_id
    return _NestedActionGodotEnv(
        env_path=env_path, port=port, seed=_seed, show_window=False, speedup=speedup
    )


def _build_args_namespace(cfg: SFConfig) -> argparse.Namespace:
    """Build the `args` namespace the godot_rl SF wrapper helpers expect.

    main() does NOT call godot_rl's sample_factory_training(); it consumes these fields directly:
    the `_make_nested_godot_env` factory (bound via partial) reads args.env_path / args.speedup /
    args.seed / args.viz, and parse_gdrl_args reads args.experiment_dir / args.experiment_name /
    args.eval. env_path=None => in-editor training: SF opens the server and waits for the Godot client.
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
    # Heavy imports are lazy: only when actually training (keeps the pure helpers import-light).
    from functools import partial

    from sample_factory.envs.env_utils import register_env
    from sample_factory.train import run_rl
    from godot_rl.wrappers.sample_factory_wrapper import parse_gdrl_args

    cfg = parse_args(argv)
    args = _build_args_namespace(cfg)
    extras = build_sf_argv(cfg)
    print(f"SampleFactory training: experiment={cfg.experiment} timesteps={cfg.timesteps} "
          f"base_port={cfg.base_port} client_port={client_port(cfg.base_port)} env_agents={cfg.env_agents}")

    # We can't use godot_rl's sample_factory_training() verbatim: its non-batched env wrapper hands
    # godot_rl's GodotEnv.from_numpy a bare scalar action for a single-key discrete space (chase),
    # which from_numpy then sub-indexes per action key and crashes. Register a module-level factory
    # (_make_nested_godot_env, picklable for SF's spawn-based env-info probe) that nests the scalar
    # before delegating. Same registered env name as godot_rl ("gdrl") so --env=gdrl is unchanged.
    register_env(
        "gdrl",
        partial(_make_nested_godot_env, args.env_path, speedup=args.speedup, seed=args.seed, viz=args.viz),
    )
    final_cfg = parse_gdrl_args(args=args, argv=extras, evaluation=args.eval)
    status = run_rl(final_cfg)
    print(f"SampleFactory finished with status={status}")
    # SF status is an enum; treat the normal-termination value as success (0).
    return int(getattr(status, "value", status) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
