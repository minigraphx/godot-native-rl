#!/usr/bin/env python3
"""Train Hide & Seek with TWO distinct policies (seeker + hider) over the godot-rl bridge.

A custom single-file multi-policy PPO (sibling of scripts/train_cleanrl.py). CleanRLGodotEnv
vectorizes over the N Godot agents as N parallel envs; this trainer reads agent_policy_names, routes
each agent index to its policy, maintains one PPO learner per distinct name, and exports each actor
to ONNX (obs/state_ins -> output/state_outs) for scripts/export_to_ncnn.py -> native ncnn.

Run this FIRST (opens the server on 11008, waits), THEN launch the Godot scene with --multi-policy.
See scripts/train_hide_seek_multipolicy.sh. Design:
docs/superpowers/specs/2026-06-05-multi-policy-trained-example-design.md

Heavy imports (torch/numpy/godot_rl) are LAZY so the pure helpers stay unit-testable. Pure PPO
helpers (compute_gae, num_updates, layer_init) + the ONNX exporter are reused from train_cleanrl.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, NamedTuple, Sequence


def policy_index_map(agent_policy_names: Sequence[str]) -> Dict[str, List[int]]:
    """Map each distinct policy name to the agent indices using it (first-seen key order,
    ascending indices). e.g. ["seeker","hider","seeker"] -> {"seeker":[0,2],"hider":[1]}."""
    out: Dict[str, List[int]] = {}
    for i, name in enumerate(agent_policy_names):
        out.setdefault(name, []).append(i)
    return out


def split_by_policy(batched, index_map: Dict[str, List[int]]):
    """Slice a (n_agents, ...) array into {name: array[indices]} per policy. Lazy numpy."""
    import numpy as np

    arr = np.asarray(batched)
    return {name: arr[idx] for name, idx in index_map.items()}


def stitch_actions(per_policy_actions, index_map: Dict[str, List[int]], n_agents: int):
    """Inverse of split_by_policy for actions: scatter each policy's (n_p, action_dim) actions back
    into a single (n_agents, action_dim) int64 array in agent order. Lazy numpy."""
    import numpy as np

    first = next(iter(per_policy_actions.values()))
    action_dim = int(np.asarray(first).shape[1])
    out = np.zeros((n_agents, action_dim), dtype=np.int64)
    for name, idx in index_map.items():
        out[idx] = np.asarray(per_policy_actions[name])
    return out


class MultiPolicyConfig(NamedTuple):
    timesteps: int
    speedup: int
    action_repeat: int
    seed: int
    num_steps: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    update_epochs: int
    num_minibatches: int
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    onnx_export_dir: str
    policy_names: tuple  # expected names, for a fail-fast sanity check against the wire


def parse_args(argv: Sequence[str] | None = None) -> "MultiPolicyConfig":
    p = argparse.ArgumentParser(allow_abbrev=False, description="Multi-policy PPO for hide & seek.")
    p.add_argument("--timesteps", type=int, default=800_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--onnx_export_dir", type=str, default="models")
    a = p.parse_args(argv)
    return MultiPolicyConfig(
        timesteps=a.timesteps, speedup=a.speedup, action_repeat=a.action_repeat, seed=a.seed,
        num_steps=a.num_steps, learning_rate=a.learning_rate, gamma=a.gamma,
        gae_lambda=a.gae_lambda, update_epochs=a.update_epochs, num_minibatches=a.num_minibatches,
        clip_coef=a.clip_coef, ent_coef=a.ent_coef, vf_coef=a.vf_coef,
        max_grad_norm=a.max_grad_norm, onnx_export_dir=a.onnx_export_dir,
        policy_names=("seeker", "hider"),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Training loop — to be implemented in a later task."""
    raise NotImplementedError("main() training loop not yet implemented")


if __name__ == "__main__":
    main()
