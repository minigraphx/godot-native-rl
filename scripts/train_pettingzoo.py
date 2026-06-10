#!/usr/bin/env python3
"""Multi-policy PPO over the PettingZoo GodotParallelEnv adapter (issue #111).

Demonstrates that our own PettingZoo ParallelEnv (scripts/godot_pettingzoo_env.py) is consumable for
multi-policy training: reads agent_policy_names, routes each agent to its policy, keeps one PPO learner
per distinct name, and exports each actor to TorchScript (+ shape sidecar) for
scripts/export_to_ncnn.py --via torchscript -> native ncnn.

Reuses the proven PPO core from train_cleanrl and the routing/export helpers from
train_hide_seek_multipolicy; the only new code here is the PettingZoo dict<->array glue. Run this
FIRST (opens the server on 11008, waits), THEN launch the Godot scene with --multi-policy. See
scripts/train_pettingzoo.sh. Design:
docs/superpowers/specs/2026-06-09-pettingzoo-multipolicy-interop-design.md

ASSUMPTION (same as train_hide_seek_multipolicy): all policies share one obs + action shape (every
learner is built from agent 0's spaces). True for Hide & Seek. Heavy imports stay lazy in main().
"""
from __future__ import annotations

import argparse
from typing import Dict, NamedTuple, Sequence


def stack_by_agent(per_agent: Dict, agents: Sequence) -> "np.ndarray":
    """Stack a {agent: vector} dict into a (n_agents, dim) array, ordered by `agents`. Lazy numpy.

    Assumes every agent in `agents` is present in `per_agent` (fixed-population semantics; see
    GodotParallelEnv). A missing agent raises KeyError — an upstream contract violation, not silent.
    """
    import numpy as np

    return np.stack([np.asarray(per_agent[a]) for a in agents])


def to_action_dict(full_action: "np.ndarray", agents: Sequence) -> Dict:
    """Scatter a (n_agents, action_dim) array into {agent: row}, ordered by `agents`."""
    return {a: full_action[i] for i, a in enumerate(agents)}


def action_nvec(action_space) -> list:
    """Per-component action sizes from a Tuple(Discrete(n), ...) space -> [n, ...]."""
    from gymnasium import spaces

    if not isinstance(action_space, spaces.Tuple):
        raise ValueError(f"action_nvec expects a Tuple space, got {type(action_space).__name__}")
    return [int(s.n) for s in action_space.spaces]


def require_positive_updates(updates: int, timesteps: int, num_steps: int, n_agents: int) -> int:
    """Fail loud when the update count rounds to 0 (issue #119).

    num_updates = timesteps // (num_steps * n_agents) hits 0 whenever timesteps is small relative
    to the rollout batch — easy on the default 8-tiled parallel scene. The loop would then be
    skipped entirely and a randomly-initialized policy exported, with only a quiet
    "running 0 updates" hint. Raise SystemExit with the actual numbers and the remedies instead.
    """
    if updates <= 0:
        batch = num_steps * n_agents
        raise SystemExit(
            f"train_pettingzoo: 0 updates — timesteps={timesteps} is below one rollout batch "
            f"(num_steps × n_agents = {num_steps} × {n_agents} = {batch}). The trainer would export "
            f"a randomly-initialized policy. Raise TIMESTEPS to at least {batch}, or lower "
            f"--num_steps (NUM_STEPS= via train_pettingzoo.sh), or use a scene with fewer agents."
        )
    return updates


def unwrap_obs(obs_dict: Dict, key: str = "obs") -> Dict:
    """godot_rl exposes Dict obs spaces and returns per-agent Dict obs ({key: vector}); pull the inner
    array so it can be stacked. {agent: {key: vec}} -> {agent: vec}. Mirrors CleanRLGodotEnv's obs[key]
    extraction (single-sensor envs). Multi-sensor obs would concatenate keys — out of scope here."""
    return {agent: per_agent[key] for agent, per_agent in obs_dict.items()}


class Config(NamedTuple):
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
    export_dir: str
    port: int


def parse_args(argv: Sequence[str] | None = None) -> "Config":
    p = argparse.ArgumentParser(allow_abbrev=False, description="Multi-policy PPO over PettingZoo adapter.")
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
    p.add_argument("--export_dir", type=str, default="models")
    p.add_argument("--port", type=int, default=11008)
    a = p.parse_args(argv)
    return Config(
        timesteps=a.timesteps, speedup=a.speedup, action_repeat=a.action_repeat, seed=a.seed,
        num_steps=a.num_steps, learning_rate=a.learning_rate, gamma=a.gamma, gae_lambda=a.gae_lambda,
        update_epochs=a.update_epochs, num_minibatches=a.num_minibatches, clip_coef=a.clip_coef,
        ent_coef=a.ent_coef, vf_coef=a.vf_coef, max_grad_norm=a.max_grad_norm,
        export_dir=a.export_dir, port=a.port,
    )


def main(argv: Sequence[str] | None = None) -> None:
    import pathlib

    import numpy as np
    import torch
    import torch.nn as nn

    from godot_pettingzoo_env import GodotParallelEnv
    from train_hide_seek_multipolicy import (
        export_actor_as_torchscript, policy_index_map, stitch_actions,
    )
    import train_cleanrl as tc

    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cpu")

    env = GodotParallelEnv(
        port=cfg.port, show_window=False, seed=cfg.seed,
        config={"action_repeat": cfg.action_repeat, "speedup": cfg.speedup},
    )
    agents_list = env.possible_agents
    n_agents = len(agents_list)
    obs_space = env.observation_space(agents_list[0])
    obs_keys = list(obs_space.spaces.keys())
    if obs_keys != ["obs"]:
        raise ValueError(
            f"train_pettingzoo supports a single 'obs' sensor; got obs keys {obs_keys}. "
            "unwrap_obs would silently drop the others — concatenate them before training."
        )
    observation_dim = int(obs_space["obs"].shape[0])
    nvec = action_nvec(env.action_space(agents_list[0]))
    total_logits = int(sum(nvec))

    index_map = policy_index_map(env.agent_policy_names)
    print(f"obs_dim={observation_dim} logits={total_logits} nvec={nvec} "
          f"n_agents={n_agents} policies={ {k: len(v) for k, v in index_map.items()} }")

    learners, opts, bufs = {}, {}, {}
    num_steps = cfg.num_steps
    for name, idx in index_map.items():
        np_ = len(idx)
        ag = tc._build_agent(observation_dim, total_logits).to(device)
        learners[name] = ag
        opts[name] = torch.optim.Adam(ag.parameters(), lr=cfg.learning_rate, eps=1e-5)
        bufs[name] = dict(
            obs=torch.zeros((num_steps, np_, observation_dim), device=device),
            actions=torch.zeros((num_steps, np_, len(nvec)), dtype=torch.long, device=device),
            logprobs=torch.zeros((num_steps, np_), device=device),
            rewards=torch.zeros((num_steps, np_), device=device),
            dones=torch.zeros((num_steps, np_), device=device),
            values=torch.zeros((num_steps, np_), device=device),
        )

    try:
        # Guard inside try so env.close() still runs when this exits loud (#119).
        updates = require_positive_updates(
            tc.num_updates(cfg.timesteps, num_steps, n_agents), cfg.timesteps, num_steps, n_agents)
        print(f"running {updates} updates over {n_agents} agents")

        obs_dict, _ = env.reset(seed=cfg.seed)
        next_obs = torch.tensor(stack_by_agent(unwrap_obs(obs_dict), agents_list).astype(np.float32), device=device)
        next_done = torch.zeros(n_agents, device=device)

        def split_t(t):
            return {name: t[idx] for name, idx in index_map.items()}

        for update in range(updates):
            for step in range(num_steps):
                no_split = split_t(next_obs)
                nd_split = split_t(next_done)
                per_policy_action = {}
                for name, idx in index_map.items():
                    ag, b, ob = learners[name], bufs[name], no_split[name]
                    b["obs"][step] = ob
                    b["dones"][step] = nd_split[name]
                    with torch.no_grad():
                        logits = ag.logits(ob)
                        value = ag.value(ob)
                    dists = tc._split_categoricals(logits, nvec)
                    sampled = [d.sample() for d in dists]
                    action = torch.stack(sampled, dim=1)
                    b["actions"][step] = action
                    b["logprobs"][step] = sum(d.log_prob(a) for d, a in zip(dists, sampled))
                    b["values"][step] = value
                    per_policy_action[name] = action.cpu().numpy().astype(np.int64)
                full_action = stitch_actions(per_policy_action, index_map, n_agents)

                obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(to_action_dict(full_action, agents_list))
                reward = stack_by_agent(rew_dict, agents_list).astype(np.float32)
                term = stack_by_agent(term_dict, agents_list).astype(np.float32)
                trunc = stack_by_agent(trunc_dict, agents_list).astype(np.float32)
                done = np.logical_or(term, trunc).astype(np.float32)
                reward_t = torch.tensor(reward, device=device)
                for name, idx in index_map.items():
                    bufs[name]["rewards"][step] = reward_t[idx]
                next_obs = torch.tensor(stack_by_agent(unwrap_obs(obs_dict), agents_list).astype(np.float32), device=device)
                next_done = torch.tensor(done, device=device)

            for name, idx in index_map.items():
                ag, opt, b = learners[name], opts[name], bufs[name]
                np_ = len(idx)
                with torch.no_grad():
                    next_value = ag.value(next_obs[idx])
                adv_np, ret_np = tc.compute_gae(
                    b["rewards"].cpu().numpy(), b["values"].cpu().numpy(), b["dones"].cpu().numpy(),
                    next_value.cpu().numpy(), next_done[idx].cpu().numpy(), cfg.gamma, cfg.gae_lambda)
                advantages = torch.tensor(adv_np, device=device)
                returns = torch.tensor(ret_np, device=device)

                b_obs = b["obs"].reshape(-1, observation_dim)
                b_actions = b["actions"].reshape(-1, len(nvec))
                b_logprobs = b["logprobs"].reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                batch_size = num_steps * np_
                minibatch_size = max(1, batch_size // cfg.num_minibatches)
                b_inds = np.arange(batch_size)
                for _ in range(cfg.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, batch_size, minibatch_size):
                        mb = b_inds[start:start + minibatch_size]
                        logits = ag.logits(b_obs[mb])
                        dists = tc._split_categoricals(logits, nvec)
                        mb_actions = b_actions[mb]
                        new_logprob = sum(d.log_prob(mb_actions[:, i]) for i, d in enumerate(dists))
                        entropy = sum(d.entropy() for d in dists)
                        new_value = ag.value(b_obs[mb])
                        logratio = new_logprob - b_logprobs[mb]
                        ratio = logratio.exp()
                        mb_adv = b_advantages[mb]
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        pg_loss = torch.max(-mb_adv * ratio,
                                            -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                        v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()
                        loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(ag.parameters(), cfg.max_grad_norm)
                        opt.step()

            msg = " ".join(f"{name}_rew={float(bufs[name]['rewards'].mean()):.3f}" for name in index_map)
            print(f"update {update + 1}/{updates} {msg}")

        outdir = pathlib.Path(cfg.export_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        for name in index_map:
            pt_path = outdir / f"pettingzoo_{name}.pt"
            export_actor_as_torchscript(learners[name], observation_dim, pt_path)
            print("Exported TorchScript to:", pt_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()
