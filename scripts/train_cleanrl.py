#!/usr/bin/env python3
"""Train the Chase The Target agent with a single-file CleanRL-style PPO over the godot-rl bridge.

This is a second training backend alongside scripts/train_chase.py (Stable-Baselines3). It uses
godot_rl's CleanRLGodotEnv wrapper, runs the canonical CleanRL PPO loop, and exports the trained
policy to ONNX with godot_rl's `obs`/`state_ins` -> `output`/`state_outs` naming so it flows
unchanged into scripts/export_to_ncnn.py -> native ncnn deploy.

Run this FIRST (it opens the server on port 11008 and waits), THEN launch the Godot training scene
which connects as the client. See scripts/train_cleanrl.sh for orchestration.

Design: docs/superpowers/specs/2026-06-02-cleanrl-backend-design.md

The deploy path (addons/.../action_decode.gd) decodes a discrete action by argmax over the policy
output's per-key logit segment, so the exported ONNX output is the raw action logits (length
sum(nvec)) — exactly like the SB3 exporter's action_net output.

Convention: heavy imports (torch / numpy / gymnasium / godot_rl) are LAZY (inside main() or the
helper that needs them) so the pure helpers below stay unit-testable without those deps installed.
Do NOT pass seed= to anything that calls env.seed() — seed via the env constructor only.
"""
from __future__ import annotations

import argparse
from typing import NamedTuple, Sequence


class PPOConfig(NamedTuple):
    """Immutable PPO hyperparameters + IO paths (built from argv by parse_args)."""

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
    save_model_path: str
    onnx_export_path: str


def parse_args(argv: Sequence[str] | None = None) -> PPOConfig:
    """Parse argv into an immutable PPOConfig. Raises SystemExit on unknown args (argparse)."""
    p = argparse.ArgumentParser(allow_abbrev=False, description="CleanRL single-file PPO for chase.")
    p.add_argument("--timesteps", type=int, default=300_000)
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
    p.add_argument("--save_model_path", type=str, default="models/chase_cleanrl_policy.pt")
    p.add_argument("--onnx_export_path", type=str, default="models/chase_cleanrl_policy.onnx")
    a = p.parse_args(argv)
    return PPOConfig(
        timesteps=a.timesteps,
        speedup=a.speedup,
        action_repeat=a.action_repeat,
        seed=a.seed,
        num_steps=a.num_steps,
        learning_rate=a.learning_rate,
        gamma=a.gamma,
        gae_lambda=a.gae_lambda,
        update_epochs=a.update_epochs,
        num_minibatches=a.num_minibatches,
        clip_coef=a.clip_coef,
        ent_coef=a.ent_coef,
        vf_coef=a.vf_coef,
        max_grad_norm=a.max_grad_norm,
        save_model_path=a.save_model_path,
        onnx_export_path=a.onnx_export_path,
    )


def discrete_action_dims(nvec: Sequence[int]) -> tuple[int, list[int]]:
    """Map a MultiDiscrete nvec to (total_logits, [n0, n1, ...]).

    The actor head emits `total_logits = sum(nvec)` values: one contiguous logit segment per
    discrete sub-action, which the deploy-side ActionDecode argmaxes per segment. Raises
    ValueError on an empty nvec or any non-positive entry (a malformed action space).
    """
    dims = [int(n) for n in nvec]
    if len(dims) == 0:
        raise ValueError("discrete_action_dims: empty nvec (no discrete actions)")
    if any(n <= 0 for n in dims):
        raise ValueError(f"discrete_action_dims: non-positive entry in nvec {dims}")
    return sum(dims), dims


def obs_dim(observation_space) -> int:
    """Observation vector length from a Box-like space (last/only dim of .shape)."""
    return int(observation_space.shape[-1])


def act_layout(action_space) -> tuple[int, list[int]]:
    """Total actor logits + per-head sizes from a MultiDiscrete-like action space (.nvec)."""
    return discrete_action_dims(list(action_space.nvec))


def num_updates(total_timesteps: int, num_steps: int, num_envs: int) -> int:
    """Number of PPO updates: floor(total_timesteps / (num_steps * num_envs)), never negative."""
    batch_size = num_steps * num_envs
    if batch_size <= 0:
        raise ValueError("num_updates: num_steps and num_envs must be positive")
    return max(0, total_timesteps // batch_size)


def compute_gae(rewards, values, dones, next_value, next_done, gamma: float, gae_lambda: float):
    """Generalized Advantage Estimation. Pure numpy; lazy numpy import keeps the module light.

    Shapes: rewards/values/dones are (num_steps, num_envs); next_value/next_done are (num_envs,).
    `dones[t]` marks whether each env was terminal at the START of step t (CleanRL convention), so
    the bootstrap into step t reads `1 - dones[t+1]`. Returns (advantages, returns), both
    (num_steps, num_envs); returns = advantages + values.
    """
    import numpy as np

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    next_value = np.asarray(next_value, dtype=np.float32)
    next_done = np.asarray(next_done, dtype=np.float32)

    num_steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    lastgaelam = np.zeros_like(next_value)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def layer_init(layer, std: float = 2.0 ** 0.5, bias_const: float = 0.0):
    """Orthogonal weight init + constant bias (CleanRL default). Returns the layer. Lazy torch."""
    import torch.nn as nn

    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _build_agent(observation_dim: int, total_logits: int):
    """Construct the actor/critic nn.Module. Lazy torch import (called from main only)."""
    import torch.nn as nn

    class Agent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Sequential(
                layer_init(nn.Linear(observation_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
            )
            self.actor = layer_init(nn.Linear(64, total_logits), std=0.01)
            self.critic = layer_init(nn.Linear(64, 1), std=1.0)

        def logits(self, obs):
            return self.actor(self.shared(obs))

        def value(self, obs):
            return self.critic(self.shared(obs)).squeeze(-1)

    return Agent()


def _split_categoricals(logits, nvec):
    """Yield one torch Categorical per discrete head over the flat logits. Lazy torch."""
    import torch
    from torch.distributions.categorical import Categorical

    dists = []
    offset = 0
    for n in nvec:
        dists.append(Categorical(logits=logits[:, offset:offset + n]))
        offset += n
    return dists


def export_actor_as_onnx(agent, observation_dim: int, path: str) -> None:
    """Export the actor head (raw action logits) to ONNX with godot_rl's IO naming.

    forward(obs, state_ins) -> (logits, state_ins). input_names=["obs","state_ins"],
    output_names=["output","state_outs"] so scripts/export_to_ncnn.py derives the inputshape and
    the in0/out0 parity check work unchanged. Output = raw logits (length sum(nvec)); the deploy
    ActionDecode argmaxes per segment.
    """
    import torch
    import torch.nn as nn

    class OnnxableActor(nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, obs, state_ins):
            return self.inner.logits(obs), state_ins

    onnxable = OnnxableActor(agent).to("cpu").eval()
    dummy_obs = torch.zeros(1, observation_dim)
    torch.onnx.export(
        onnxable,
        args=(dummy_obs, torch.zeros(1).float()),
        f=path,
        opset_version=17,
        input_names=["obs", "state_ins"],
        output_names=["output", "state_outs"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "state_ins": {0: "batch_size"},
            "output": {0: "batch_size"},
            "state_outs": {0: "batch_size"},
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    import pathlib

    import numpy as np
    import torch
    import torch.nn as nn

    from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv

    cfg = parse_args(argv)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cpu")

    # env_path=None => in-editor training: opens the server and waits for a Godot client.
    # The seed is passed only through the constructor (CleanRLGodotEnv never calls env.seed()).
    env = CleanRLGodotEnv(
        env_path=None,
        show_window=False,
        seed=cfg.seed,
        n_parallel=1,
        speedup=cfg.speedup,
        action_repeat=cfg.action_repeat,
    )

    n_envs = env.num_envs
    observation_dim = obs_dim(env.single_observation_space)
    total_logits, nvec = act_layout(env.single_action_space)
    print(f"obs_dim={observation_dim} action_logits={total_logits} nvec={nvec} num_envs={n_envs}")

    agent = _build_agent(observation_dim, total_logits).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    num_steps = cfg.num_steps
    batch_size = num_steps * n_envs
    minibatch_size = max(1, batch_size // cfg.num_minibatches)
    updates = num_updates(cfg.timesteps, num_steps, n_envs)
    print(f"running {updates} updates (batch_size={batch_size}, minibatch_size={minibatch_size})")

    # Rollout storage.
    obs_buf = torch.zeros((num_steps, n_envs, observation_dim), device=device)
    actions_buf = torch.zeros((num_steps, n_envs, len(nvec)), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((num_steps, n_envs), device=device)
    rewards_buf = torch.zeros((num_steps, n_envs), device=device)
    dones_buf = torch.zeros((num_steps, n_envs), device=device)
    values_buf = torch.zeros((num_steps, n_envs), device=device)

    next_obs_np, _ = env.reset(cfg.seed)
    next_obs = torch.tensor(np.asarray(next_obs_np, dtype=np.float32), device=device)
    next_done = torch.zeros(n_envs, device=device)

    for update in range(updates):
        for step in range(num_steps):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                logits = agent.logits(next_obs)
                value = agent.value(next_obs)
            dists = _split_categoricals(logits, nvec)
            sampled = [d.sample() for d in dists]
            action = torch.stack(sampled, dim=1)  # (n_envs, len(nvec))
            logprob = sum(d.log_prob(a) for d, a in zip(dists, sampled))

            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value

            # Integer actions => the wrapper's to_original_dist takes the discrete branch.
            action_np = action.cpu().numpy().astype(np.int64)
            next_obs_np, reward, terminations, truncations, _ = env.step(action_np)
            done = np.logical_or(np.asarray(terminations), np.asarray(truncations)).astype(np.float32)

            rewards_buf[step] = torch.tensor(np.asarray(reward, dtype=np.float32), device=device)
            next_obs = torch.tensor(np.asarray(next_obs_np, dtype=np.float32), device=device)
            next_done = torch.tensor(done, device=device)

        with torch.no_grad():
            next_value = agent.value(next_obs)
        advantages_np, returns_np = compute_gae(
            rewards_buf.cpu().numpy(),
            values_buf.cpu().numpy(),
            dones_buf.cpu().numpy(),
            next_value.cpu().numpy(),
            next_done.cpu().numpy(),
            cfg.gamma,
            cfg.gae_lambda,
        )
        advantages = torch.tensor(advantages_np, device=device)
        returns = torch.tensor(returns_np, device=device)

        # Flatten the batch.
        b_obs = obs_buf.reshape(-1, observation_dim)
        b_actions = actions_buf.reshape(-1, len(nvec))
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        b_inds = np.arange(batch_size)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]

                logits = agent.logits(b_obs[mb_inds])
                dists = _split_categoricals(logits, nvec)
                mb_actions = b_actions[mb_inds]
                new_logprob = sum(d.log_prob(mb_actions[:, i]) for i, d in enumerate(dists))
                entropy = sum(d.entropy() for d in dists)
                new_value = agent.value(b_obs[mb_inds])

                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        steps_done = (update + 1) * batch_size
        print(f"update {update + 1}/{updates} steps={steps_done} "
              f"mean_reward={float(rewards_buf.mean()):.4f} value_loss={float(v_loss.detach()):.4f}")

    # Save the torch policy and export the actor logits to ONNX for the ncnn deploy pipeline.
    pt_path = pathlib.Path(cfg.save_model_path)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), pt_path)
    print("Saved torch policy to:", pt_path)

    onnx_path = pathlib.Path(cfg.onnx_export_path).with_suffix(".onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_actor_as_onnx(agent, observation_dim, str(onnx_path))
    print("Exported ONNX to:", onnx_path)

    env.close()


if __name__ == "__main__":
    main()
