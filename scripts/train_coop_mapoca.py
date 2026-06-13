#!/usr/bin/env python3
"""MA-POCA-style cooperative trainer for coop_collect (#30 M2) over godot_rl's CleanRLGodotEnv.

Single-file, in the spirit of train_cleanrl.py, with the multi-agent-credit pieces MA-POCA adds:

  * a SHARED decentralized actor (parameter sharing) — every team agent runs the same policy on its
    own local obs, and only this actor is exported to ncnn (the critic is training-only);
  * a CENTRALIZED critic: a permutation-invariant attention encoder over the whole team's
    observations emits one shared team value V(s) (lower-variance returns than a decentralized
    value — the main win over the M1 shared-PPO baseline);
  * a per-agent COUNTERFACTUAL baseline (leave-one-out team value): baseline_a = V(team minus a), so
    agent a's advantage A_a = return - baseline_a isolates a's marginal contribution to the shared
    reward. (The full action-marginal counterfactual is a documented M2.1 refinement; leave-one-out
    is the stable, defensible baseline that gets M2 learning.)

The pure credit/masking math is in coop_mapoca.py (unit-tested). Run via scripts/train_coop_mapoca.sh.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import NamedTuple, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import coop_mapoca as mp  # noqa: E402  (numpy-only at import; torch stays lazy)


class Config(NamedTuple):
    timesteps: int
    team_size: int
    num_steps: int
    num_minibatches: int
    update_epochs: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    learning_rate: float
    max_grad_norm: float
    speedup: int
    action_repeat: int
    seed: int
    save_model_path: str
    pt_export_path: str


def parse_args(argv: Sequence[str] | None = None) -> Config:
    p = argparse.ArgumentParser(allow_abbrev=False, description="MA-POCA coop trainer (#30 M2).")
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--team-size", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=256)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action-repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-model-path", type=str, default="models/coop_mapoca.pt")
    p.add_argument("--pt-export-path", type=str, default="models/coop_mapoca.pt")
    a = p.parse_args(argv)
    return Config(
        timesteps=a.timesteps, team_size=a.team_size, num_steps=a.num_steps,
        num_minibatches=a.num_minibatches, update_epochs=a.update_epochs, gamma=a.gamma,
        gae_lambda=a.gae_lambda, clip_coef=a.clip_coef, ent_coef=a.ent_coef, vf_coef=a.vf_coef,
        learning_rate=a.learning_rate, max_grad_norm=a.max_grad_norm, speedup=a.speedup,
        action_repeat=a.action_repeat, seed=a.seed, save_model_path=a.save_model_path,
        pt_export_path=a.pt_export_path)


def layer_init(layer, std: float = 2.0 ** 0.5, bias_const: float = 0.0):
    import torch.nn as nn

    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def build_actor(obs_dim: int, n_actions: int):
    """Shared decentralized actor: local obs -> action logits. The only net exported to ncnn."""
    import torch.nn as nn

    class Actor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
                layer_init(nn.Linear(64, 64)), nn.Tanh(),
            )
            self.head = layer_init(nn.Linear(64, n_actions), std=0.01)

        def forward(self, obs):  # obs (..., obs_dim) -> logits (..., n_actions)
            return self.head(self.body(obs))

    return Actor()


def build_critic(obs_dim: int):
    """Centralized critic: permutation-invariant attention over a team's per-agent obs -> value.

    Encodes each agent's obs to an entity embedding, self-attends across the team (so it generalizes
    over team size and can mask absent agents for M3), mean-pools, and reads out one scalar value.
    Calling it on a team subset (leave-one-out) yields the counterfactual baseline.
    """
    import torch
    import torch.nn as nn

    class Critic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh())
            self.attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
            self.value = nn.Sequential(
                layer_init(nn.Linear(64, 64)), nn.Tanh(), layer_init(nn.Linear(64, 1), std=1.0))

        def forward(self, team_obs, key_padding_mask=None):
            # team_obs: (batch, team, obs_dim); key_padding_mask: (batch, team) True = absent (M3).
            e = self.embed(team_obs)
            a, _ = self.attn(e, e, e, key_padding_mask=key_padding_mask)
            h = e + a
            if key_padding_mask is not None:
                keep = (~key_padding_mask).float().unsqueeze(-1)
                pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
            else:
                pooled = h.mean(1)
            return self.value(pooled).squeeze(-1)  # (batch,)

    return Critic()


def export_actor_torchscript(actor, obs_dim: int, pt_path: pathlib.Path):
    """Trace the shared actor to TorchScript + a shape sidecar, ready for export_to_ncnn.py."""
    import torch
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from export_to_ncnn import write_shape_sidecar

    actor.eval()
    dummy = torch.zeros(1, obs_dim)
    with torch.no_grad():
        scripted = torch.jit.trace(actor, dummy)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(pt_path))
    write_shape_sidecar(pt_path, [1, obs_dim])


def main(argv: Sequence[str] | None = None) -> None:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.distributions.categorical import Categorical
    from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv

    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cpu")

    env = CleanRLGodotEnv(env_path=None, show_window=False, seed=cfg.seed, n_parallel=1,
                          speedup=cfg.speedup, action_repeat=cfg.action_repeat)
    n_agents = env.num_envs
    n_teams = mp.validate_team_layout(n_agents, cfg.team_size)
    slices = mp.team_slices(n_agents, cfg.team_size)

    obs_space = env.single_observation_space
    obs_dim = int(np.prod(obs_space.shape))
    act_space = env.single_action_space
    n_actions = int(getattr(act_space, "n", None) or act_space.nvec[0])
    print(f"obs_dim={obs_dim} n_actions={n_actions} n_agents={n_agents} "
          f"team_size={cfg.team_size} n_teams={n_teams}")

    actor = build_actor(obs_dim, n_actions).to(device)
    critic = build_critic(obs_dim).to(device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                           lr=cfg.learning_rate, eps=1e-5)

    S, A, T, K = cfg.num_steps, n_agents, n_teams, cfg.team_size
    obs_buf = torch.zeros((S, A, obs_dim), device=device)
    act_buf = torch.zeros((S, A), dtype=torch.long, device=device)
    logp_buf = torch.zeros((S, A), device=device)
    rew_buf = torch.zeros((S, T), device=device)         # one shared reward per team
    done_buf = torch.zeros((S, T), device=device)
    val_buf = torch.zeros((S, T), device=device)
    base_buf = torch.zeros((S, T, K), device=device)     # per-agent leave-one-out baseline

    def team_obs_tensor(flat_obs):  # (A, obs_dim) -> (T, K, obs_dim)
        return flat_obs.view(T, K, obs_dim)

    def critic_value(team_obs):  # (T, K, obs_dim) -> (T,)
        return critic(team_obs)

    def loo_baselines(team_obs):  # leave-one-out: (T, K) per-agent baseline
        outs = []
        for a in range(K):
            keep = [j for j in range(K) if j != a]
            outs.append(critic(team_obs[:, keep, :]))
        return torch.stack(outs, dim=1)  # (T, K)

    next_obs_np, _ = env.reset(cfg.seed)
    next_obs = torch.tensor(np.asarray(next_obs_np, dtype=np.float32), device=device).reshape(A, obs_dim)
    next_done = torch.zeros(T, device=device)

    batch_size = S * T
    minibatch_size = max(1, batch_size // cfg.num_minibatches)
    updates = max(1, cfg.timesteps // (S * A))
    print(f"running {updates} updates (team batch_size={batch_size}, minibatch={minibatch_size})")

    global_step = 0
    ep_returns: list[float] = []
    ep_accum = np.zeros(T, dtype=np.float32)  # running team-return accumulator (per team)
    _grouping_verified = [False]  # one-shot world-major team-grouping check (see the rollout loop)
    for update in range(1, updates + 1):
        for step in range(S):
            global_step += A
            obs_buf[step] = next_obs
            t_obs = team_obs_tensor(next_obs)
            with torch.no_grad():
                logits = actor(next_obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp_buf[step] = dist.log_prob(action)
                val_buf[step] = critic_value(t_obs)
                base_buf[step] = loo_baselines(t_obs)
            act_buf[step] = action
            done_buf[step] = next_done

            # The wrapper indexes action[:, head], so it needs (n_agents, n_action_heads) = (A, 1).
            obs2, reward, term, trunc, _ = env.step(action.cpu().numpy().astype(np.int64).reshape(A, 1))
            done = np.logical_or(np.asarray(term), np.asarray(trunc)).astype(np.float32)
            next_obs = torch.tensor(np.asarray(obs2, dtype=np.float32), device=device).reshape(A, obs_dim)
            # All agents on a team share the IDENTICAL per-frame team reward (coop_collect's defining
            # property). Use that to verify the flat-slot -> team grouping is correct on the first
            # nonzero-reward step: if the scene's AGENT order isn't world-major, teammates would be
            # from different worlds and their rewards would differ. Fail loud rather than train a
            # centralized critic on mis-grouped teams. (Checked once to keep the hot loop clean.)
            rew_grid = np.asarray(reward, dtype=np.float32).reshape(T, K)
            if not _grouping_verified[0] and np.abs(rew_grid).sum() > 0:
                spread = float(np.abs(rew_grid - rew_grid[:, :1]).max())
                if spread > 1e-6:
                    raise RuntimeError(
                        f"team grouping looks wrong: teammates' shared reward differs by {spread} "
                        f"(team_size={K}). The scene's AGENT order may not be world-major; use the "
                        f"single-world scene or fix --team-size.")
                _grouping_verified[0] = True
            rew_team = rew_grid[:, 0]
            done_team = done.reshape(T, K)[:, 0]
            rew_buf[step] = torch.tensor(rew_team, device=device)
            next_done = torch.tensor(done_team, device=device)
            # Per-team episode-return bookkeeping: accumulate, flush on done.
            ep_accum += rew_team
            for ti in range(T):
                if done_team[ti] > 0:
                    ep_returns.append(float(ep_accum[ti]))
                    ep_accum[ti] = 0.0

        with torch.no_grad():
            next_value = critic_value(team_obs_tensor(next_obs)).cpu().numpy()
        adv_team, ret_team = mp.compute_gae(
            rew_buf.cpu().numpy(), val_buf.cpu().numpy(), done_buf.cpu().numpy(),
            next_value, next_done.cpu().numpy(), cfg.gamma, cfg.gae_lambda)
        # Per-agent counterfactual advantage = team_return - leave-one-out baseline.
        adv_agent = mp.counterfactual_advantage(ret_team, base_buf.cpu().numpy())  # (S,T,K)
        ret_team_t = torch.tensor(ret_team, device=device)
        adv_agent_t = torch.tensor(mp.normalize(adv_agent).reshape(S, A), device=device)

        b_obs = obs_buf.reshape(S * A, obs_dim)
        b_act = act_buf.reshape(S * A)
        b_logp = logp_buf.reshape(S * A)
        b_adv = adv_agent_t.reshape(S * A)
        b_team_obs = obs_buf.reshape(S * T, K, obs_dim)
        b_ret = ret_team_t.reshape(S * T)

        agent_idx = np.arange(S * A)
        team_idx = np.arange(S * T)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(agent_idx)
            for start in range(0, S * A, minibatch_size * K):
                mb = agent_idx[start:start + minibatch_size * K]
                logits = actor(b_obs[mb])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[mb])
                ratio = (new_logp - b_logp[mb]).exp()
                a_mb = b_adv[mb]
                l1 = -a_mb * ratio
                l2 = -a_mb * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(l1, l2).mean()
                ent = dist.entropy().mean()
                actor_loss = pg_loss - cfg.ent_coef * ent
                # Critic update on team minibatch (centralized value -> team return).
                np.random.shuffle(team_idx)
                tmb = team_idx[:minibatch_size]
                v = critic(b_team_obs[tmb])
                v_loss = ((v - b_ret[tmb]) ** 2).mean()
                loss = actor_loss + cfg.vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()),
                                         cfg.max_grad_norm)
                opt.step()

        if ep_returns:
            recent = ep_returns[-50:]
            print(f"update {update}/{updates} global_step {global_step} "
                  f"ep_rew_mean {sum(recent)/len(recent):.3f}")
        else:
            print(f"update {update}/{updates} global_step {global_step} (no completed episodes yet)")

    zip_path = pathlib.Path(cfg.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict()}, zip_path)
    print("Saved actor state to:", zip_path)
    pt_path = pathlib.Path(cfg.pt_export_path).with_suffix(".pt")
    export_actor_torchscript(actor, obs_dim, pt_path)
    print("Exported TorchScript actor to:", pt_path)
    print("Convert with: export_to_ncnn.py", pt_path)
    env.close()


if __name__ == "__main__":
    main()
