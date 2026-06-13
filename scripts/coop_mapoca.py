#!/usr/bin/env python3
"""Pure, torch-free helpers for the MA-POCA cooperative trainer (#30 M2/M3).

The credit-assignment + masking math lives here so it can be unit-tested without torch or a Godot
socket (mirrors how `train_cleanrl.py` keeps `compute_gae` / `num_updates` pure). The trainer
(`train_coop_mapoca.py`) imports these and adds the torch pieces (attention critic, actors).

MA-POCA in one paragraph: a *centralized* critic sees the whole team's observations and estimates
the shared return V(s); a per-agent *counterfactual baseline* Q_a marginalizes agent a's own action
out of that value, so each agent's advantage A_a = return - Q_a isolates the credit for *its* action
from the team reward. Posthumous credit (M3) falls out when the critic masks agents that have already
left the episode, so their earlier actions still get credit for the team's later reward.
"""
from __future__ import annotations

from typing import Sequence


def validate_team_layout(num_envs: int, team_size: int) -> int:
    """Return the number of teams given `num_envs` flat agent slots and `team_size`.

    CleanRLGodotEnv presents every training agent as one flat vector-env slot. For coop_collect a
    "team" is `team_size` consecutive slots (one tiled world). Raises ValueError unless team_size
    divides num_envs evenly and both are positive.
    """
    if num_envs <= 0 or team_size <= 0:
        raise ValueError(f"num_envs and team_size must be positive, got {num_envs}, {team_size}")
    if num_envs % team_size != 0:
        raise ValueError(
            f"num_envs ({num_envs}) must be a multiple of team_size ({team_size}); "
            "check the scene's agent count vs the trainer's --team-size"
        )
    return num_envs // team_size


def team_slices(num_envs: int, team_size: int) -> list[slice]:
    """The flat-slot slices, one per team. `team_slices(4, 2) -> [slice(0,2), slice(2,4)]`.

    Assumes world-major ordering (team 0's agents first, then team 1's, ...). The training scene
    must add agents to the AGENT group world-by-world for this to hold — asserted by the smoke.
    """
    n_teams = validate_team_layout(num_envs, team_size)
    return [slice(t * team_size, (t + 1) * team_size) for t in range(n_teams)]


def compute_gae(rewards, values, dones, next_value, next_done, gamma: float, gae_lambda: float):
    """Generalized Advantage Estimation over the *shared team return*. Pure numpy.

    Identical convention to train_cleanrl.compute_gae: shapes (num_steps, n) with `dones[t]` the
    start-of-step terminal flag. Here `n` indexes teams (one shared value/return per team), not
    individual agents — the centralized critic emits one value per team.
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


def counterfactual_advantage(team_returns, baselines):
    """Per-agent MA-POCA advantage A_a = team_return - counterfactual_baseline_a.

    `team_returns` is (num_steps, n_teams); `baselines` is (num_steps, n_teams, team_size) — the
    critic's per-agent counterfactual value (the team value with agent a's action marginalized out).
    Broadcasts the shared return across the team axis. Returns (num_steps, n_teams, team_size).
    """
    import numpy as np

    team_returns = np.asarray(team_returns, dtype=np.float32)
    baselines = np.asarray(baselines, dtype=np.float32)
    if baselines.ndim != 3:
        raise ValueError(f"baselines must be (steps, teams, team_size), got {baselines.shape}")
    if team_returns.shape != baselines.shape[:2]:
        raise ValueError(
            f"team_returns {team_returns.shape} must match baselines[:2] {baselines.shape[:2]}"
        )
    return team_returns[..., None] - baselines


def normalize(adv):
    """Standardize advantages to zero-mean/unit-std (PPO norm). Pure numpy; safe on tiny std."""
    import numpy as np

    adv = np.asarray(adv, dtype=np.float32)
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def alive_mask(dones_per_agent):
    """Posthumous-credit mask (M3): 1.0 while an agent is present, 0.0 once it has left.

    `dones_per_agent` is a boolean/0-1 array (..., team_size) marking the step an agent finished
    early. The mask stays 0 for every later step (an agent that banked and left does not re-enter),
    so the centralized critic ignores absent agents when pooling the team. Pure numpy.
    """
    import numpy as np

    d = np.asarray(dones_per_agent, dtype=np.float32)
    left = np.cumsum(d, axis=0) > 0  # True from the finishing step onward
    return (~left).astype(np.float32)


def masked_mean(values, mask):
    """Mean of `values` over entries where `mask` is nonzero; 0.0 if the mask is all-zero. Pure."""
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    denom = mask.sum()
    if denom == 0:
        return 0.0
    return float((values * mask).sum() / denom)
